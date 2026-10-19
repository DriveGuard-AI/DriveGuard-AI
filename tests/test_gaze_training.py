"""Synthetic joint-training contracts; these tests make no DMD quality claims."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from training.common.config import TASKS, canonical_hash
from training.common.checkpoint import load_checkpoint,save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def gaze_fixture(root, config):
    root = Path(root)
    root.mkdir(parents=True)
    names = config['data']['feature_names']
    manifest = {'schema_version':'driveguard-feature-manifest-v3', 'dataset_version':'synthetic-gaze-test',
        'feature_schema':{'version':'driveguard-interaction-v2','names':names,'sample_rate_hz':25},
        'extractor_profile':{'name':'synthetic-only','synthetic':True,'feature_definitions':dict.fromkeys(names,'synthetic test value')},
        'label_mapping':{'version':'driveguard-behavior-v1','tasks':list(TASKS),'rules':{'test':'synthetic +/-'}},
        'gaze_mapping':{'version':'driveguard-gaze-v1','classes':config['data']['gaze_classes'],'rules':{'test':'synthetic nine classes'}},
        'group_profile':config['data']['group_profile'], 'group_profile_sha256':canonical_hash(config['data']['group_profile']), 'records':[]}
    rng = np.random.default_rng(42)
    for split in ('train','val'):
        n = 300
        record = {'record_id':split,'subject_id':split,'recording_id':split,'view':'face','subset':'synthetic','split':split}
        arrays = {'values':rng.normal(size=(n,23)).astype(np.float32), 'masks':np.ones((n,23),np.uint8),
            'targets':np.repeat((np.arange(n)//25 % 2)[:,None],4,axis=1).astype(np.float32),
            'label_masks':np.ones((n,4),np.uint8),'expert_age_ms':np.zeros((n,3),np.float32),
            'gaze_targets':((np.arange(n)//25-3)%9).astype(np.int64),'gaze_label_masks':np.ones(n,np.uint8)}
        for name, array in arrays.items():
            record[name] = f'{split}_{name}.npy'
            np.save(root/record[name],array)
        manifest['records'].append(record)
    path = root/'manifest.json'
    path.write_text(json.dumps(manifest),encoding='utf-8')
    return path, manifest


class GazeTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self):
        self.assertTrue((ROOT/'configs/training/relation_gaze.json').exists(), 'joint config missing')
        from training.gaze.config import load_config
        config = load_config()
        config['model'].update(channels=[4],dilations=[1],projection_dim=4,gate_hidden_dim=4)
        config['training'].update(epochs=1,batch_size=3)
        return config

    def test_config_shape_order_and_full_tasks(self):
        c = self.config()
        from training.gaze.config import validate_config
        from training.gaze.profile import FEATURE_NAMES, GAZE_CLASSES
        self.assertEqual(len(FEATURE_NAMES),23)
        self.assertEqual(c['model']['input_channels'],46)
        self.assertEqual(len(GAZE_CLASSES),9)
        self.assertEqual(c['training']['active_tasks'],list(TASKS))
        for mutate in (lambda x:x['data']['feature_names'].reverse(), lambda x:x['data']['gaze_classes'].reverse(),
                       lambda x:x['training'].update(active_tasks=['fatigue']),lambda x:x['training'].update(gaze_enabled=False)):
            bad = deepcopy(c); mutate(bad)
            with self.assertRaises(ValueError): validate_config(bad)

    def test_gaze_shared_gradient_and_availability(self):
        c = self.config()
        from training.common.model_factory import build_model
        model = build_model(c)
        x = torch.randn(3,100,46); x[:,:,23:] = 1
        inputs = {'x':x,'quality':torch.ones(3,4,2),'eligible':torch.ones(3,4,dtype=torch.bool)}
        out = model(**inputs)
        self.assertEqual(tuple(out['behavior_logits'].shape),(3,4))
        self.assertEqual(tuple(out['gaze_logits'].shape),(3,9))
        out['gaze_logits'].sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.blocks.parameters() if p.grad is not None),0)
        self.assertGreater(float(model.projections[0][0].weight.grad.abs().sum()),0)
        self.assertIsNone(model.heads[0].weight.grad)
        x[0,:,23+19:] = 0
        x[1,-1,23+5] = 0
        x[2,:51,23+19:] = 0
        self.assertEqual(model(**inputs)['gaze_available'].tolist(),[False]*3)
        x[:,:,23:] = 1; inputs['eligible'][:,0] = False
        self.assertFalse(model(**inputs)['gaze_available'].any())

    def test_loss_masks_task_means_and_missing_gradients(self):
        self.config()
        from training.gaze.losses import joint_loss
        b = torch.zeros(2,4,requires_grad=True); g = torch.zeros(2,9,requires_grad=True)
        target = torch.tensor([[1.,0,0,0],[0,0,0,0]])
        mask = torch.tensor([[1,0,0,0],[1,0,0,0]],dtype=torch.bool)
        gt = torch.tensor([2,0]); gm = torch.tensor([True,False])
        loss,sums,counts = joint_loss(b,g,target,mask,gt,gm)
        self.assertAlmostEqual(float(loss.detach()), (np.log(2)+np.log(9))/2,places=6)
        self.assertEqual(counts.tolist(),[2,0,0,0,1])
        loss.backward(); self.assertTrue((b.grad[:,1:] == 0).all()); self.assertTrue((g.grad[1] == 0).all())
        g.grad = None; b.grad = None
        loss,_,_ = joint_loss(b,g,target,mask,torch.zeros(2,dtype=torch.int64),torch.zeros(2,dtype=torch.bool))
        loss.backward(); self.assertIsNone(g.grad)
        loss,_,counts = joint_loss(b,g,torch.zeros_like(target),torch.zeros_like(mask),torch.zeros_like(gt),torch.zeros_like(gm))
        self.assertIsNone(loss); self.assertEqual(counts.sum(),0)

    def test_corpus_validation_coverage_and_train_only_normalization(self):
        c = self.config()
        from training.gaze.data import FeatureCorpus
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root=Path(temporary); path,manifest=gaze_fixture(root/'data',c)
            corpus=FeatureCorpus(c,path,root/'data')
            try:
                norm=corpus.fit_normalizer(); item=corpus.dataset('val',norm)[0]
                self.assertEqual(item[0]['x'].shape,(100,46)); self.assertEqual(len(item),5)
                self.assertEqual(corpus.check_supervision(list(TASKS),norm)['train']['gaze_support'],[1]*9)
                self.assertFalse(corpus.arrays(corpus.records[0])['values'].flags.writeable)
                np.testing.assert_allclose(norm['mean'],np.load(root/'data/train_values.npy').mean(0),atol=1e-6)
            finally: corpus.close()
            for mutate in (lambda m:m.update(group_profile_sha256='bad'),lambda m:m['gaze_mapping']['classes'].reverse(),
                           lambda m:m['records'][0].update(gaze_targets='../escape.npy'),lambda m:m['records'][1].update(subject_id='train')):
                bad=deepcopy(manifest); mutate(bad); path.write_text(json.dumps(bad),encoding='utf-8')
                with self.assertRaises(ValueError): FeatureCorpus(c,path,root/'data')
            path.write_text(json.dumps(manifest),encoding='utf-8')
            np.save(root/'data/train_gaze_targets.npy',np.full(300,9,np.int64))
            with self.assertRaises(ValueError): FeatureCorpus(c,path,root/'data')

    def test_cpu_resume_reload_exact_and_checkpoints(self):
        c=self.config()
        from training.gaze.train import run_training
        from training.common.model_factory import load_model
        from training.gaze.data import FeatureCorpus
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root=Path(temporary); path,_=gaze_fixture(root/'data',c)
            c['runtime'].update(manifest=str(path),data_root=str(root/'data'),output_dir=str(root/'first'))
            first=run_training(c)
            c['training']['epochs']=2; c['runtime']['output_dir']=str(root/'resumed')
            resumed=run_training(c,resume=first['last_checkpoint'])
            c['runtime']['output_dir']=str(root/'full'); full=run_training(c)
            a,b=[load_checkpoint(r['last_checkpoint']) for r in (resumed,full)]
            self.assertEqual(a['history'],b['history'])
            for name in a['model_state']: torch.testing.assert_close(a['model_state'][name],b['model_state'][name],rtol=0,atol=0)
            self.assertEqual(a['gaze_classes'],c['data']['gaze_classes'])
            model,payload=load_model(resumed['best_checkpoint'])
            corpus=FeatureCorpus(c,path,root/'data')
            try:
                inputs,*_=corpus.dataset('val',payload['normalizer'])[0]
                out=model(**{k:torch.as_tensor(v)[None] for k,v in inputs.items()})
                self.assertTrue(torch.isfinite(out['gaze_logits']).all())
            finally: corpus.close()
            report=a['history'][-1]['val']
            self.assertEqual(len(report['gaze_metrics']['confusion_matrix']),9)
            self.assertIn('face',report['groups']['view'])
            broken=deepcopy(a); broken['gaze_mapping']['rules']={'corrupt':'changed'}
            save_checkpoint(root/'broken.pt',broken)
            with self.assertRaisesRegex(ValueError,'mapping|metadata'): load_model(root/'broken.pt')
            broken=deepcopy(a); broken['gaze_classes'].reverse()
            save_checkpoint(root/'broken.pt',broken)
            c['training']['epochs']=3; c['runtime']['output_dir']=str(root/'bad_resume')
            with self.assertRaisesRegex(ValueError,'class|metadata'): run_training(c,resume=root/'broken.pt')
            broken=deepcopy(a); broken.pop('best_snapshot')
            save_checkpoint(root/'broken.pt',broken)
            with self.assertRaisesRegex(ValueError,'snapshot|metadata'): run_training(c,resume=root/'broken.pt')
            self.assertIsNotNone(importlib.util.find_spec('training.gaze.predict'),'prediction adapter missing')
            from training.gaze.predict import JointPredictor
            predictor=JointPredictor(resumed['best_checkpoint'])
            values=np.load(root/'data/val_values.npy'); masks=np.load(root/'data/val_masks.npy'); ages=np.zeros((300,3),np.float32)
            prediction=predictor.predict_window(values,masks,ages)
            self.assertTrue(prediction['gaze_available']); self.assertEqual(len(prediction['gaze_probabilities']),9)
            self.assertAlmostEqual(sum(prediction['gaze_probabilities'].values()),1.,places=6)
            self.assertEqual(list(prediction['behavior_probabilities']),list(TASKS))
            masks[:,19:]=0
            prediction=predictor.predict_window(values,masks,ages)
            self.assertTrue(prediction['available']); self.assertFalse(prediction['gaze_available'])
            self.assertIsNone(prediction['gaze_class']); self.assertIsNone(prediction['gaze_probabilities'])
            altered=np.load(root/'data/train_gaze_targets.npy'); altered[0]=1
            np.save(root/'data/train_gaze_targets.npy',altered)
            with self.assertRaisesRegex(ValueError,'fingerprint|data'): run_training(c,resume=resumed['last_checkpoint'])

    def test_stdlib_platform_prechecks(self):
        self.config()
        for backend in ('cpu','gpu','npu'):
            result=subprocess.run([sys.executable,'-S','-m',f'training.gaze.{backend}_train','--check-config'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('relation_gaze_v1',result.stdout)

    def test_diagnostic_coverage_preserves_raw_labels_and_missing_classes(self):
        c=self.config()
        from training.gaze.data import FeatureCorpus
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root=Path(temporary); path,_=gaze_fixture(root/'data',c)
            np.save(root/'data/train_masks.npy',np.zeros((300,23),np.uint8))
            corpus=FeatureCorpus(c,path,root/'data')
            try:
                report=corpus.check_supervision([])
                self.assertEqual(report['train']['positive'],[0]*4)
                self.assertGreater(sum(report['train']['raw_positive']),0)
                self.assertEqual(report['train']['gaze_raw_support'],[1]*9)
                self.assertEqual(report['train']['gaze_support'],[0]*9)
                self.assertIn('train',report['train']['groups']['subject_id'])
                with self.assertRaisesRegex(ValueError,'coverage'): corpus.check_supervision(list(TASKS))
            finally: corpus.close()

    def test_gaze_only_batches_preserve_behavior_heads_and_skip_empty_batch(self):
        c=self.config()
        from training.common.model_factory import build_model
        from training.gaze.train import run_epoch
        from torch.utils.data import DataLoader
        model=build_model(c)
        optimizer=torch.optim.Adam(model.parameters(),weight_decay=.1)
        empty=({'x':torch.zeros(100,46),'quality':torch.zeros(4,2),'eligible':torch.zeros(4,dtype=torch.bool)},
               torch.zeros(4),torch.zeros(4,dtype=torch.bool),torch.tensor(0,dtype=torch.int64),torch.tensor(False))
        good=deepcopy(empty); good[0]['x'][:,23:]=1; good[0]['quality'][:]=1; good[0]['eligible'][:]=True
        good=(*good[:-1],torch.tensor(True))
        before=deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError,'supervision'): run_epoch(model,DataLoader([empty]),torch.device('cpu'),optimizer=optimizer)
        self.assertFalse(optimizer.state)
        report=run_epoch(model,DataLoader([empty,good],batch_size=1),torch.device('cpu'),optimizer=optimizer)
        self.assertEqual(report['known_counts'],[0,0,0,0,1]); self.assertEqual(report['skipped_batches'],1)
        for name,value in model.state_dict().items():
            if name.startswith(('heads.','gates.')): torch.testing.assert_close(value,before[name],atol=0,rtol=0)
        before=deepcopy(model.state_dict())
        single=(*good[:2],torch.tensor([True,False,False,False]),torch.tensor(0,dtype=torch.int64),torch.tensor(False))
        run_epoch(model,DataLoader([single]),torch.device('cpu'),optimizer=optimizer)
        for name,value in model.state_dict().items():
            if name.startswith(('heads.1.','heads.2.','heads.3.','gates.1.','gates.2.','gates.3.','gaze_head.')):
                torch.testing.assert_close(value,before[name],atol=0,rtol=0)

    def test_corrupted_gaze_array_contracts(self):
        c=self.config()
        from training.gaze.data import FeatureCorpus
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root=Path(temporary); path,_=gaze_fixture(root/'data',c)
            original=np.load(root/'data/train_gaze_targets.npy')
            for bad in (original.astype(np.float32),original[:,None],np.full(300,-1,np.int64)):
                np.save(root/'data/train_gaze_targets.npy',bad)
                with self.assertRaises(ValueError): FeatureCorpus(c,path,root/'data')
            np.save(root/'data/train_gaze_targets.npy',original)
            np.save(root/'data/train_gaze_label_masks.npy',np.zeros(300,np.uint8))
            with self.assertRaisesRegex(ValueError,'Unknown'): FeatureCorpus(c,path,root/'data')

    def test_skipped_batches_count_unknowns_without_changing_training_rng(self):
        from training.common.model_factory import build_model
        from training.gaze.train import run_epoch
        from torch.utils.data import DataLoader
        model=build_model(self.config()); reference=deepcopy(model)
        empty=({'x':torch.zeros(100,46),'quality':torch.zeros(4,2),'eligible':torch.zeros(4,dtype=torch.bool)},
               torch.zeros(4),torch.zeros(4,dtype=torch.bool),torch.tensor(0,dtype=torch.int64),torch.tensor(False))
        good=deepcopy(empty); good[0]['x'][:,23:]=1; good[0]['quality'][:]=1; good[0]['eligible'][:]=True
        good=(*good[:-1],torch.tensor(True))
        def train(candidate,items):
            torch.manual_seed(123)
            optimizer=torch.optim.Adam(candidate.parameters())
            loader=DataLoader(items,batch_size=1,generator=torch.Generator().manual_seed(456))
            report=run_epoch(candidate,loader,torch.device('cpu'),optimizer=optimizer)
            return report,torch.get_rng_state().clone()
        report,rng=train(model,[empty,good])
        self.assertEqual(report['label_coverage']['unknown'],[2]*4)
        self.assertEqual([report['metrics']['per_task'][name]['unknown'] for name in TASKS],[2]*4)
        self.assertEqual(report['gaze_metrics']['unknown_or_unavailable'],1)
        self.assertEqual(report['gaze_metrics']['known'],1)
        self.assertEqual(report['gaze_unavailable_windows'],1)
        self.assertEqual(report['skipped_batches'],1)
        reference_report,reference_rng=train(reference,[good])
        self.assertEqual(report['loss_sums'],reference_report['loss_sums'])
        torch.testing.assert_close(rng,reference_rng,atol=0,rtol=0)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,reference.state_dict()[name],atol=0,rtol=0)

    def test_numpy_torch_gaze_availability_parity_and_boundary(self):
        self.config()
        from training.gaze.preprocessing import gaze_available
        from training.gaze.model import RelationGazeTCN
        rng=np.random.default_rng(7)
        masks=rng.integers(0,2,(200,100,23)).astype(bool)
        eligible=rng.integers(0,2,(200,4)).astype(bool)
        masks[:100,:,5:8]=True
        masks[:100,:,19:23]=rng.random((100,100,4))>.1
        x=torch.from_numpy(np.concatenate((np.zeros(masks.shape,np.float32),masks.astype(np.float32)),axis=-1))
        got=RelationGazeTCN.gaze_availability(x,torch.from_numpy(eligible)).tolist()
        self.assertEqual(got,[gaze_available(m,e) for m,e in zip(masks,eligible)])
        for valid,expected in ((49,False),(50,True),(51,True)):
            m=np.zeros((100,23),bool); m[-valid:,5:8]=True; m[-valid:,19:21]=True
            e=np.ones(4,bool)
            self.assertEqual(gaze_available(m,e),expected)
            x=torch.from_numpy(np.concatenate((np.zeros(m.shape,np.float32),m.astype(np.float32)),axis=1))[None]
            self.assertEqual(RelationGazeTCN.gaze_availability(x,torch.from_numpy(e)[None]).item(),expected)


if __name__ == '__main__': unittest.main()
