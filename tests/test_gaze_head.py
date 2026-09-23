"""Independent head evidence must survive F rejection without enabling other tasks."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from training.gaze.config import load_config
from training.common.model_factory import build_model, load_model
from training.common.checkpoint import load_checkpoint
from tests.test_gaze_training import gaze_fixture

ROOT = Path(__file__).resolve().parents[1]


class HeadBranchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self):
        config = load_config()
        config['model'].update(kind='relation_gaze_head_v1',channels=[4],dilations=[1],
                               projection_dim=4,gate_hidden_dim=4,dropout=0.)
        config['training'].update(epochs=1,batch_size=3)
        return config

    def inputs(self, batch=1):
        x = torch.zeros(batch,100,46)
        x[:,:,5:8] = torch.randn(batch,100,3)
        x[:,:,28:31] = 1
        return dict(x=x,quality=torch.zeros(batch,4,2),eligible=torch.zeros(batch,4,dtype=torch.bool))

    def test_head_only_enables_distraction_and_not_gaze(self):
        model=build_model(self.config()).eval()
        inputs=self.inputs()
        out=model(**inputs)
        self.assertEqual(out['task_available'].tolist(),[[False,False,False,True]])
        self.assertTrue(out['head_available'].item())
        self.assertFalse(out['gaze_available'].item())
        self.assertEqual(out['head_dist_weight'].item(),1.)
        self.assertFalse(out['weights'].any())
        out['behavior_logits'][0,3].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.head_projection.parameters()),0.)
        self.assertEqual(out['behavior_logits'][0,:3].tolist(),[0.,0.,0.])

    def test_head_threshold_current_axis_and_numpy_torch_agree(self):
        model=build_model(self.config()).eval()
        from training.gaze.head_evidence import behavior_availability
        inputs=self.inputs(4)
        inputs['x'][0,:50,28:31]=0
        inputs['x'][1,:51,28:31]=0
        inputs['x'][2,-1,28]=0
        inputs['x'][3,:,30]=0
        out=model(**inputs)
        self.assertEqual(out['head_available'].tolist(),[True,False,False,False])
        np_inputs={k:v.numpy() for k,v in inputs.items()}
        np.testing.assert_array_equal(behavior_availability(np_inputs,'relation_gaze_head_v1'),
                                      out['task_available'].numpy())

    def test_stale_or_unsupported_head_is_rejected_after_preprocessing(self):
        model=build_model(self.config()).eval()
        from training.gaze.preprocessing import dependency_masks,prepare_window
        from training.common.normalization import fit_normalizer
        names=self.config()['data']['feature_names']; profile=self.config()['data']['group_profile']
        values=np.ones((100,23),np.float32); masks=np.ones_like(values,np.uint8); ages=np.zeros((100,3),np.float32)
        normalizer=fit_normalizer([(values,masks)],names)
        ages[-1,1]=81
        inputs=prepare_window(values,dependency_masks(masks,ages,profile),ages,normalizer,profile)
        self.assertFalse(model(**{k:torch.as_tensor(v)[None] for k,v in inputs.items()})['head_available'].item())
        ages[:]=0; normalizer['supported'][5]=False
        inputs=prepare_window(values,masks,ages,normalizer,profile)
        self.assertFalse(model(**{k:torch.as_tensor(v)[None] for k,v in inputs.items()})['head_available'].item())

    def test_ineligible_head_values_cannot_change_predictions(self):
        model=build_model(self.config()).eval()
        inputs=self.inputs(); inputs['eligible'][:,1]=True
        inputs['x'][:,-1,28:31]=0
        out=model(**inputs)
        inputs['x'][:,:50,5:8]=999.
        changed=model(**inputs)
        torch.testing.assert_close(out['behavior_logits'],changed['behavior_logits'],rtol=0,atol=0)
        self.assertEqual(changed['head_dist_weight'].item(),0.)

    def test_ineligible_head_quality_cannot_reweight_other_groups(self):
        torch.manual_seed(123)
        model=build_model(self.config()).eval()
        from training.gaze.preprocessing import prepare_window
        from training.common.normalization import fit_normalizer
        config=self.config()
        values=np.random.default_rng(123).normal(size=(100,23)).astype(np.float32)
        masks=np.ones((100,23),np.uint8); masks[:,:8]=0; masks[:,12]=0; masks[:,19:]=0
        masks[:40,5:8]=1
        normalizer=fit_normalizer([(values,np.ones_like(masks))],config['data']['feature_names'])
        def prepare():
            return prepare_window(values,masks,np.zeros((100,3),np.float32),normalizer,config['data']['group_profile'])
        first=prepare()
        masks[:,6]=0; masks[40:80,6]=1
        second=prepare()
        np.testing.assert_array_equal(first['quality'],second['quality'])
        np.testing.assert_array_equal(first['eligible'],second['eligible'])
        original=model(**{k:torch.as_tensor(v)[None] for k,v in first.items()})
        changed=model(**{k:torch.as_tensor(v)[None] for k,v in second.items()})
        self.assertFalse(changed['head_available'].item())
        torch.testing.assert_close(original['weights'],changed['weights'],rtol=0,atol=0)
        torch.testing.assert_close(original['behavior_logits'],changed['behavior_logits'],rtol=0,atol=0)

    def test_head_causality_and_does_not_change_other_tasks(self):
        model=build_model(self.config()).eval()
        inputs=self.inputs(); inputs['eligible'][:,1]=True
        original=model(**inputs); encoded=model.encode_head(inputs['x'])
        inputs['x'][:,60:,5:8] += 5
        changed=model(**inputs); later=model.encode_head(inputs['x'])
        torch.testing.assert_close(encoded[:,:,:60],later[:,:,:60],rtol=0,atol=0)
        torch.testing.assert_close(original['behavior_logits'][:,:3],changed['behavior_logits'][:,:3],rtol=0,atol=0)
        torch.testing.assert_close(original['gaze_logits'],changed['gaze_logits'],rtol=0,atol=0)
        self.assertAlmostEqual(float((changed['weights'][0,3].sum()+changed['head_dist_weight'][0]).detach()),1.,places=6)

    def test_head_parameters_do_not_decay_without_distraction_supervision(self):
        from training.gaze.train import run_epoch
        from torch.utils.data import DataLoader
        model=build_model(self.config())
        optimizer=torch.optim.Adam(model.parameters(),weight_decay=.1)
        inputs={k:v[0] for k,v in self.inputs().items()}
        row=(inputs,torch.ones(4),torch.ones(4,dtype=torch.bool),torch.tensor(0),torch.tensor(False))
        run_epoch(model,DataLoader([row]),torch.device('cpu'),optimizer=optimizer)
        before=deepcopy(model.state_dict())
        inputs['eligible'][1]=True
        row=(inputs,torch.ones(4),torch.tensor([True,False,False,False]),torch.tensor(0),torch.tensor(False))
        run_epoch(model,DataLoader([row]),torch.device('cpu'),optimizer=optimizer)
        for name,value in model.state_dict().items():
            if name.startswith(('head_projection.','head_blocks.','gates.3.','heads.3.')):
                torch.testing.assert_close(value,before[name],rtol=0,atol=0)

    def test_corpus_counts_head_only_windows_per_task(self):
        from training.gaze.data import FeatureCorpus
        config=self.config()
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp); path,_=gaze_fixture(root/'data',config)
            masks=np.zeros((300,23),np.uint8); masks[:,5:8]=1
            np.save(root/'data/val_masks.npy',masks)
            corpus=FeatureCorpus(config,path,root/'data')
            try:
                report=corpus.check_supervision([])['val']
                self.assertEqual(report['positive'][:3],[0,0,0])
                self.assertGreater(report['positive'][3],0)
                self.assertEqual(report['task_unavailable_windows'],[9,9,9,0])
                self.assertEqual(report['gaze_support'],[0]*9)
            finally: corpus.close()

    def test_head_only_training_supervises_only_distraction(self):
        from training.gaze.train import run_epoch
        from torch.utils.data import DataLoader
        model=build_model(self.config())
        optimizer=torch.optim.Adam(model.parameters(),weight_decay=.1)
        inputs={k:v[0] for k,v in self.inputs().items()}
        row=(inputs,torch.ones(4),torch.ones(4,dtype=torch.bool),torch.tensor(0),torch.tensor(True))
        before=deepcopy(model.state_dict())
        report=run_epoch(model,DataLoader([row]),torch.device('cpu'),optimizer=optimizer)
        self.assertEqual(report['known_counts'],[0,0,0,1,0])
        self.assertEqual(report['task_unavailable_windows'],[1,1,1,0])
        for name,value in model.state_dict().items():
            if name.startswith(('heads.0.','heads.1.','heads.2.','gates.0.','gates.1.','gates.2.','gaze_head.')):
                torch.testing.assert_close(value,before[name],rtol=0,atol=0)

    def test_training_resume_reload_and_diagnostics(self):
        from training.gaze.train import run_training
        from training.gaze.predict import JointPredictor
        from training.gaze.diagnostics import diagnose
        config=self.config()
        config['model']['dropout']=.1
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp); path,_=gaze_fixture(root/'data',config)
            config['runtime'].update(manifest=str(path),data_root=str(root/'data'),output_dir=str(root/'first'))
            first=run_training(config)
            self.assertEqual(load_checkpoint(first['best_checkpoint'])['model_kind'],'relation_gaze_head_v1')
            config['training']['epochs']=2; config['runtime']['output_dir']=str(root/'resumed')
            resumed=run_training(config,resume=first['last_checkpoint'])
            config['runtime']['output_dir']=str(root/'full')
            full=run_training(config)
            a,b=[load_checkpoint(r['last_checkpoint']) for r in (resumed,full)]
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key],b['model_state'][key],rtol=0,atol=0)
            self.assertEqual(a['history'],b['history'])
            torch.testing.assert_close(a['optimizer_state'],b['optimizer_state'],rtol=0,atol=0)
            predictor=JointPredictor(resumed['best_checkpoint'])
            values=np.load(root/'data/val_values.npy'); masks=np.zeros((300,23),np.uint8); masks[:,5:8]=1
            prediction=predictor.predict_window(values,masks,np.zeros((300,3),np.float32))
            self.assertEqual(prediction['behavior_available'],dict(fatigue=False,phone_use=False,drinking=False,distraction=True))
            self.assertIsNone(prediction['behavior_probabilities']['fatigue'])
            self.assertIsNotNone(prediction['behavior_probabilities']['distraction'])
            self.assertTrue(prediction['distraction_head_evidence_available'])
            self.assertFalse(prediction['gaze_available'])
            report=diagnose(resumed['best_checkpoint'],output_dir=root/'diagnostics',view='all',subset='all')
            self.assertEqual(report['model_kind'],'relation_gaze_head_v1')
            with (root/'diagnostics/windows.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertIn('h_eligible',rows[0]); self.assertIn('h_dist_weight',rows[0])
            self.assertIn('h_eligibility',report['groups'])
            config['model']['kind']='relation_gaze_v1'; config['training']['epochs']=3
            config['runtime']['output_dir']=str(root/'wrong-kind')
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                run_training(config,resume=resumed['last_checkpoint'])
            config['training']['epochs']=1; config['runtime']['output_dir']=str(root/'old-kind')
            old=run_training(config)
            config['model']['kind']='relation_gaze_head_v1'; config['training']['epochs']=2
            config['runtime']['output_dir']=str(root/'wrong-kind-reverse')
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                run_training(config,resume=old['last_checkpoint'])


if __name__ == '__main__': unittest.main()
