"""Independent eye supervision, isolation and exact continuation contracts."""
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tests.test_head_adapt_train import TinyEncoder, probe
from training.common.checkpoint import load_checkpoint, save_checkpoint
from training.visual.action_model import ActionModel
from training.visual import head_adapt_train as adapt


class Cache:
    def __init__(self, eye=False):
        self.root=Path('eye' if eye else 'head').resolve()
        self.manifest_sha256='eye-digest' if eye else 'head-digest'
        self.records={}; self.values={}
        for rid,subject,split,n in [('a','s1','train',5),('b','s2','train',3),('c','s3','val',3)]:
            self.records[rid]=dict(record_id=rid,subject_id=subject,split=split)
            self.values[rid]=dict(images=np.full((n,3,224,224),90 if eye else 30,np.uint8),
                slot_indices=np.arange(n,dtype=np.int64)*3,source_pts_ms=np.arange(n,dtype=np.int64)*120,
                valid=np.ones(n,dtype=bool))
        self.values['a']['valid'][-1]=False
    def arrays(self,rid): return self.values[rid]


def fixtures(api):
    head,eye=Cache(),Cache(True)
    labels=SimpleNamespace(records={},digest='labels-digest')
    for rid,a in head.values.items():
        n=len(a['valid']); y=np.zeros((n,2),np.float32); y[1::2,0]=1
        known=np.ones((n,2),bool); known[-1,0]=False
        labels.records[rid]=dict(targets=y,masks=known)
    return head,eye,labels


class EyeTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.eye_train'), 'Independent eye trainer missing')
        return importlib.import_module('training.visual.eye_train')
    def reference(self):
        with patch.object(adapt,'ResNet18',TinyEncoder):
            return ActionModel(adapt.AdaptModel(probe(),TinyEncoder().state_dict(),32,True,2)).eval()
    def test_unique_matched_known_cohort_and_subject_weights(self):
        api=self.api(); h,e,l=fixtures(api)
        a=api.EyeDataset(h,e,l,'train','head'); b=api.EyeDataset(h,e,l,'train','eye')
        self.assertEqual(a.indices,b.indices)
        self.assertEqual(len(a),6)
        self.assertEqual(a.summary,b.summary)
        self.assertEqual(a.metadata(0)['slot_index'],0)
        self.assertEqual(a[0][0].dtype,torch.uint8)
        self.assertEqual(int(a[0][0][0,0,0]),30)
        self.assertEqual(int(b[0][0][0,0,0]),90)
        weights=api.subject_weights(a)
        self.assertAlmostEqual(sum(weights[:4]),sum(weights[4:]))
        l.records['a']['targets'][-1,0]=float('nan')
        self.assertEqual(api.EyeDataset(h,e,l,'train','head').indices,a.indices)
        e.values['a']['source_pts_ms'][0]+=1
        with self.assertRaisesRegex(ValueError,'align|identity'):
            api.EyeDataset(h,e,l,'train','eye')
    def test_warm_start_exact_and_private_encoder_bn_frozen(self):
        api=self.api(); reference=self.reference(); original=deepcopy(reference.state_dict())
        model=api.EyeModel(reference,frame_chunk=2)
        pixels=torch.randint(256,(3,3,224,224),dtype=torch.uint8)
        with torch.no_grad():
            torch.testing.assert_close(model(pixels),reference.frame_actions(pixels)[:,0],atol=0,rtol=0)
        model.train(); model(pixels).sum().backward()
        self.assertIsNotNone(model.encoder.layer4[0].weight.grad)
        self.assertFalse(model.encoder.layer4[1].training)
        self.assertIsNone(model.encoder.layer4[1].weight.grad)
        with torch.no_grad(): model.encoder.layer4[0].weight.add_(1)
        for k,v in original.items(): torch.testing.assert_close(v,reference.state_dict()[k],atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in reference.parameters()))
    def test_exact_resume_both_sampling_modes_and_checkpoint_identity(self):
        api=self.api(); h,e,l=fixtures(api)
        train=api.EyeDataset(h,e,l,'train','eye'); val=api.EyeDataset(h,e,l,'val','eye')
        reference=self.reference()
        settings=dict(adapt.DEFAULTS,epochs=2,batch_size=2,accumulation_steps=2,early_stopping_patience=10)
        with tempfile.TemporaryDirectory() as tmp,patch.object(api,'ResNet18',TinyEncoder):
            for balanced in (False,True):
                root=Path(tmp)/str(balanced)
                full=api.train_eye(train,val,reference,settings,root/'full',{},balanced)
                one=api.train_eye(train,val,reference,dict(settings,epochs=1),root/'one',{},balanced)
                resumed=api.train_eye(train,val,reference,settings,root/'resumed',{},balanced,one['last_checkpoint'])
                a=load_checkpoint(full['last_checkpoint']); b=load_checkpoint(resumed['last_checkpoint'])
                self.assertEqual(a['history'],b['history'])
                for k,v in a['model_state'].items(): torch.testing.assert_close(v,b['model_state'][k],atol=0,rtol=0)
                loaded,payload=api.load_eye(full['best_checkpoint'])
                self.assertEqual(payload['model_kind'],'eye_v1')
                self.assertEqual(payload['data_summary']['train']['count'],6)
                bad=deepcopy(a); bad['normalizer']['mean'][0]+=1
                save_checkpoint(root/'bad.pt',bad)
                with self.assertRaisesRegex(ValueError,'normaliz'): api.load_eye(root/'bad.pt')
                bad=deepcopy(a); bad['provenance']['changed']=True
                save_checkpoint(root/'bad.pt',bad)
                with self.assertRaisesRegex(ValueError,'identity|provenance'): api.load_eye(root/'bad.pt')
    def test_accumulation_weighted_final_tail_and_validation_subjects(self):
        api=self.api()
        class Rows(torch.utils.data.Dataset):
            def __len__(self): return 3
            def __getitem__(self,i): return torch.tensor([i],dtype=torch.float32),float(i==0)
            def metadata(self,i): return dict(subject_id='s',target=float(i==0))
        class Scalar(nn.Module):
            def __init__(self): super().__init__(); self.weight=nn.Parameter(torch.tensor(.3))
            def forward(self,x): return self.weight.expand(len(x))
        a,b=Scalar(),Scalar()
        for model,batch,accum in ((a,2,2),(b,3,1)):
            api.run_epoch(model,DataLoader(Rows(),batch_size=batch),torch.device('cpu'),pos_weight=2.,
                optimizer=torch.optim.SGD(model.parameters(),lr=.2),accumulation_steps=accum)
        torch.testing.assert_close(a.weight,b.weight,atol=1e-7,rtol=0)
        report=api.evaluate_eye(a,DataLoader(Rows(),batch_size=2),torch.device('cpu'))
        self.assertAlmostEqual(report['loss'],float(((torch.nn.functional.softplus(-a.weight)+torch.nn.functional.softplus(a.weight))/2).detach()),places=6)

    def test_epoch_zero_selection_single_class_reporting_and_strict_metadata(self):
        api=self.api(); h,e,l=fixtures(api)
        train=api.EyeDataset(h,e,l,'train','head'); val=api.EyeDataset(h,e,l,'val','head')
        reference=self.reference()
        settings=dict(adapt.DEFAULTS,epochs=1,batch_size=2,early_stopping_patience=10)
        with tempfile.TemporaryDirectory() as tmp,patch.object(api,'ResNet18',TinyEncoder):
            root=Path(tmp)
            with patch.object(api,'evaluate_eye',side_effect=[dict(loss=.1),dict(loss=.9)]):
                result=api.train_eye(train,val,reference,settings,root/'run',{})
            self.assertEqual(result['best_epoch'],0)
            payload=load_checkpoint(result['last_checkpoint'])
            for field,value in [('sampling',dict(policy='bad',epoch_draws=6,replacement=False)),
                                ('data_summary',dict(payload['data_summary'],class_weighting={'positive':500}))]:
                bad=deepcopy(payload); bad[field]=value; save_checkpoint(root/'bad.pt',bad)
                with self.assertRaisesRegex(ValueError,'identity|metadata|weight|sampling'):
                    api.load_eye(root/'bad.pt')
            bad=deepcopy(payload); del bad['optimizer_state']; save_checkpoint(root/'bad.pt',bad)
            with self.assertRaisesRegex(ValueError,'identity|metadata|training'):
                api.load_eye(root/'bad.pt')
            l.records['c']['targets'][:]=0
            only_negative=api.EyeDataset(h,e,l,'val','head')
            with self.assertRaisesRegex(ValueError,'both positive and negative'):
                api.evaluate_eye(api.EyeModel(reference),DataLoader(only_negative,batch_size=2),torch.device('cpu'))

    def test_best_checkpoint_is_independent_of_early_stopping_tolerance(self):
        api=self.api(); h,e,l=fixtures(api)
        train=api.EyeDataset(h,e,l,'train','eye'); val=api.EyeDataset(h,e,l,'val','eye')
        settings=dict(adapt.DEFAULTS,epochs=3,batch_size=2,early_stopping_patience=2,early_stopping_min_delta=.1)
        with tempfile.TemporaryDirectory() as tmp,patch.object(api,'ResNet18',TinyEncoder):
            root=Path(tmp); reference=self.reference()
            # Epoch 1 is a real minimum, although too small to reset patience.
            with patch.object(api,'evaluate_eye',side_effect=[dict(loss=1.),dict(loss=.95)]):
                one=api.train_eye(train,val,reference,dict(settings,epochs=1),root/'one',{})
            self.assertEqual(one['best_epoch'],1)
            self.assertEqual(load_checkpoint(one['last_checkpoint'])['bad_epochs'],1)
            # Cumulative improvement versus the last significant value resets
            # patience after resume, even though improvement from best is <delta.
            with patch.object(api,'evaluate_eye',side_effect=[dict(loss=.89),dict(loss=.88)]):
                result=api.train_eye(train,val,reference,settings,root/'resumed',{},resume=one['last_checkpoint'])
            self.assertEqual(result['completed_epochs'],3)
            self.assertEqual(result['best_epoch'],3)
            self.assertFalse(result['stopped_early'])
            payload=load_checkpoint(result['last_checkpoint'])
            self.assertEqual(payload['bad_epochs'],1)
            self.assertEqual(payload['stopping_loss'],.89)
            self.assertEqual(api.load_eye(result['best_checkpoint'])[1]['selected_loss'],.88)


if __name__=='__main__': unittest.main()
