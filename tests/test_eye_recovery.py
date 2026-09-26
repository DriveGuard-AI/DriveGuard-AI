"""Frozen inference comparison: real event decoder, no training/label inference."""
from contextlib import contextmanager,redirect_stdout
from copy import deepcopy
import importlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_eye_diagnostics import event_fixture
from training.visual.eye_composition import compose_events, infer_eyes, frame_metrics
from training.visual.fatigue_spatial_suite import _summary
from training.visual.spatial import crop_policy
from training.visual.action_eval import replay_events,action_metrics
from training.common.data import file_sha256
import json


class BrightEye(torch.nn.Module):
    def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.tensor(0.))
    def forward(self,images):return images[:,0,0,0].float()-100+self.bias


class RecoverySuiteTests(unittest.TestCase):
    def test_reference_binds_exact_model_source_cache_and_rejects_tampering(self):
        api=importlib.import_module('training.visual.eye_recovery')
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve();checkpoint=root/'best.pt';checkpoint.write_bytes(b'fixture')
            action=root/'action.json';action.write_text('{}')
            original=dict(action_report=str(action),action_report_sha256=file_sha256(action))
            provenance=dict(source_reference=original,head_cache_sha256='head',eye_cache=str(root/'old'),eye_cache_sha256='old')
            selected=dict(variant=api.VARIANT,checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
                arm='B_eye_native',seed=42,readout='combined',checkpoint_kind='best',epoch=6)
            report=dict(schema_version='driveguard-independent-eye-comparison-v1',status='complete',task='fatigue',split='val',
                reference_reproduced=True,comparisons=[selected],plan=dict(provenance=provenance,reference={'variant':'actions'}))
            source=root/'comparison.json';source.write_text(json.dumps(report))
            payload=dict(input_mode='eye',subject_balanced=False,kind='weights',checkpoint_role='best',epoch=6,
                settings={'seed':42},provenance=provenance)
            data=dict(provenance=original,cache=SimpleNamespace(manifest_sha256='head'),sources=[])
            old=SimpleNamespace(manifest_sha256='old',report={'run':{}},root=root/'old',close=lambda:None)
            @contextmanager
            def context(*args):yield data
            with patch.object(api,'load_eye',return_value=(BrightEye(),payload)),patch.object(api,'_inputs',context),patch.object(api,'EyeCache',return_value=old):
                with api._reference(source,api.VARIANT,root,'cpu',0) as loaded:
                    self.assertEqual(loaded['selected_eye']['epoch'],6)
                with self.assertRaisesRegex(ValueError,'variant'):
                    with api._reference(source,'C_eye_subject_balanced-seed42-best-combined',root,'cpu',0):pass
                payload['subject_balanced']=True
                with self.assertRaisesRegex(ValueError,'provenance'):
                    with api._reference(source,api.VARIANT,root,'cpu',0):pass
                payload['subject_balanced']=False;old.manifest_sha256='different'
                with self.assertRaisesRegex(ValueError,'cache identity'):
                    with api._reference(source,api.VARIANT,root,'cpu',0):pass
                old.manifest_sha256='old';checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'hash'):
                    with api._reference(source,api.VARIANT,root,'cpu',0):pass

    def test_fixed_weights_missing_control_recovery_and_terminal_reuse(self):
        api=importlib.import_module('training.visual.eye_recovery')
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve();source=root/'source';source.mkdir()
            head,rows,original,resets,a=event_fixture()
            rows[0].update(subject_id='14',split='val',f_eligible=False)
            a['images']=np.full((32,3,224,224),110,np.uint8)
            head.root=source/'head';head.manifest_sha256='head'
            head.records={'a':dict(record_id='a',subject_id='14',split='val',source={'driver_roi':[0,0,224,224]})}
            head.report={'crop':crop_policy([0,0,1,1])}
            original['a'][:,1]=-2.;original['a'][:,0]=-2.
            rows=replay_events(head,original,rows,resets,500)
            targets=np.zeros((32,2),np.float32);targets[:,0]=1
            labels=SimpleNamespace(records={'a':dict(targets=targets,masks=np.ones((32,2),bool),
                reset=resets['a'],subtypes=np.full(100,'closed_eye'))},closed_eye_min_ms=500)
            def cache(name,valid,mode=None):
                b=dict(a,valid=np.full(32,valid,bool),roi_xyxy=np.tile([30.,30.,60.,60.],(32,1)),native_eye_width=np.full(32,15.))
                return SimpleNamespace(root=source/name,manifest_sha256=name,records=head.records,arrays=lambda rid:b,
                    report={'run':dict(recovery=dict(mode=mode,split='val')) if mode else {},'records':[]},close=lambda:None)
            old=cache('old',False);model=BrightEye();state=deepcopy(model.state_dict())
            pred=infer_eyes(head,old,model,rows,'eye'); combined,_=compose_events(head,old,original,pred,rows,resets,500)
            selected=_summary(combined)
            data=dict(cache=head,labels=labels,base=SimpleNamespace(rows={'val':rows}),model=BrightEye(),
                selected=_summary(rows,variant='reference-event'),eye_model=model,old_eyes=old,selected_eye=selected,
                report={'action_metrics':{'reference':action_metrics(original,labels,head)}},
                source_eye_report={'frame_metrics':{'B_eye_native-seed42-best':frame_metrics(pred,head,old,labels)}},
                provenance={'fixture':True},sources=[source])
            variants={mode:cache(mode,mode!='sampled8',mode) for mode in ('sampled8','dense','dense_track')}
            @contextmanager
            def reference(*args,**kwargs):yield data
            def read_cache(path,*args):return variants[Path(path).name]
            with patch.object(api,'_reference',reference),patch.object(api,'EyeCache',side_effect=read_cache), \
                 patch.object(api,'evaluate_events',return_value=(rows,original)),redirect_stdout(io.StringIO()):
                report=api.run(eye_report=source/'report.json',dataset_root=source,recovery_root=source,output_dir=root/'out')
                again=api.run(eye_report=source/'report.json',dataset_root=source,recovery_root=source,output_dir=root/'out',continue_run=True)
            self.assertEqual(report,again);self.assertEqual(report['training_runs'],0)
            self.assertEqual(len(report['comparisons']),8)
            control=next(r for r in report['comparisons'] if r['variant']=='sampled8-combined')
            recovered=next(r for r in report['comparisons'] if r['variant']=='dense-combined')
            self.assertEqual(control['overall']['tp'],0);self.assertEqual(recovered['overall']['tp'],1)
            self.assertEqual(recovered['by_subtype']['closed_eye']['tp'],1)
            self.assertEqual(report['input_support']['sampled8']['overall']['duration_eligible'],0)
            self.assertEqual(report['input_support']['dense']['overall']['duration_eligible'],1)
            for name,v in model.state_dict().items():torch.testing.assert_close(v,state[name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
