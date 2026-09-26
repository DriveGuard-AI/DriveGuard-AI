"""Frozen yawn composition and complete-cohort independent eye comparisons."""
import importlib.abc
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class PerceptionEnvironmentTests(unittest.TestCase):
    def test_visual_metadata_validation_does_not_import_torch(self):
        from tests.test_head_frames import frame_fixture
        from training.visual.data import VisualCache
        with tempfile.TemporaryDirectory(dir='runs') as tmp, frame_fixture(tmp) as (prepared, _, _):
            expected = prepared.cache.report['encoder']
            script = '''
import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Perception cache validation imported Torch')
sys.meta_path.insert(0, NoTorch())
from training.visual.data import VisualCache
cache = VisualCache(sys.argv[1]); cache.close()
'''
            result = subprocess.run([sys.executable, '-c', script, str(prepared.cache.root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class CompositionTests(unittest.TestCase):
    def test_missing_eye_preserves_yawn_and_does_not_become_available_eye(self):
        from training.visual.eye_composition import compose_events
        from tests.test_eye_diagnostics import event_fixture
        cache, rows, original, resets, arrays = event_fixture()
        original['a'][:, 1] = 2.
        eye_arrays = dict(arrays, valid=np.zeros(32, bool))
        eyes = SimpleNamespace(arrays=lambda rid: eye_arrays)
        new_eye = {'a': np.full(32, np.nan)}
        combined, eye_only = compose_events(cache, eyes, original, new_eye, rows, resets, 500)
        self.assertEqual(combined[0]['logit'], 2.)
        self.assertTrue(combined[0]['available'])
        self.assertFalse(eye_only[0]['available'])
        self.assertIsNone(eye_only[0]['logit'])
        self.assertTrue(np.all(original['a'][:, 0] == -1.))

    def test_new_eye_adds_events_resets_at_missing_pixels_and_retains_labels(self):
        from training.visual.eye_composition import compose_events
        from tests.test_eye_diagnostics import event_fixture
        cache, rows, original, resets, arrays = event_fixture()
        eye_arrays = dict(arrays, valid=np.ones(32, bool))
        eyes = SimpleNamespace(arrays=lambda rid: eye_arrays)
        new_eye = {'a': np.ones(32)}
        combined, eye_only = compose_events(cache, eyes, original, new_eye, rows, resets, 500)
        self.assertEqual(combined[0]['logit'], 1.)
        self.assertEqual(eye_only[0]['logit'], 1.)
        self.assertEqual(combined[0]['target'], rows[0]['target'])
        eye_arrays['valid'][-2] = False
        combined, eye_only = compose_events(cache, eyes, original, new_eye, rows, resets, 500)
        self.assertLess(combined[0]['logit'], 0)
        new_eye['a'][-1] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            compose_events(cache, eyes, original, new_eye, rows, resets, 500)

    def test_audit_separates_missing_positive_from_negative(self):
        from training.visual.eye_audit import write_audit
        from tests.test_eye_diagnostics import event_fixture
        from training.visual.spatial import crop_policy
        cache, rows, _, resets, arrays = event_fixture()
        rows[0].update(subject_id='14',split='val',f_eligible=False)
        cache.records={'a':dict(subject_id='14',split='val',source=dict(driver_roi=[0,0,224,224]))}
        cache.report={'crop':crop_policy([0,0,1,1])}
        arrays['images']=np.zeros((32,3,224,224),np.uint8)
        eye_arrays=dict(arrays,valid=np.ones(32,bool),roi_xyxy=np.tile([30,30,60,60],(32,1)),
                        native_eye_width=np.full(32,15.))
        eye_arrays['valid'][-8:]=False
        eyes=SimpleNamespace(arrays=lambda rid:eye_arrays)
        targets=np.zeros((32,2),np.float32); targets[-16:,0]=1
        masks=np.ones((32,2),bool); masks[:4,0]=False
        labels=SimpleNamespace(records={'a':dict(targets=targets,masks=masks,reset=resets['a'],
                                                 subtypes=np.full(100,'closed_eye'))})
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            result=write_audit(Path(tmp),cache,eyes,labels,rows,500)
            record=result['records'][0]
            self.assertEqual(record['positive'],16)
            self.assertEqual(record['positive_with_eye'],8)
            self.assertEqual(record['negative'],12)
            self.assertEqual(result['windows'][0]['eye_duration_eligible'],False)
            self.assertTrue((Path(tmp)/'input-audit/index.html').exists())
            self.assertTrue(any(s['category']=='positive_missing_eye' for s in result['samples']))


class SuiteTests(unittest.TestCase):
    def test_six_matched_runs_frozen_yawn_full_cohort_and_verified_resume(self):
        import torch
        from training.visual.fatigue_eyes import run
        from training.visual.head_frames import selected_slots
        from training.visual.head_adapt_train import AdaptModel, DEFAULTS
        from training.visual.action_model import ActionModel
        from training.visual.action_labels import ActionLabels
        from training.visual.action_eval import evaluate_events,action_metrics
        from training.visual.fatigue_spatial_suite import _summary
        from training.visual.spatial import crop_policy
        from training.common.config import canonical_hash,read_json
        from tests.test_head_adapt_train import TinyEncoder,probe
        torch.set_num_threads(1); torch.manual_seed(5)
        with tempfile.TemporaryDirectory(dir='runs') as tmp, \
                patch('training.visual.head_adapt_train.ResNet18',TinyEncoder), \
                patch('training.visual.eye_train.ResNet18',TinyEncoder):
            root=Path(tmp).resolve(); source=root/'source'; source.mkdir()
            report_path=source/'comparison.json'; report_path.write_text('{}')
            model=ActionModel(AdaptModel(probe(),TinyEncoder().state_dict(),32,True,16)).eval()
            with torch.no_grad():
                model.actions.weight[1].zero_();model.actions.bias[1].fill_(1.)
            original=deepcopy(model.state_dict())
            heads={};eyes={};metadata={};records={};rows={}
            for split,rid,subject in [('train','t','1'),('val','v','14')]:
                slots=selected_slots(150);n=len(slots)
                head=dict(images=np.full((n,3,224,224),90,np.uint8),slot_indices=slots,
                          source_pts_ms=slots*40,valid=np.ones(n,bool))
                eye=dict(head,images=np.full((n,3,224,224),110,np.uint8),valid=np.ones(n,bool),
                         roi_xyxy=np.tile([30.,30.,60.,60.],(n,1)),native_eye_width=np.full(n,15.))
                eye['valid'][-8:]=False
                heads[rid],eyes[rid]=head,eye
                metadata[rid]=dict(record_id=rid,subject_id=subject,split=split,source={'driver_roi':[0,0,224,224]})
                targets=np.zeros((n,2),np.float32);targets[n//2:,0]=1
                records[rid]=dict(targets=targets,masks=np.ones((n,2),bool),subtypes=np.full(150,'closed_eye'),
                                  reset=np.r_[True,np.zeros(n-1,bool)])
                rows[split]=[dict(record_id=rid,subject_id=subject,split=split,start_slot=i*25,end_slot=i*25+99,
                                  label_known=i!=2,target=i%2,f_eligible=False) for i in range(3)]
            cache=SimpleNamespace(root=source/'head',manifest_sha256='head',records=metadata,
                                  arrays=lambda rid:heads[rid],report={'crop':crop_policy([0,0,1,1])})
            eye_cache=SimpleNamespace(root=source/'eyes',manifest_sha256='eye',records=metadata,
                                      arrays=lambda rid:eyes[rid],report={'fixture':True},close=lambda:None)
            labels=ActionLabels.__new__(ActionLabels);labels.records=records;labels.closed_eye_min_ms=500
            labels.provenance={'fixture':True};labels.digest=canonical_hash(labels.provenance)
            resets={rid:r['reset'] for rid,r in records.items()}
            baseline,pred=evaluate_events(cache,model,rows['val'],resets,500)
            variant='B_action_supervision-seed42-best-event'
            selected=_summary(baseline,variant=variant,arm='B_action_supervision_event',seed=42,
                              checkpoint_kind='best',checkpoint='fixture.pt',checkpoint_sha256='fixture',epoch=4)
            source_report={'action_metrics':{variant.removesuffix('-event'):action_metrics(pred,labels,cache)}}
            @contextmanager
            def inputs(*args):
                yield dict(base=SimpleNamespace(rows=rows),cache=cache,labels=labels,model=model,
                           selected=selected,report=source_report,provenance={'fixture':True},sources=[source])
            settings=dict(DEFAULTS,epochs=1,batch_size=16,accumulation_steps=1,early_stopping_patience=7,backend='cpu')
            with patch('training.visual.fatigue_eyes._inputs',inputs), \
                    patch('training.visual.fatigue_eyes.EyeCache',return_value=eye_cache),redirect_stdout(io.StringIO()):
                report=run(action_report=report_path,dataset_root=source,eye_cache=source/'eyes',
                           output_dir=root/'out',settings=settings)
                repeated=run(action_report=report_path,dataset_root=source,eye_cache=source/'eyes',
                             output_dir=root/'out',settings=settings,continue_run=True)
                self.assertEqual(report,repeated)
                self.assertEqual(report['training_runs'],6)
                self.assertEqual(len(report['comparisons']),25)
                self.assertIn('original_reference_roi_matched',report['frame_metrics'])
                self.assertEqual(report['reference_action_metrics'],source_report['action_metrics'][variant.removesuffix('-event')])
                self.assertEqual(read_json(root/'out/comparison.json'),report)
                for item in report['comparisons']:
                    self.assertEqual(item['window_label_hash'],selected['window_label_hash'])
                    self.assertEqual(item['overall']['positive'],1)
                    self.assertEqual(item['overall']['negative'],1)
                    if item.get('readout')=='combined':
                        self.assertEqual(item['changes_from_reference']['overall']['lost_positive'],0)
                self.assertTrue((root/'out/input-audit/index.html').exists())
                for key,value in original.items(): torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
                self.assertEqual(report['matched_cohort']['train']['head'],report['matched_cohort']['train']['eye'])
                complete_path=root/'out/A_head_independent-seed42/train-001/result.json'
                altered=read_json(complete_path);altered['completed_epochs']=99
                complete_path.write_text(json.dumps(altered),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'terminal|result'):
                    run(action_report=report_path,dataset_root=source,eye_cache=source/'eyes',
                        output_dir=root/'out',settings=settings,continue_run=True)


if __name__ == '__main__': unittest.main()
