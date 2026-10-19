"""Causal diagnostics must agree with the production event decoder."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager,redirect_stdout
from copy import deepcopy
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class BreakpointTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.eye_breakpoints'))
        from training.visual import eye_breakpoints
        return eye_breakpoints

    def fixture(self, n=12):
        head=dict(slot_indices=np.arange(n)*3,source_pts_ms=np.arange(n)*120,valid=np.ones(n,bool))
        eye=dict(valid=np.ones(n,bool),origin=np.ones(n,np.uint8),track_age_ms=np.zeros(n,np.int64))
        return head,eye,np.ones(n),np.zeros(n,bool)

    def test_boundary_and_repeated_pts_do_not_inflate_duration(self):
        api=self.api();head,eye,scores,resets=self.fixture()
        trace=api.trace_record(head,eye,scores,resets,1000)
        self.assertFalse(trace[8]['eye_event']);self.assertTrue(trace[9]['eye_event'])
        head['source_pts_ms'][9:]=960
        trace=api.trace_record(head,eye,scores,resets,1000)
        self.assertFalse(any(r['eye_event'] for r in trace))

    def test_missing_negative_and_reset_have_distinct_breaks(self):
        api=self.api();head,eye,scores,resets=self.fixture(30)
        eye['valid'][10]=False;scores[10]=np.nan;scores[20]=-.2;resets[25]=True
        trace=api.trace_record(head,eye,scores,resets,1000)
        self.assertEqual(trace[10]['break_reason'],'missing_eye')
        self.assertEqual(trace[20]['break_reason'],'below_threshold')
        self.assertEqual(trace[25]['break_reason'],'source_reset')
        self.assertEqual(trace[25]['closed_run_ms'],0)
        self.assertEqual(trace[20]['input_run_ms'],1080)
        self.assertIsNone(trace[10]['eye_logit'])

    def test_endpoint_diagnosis_does_not_use_earlier_event(self):
        api=self.api();head,eye,scores,resets=self.fixture(32)
        scores[-1]=-.1
        trace=api.trace_record(head,eye,scores,resets,1000)
        row=dict(record_id='a',subject_id='14',start_slot=0,end_slot=93,label_known=True,target=1,available=True,logit=-1.)
        detail=api.describe_window(row,'closed_eye',trace)
        self.assertEqual(detail['reason'],'predicted_open_endpoint')
        self.assertTrue(detail['input_duration_eligible']);self.assertTrue(detail['false_negative'])
        self.assertGreaterEqual(detail['longest_closed_run_seen_ms'],1000)
        self.assertFalse(detail['eye_event'])
        unknown=api.describe_window(dict(row,label_known=False),'closed_eye',trace)
        self.assertFalse(unknown['false_negative'])

    def test_carried_history_not_reset_at_window_start(self):
        api=self.api();head,eye,scores,resets=self.fixture(40)
        trace=api.trace_record(head,eye,scores,resets,1000)
        row=dict(record_id='a',subject_id='7',start_slot=72,end_slot=117,label_known=True,target=1,available=True,logit=1.)
        detail=api.describe_window(row,'closed_eye',trace)
        self.assertEqual(detail['endpoint_closed_run_ms'],4680)
        self.assertEqual(detail['reason'],'eye_detected')

    def test_playback_writes_seekable_frames_and_missing_eye_placeholder(self):
        api=self.api();head,eye,scores,resets=self.fixture(12)
        head['images']=np.full((12,3,224,224),130,np.uint8)
        eye['images']=np.full((12,3,224,224),80,np.uint8)
        eye['roi_xyxy']=np.tile([40.,40.,60.,60.],(12,1))
        eye['valid'][2]=False;scores[2]=np.nan
        trace=api.trace_record(head,eye,scores,resets,1000)
        with tempfile.TemporaryDirectory() as tmp:
            path=api.write_playback(Path(tmp),'window-a',head,eye,trace,0,11,[0,0,224,224])
            self.assertTrue((Path(tmp)/path).is_file())
            self.assertTrue((Path(tmp)/'window-a/replay.gif').is_file())
            html=(Path(tmp)/path).read_text(encoding='utf-8')
            self.assertIn('type="range"',html)
            self.assertIn('missing_eye',html)

    def test_pipeline_reproduces_full_scope_and_exports_request_without_training(self):
        api=self.api()
        from tests.test_eye_diagnostics import event_fixture
        from tests.test_eye_recovery import BrightEye
        from training.visual.action_eval import replay_events
        from training.visual.eye_composition import infer_eyes,compose_events,frame_metrics
        from training.visual.fatigue_spatial_suite import _summary
        from training.visual.spatial import crop_policy
        from training.visual.eye_frames import recovery_policy
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve();source=root/'source';source.mkdir()
            head,rows,actions,resets,a=event_fixture()
            a['images']=np.full((32,3,224,224),110,np.uint8)
            head.root=source/'head';head.manifest_sha256='head'
            ids=['a','b','c']
            head.records={rid:dict(record_id=rid,subject_id='14',split='val',source={'driver_roi':[0,0,224,224]}) for rid in ids}
            head.report={'crop':crop_policy([0,0,1,1])}
            rows=[dict(rows[0],record_id=rid,subject_id='14',split='val',target=int(rid=='a'),label_known=rid!='c',f_eligible=False) for rid in ids]
            actions={rid:np.tile([-2.,-2.],(32,1)) for rid in ids};resets={rid:np.zeros(32,bool) for rid in ids}
            baseline=replay_events(head,actions,rows,resets,1000)
            labels=SimpleNamespace(records={rid:dict(targets=np.tile([int(rid=='a'),0],(32,1)),masks=np.ones((32,2),bool),
                reset=resets[rid],subtypes=np.full(100,'closed_eye')) for rid in ids},closed_eye_min_ms=1000)
            b=dict(a,roi_xyxy=np.tile([40.,40.,60.,60.],(32,1)),origin=np.ones(32,np.uint8),track_age_ms=np.zeros(32,np.int64))
            eyes=SimpleNamespace(root=source/'eyes',manifest_sha256='eyes',records=head.records,arrays=lambda rid:b,
                report={'run':{'recovery':recovery_policy('dense_track','val')}},close=lambda:None)
            model=BrightEye();state=deepcopy(model.state_dict())
            prediction=infer_eyes(head,eyes,model,baseline,'eye')
            combined,eye_only=compose_events(head,eyes,actions,prediction,baseline,resets,1000)
            provenance={'fixture':True,'eye_report':str(source/'old.json')}
            report=dict(schema_version='driveguard-eye-recovery-comparison-v1',status='complete',split='val',reference_reproduced=True,
                training_runs=0,frozen_weights_unchanged=True,plan=dict(provenance=provenance,
                event_policy=dict(closed_eye_min_ms=1000,threshold_logit=0.,missing='reset-eye-only'),
                caches={'dense_track':dict(path=str(eyes.root),manifest_sha256='eyes',policy=recovery_policy('dense_track','val'))}),
                comparisons=[_summary(combined,variant='dense_track-combined'),_summary(eye_only,variant='dense_track-eye_only')],
                frame_metrics={'dense_track':frame_metrics(prediction,head,eyes,labels)},
                input_support={'dense_track':{'by_subtype':{'closed_eye':dict(positive=1,duration_eligible=1,endpoint_available=1)}}})
            path=source/'comparison.json';path.write_text(json.dumps(report))
            data=dict(cache=head,labels=labels,base=SimpleNamespace(rows={'val':rows}),model=BrightEye(),eye_model=model,
                selected=_summary(baseline),provenance=provenance,sources=[source])
            @contextmanager
            def reference(*args):yield data
            with patch.object(api,'_reference',reference),patch.object(api,'EyeCache',return_value=eyes), \
                 patch.object(api,'evaluate_events',return_value=(baseline,actions)),redirect_stdout(io.StringIO()):
                result=api.run(recovery_report=path,dataset_root=source,output_dir=root/'out')
                with patch.object(api,'evaluate_events',side_effect=AssertionError('completed outputs must be reused')):
                    again=api.run(recovery_report=path,dataset_root=source,output_dir=root/'out',continue_run=True)
                self.assertEqual(result,again)
                (root/'out/frames.csv').write_text('damaged')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    api.run(recovery_report=path,dataset_root=source,output_dir=root/'out',continue_run=True)
                original_write=Path.write_text
                def fail_entry(p,*args,**kwargs):
                    if p==root/'failed/index.html':raise OSError('simulated disk failure')
                    return original_write(p,*args,**kwargs)
                with patch.object(Path,'write_text',fail_entry):
                    with self.assertRaisesRegex(OSError,'disk failure'):
                        api.run(recovery_report=path,dataset_root=source,output_dir=root/'failed')
                self.assertFalse((root/'failed/summary.json').exists())
                report['comparisons'][0]['overall']['tp']=0;path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'predictions differ'):
                    api.run(recovery_report=path,dataset_root=source,output_dir=root/'bad')
            self.assertEqual(result['windows'],3);self.assertEqual(result['pure_eye_positive'],1)
            self.assertEqual(result['training_runs'],0);self.assertEqual(result['pure_eye_false_negative'],0)
            request=json.loads((root/'out/replay-request.json').read_text())
            self.assertEqual([r['record_id'] for r in request['records']],['a'])
            self.assertTrue((root/'out/frames.csv').exists());self.assertTrue((root/'out/index.html').exists())
            self.assertFalse((root/'bad/summary.json').exists())
            for name,value in model.state_dict().items():torch.testing.assert_close(value,state[name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
