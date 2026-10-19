"""Source telemetry is trusted only after reproducing cached native pixels."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tests.test_head_frames import frame_fixture
from tests.test_eye_frames import FakeFace
from training.visual.head_frames import FrameCache,build_frame_cache
from training.visual.eye_frames import build_eye_cache
from training.visual.spatial import crop_policy
from training.common.data import file_sha256


class ReplayTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.eye_tracking_replay'))
        from training.visual import eye_tracking_replay
        return eye_tracking_replay

    def test_source_replay_reproduces_pixels_resumes_and_detects_tampering(self):
        api=self.api()
        import cv2
        import sys
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp,source_count=101,source_step_ms=40) as (prepared,dataset,_):
            root=Path(tmp).resolve();build_frame_cache(prepared,crop_policy(),dataset,root/'head')
            head=FrameCache(root/'head')
            try:
                with patch.dict(sys.modules,{'cv2':cv2}),patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace):
                    build_eye_cache(head,dataset,root/'eye',recovery='dense_track',split='val')
                    (root/'inputs').mkdir();source=root/'inputs/recovery.json'
                    source.write_text(json.dumps(dict(schema_version='driveguard-eye-recovery-comparison-v1',status='complete',
                        plan={'caches':{'dense_track':dict(path=str(root/'eye'),manifest_sha256=file_sha256(root/'eye/manifest.json'))}})))
                    request=dict(schema_version='driveguard-eye-breakpoint-replay-request-v1',status='complete',
                        head_cache=str(root/'head'),head_cache_sha256=head.manifest_sha256,eye_cache=str(root/'eye'),
                        eye_cache_sha256=file_sha256(root/'eye/manifest.json'),source_report=str(source),source_report_sha256=file_sha256(source),
                        records=[dict(record_id='job1',windows=[dict(window_id='0'*24,start_pts_ms=0,end_pts_ms=1000)])])
                    path=root/'inputs/request.json';path.write_text(json.dumps(request))
                    result=api.run(request=path,dataset_root=dataset,output_dir=root/'replay')
                    self.assertTrue(result['cache_reproduced']);self.assertEqual(result['source_frames'],101)
                    self.assertEqual(result['records'][0]['verified_samples'],32)
                    with patch('inference.perception.video_source.decode_video',side_effect=AssertionError('no decode on resume')):
                        self.assertEqual(api.run(request=path,dataset_root=dataset,output_dir=root/'replay',continue_run=True),result)
                    stops=root/'replay/stop-reasons.csv';saved=stops.read_bytes();stops.write_text('damaged')
                    with self.assertRaisesRegex(ValueError,'checksum'):
                        api.run(request=path,dataset_root=dataset,output_dir=root/'replay',continue_run=True)
                    stops.write_bytes(saved)
                    csvpath=root/'replay'/result['records'][0]['trace_file'];csvpath.write_text('changed')
                    with self.assertRaisesRegex(ValueError,'checksum'):
                        api.run(request=path,dataset_root=dataset,output_dir=root/'replay',continue_run=True)
                    with patch.object(FakeFace,'predict',return_value=dict(status='missing',driver_confirmed=False,landmarks=None)):
                        with self.assertRaisesRegex(ValueError,'reproduce'):
                            api.run(request=path,dataset_root=dataset,output_dir=root/'mismatch')
                    self.assertFalse((root/'mismatch/summary.json').exists())
                    with patch.object(cv2,'__version__','different'):
                        with self.assertRaisesRegex(ValueError,'runtime'):
                            api.run(request=path,dataset_root=dataset,output_dir=root/'runtime')
            finally:head.close()

    def test_real_flow_replay_matches_cached_pixels_and_records_first_expiry(self):
        api=self.api()
        import cv2,sys
        from tests.test_eye_frames import face
        from inference.perception.video_source import VideoFrame
        from training.visual.eye_frames import EyeCache,_inputs
        class DetectOnce:
            def __init__(self,*args,**kwargs):pass
            def predict(self,bgr,t):
                self.last_face_diagnostics=dict(reason='ok' if t==0 else 'no_face_detected',detected_faces=int(t==0))
                return face() if t==0 else dict(status='ok',driver_confirmed=False,landmarks=None)
            def close(self):pass
        texture=np.random.default_rng(3).integers(0,256,(500,1000,3),np.uint8)
        texture=cv2.GaussianBlur(texture,(3,3),.6)
        def decode(*args):
            for t in range(0,4001,40):
                yield VideoFrame(t,cv2.warpAffine(texture,np.float32([[1,0,t/80],[0,1,0]]),(1000,500)))
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp,source_count=101,source_step_ms=40) as (prepared,dataset,_):
            root=Path(tmp).resolve();build_frame_cache(prepared,crop_policy(),dataset,root/'head');head=FrameCache(root/'head')
            try:
                with patch.dict(sys.modules,{'cv2':cv2}),patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',DetectOnce), \
                     patch('inference.perception.video_source.decode_video',decode):
                    manifest=build_eye_cache(head,dataset,root/'eyes',recovery='dense_track',split='val')
                    eyes=EyeCache(root/'eyes',head)
                    try:
                        _,items=_inputs(head,dataset,manifest['run']['detector_runtime'],manifest['run']['recovery'])
                        result=api._replay(items[0],head,eyes,[dict(window_id='0'*24,start_pts_ms=0,end_pts_ms=1000)],root/'trace.csv')
                        self.assertTrue(result['cache_reproduced']);self.assertEqual(result['verified_samples'],32)
                        counts={(r['tracking_status'],r['reason']):r['count'] for r in result['counters'] if r['scope']=='record_all_frames'}
                        self.assertEqual(counts['stopped','age_expired'],1)
                        self.assertGreater(counts['tracked','tracked'],0)
                        self.assertGreater(counts['unavailable','no_seed'],0)
                    finally:eyes.close()
            finally:head.close()


if __name__=='__main__':unittest.main()
