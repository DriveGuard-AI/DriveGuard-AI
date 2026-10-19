"""Recovery caches preserve source clocks and causal localization snapshots."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import importlib.util

import numpy as np

from tests.test_head_frames import frame_fixture
from tests.test_eye_frames import FakeFace,face
from inference.perception.video_source import VideoFrame
from training.visual.head_frames import FrameCache, build_frame_cache
from training.visual.spatial import crop_policy
from training.visual import eye_frames as api


class Locator:
    calls=[]
    def __init__(self,*args,**kwargs):pass
    def step(self,bgr,timestamp_ms):
        self.calls.append(timestamp_ms)
        # Every source frame has a different ROI: catch look-ahead leakage.
        x=100+timestamp_ms/100
        return dict(roi=[x,100,x+40,140],width=20.,reason='accepted',
                    detector_reason='ok',origin='detected',track_age_ms=0)


class RecoveryCacheTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('cv2'),'flow extraction requires OpenCV')
    def test_real_locator_flow_cache_uses_current_native_pixels(self):
        import cv2
        from inference.experts.mediapipe_adapter import face_observation
        class DetectOnce:
            def __init__(self,*args,**kwargs):pass
            def predict(self,bgr,t):
                self.last_face_diagnostics={'reason':'ok' if t==0 else 'no_face_detected','detected_faces':int(t==0)}
                return face() if t==0 else face_observation([],1000,500,[40,0,520,325],True,t,'fixture',matrices=[])
            def close(self):pass
        texture=np.random.default_rng(3).integers(0,256,(500,1000,3),np.uint8)
        texture=cv2.GaussianBlur(texture,(3,3),.6)
        def image(t):return cv2.warpAffine(texture,np.float32([[1,0,t/80],[0,1,0]]),(1000,500))
        def decode(*args):
            for t in range(0,4001,40):yield VideoFrame(t,image(t))
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp,source_count=101,source_step_ms=40) as (prepared,dataset,_):
            root=Path(tmp);build_frame_cache(prepared,crop_policy(),dataset,root/'heads');head=FrameCache(root/'heads')
            try:
                with patch.dict(sys.modules,{'cv2':cv2}),patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',DetectOnce),patch('inference.perception.video_source.decode_video',decode):
                    api.build_eye_cache(head,dataset,root/'eyes',recovery='dense_track',split='val')
                with patch.dict(sys.modules,{'cv2':None,'mediapipe':None}):
                    cache=api.EyeCache(root/'eyes',head)
                try:
                    a=cache.arrays('job1');tracked=np.flatnonzero(a['origin']==2)
                    self.assertGreater(len(tracked),0)
                    self.assertTrue((a['track_age_ms'][tracked]<=250).all())
                    self.assertFalse(a['valid'][a['source_pts_ms']>250].any())
                    for i in tracked:
                        np.testing.assert_array_equal(a['images'][i],api._pixels(image(int(a['source_pts_ms'][i])),a['roi_xyxy'][i]))
                finally:cache.close()
            finally:head.close()

    def test_dense_detection_uses_all_source_frames_but_past_sample_geometry(self):
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
            root=Path(tmp);build_frame_cache(prepared,crop_policy(),dataset,root/'heads');head=FrameCache(root/'heads')
            Locator.calls=[]
            stub=SimpleNamespace(EyeLocator=Locator,POLICY={'test':True})
            try:
                with patch.dict(sys.modules,{'training.visual.eye_tracking':stub}),patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace):
                    report=api.build_eye_cache(head,dataset,root/'eyes',recovery='dense',split='all')
                    cache=api.EyeCache(root/'eyes',head)
                    try:
                        a=cache.arrays('job1'); b=head.arrays('job1')
                        self.assertGreater(len(Locator.calls),len(a['valid']))
                        np.testing.assert_array_equal(a['source_pts_ms'],b['source_pts_ms'])
                        np.testing.assert_allclose(a['roi_xyxy'][:,0],100+a['source_pts_ms']/100,rtol=1e-6)
                        self.assertTrue((a['origin']==1).all());self.assertTrue((a['track_age_ms']==0).all())
                        self.assertEqual(report['records'][0]['audit']['localization']['detector_calls']['ok'],len(Locator.calls))
                    finally:cache.close()
                    with patch.object(Locator,'step',side_effect=AssertionError('resume must reuse')):
                        self.assertEqual(api.build_eye_cache(head,dataset,root/'eyes',recovery='dense',split='all',continue_run=True),report)
                    with self.assertRaisesRegex(ValueError,'configuration|provenance'):
                        api.build_eye_cache(head,dataset,root/'eyes',recovery='sampled8',split='all',continue_run=True)
            finally:head.close()

    def test_detailed_missing_reason_and_explicit_scope(self):
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
            root=Path(tmp);build_frame_cache(prepared,crop_policy(),dataset,root/'heads');head=FrameCache(root/'heads')
            missing=dict(roi=None,width=0.,reason='unconfirmed_or_missing_face',detector_reason='outside_roi',origin='missing',track_age_ms=0)
            stub=SimpleNamespace(EyeLocator=Locator,POLICY={'test':True})
            try:
                with patch.dict(sys.modules,{'training.visual.eye_tracking':stub}),patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace),patch.object(Locator,'step',return_value=missing):
                    report=api.build_eye_cache(head,dataset,root/'eyes',recovery='sampled8',split='all')
                    audit=report['records'][0]['audit']
                    self.assertEqual(audit['reasons'],{'outside_roi':audit['sample_count']})
                    cache=api.EyeCache(root/'eyes',head)
                    self.assertFalse(cache.arrays('job1')['valid'].any());cache.close()
                with self.assertRaisesRegex(ValueError,'split|scope|records'):
                    api.build_eye_cache(head,dataset,root/'empty',recovery='dense',split='train')
            finally:head.close()


if __name__=='__main__':unittest.main()
