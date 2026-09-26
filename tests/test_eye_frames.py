"""Native eye geometry, causal clocks, immutable provenance and bounded replay."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inference.experts.assets import file_sha256
from inference.perception.video_source import VideoFrame
from tests.test_head_frames import frame_fixture
from training.visual.head_frames import FrameCache, build_frame_cache
from training.visual.spatial import crop_policy


def face():
    points = np.tile([250.,150.],(478,1))
    points[33],points[133] = [200,140],[220,140]
    points[362],points[263] = [270,140],[300,140]
    return dict(status='ok',driver_confirmed=True,landmarks=points.tolist())


class FakeFace:
    calls = []
    options = []
    response = None
    def __init__(self,asset,**options):
        self.options.append(options)
    def predict(self,bgr,timestamp_ms):
        self.calls.append(timestamp_ms)
        return deepcopy(self.response if self.response is not None else face())
    def close(self):pass


class EyeFramesTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.eye_frames'),'native eye cache missing')
        from training.visual import eye_frames
        return eye_frames

    def setUp(self):
        FakeFace.calls=[]; FakeFace.options=[]; FakeFace.response=None

    def test_geometry_uses_larger_eye_without_openness_and_rejects_outside(self):
        api=self.api(); observation=face(); head=[40,0,520,325]
        roi,width,reason=api.eye_geometry(observation,head,[1000,500])
        np.testing.assert_array_equal(roi,[255,110,315,170]); self.assertEqual(width,30)
        self.assertEqual(reason,'accepted')
        for change in ('missing','unconfirmed','tiny','hand','outside','padded_outside','nonfinite'):
            bad=deepcopy(observation)
            if change=='missing':bad['landmarks']=None
            elif change=='unconfirmed':bad['driver_confirmed']=False
            elif change=='tiny':bad['landmarks']=np.tile([250.,150.],(478,1)).tolist()
            elif change=='hand':bad['landmarks']=bad['landmarks'][:21]
            elif change=='outside':bad['landmarks'][0]=[521,100]
            elif change=='padded_outside':bad['landmarks'][362:363]=[[480,140]];bad['landmarks'][263]=[510,140]
            else:bad['landmarks'][33][0]=float('nan')
            with self.subTest(change=change):self.assertIsNone(api.eye_geometry(bad,head,[1000,500])[0])

    def test_native_rgb_sampling_freshness_and_unique_source_inference(self):
        api=self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp,source_count=21,source_step_ms=200) as (prepared,dataset,_):
            base=Path(tmp);build_frame_cache(prepared,crop_policy(),dataset,base/'heads');head=FrameCache(base/'heads')
            try:
                def decoded(*args):
                    for i in range(21):
                        frame=np.zeros((500,1000,3),np.uint8);frame[110:170,255:315]=[i,40,90]
                        yield VideoFrame(i*200,frame)
                with patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace),patch('inference.perception.video_source.decode_video',decoded):
                    report=api.build_eye_cache(head,dataset,base/'eyes')
                cache=api.EyeCache(base/'eyes',head)
                try:
                    a,b=cache.arrays('job1'),head.arrays('job1')
                    for key in ('slot_indices','source_pts_ms','valid'):np.testing.assert_array_equal(a[key],b[key])
                    self.assertEqual(FakeFace.calls,sorted(set(b['source_pts_ms'][b['valid']].tolist())))
                    self.assertEqual(FakeFace.options[0]['variant']['mode'],'VIDEO')
                    np.testing.assert_allclose(FakeFace.options[0]['variant']['crop_xyxy'],[.04,0,.52,.65])
                    valid=a['valid'];self.assertTrue((~valid).any())
                    self.assertTrue((a['images'][~valid]==0).all());self.assertTrue((a['roi_xyxy'][~valid]==0).all())
                    self.assertTrue((a['native_eye_width'][valid]==30).all())
                    np.testing.assert_array_equal(a['images'][valid,0,112,112],90)
                    np.testing.assert_array_equal(a['images'][valid,2,112,112],a['source_pts_ms'][valid]//200)
                    self.assertEqual(report['records'][0]['audit']['available'],int(valid.sum()))
                finally:cache.close()
                before={str(p):file_sha256(p) for p in (base/'eyes').rglob('*') if p.is_file()}
                with patch('inference.perception.video_source.decode_video',side_effect=AssertionError('must reuse')):
                    self.assertEqual(api.build_eye_cache(head,dataset,base/'eyes',continue_run=True),report)
                    (base/'eyes/manifest.json').unlink()
                    self.assertEqual(api.build_eye_cache(head,dataset,base/'eyes',continue_run=True),report)
                self.assertEqual(before,{p:file_sha256(p) for p in before})
            finally:head.close()

    def test_failed_attempt_retained_and_geometry_missing_is_not_an_error(self):
        api=self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
            base=Path(tmp);build_frame_cache(prepared,crop_policy(),dataset,base/'heads');head=FrameCache(base/'heads')
            try:
                with patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace),patch.object(FakeFace,'predict',side_effect=RuntimeError('detector failure')):
                    with self.assertRaisesRegex(RuntimeError,'detector failure'):api.build_eye_cache(head,dataset,base/'eyes')
                failures=list((base/'eyes').rglob('failure.json'));self.assertEqual(len(failures),1)
                FakeFace.response=dict(status='missing',driver_confirmed=False,landmarks=None)
                with patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace):
                    report=api.build_eye_cache(head,dataset,base/'eyes',continue_run=True)
                self.assertTrue(failures[0].exists());self.assertEqual(report['records'][0]['audit']['available'],0)
                cache=api.EyeCache(base/'eyes',head)
                try:self.assertFalse(cache.arrays('job1')['valid'].any())
                finally:cache.close()
            finally:head.close()

    def test_rehashed_geometry_and_pts_and_asset_tampering_rejected(self):
        api=self.api()
        for change in ('geometry','pts','asset','pixels'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
                base=Path(tmp);build_frame_cache(prepared,crop_policy(),dataset,base/'heads');head=FrameCache(base/'heads')
                try:
                    with patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace):report=api.build_eye_cache(head,dataset,base/'eyes')
                    row=report['records'][0]
                    if change=='asset':(Path(dataset)/'synthetic.task').write_bytes(b'changed')
                    else:
                        key={'geometry':'roi_xyxy','pts':'source_pts_ms','pixels':'images'}[change];path=base/'eyes'/row['arrays'][key]
                        a=np.load(path)
                        if change=='geometry':a[0]=[0,0,60,60]
                        elif change=='pts':a[0]-=1
                        else:a[0]=0
                        np.save(path,a)
                        if change!='pixels':
                            row['sha256'][key]=file_sha256(path)
                            (base/'eyes'/row['directory']/'record.json').write_text(json.dumps(row))
                            (base/'eyes/manifest.json').write_text(json.dumps(report))
                    with self.assertRaises(ValueError):api.EyeCache(base/'eyes',head)
                    with self.assertRaises(ValueError):api.build_eye_cache(head,dataset,base/'eyes',continue_run=True)
                finally:head.close()

    def test_cli_import_does_not_import_torch(self):
        self.api()
        command="import sys; sys.modules['torch']=None; from training.visual.eye_frames import main; main(['--help'])"
        result=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--head-cache',result.stdout)

    def test_model_version_bound_on_resume_but_reader_needs_no_mediapipe(self):
        api=self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
            base=Path(tmp);build_frame_cache(prepared,crop_policy(),dataset,base/'heads');head=FrameCache(base/'heads')
            try:
                with patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace):
                    report=api.build_eye_cache(head,dataset,base/'eyes')
                self.assertEqual(report['run'].get('detector_runtime',{}).get('mediapipe_version'),'test-only')
                with patch.dict(sys.modules,mediapipe=None):
                    cache=api.EyeCache(base/'eyes',head);cache.close()
                with patch.object(sys.modules['mediapipe'],'__version__','changed'):
                    with self.assertRaisesRegex(ValueError,'configuration'):api.build_eye_cache(head,dataset,base/'eyes',continue_run=True)
            finally:head.close()

    def test_pixel_preprocessing_matches_original_without_torch(self):
        api=self.api()
        from training.visual.encoder import preprocess_pixels
        image=np.random.default_rng(4).integers(0,256,(49,91,3),dtype=np.uint8)
        for roi in ([4.2,6.7,40.8,43.3],[10,20,12,22]):
            np.testing.assert_array_equal(api._pixels(image,roi),preprocess_pixels(image,roi))

    def test_bad_clock_and_disk_space_are_resumable_and_output_overlap_rejected(self):
        api=self.api()
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,source):
            base=Path(tmp);build_frame_cache(prepared,crop_policy(),dataset,base/'heads');head=FrameCache(base/'heads')
            try:
                for directory in (head.root,Path(dataset),prepared.root,prepared.cache.root,Path(prepared.report['data_root'])):
                    with self.assertRaisesRegex(ValueError,'separate'):api.build_eye_cache(head,dataset,directory/'eyes')
                with patch('training.visual.eye_frames.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
                    with self.assertRaisesRegex(OSError,'disk space'):api.build_eye_cache(head,dataset,base/'eyes')
                self.assertEqual(list((base/'eyes').rglob('images.npy')),[])
                def bad_decode(*args):
                    for i in range(100):yield VideoFrame(i*40+int(i==1),np.zeros((500,1000,3),np.uint8))
                with patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace),patch('inference.perception.video_source.decode_video',bad_decode):
                    with self.assertRaisesRegex(ValueError,'PTS'):api.build_eye_cache(head,dataset,base/'eyes',continue_run=True)
                self.assertEqual(len(list((base/'eyes').rglob('failure.json'))),1)
                with patch('inference.experts.face_experiment_adapter.FaceExperimentAdapter',FakeFace):api.build_eye_cache(head,dataset,base/'eyes',continue_run=True)
                with source.open('ab') as handle:handle.write(b'changed')
                with self.assertRaises(ValueError):api.EyeCache(base/'eyes',head)
            finally:head.close()


if __name__=='__main__':unittest.main()
