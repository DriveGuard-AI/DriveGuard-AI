"""Synthetic pixel replay, provenance and exact sparse sampling contracts."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inference.experts.assets import file_sha256
from inference.perception.video_source import VideoFrame
from tests.test_face_diagnostic_replay import replay_fixture
from tests.test_visual_extract import FakeEncoder
from training.common.config import read_json
from training.visual.spatial import crop_policy


@contextmanager
def frame_fixture(tmp, **options):
    """Real verified preparation over the immutable synthetic replay fixture."""
    from training.visual.data import PreparedData
    from training.visual.extract import extract
    with replay_fixture(tmp, **options) as (kwargs, _, source):
        base = Path(tmp)
        with patch('training.visual.encoder.FrozenEncoder', FakeEncoder):
            visual = extract(**kwargs, output_dir=base/'visual', weights=base/'Dataset/synthetic.pth',
                             weights_sha256=file_sha256(base/'Dataset/synthetic.pth'))
        root = base/'prepared'; root.mkdir()
        record = visual['records'][0]
        row = {k: record[k] for k in ('record_id','recording_id','subject_id','view','subset','split')}
        row.update(start_slot=0,end_slot=99,visual_start=0,label_known=False,target=0,
                   baseline_available=False,baseline_logit=None,h_eligible=False,f_eligible=False)
        (root/'val.json').write_text(json.dumps([row]),encoding='utf-8')
        np.save(root/'val.npy',np.zeros((1,1),np.float32))
        report = dict(schema_version='driveguard-visual-prepared-v2',status='complete',task='fatigue',
                      manifest=str(Path(kwargs['manifest']).resolve()),data_root=str(Path(kwargs['data_root']).resolve()),
                      visual_cache=str((base/'visual').resolve()),visual_cache_sha256=file_sha256(base/'visual/manifest.json'),
                      context_dim=1,normalization={},splits=dict(val=dict(context='val.npy',rows='val.json',
                      context_sha256=file_sha256(root/'val.npy'),rows_sha256=file_sha256(root/'val.json'))))
        (root/'prepared.json').write_text(json.dumps(report),encoding='utf-8')
        prepared = PreparedData(root)
        try: yield prepared, kwargs['dataset_root'], source
        finally: prepared.close()


class HeadFramesTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.head_frames'),'pixel cache implementation missing')
        from training.visual.head_frames import build_frame_cache, FrameCache, FrameDataset
        return build_frame_cache, FrameCache, FrameDataset

    def test_uint8_preprocessing_is_exactly_existing_normalization(self):
        from training.visual import encoder
        self.assertTrue(hasattr(encoder,'preprocess_pixels'),'reusable uint8 preprocessing missing')
        rng = np.random.default_rng(42)
        for shape,roi in [((31,53,3),[1.2,2.8,50.1,29.9]),((55,19,3),[0,0,19,55])]:
            bgr = rng.integers(0,256,shape,dtype=np.uint8)
            pixels = encoder.preprocess_pixels(bgr,roi)
            self.assertEqual(pixels.dtype,np.uint8)
            self.assertEqual(pixels.shape,(3,224,224))
            normalized = (pixels.astype(np.float32)/255-encoder.MEAN[:,None,None])/encoder.STD[:,None,None]
            np.testing.assert_array_equal(normalized,encoder.preprocess(bgr,roi))

    def test_dense_sparse_sampling_labels_and_immutable_reuse(self):
        build,Cache,Dataset = self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
            out=Path(tmp)/'pixels'
            def colored_decode(*args):
                for i in range(100):
                    image=np.zeros((500,1000,3),np.uint8); image[:]=[i,2*i,255-i]
                    yield VideoFrame(i*40,image)
            with patch('inference.perception.video_source.decode_video',colored_decode):
                report=build(prepared,crop_policy(),dataset,out)
            self.assertEqual(report['status'],'complete')
            self.assertEqual([r['split'] for r in report['records']],['val'])
            cache=Cache(out)
            try:
                dense=Dataset(prepared,cache,'val',32); sparse=Dataset(prepared,cache,'val',8)
                self.assertEqual(dense.frames,32); self.assertEqual(sparse.frames,8)
                self.assertEqual(dense.metadata(0),prepared.rows['val'][0])
                a,target,known=dense[0]; b,_,_=sparse[0]
                self.assertEqual(a['images'].shape,(32,3,224,224)); self.assertEqual(a['images'].dtype,np.uint8)
                self.assertEqual(target,np.float32(0)); self.assertFalse(known)
                np.testing.assert_array_equal(a['images'][3::4],b['images'])
                np.testing.assert_array_equal(a['visual_mask'][3::4],b['visual_mask'])
                arrays=cache.arrays('job1')
                expected=np.array([i for i in range(100) if i%25 in (2,5,8,12,15,18,21,24)])
                np.testing.assert_array_equal(arrays['slot_indices'],expected)
                np.testing.assert_array_equal(arrays['source_pts_ms'],expected*40)
                np.testing.assert_array_equal(a['images'][:,:,112,112],np.stack([255-expected,2*expected,expected],axis=1))
                self.assertIsInstance(arrays['images'],np.memmap)
                for frames in (0,16,True):
                    with self.assertRaises(ValueError):Dataset(prepared,cache,'val',frames)
                with self.assertRaises(ValueError):Dataset(prepared,cache,'test',8)
            finally:cache.close()
            before={str(p):file_sha256(p) for p in out.rglob('*') if p.is_file()}
            with patch('inference.perception.video_source.decode_video',side_effect=AssertionError('must reuse')):
                self.assertEqual(build(prepared,crop_policy(),dataset,out,continue_run=True),report)
            self.assertEqual(before,{p:file_sha256(p) for p in before})
            with self.assertRaises(FileExistsError):build(prepared,crop_policy(),dataset,out)
            with self.assertRaisesRegex(ValueError,'configuration|identity'):
                build(prepared,crop_policy([0,0,1,1]),dataset,out,continue_run=True)

    def test_full_source_clock_dimensions_and_new_attempt_after_failure(self):
        build,_,_=self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
            for count,height,changed in ((99,500,False),(101,500,False),(100,499,False),(100,500,True)):
                out=Path(tmp)/f'bad-{count}-{height}-{changed}'
                def wrong_decode(*args):
                    for i in range(count):
                        yield VideoFrame(i*40+int(changed and i==1),np.zeros((height,1000,3),np.uint8))
                with patch('inference.perception.video_source.decode_video',wrong_decode):
                    with self.assertRaisesRegex(ValueError,'count|PTS|dimensions'):build(prepared,crop_policy(),dataset,out)
                failure=list(out.rglob('failure.json')); self.assertEqual(len(failure),1)
                result=build(prepared,crop_policy(),dataset,out,continue_run=True)
                self.assertNotEqual(failure[0].parent,out/result['records'][0]['directory'])
                self.assertTrue(failure[0].is_file())

    def test_stale_frames_zero_and_publication_resume(self):
        build,Cache,_=self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp,source_count=21,source_step_ms=200) as (prepared,dataset,_):
            out=Path(tmp)/'pixels'; report=build(prepared,crop_policy(),dataset,out)
            cache=Cache(out)
            try:
                arrays=cache.arrays('job1'); valid=arrays['slot_indices']*40-arrays['source_pts_ms']<=80
                self.assertTrue((~valid).any()); np.testing.assert_array_equal(arrays['valid'],valid)
                self.assertTrue((arrays['images'][~valid]==0).all())
            finally:cache.close()
            (out/'manifest.json').unlink()
            with patch('inference.perception.video_source.decode_video',side_effect=AssertionError('reuse completed record')):
                self.assertEqual(build(prepared,crop_policy(),dataset,out,continue_run=True),report)

    def test_source_and_pixel_corruption_rejected(self):
        build,Cache,_=self.api()
        for change in ('source','pixels','original'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,source):
                out=Path(tmp)/'pixels'; report=build(prepared,crop_policy(),dataset,out)
                path=source if change=='source' else (out/report['records'][0]['arrays']['images'] if change=='pixels'
                                                       else next(Path(prepared.report['data_root']).rglob('targets.npy')))
                with path.open('ab') as handle:handle.write(b'corrupt')
                with self.assertRaises(ValueError):build(prepared,crop_policy(),dataset,out,continue_run=True)
                with self.assertRaises(ValueError):Cache(out)

    def test_output_overlap_and_changed_preparation_rejected(self):
        build,_,_=self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
            for root in (dataset,prepared.root,prepared.cache.root,Path(prepared.report['data_root'])):
                with self.assertRaisesRegex(ValueError,'separate|overlap'):build(prepared,crop_policy(),dataset,root/'pixels')
            with (prepared.root/'prepared.json').open('a',encoding='utf-8') as handle:handle.write(' ')
            with self.assertRaisesRegex(ValueError,'prepar'):build(prepared,crop_policy(),dataset,Path(tmp)/'bad')

    def test_rehashed_pts_and_shape_tampering_rejected_against_original(self):
        build,Cache,_=self.api()
        for change in ('pts','shape'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
                out=Path(tmp)/'pixels'; report=build(prepared,crop_policy(),dataset,out); row=report['records'][0]
                key='source_pts_ms' if change=='pts' else 'images'; path=out/row['arrays'][key]
                array=np.load(path)
                if change=='pts':array[0]-=1
                else:array=array[:,:,:,1:]
                np.save(path,array); row['sha256'][key]=file_sha256(path)
                (out/row['directory']/'record.json').write_text(json.dumps(row),encoding='utf-8')
                (out/'manifest.json').write_text(json.dumps(report),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'PTS|dimensions'):Cache(out)

    def test_disk_exhaustion_is_resumable_without_partial_record(self):
        build,_,_=self.api()
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir='runs') as tmp,frame_fixture(tmp) as (prepared,dataset,_):
            out=Path(tmp)/'pixels'
            with patch('training.visual.head_frames.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError,'disk space'):build(prepared,crop_policy(),dataset,out)
            self.assertFalse((out/'manifest.json').exists()); self.assertEqual(list(out.rglob('images.npy')),[])
            self.assertEqual(build(prepared,crop_policy(),dataset,out,continue_run=True)['status'],'complete')


if __name__=='__main__':unittest.main()
