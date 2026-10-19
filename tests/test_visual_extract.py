"""Synthetic extraction integration; no video/model downloads or quality claims."""
import importlib.util
import builtins
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inference.experts.assets import file_sha256
from inference.perception.video_source import VideoFrame
from tests.test_face_diagnostic_replay import replay_fixture


class FakeEncoder:
    batches = []
    def __init__(self, weights, weights_sha256, **kwargs):
        from training.visual.encoder import encoder_metadata
        self.metadata = encoder_metadata(weights_sha256)
    def encode(self, batch):
        self.batches.append(len(batch))
        return np.ones((len(batch),512),np.float32)


class VisualExtractTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.extract'), 'extract missing')
        from training.visual.extract import extract
        return extract

    def run_extract(self, kwargs, output, **extra):
        with patch('training.visual.encoder.FrozenEncoder',FakeEncoder):
            return self.api()(**kwargs, output_dir=output,weights=Path(kwargs['dataset_root'])/'synthetic.pth',
                              weights_sha256=file_sha256(Path(kwargs['dataset_root'])/'synthetic.pth'),batch_size=3,**extra)

    def test_causal_slots_layout_and_resume(self):
        self.api()
        FakeEncoder.batches = []
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (kwargs,_,_):
            output = Path(tmp)/'visual'
            original_import = builtins.__import__
            def no_mediapipe(name,*args,**options):
                if name.startswith('mediapipe') or name.endswith('mediapipe_adapter'):
                    raise AssertionError('RGB extraction must not import MediaPipe')
                return original_import(name,*args,**options)
            with patch('builtins.__import__',no_mediapipe):
                report = self.run_extract(kwargs,output)
            self.assertEqual(report['status'],'complete')
            row = report['records'][0]
            self.assertEqual(row['slots'],100)
            self.assertEqual(row['sample_count'],8)
            self.assertEqual(FakeEncoder.batches,[3,3,2])
            arrays = {key:np.load(output/value,allow_pickle=False) for key,value in row['arrays'].items()}
            np.testing.assert_array_equal(arrays['slot_indices'],[12,24,37,49,62,74,87,99])
            np.testing.assert_array_equal(arrays['source_pts_ms'],arrays['slot_indices']*40)
            self.assertTrue(arrays['valid'].all())
            self.assertEqual(arrays['embeddings'].shape,(8,512))
            self.assertTrue((output/row['directory']/'record.json').is_file())
            before = {str(p):file_sha256(p) for p in output.rglob('*') if p.is_file()}
            with patch('inference.perception.video_source.decode_video',side_effect=AssertionError('resume must not decode')):
                self.assertEqual(self.run_extract(kwargs,output,continue_run=True),report)
            self.assertEqual(before,{p:file_sha256(p) for p in before})
            with self.assertRaises(FileExistsError): self.run_extract(kwargs,output)

    def test_original_dimensions_and_full_decoded_frame_count(self):
        self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (kwargs,_,_):
            for count,size in ((99,(500,1000,3)),(101,(500,1000,3)),(100,(499,1000,3))):
                def wrong_decode(*args):
                    for index in range(count):
                        yield VideoFrame(index*40,np.zeros(size,np.uint8))
                with self.subTest(count=count,size=size),patch('inference.perception.video_source.decode_video',wrong_decode):
                    with self.assertRaisesRegex(ValueError,'count|PTS|dimensions'):
                        self.run_extract(kwargs,Path(tmp)/f'wrong-{count}-{size[0]}')

    def test_completed_record_reused_after_interrupted_publication(self):
        self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (kwargs,_,_):
            output = Path(tmp)/'visual'
            report = self.run_extract(kwargs,output)
            (output/'manifest.json').unlink()
            with patch('inference.perception.video_source.decode_video',side_effect=AssertionError('must reuse record')):
                self.assertEqual(self.run_extract(kwargs,output,continue_run=True),report)

    def test_interrupted_manifest_publication_never_leaves_partial_manifest(self):
        self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (kwargs,_,_):
            output=Path(tmp)/'visual';link=os.link
            def interrupt(source,destination):
                if Path(destination).name=='manifest.json':raise OSError('interrupted publication')
                return link(source,destination)
            with patch('training.visual.extract.os.link',side_effect=interrupt):
                with self.assertRaisesRegex(OSError,'interrupted'):self.run_extract(kwargs,output)
            self.assertFalse((output/'manifest.json').exists())
            markers=list(output.rglob('record.json'));self.assertEqual(len(markers),1)
            with patch('inference.perception.video_source.decode_video',side_effect=AssertionError('reuse completed record')):
                report=self.run_extract(kwargs,output,continue_run=True)
            self.assertEqual(report['status'],'complete')
            self.assertEqual((output/report['records'][0]['directory']/'record.json').resolve(),markers[0].resolve())

    def test_sampling_boundary_causality(self):
        self.api()
        from training.visual.extract import selected_slots
        self.assertEqual(selected_slots(12).tolist(),[])
        self.assertEqual(selected_slots(13).tolist(),[12])
        self.assertEqual(selected_slots(25).tolist(),[12,24])
        self.assertEqual(selected_slots(100).tolist(),[12,24,37,49,62,74,87,99])

    def test_invalid_record_marker_gets_new_attempt(self):
        self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (kwargs,_,_):
            output = Path(tmp)/'visual'
            report = self.run_extract(kwargs,output)
            old = output/report['records'][0]['directory']
            (output/'manifest.json').unlink()
            (old/'record.json').write_text('[]',encoding='utf-8')
            resumed = self.run_extract(kwargs,output,continue_run=True)
            self.assertNotEqual(resumed['records'][0]['directory'],report['records'][0]['directory'])
            self.assertEqual((old/'record.json').read_text(encoding='utf-8'),'[]')

    def test_stale_selected_slots_zeroed(self):
        self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp,source_count=21,source_step_ms=200) as (kwargs,_,_):
            report = self.run_extract(kwargs,Path(tmp)/'visual')
            row = report['records'][0]
            arrays = {k:np.load(Path(tmp)/'visual'/v) for k,v in row['arrays'].items()}
            expected = arrays['slot_indices']*40-arrays['source_pts_ms'] <= 80
            self.assertTrue((~expected).any())
            np.testing.assert_array_equal(arrays['valid'],expected)
            self.assertTrue((arrays['embeddings'][~expected] == 0).all())

    def test_every_slot_pts_and_image_count_validated_failed_attempt_retained(self):
        self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (kwargs,_,_):
            output = Path(tmp)/'visual'
            def wrong_decode(*args):
                for index in range(100):
                    yield VideoFrame(index*40+(1 if index == 1 else 0),np.zeros((500,1000,3),np.uint8))
            with patch('inference.perception.video_source.decode_video',wrong_decode):
                with self.assertRaisesRegex(ValueError,'PTS'): self.run_extract(kwargs,output)
            self.assertFalse((output/'manifest.json').exists())
            failures = list(output.rglob('failure.json'))
            self.assertEqual(len(failures),1)
            report = self.run_extract(kwargs,output,continue_run=True)
            self.assertEqual(report['status'],'complete')
            self.assertTrue(failures[0].exists())
            self.assertNotEqual(failures[0].parent,output/report['records'][0]['directory'])

    def test_original_source_and_cache_corruption_rejected(self):
        self.api()
        for change in ('source','cache'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (kwargs,_,source):
                if change == 'source': source.write_bytes(b'changed')
                else:
                    target = next(Path(kwargs['data_root']).rglob('targets.npy'))
                    with target.open('ab') as handle: handle.write(b'corrupt')
                with self.assertRaises(ValueError): self.run_extract(kwargs,Path(tmp)/'visual')

    def test_changed_run_config_and_corrupt_completed_arrays_rejected(self):
        self.api()
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (kwargs,_,_):
            output = Path(tmp)/'visual'
            report = self.run_extract(kwargs,output)
            with patch('training.visual.encoder.FrozenEncoder',FakeEncoder):
                with self.assertRaisesRegex(ValueError,'configuration'):
                    self.api()(**kwargs,output_dir=output,weights=Path(kwargs['dataset_root'])/'synthetic.pth',
                               weights_sha256='0'*64,batch_size=3,continue_run=True)
            with (output/report['records'][0]['arrays']['embeddings']).open('ab') as handle: handle.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'completed'): self.run_extract(kwargs,output,continue_run=True)


if __name__ == '__main__': unittest.main()
