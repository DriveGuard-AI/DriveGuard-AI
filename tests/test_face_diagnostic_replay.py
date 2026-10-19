"""Exercise actual replay with synthetic external decoder/model APIs only."""
from contextlib import contextmanager
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inference.experts.assets import file_sha256
from inference.perception.video import run_video
from inference.perception.gaze_profile import gaze_profile
from training.common.config import canonical_hash
from training.dmd.cache import ARRAYS, METADATA, write_json
from training.gaze.profile import default_group_profile
from tests.test_expert_runtime_flow import runtime_fixture


@contextmanager
def replay_fixture(base, **options):
    base=Path(base); dataset=base/'Dataset'; dataset.mkdir()
    with runtime_fixture(dataset,gaze=True,**options) as (source, config, closed):
        config['view']='body'
        cache=base/'cache'; attempt=cache/'records/job1/attempt1'; attempt.mkdir(parents=True)
        report=run_video(source,attempt/'features',config,feature_profile=gaze_profile(),group_profile=default_group_profile())
        n=report['slots']
        job=dict(job_id='job1',subject_id='14',recording_id='synthetic',view='body',subset='distraction',
                 split='val',video='synthetic.mp4',video_size_bytes=source.stat().st_size)
        inputs=dict(job=job,video_sha256=file_sha256(source))
        write_json(attempt/'input.json',inputs)
        arrays={**ARRAYS,'gaze_targets':'gaze_targets.npy','gaze_label_masks':'gaze_label_masks.npy'}
        data={'targets':np.zeros((n,4),np.float32),'label_masks':np.ones((n,4),np.uint8),
              'source_pts_ms':np.arange(options.get('source_count',100),dtype=np.int64)*options.get('source_step_ms',40),
              'gaze_targets':np.zeros(n,np.int64),'gaze_label_masks':np.zeros(n,np.uint8)}
        for name,array in data.items(): np.save(attempt/arrays[name],array)
        complete=dict(schema_version='driveguard-dmd-complete-v2',fingerprint=canonical_hash(inputs),job=job,
                      feature_record=report,arrays=arrays,array_sha256={k:file_sha256(attempt/v) for k,v in arrays.items()},label_summary={})
        write_json(attempt/'complete.json',complete)
        record=dict(record_id='job1',**{k:job[k] for k in METADATA},
                    **{k:(attempt/v).relative_to(cache).as_posix() for k,v in arrays.items() if k!='source_pts_ms'})
        manifest={'schema_version':'driveguard-feature-manifest-v3','records':[record]}
        write_json(cache/'manifest.json',manifest)
        yield dict(manifest=cache/'manifest.json',data_root=cache,dataset_root=dataset),closed,source


class FaceReplayTests(unittest.TestCase):
    def runner(self):
        self.assertIsNotNone(importlib.util.find_spec('training.dmd.face_diagnostics'),'replay tool missing')
        from training.dmd.face_diagnostics import diagnose
        return diagnose

    def test_replay_matches_cache_and_preserves_inputs(self):
        run=self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base) as (kwargs,closed,source):
            initial={str(p):file_sha256(p) for p in Path(base).rglob('*') if p.is_file()}
            report=run(**kwargs,output_dir=Path(base)/'report',samples_per_reason=0)
            self.assertEqual(report['status'],'complete')
            record=report['records'][0]
            self.assertEqual(record['slots'],100)
            self.assertEqual(record['unique_inference_frames'],100)
            self.assertEqual(record['reason_counts'],{'ok':100})
            self.assertEqual(record['head_mask_mismatches'],0)
            self.assertEqual(record['iris_mask_mismatches'],0)
            self.assertTrue(record['model_id_matches'])
            self.assertFalse(record['partial'])
            self.assertTrue(report['groups']['view']['body']['runtime_and_masks_match'])
            self.assertEqual(initial,{p:file_sha256(p) for p in initial})
            self.assertTrue((Path(base)/'report/index.html').is_file())
            with (Path(base)/'report/record-001/frames.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(rows[-1]['slot_ms'],'3960')
            self.assertEqual(rows[-1]['cached_head_valid'],'True')
            self.assertEqual(closed.count('task'),4)

    def test_partial_run_and_source_reuse(self):
        run=self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base,source_count=34,source_step_ms=120) as (kwargs,_,_):
            report=run(**kwargs,output_dir=Path(base)/'report',samples_per_reason=0,max_seconds=1)
            record=report['records'][0]
            self.assertTrue(record['partial'])
            self.assertEqual(record['slots'],25)
            self.assertEqual(record['unique_inference_frames'],9)
            self.assertEqual(record['head_mask_mismatches'],0)

    def test_bad_provenance_and_existing_output_rejected(self):
        run=self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base) as (kwargs,_,source):
            output=Path(base)/'report'; output.mkdir()
            with self.assertRaises(FileExistsError): run(**kwargs,output_dir=output)
            source.write_bytes(b'changed video')
            with self.assertRaisesRegex(ValueError,'video|Video'): run(**kwargs,output_dir=Path(base)/'bad')
            self.assertFalse((Path(base)/'bad').exists())

    def test_cache_corruption_and_empty_selection_rejected(self):
        run=self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base) as (kwargs,_,_):
            with self.assertRaisesRegex(ValueError,'No records'): run(**kwargs,output_dir=Path(base)/'empty',view='face')
            path=kwargs['data_root']/'records/job1/attempt1/features/masks.npy'
            data=np.load(path); data[0,5]=0; np.save(path,data)
            with self.assertRaisesRegex(ValueError,'cache|Cache'): run(**kwargs,output_dir=Path(base)/'bad')
            self.assertFalse((Path(base)/'bad').exists())

    def test_runtime_failure_has_no_success_marker(self):
        run=self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base) as (kwargs,closed,_):
            with patch('inference.experts.mediapipe_adapter.MediaPipeAdapter.predict',side_effect=RuntimeError('replay failed')):
                with self.assertRaisesRegex(RuntimeError,'replay failed'):
                    run(**kwargs,output_dir=Path(base)/'bad',samples_per_reason=0)
            self.assertFalse((Path(base)/'bad/summary.json').exists())
            self.assertEqual(closed.count('task'),4)
            self.assertEqual(closed.count('decoder'),2)

    def test_pose_failure_is_reported_as_cache_disagreement(self):
        run=self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base) as (kwargs,_,_):
            with patch('inference.experts.gaze_geometry.head_pose',return_value=None):
                report=run(**kwargs,output_dir=Path(base)/'report',samples_per_reason=0)
            record=report['records'][0]
            self.assertEqual(record['reason_counts'],{'pose_matrix_invalid':100})
            self.assertEqual(record['head_mask_mismatches'],100)
            self.assertEqual(record['iris_mask_mismatches'],0)
            self.assertFalse(record['head_iris_masks_reproduced'])

    def test_high_rate_source_matches_original_sampling(self):
        run=self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base,source_count=200,source_step_ms=20) as (kwargs,_,_):
            record=run(**kwargs,output_dir=Path(base)/'report',samples_per_reason=0)['records'][0]
            self.assertEqual(record['unique_inference_frames'],100)
            self.assertEqual(record['source_pts_mismatches'],0)
            self.assertTrue(record['head_iris_masks_reproduced'])

    def test_cleanup_failure_has_no_success_marker(self):
        run=self.runner()
        from inference.experts.mediapipe_adapter import MediaPipeAdapter
        original=MediaPipeAdapter.close
        def fail(adapter):
            original(adapter)
            raise RuntimeError('close failed')
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base) as (kwargs,closed,_):
            with patch.object(MediaPipeAdapter,'close',fail):
                with self.assertRaisesRegex(RuntimeError,'close failed'):
                    run(**kwargs,output_dir=Path(base)/'bad',samples_per_reason=0)
            self.assertFalse((Path(base)/'bad/summary.json').exists())
            self.assertEqual(closed.count('task'),4)

    def test_source_escape_and_output_inside_cache_rejected(self):
        run=self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base) as (kwargs,_,_):
            with self.assertRaisesRegex(ValueError,'separate'):
                run(**kwargs,output_dir=kwargs['data_root']/'report',samples_per_reason=0)
            manifest=json.loads(kwargs['manifest'].read_text())
            manifest['records'][0]['values']='../Dataset/synthetic.mp4'
            kwargs['manifest'].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'outside'):
                run(**kwargs,output_dir=Path(base)/'report',samples_per_reason=0)

    def test_source_change_between_validation_and_replay_rejected(self):
        run=self.runner()
        from training.dmd.face_diagnostics import _replay
        def change(item,*args):
            item['source'].write_bytes(b'changed after validation')
            return _replay(item,*args)
        with tempfile.TemporaryDirectory(dir='runs') as base,replay_fixture(base) as (kwargs,_,_):
            with patch('training.dmd.face_diagnostics._replay',side_effect=change):
                with self.assertRaisesRegex(ValueError,'changed after checksum'):
                    run(**kwargs,output_dir=Path(base)/'bad',samples_per_reason=0)
            self.assertFalse((Path(base)/'bad/summary.json').exists())


if __name__ == '__main__': unittest.main()
