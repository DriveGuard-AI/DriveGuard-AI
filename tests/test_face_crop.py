"""Crop binding and temporal evidence accounting without real model quality claims."""
from copy import deepcopy
from pathlib import Path
import builtins
import json
import tempfile
import unittest
from unittest.mock import patch

from training.dmd.face_experiment_config import BASELINE, validate_config
from tests.test_face_diagnostic_replay import replay_fixture
from tests.test_face_experiments import capture_tasks


class FaceCropTests(unittest.TestCase):
    def test_scope_validation_and_record_binding(self):
        from training.dmd.face_experiments import run_experiments
        from inference.experts.assets import file_sha256
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, _, source):
            config = dict(schema_version='driveguard-face-experiments-v1',
                          variants=[dict(BASELINE), dict(BASELINE, name='head_crop', crop_xyxy=[.1,.1,.9,.9])],
                          crop_scope=dict(record_id='job1', video_sha256=file_sha256(source), image_size=[1000,500]))
            self.assertEqual(validate_config(config), config)
            for key, value in [('record_id','wrong'), ('video_sha256','0'*64), ('image_size',[999,500])]:
                wrong = deepcopy(config); wrong['crop_scope'][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'scope'):
                    run_experiments(**kwargs, experiment_config=wrong, output_dir=Path(tmp)/key, max_seconds=.04, samples_per_reason=0)
                self.assertFalse((Path(tmp)/key).exists())
            with capture_tasks():
                report=run_experiments(**kwargs, experiment_config=config, output_dir=Path(tmp)/'ok', max_seconds=.04, samples_per_reason=0)
            self.assertEqual(report['experiment_config']['crop_scope'],config['crop_scope'])
            windows=Path(tmp)/'ok'/report['records'][0]['directory']/'head_windows.csv'
            self.assertIn('variant_coverage25_eligible',windows.read_text(encoding='utf-8-sig'))

    def test_temporal_stats_window_boundaries_and_missing_runs(self):
        from training.dmd.face_head_continuity import HeadContinuity
        stats = HeadContinuity()
        rows=[]
        for i in range(125):
            row=stats.update(i, True, i not in (98,99,100), True, int(i>=100))
            if row is not None: rows.append(row)
        self.assertEqual([r['end_slot'] for r in rows],[99,124])
        self.assertFalse(rows[0]['variant_strict_eligible'])
        self.assertTrue(rows[0]['variant_gap80_eligible'])
        self.assertTrue(rows[1]['variant_strict_eligible'])
        summary=stats.report()
        self.assertEqual(summary['variant']['missing_runs'],1)
        self.assertEqual(summary['variant']['longest_missing_slots'],3)
        strict=summary['windows']['strict']['known_negative']
        self.assertEqual((strict['windows'],strict['cached_eligible'],strict['variant_eligible'],strict['lost']),(1,1,0,1))
        self.assertEqual(summary['windows']['strict']['known_positive']['variant_eligible'],1)

    def test_temporal_no_head_trailing_gap_and_unknown(self):
        from training.dmd.face_head_continuity import HeadContinuity
        stats=HeadContinuity()
        for i in range(100): row=stats.update(i, False, False, False, None)
        self.assertIsNone(row['variant_gap_steps'])
        self.assertFalse(row['variant_gap80_eligible'])
        report=stats.report()
        self.assertEqual(report['variant']['longest_missing_slots'],100)
        self.assertEqual(report['variant']['missing_runs'],1)
        self.assertEqual(report['windows']['strict']['unknown']['windows'],1)
        self.assertEqual(stats.report(),report)  # Reporting is non-mutating.
        with self.assertRaises(ValueError): stats.update(102,False,False,False,None)

    def test_continuity_aggregate_does_not_join_record_gaps(self):
        from training.dmd.face_head_continuity import HeadContinuity,aggregate_continuity
        records=[]
        for _ in range(2):
            stats=HeadContinuity()
            for i in range(125): stats.update(i,False,False,True,1)
            records.append(stats.report())
        result=aggregate_continuity(records)
        self.assertEqual(result['variant']['missing_slots'],250)
        self.assertEqual(result['variant']['missing_runs'],2)
        self.assertEqual(result['variant']['longest_missing_slots'],125)
        self.assertEqual(result['windows']['strict']['known_positive']['windows'],4)

    def test_preview_samples_entire_record_without_models_or_input_edits(self):
        from training.dmd.face_crop_preview import preview_crop
        from inference.experts.assets import file_sha256
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, closed, _):
            root=Path(tmp); before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
            tasks_before=closed.count('task')
            original_import=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','torch_npu','mediapipe','ultralytics','yolox'):
                    raise AssertionError('Unexpected model runtime import: '+name)
                return original_import(name,*args,**kwargs)
            def save(path,frame): path.write_bytes(b'preview jpeg stub')
            with patch('training.dmd.face_crop_preview._save_frame',save),patch('builtins.__import__',guarded):
                report=preview_crop(**kwargs,output_dir=root/'preview',record_id='job1',samples=4)
            self.assertEqual([r['slot_index'] for r in report['samples']],[0,33,66,99])
            self.assertEqual(report['crop_scope']['record_id'],'job1')
            self.assertEqual(before,{p:file_sha256(p) for p in before})
            self.assertEqual(closed.count('task'),tasks_before)
            self.assertEqual(closed.count('decoder'),2)
            self.assertTrue((root/'preview/index.html').exists())
            self.assertNotIn('crop_xyxy',report)  # No guessed rectangle.
            with self.assertRaises(FileExistsError):
                preview_crop(**kwargs,output_dir=root/'preview',record_id='job1')

    def test_preview_failure_does_not_publish_complete(self):
        from training.dmd.face_crop_preview import preview_crop
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (kwargs,closed,_):
            with patch('training.dmd.face_crop_preview._save_frame',side_effect=OSError('image write failed')):
                with self.assertRaisesRegex(OSError,'image write failed'):
                    preview_crop(**kwargs,output_dir=Path(tmp)/'bad',record_id='job1')
            self.assertFalse((Path(tmp)/'bad/summary.json').exists())
            self.assertEqual(closed.count('decoder'),2)


if __name__ == '__main__': unittest.main()
