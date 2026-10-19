"""Fatigue replay must measure canonical eye evidence, not head proxies."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inference.experts.assets import file_sha256
from tests.test_face_diagnostic_replay import replay_fixture
from tests.test_face_experiments import config, capture_tasks
from training.gaze.profile import FEATURE_NAMES, default_group_profile


def diagnostic(kwargs):
    return dict(schema_version='driveguard-fatigue-diagnostics-v1', status='complete',
                original_inputs_unchanged=True, manifest=str(kwargs['manifest'].resolve()),
                data_root=str(kwargs['data_root'].resolve()),
                selection=dict(split='val', view='all', subset='all'),
                group_profile=default_group_profile(),
                normalizer_supported=dict.fromkeys(FEATURE_NAMES, True),
                window_steps=100, stride_steps=25)


class FatigueReplayTests(unittest.TestCase):
    def runner(self):
        self.assertIsNotNone(importlib.util.find_spec('training.dmd.fatigue_recovery'),
                             'Fatigue-specific recovery runner is missing')
        from training.dmd.fatigue_recovery import run
        return run

    def test_baseline_fatigue_masks_values_and_window_parity(self):
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, _, _):
            initial = {str(p): file_sha256(p) for p in Path(base).rglob('*') if p.is_file()}
            evidence = Path(base)/'diagnostic.json'
            evidence.write_text(json.dumps(diagnostic(kwargs)))
            variants = config(); variants['variants'] = variants['variants'][:1]
            with capture_tasks():
                report = run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                             output_dir=Path(base)/'report', experiment_config=variants,
                             samples_per_reason=0)
            audit = report['records'][0]['fatigue_evidence']
            self.assertEqual(audit['raw_mask_mismatches'], 0)
            self.assertEqual(audit['raw_value_mismatches'], 0)
            self.assertTrue(audit['baseline_fatigue_reproduced'])
            counts = audit['windows']['known_negative']
            self.assertEqual(counts['windows'], 1)
            self.assertEqual(counts['variant_f_eligible'], 1)
            self.assertEqual(audit['frames']['known_negative']['slots'], 100)
            self.assertEqual(initial, {p: file_sha256(p) for p in initial})
            aggregate = report['variants']['baseline']['fatigue_evidence']
            self.assertEqual(aggregate['windows']['all']['windows'], 1)
            self.assertTrue(aggregate['baseline_fatigue_reproduced'])
            directory = Path(base)/'report'/report['records'][0]['directory']
            with (directory/'fatigue_windows.csv').open(encoding='utf-8-sig') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['end_slot'], '99')
            self.assertEqual(row['fatigue_target'], '0')
            self.assertTrue((directory/'fatigue_frames.csv').exists())
            self.assertEqual(report['label_task'], 'fatigue')

    def test_uses_fatigue_not_distraction_labels_and_final_slot(self):
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base, source_count=150) as (kwargs, _, _):
            attempt = kwargs['data_root']/'records/job1/attempt1'
            targets = np.load(attempt/'targets.npy'); targets[:125, 0] = 1
            labels = np.load(attempt/'label_masks.npy'); labels[149, 0] = 0
            np.save(attempt/'targets.npy', targets); np.save(attempt/'label_masks.npy', labels)
            complete = json.loads((attempt/'complete.json').read_text())
            for key in ('targets', 'label_masks'):
                complete['array_sha256'][key] = file_sha256(attempt/complete['arrays'][key])
            (attempt/'complete.json').write_text(json.dumps(complete))
            evidence = Path(base)/'diagnostic.json'; evidence.write_text(json.dumps(diagnostic(kwargs)))
            variants = config(); variants['variants'] = variants['variants'][:1]
            with capture_tasks():
                report = run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                             output_dir=Path(base)/'report', experiment_config=variants, samples_per_reason=0)
            audit = report['records'][0]['fatigue_evidence']
            self.assertEqual(audit['frames']['known_positive']['slots'], 125)
            self.assertEqual(audit['frames']['unknown']['slots'], 1)
            self.assertEqual(audit['windows']['known_positive']['windows'], 2)
            self.assertEqual(audit['windows']['known_negative']['windows'], 0)
            self.assertEqual(audit['windows']['unknown']['windows'], 1)

    def test_old_summary_requires_explicit_window_geometry(self):
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, _, _):
            data = diagnostic(kwargs); data.pop('window_steps'); data.pop('stride_steps')
            evidence = Path(base)/'diagnostic.json'; evidence.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'window|stride'):
                run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                    output_dir=Path(base)/'report', samples_per_reason=0)
            self.assertFalse((Path(base)/'report').exists())

    def test_gap_is_not_interpolated_and_head_failure_is_not_eye_failure(self):
        run = self.runner()
        from inference.experts.face_experiment_adapter import FaceExperimentAdapter
        original = FaceExperimentAdapter.predict
        def predict(adapter, image, time):
            result = original(adapter, image, time)
            result['head_pose'] = None
            adapter.last_face_diagnostics.update(head_usable=False, reason='pose_matrix_invalid')
            if time == 400:
                result.update(status='missing', driver_confirmed=False, landmarks=None)
                adapter.last_face_diagnostics['reason'] = 'no_face_detected'
            return result
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base, source_count=150) as (kwargs, _, _):
            evidence = Path(base)/'diagnostic.json'; evidence.write_text(json.dumps(diagnostic(kwargs)))
            variants = config(); variants['variants'] = variants['variants'][:1]
            with capture_tasks(), patch.object(FaceExperimentAdapter, 'predict', predict):
                report = run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                             output_dir=Path(base)/'report', experiment_config=variants, samples_per_reason=0)
            directory = Path(base)/'report'/report['records'][0]['directory']
            with (directory/'fatigue_frames.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[9]['variant_ear_left_valid'], 'True')
            self.assertEqual(rows[10]['variant_ear_left_valid'], 'False')
            self.assertEqual(rows[99]['variant_perclos_4s_valid'], 'True')
            self.assertEqual(rows[99]['variant_perclos_4s_effective'], 'False')
            self.assertEqual(rows[110]['variant_perclos_4s_effective'], 'True')
            self.assertFalse(report['records'][0]['fatigue_evidence']['baseline_fatigue_reproduced'])

    def test_support_filter_partial_run_and_no_window_header_loss(self):
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, _, _):
            data = diagnostic(kwargs); data['normalizer_supported']['ear_left'] = False
            evidence = Path(base)/'diagnostic.json'; evidence.write_text(json.dumps(data))
            variants = config(); variants['variants'] = variants['variants'][:1]
            with capture_tasks():
                report = run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                             output_dir=Path(base)/'report', experiment_config=variants,
                             max_seconds=1, samples_per_reason=0)
            self.assertTrue(report['partial'])
            counts = report['variants']['baseline']['fatigue_evidence']
            self.assertEqual(counts['windows']['all']['windows'], 0)
            self.assertEqual(counts['frames']['all']['variant_ear_left_raw'], 25)
            self.assertEqual(counts['frames']['all']['variant_ear_left_effective'], 0)
            path = Path(base)/'report'/report['records'][0]['directory']/'fatigue_windows.csv'
            with path.open(encoding='utf-8-sig') as stream:
                reader = csv.DictReader(stream)
                self.assertIn('end_slot', reader.fieldnames)
                self.assertEqual(list(reader), [])

    def test_full_window_counts_must_match_original_diagnostic(self):
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, _, _):
            data = diagnostic(kwargs)
            data['groups'] = {'view': {'body': dict(windows=2, positive=0, negative=2, unknown=0,
                                                   evidence={'f_eligible': 2})}}
            evidence = Path(base)/'diagnostic.json'; evidence.write_text(json.dumps(data))
            variants = config(); variants['variants'] = variants['variants'][:1]
            with capture_tasks(), self.assertRaisesRegex(ValueError, 'window.*diagnostic'):
                run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                    output_dir=Path(base)/'report', experiment_config=variants, samples_per_reason=0)
            self.assertFalse((Path(base)/'report/summary.json').exists())

    def test_changed_summary_or_manifest_and_invalid_support_rejected(self):
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, _, _):
            variants = config(); variants['variants'] = variants['variants'][:1]
            evidence = Path(base)/'diagnostic.json'
            for key, value, message in [('status', 'incomplete', 'complete'),
                                        ('normalizer_supported', {}, 'support'),
                                        ('manifest_file_sha256', '0'*64, 'checksum')]:
                data = diagnostic(kwargs); data[key] = value
                evidence.write_text(json.dumps(data))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                    run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                        output_dir=Path(base)/'report', experiment_config=variants, samples_per_reason=0)
                self.assertFalse((Path(base)/'report').exists())

    def test_five_variants_reused_source_frames_and_no_training_import(self):
        import builtins
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(
                base, source_count=34, source_step_ms=120) as (kwargs, _, _):
            evidence = Path(base)/'diagnostic.json'; evidence.write_text(json.dumps(diagnostic(kwargs)))
            original_import = builtins.__import__
            def guarded(name, *args, **kw):
                if name.split('.')[0] in ('torch', 'torch_npu', 'ultralytics', 'yolox'):
                    raise AssertionError('Fatigue replay must not import training/detector runtime: '+name)
                return original_import(name, *args, **kw)
            with capture_tasks() as tasks, patch('builtins.__import__', guarded):
                report = run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                             output_dir=Path(base)/'report', experiment_config=config(),
                             max_seconds=1, samples_per_reason=0)
            self.assertEqual(len(tasks), 5)
            self.assertTrue(all(len(calls) == 9 for _, calls in tasks))
            self.assertTrue(report['variants']['baseline']['fatigue_evidence']['baseline_fatigue_reproduced'])
            for row in report['records']:
                self.assertEqual(row['fatigue_evidence']['raw_mask_mismatches'], 0)
                self.assertEqual(row['fatigue_evidence']['raw_value_mismatches'], 0)
                self.assertEqual(row['fatigue_evidence']['frames']['all']['slots'], 25)
            self.assertTrue((Path(base)/'report/fatigue_comparison.csv').is_file())


if __name__ == '__main__':
    unittest.main()
