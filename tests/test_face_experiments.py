"""Real cache/replay plumbing with only external MediaPipe and decoder APIs faked."""
from contextlib import contextmanager
from copy import deepcopy
import builtins
import csv
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inference.experts.assets import file_sha256
from tests.test_face_diagnostic_replay import replay_fixture


def config():
    base = dict(name='baseline', mode='VIDEO', detection=.5, presence=.5, tracking=.5, crop_xyxy=None)
    return dict(schema_version='driveguard-face-experiments-v1', variants=[base,
        dict(base, name='image', mode='IMAGE'), dict(base, name='detection', detection=.3),
        dict(base, name='presence', presence=.3), dict(base, name='tracking', tracking=.3)])


@contextmanager
def capture_tasks():
    vision = sys.modules['mediapipe'].tasks.vision
    original = vision.FaceLandmarker.create_from_options
    created = []
    def create(options):
        task = original(options)
        calls = []
        original_video = task.detect_for_video
        def video(image, time):
            calls.append(('VIDEO', time, image.data.shape))
            return original_video(image, time)
        def still(image):
            calls.append(('IMAGE', None, image.data.shape))
            return original_video(image, 0)
        task.detect_for_video, task.detect = video, still
        created.append((options, calls))
        return task
    with patch.object(vision.RunningMode, 'IMAGE', 'IMAGE', create=True), \
         patch.object(vision.FaceLandmarker, 'create_from_options', create):
        yield created


class FaceExperimentTests(unittest.TestCase):
    def runner(self):
        self.assertIsNotNone(importlib.util.find_spec('training.dmd.face_experiments'), 'experiment CLI missing')
        from training.dmd.face_experiments import run_experiments
        return run_experiments

    def test_baseline_parity_independent_lifetimes_and_readonly(self):
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, closed, _):
            initial = {str(p): file_sha256(p) for p in Path(base).rglob('*') if p.is_file()}
            with capture_tasks() as tasks:
                report = run(**kwargs, experiment_config=config(), output_dir=Path(base)/'report', samples_per_reason=0)
            rows = report['records']
            self.assertEqual(len(rows), 5)
            self.assertTrue(rows[0]['baseline_masks_reproduced'])
            self.assertTrue(rows[0]['model_id_matches'])
            self.assertTrue(all(r['baseline_masks_reproduced'] is None for r in rows[1:]))
            self.assertEqual(rows[0]['coverage']['known_negative']['slots'], 100)
            self.assertEqual(rows[0]['coverage']['known_negative']['variant_head_available'], 100)
            self.assertEqual(rows[0]['coverage']['known_positive']['slots'], 0)
            self.assertEqual(rows[0]['angle_jumps']['pairs'], 99)
            self.assertEqual(set(report['variants']), {'baseline', 'image', 'detection', 'presence', 'tracking'})
            totals = report['variants']['baseline']
            self.assertEqual(totals['slots'], 100)
            self.assertEqual(totals['coverage']['known_negative']['variant_head_available'], 100)
            self.assertTrue(totals['baseline_masks_reproduced'])
            self.assertIsNone(report['variants']['image']['baseline_masks_reproduced'])
            self.assertEqual(totals['timing']['inference_total_ms'], rows[0]['timing']['inference_total_ms'])
            self.assertEqual(len(tasks), 5)
            self.assertEqual(closed.count('task'), 7)  # original face/hands, then face only
            self.assertTrue(all(len(calls) == 100 for _, calls in tasks))
            self.assertTrue(all(call[0] == 'IMAGE' for call in tasks[1][1]))
            self.assertEqual(tasks[2][0].min_face_detection_confidence, .3)
            self.assertEqual(initial, {p: file_sha256(p) for p in initial})
            self.assertNotIn('winner', report)
            self.assertFalse(any(p.suffix == '.npy' for p in (Path(base)/'report').rglob('*')))
            with (Path(base)/'report'/rows[0]['directory']/'frames.csv').open(encoding='utf-8-sig', newline='') as f:
                frame = next(csv.DictReader(f))
            self.assertEqual(frame['cached_head_mask'], '[1, 1, 1]')
            self.assertEqual(frame['variant_head_mask'], '[1, 1, 1]')
            self.assertEqual(float(frame['head_yaw_deg']), 0.)
            self.assertIn('inference_ms', frame)
            self.assertTrue((Path(base)/'report/summary.json').exists())

    def test_reused_pts_partial_and_threshold_identity(self):
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base, source_count=34, source_step_ms=120) as (kwargs, _, _):
            with capture_tasks() as tasks:
                report = run(**kwargs, experiment_config=config(), output_dir=Path(base)/'report', max_seconds=1, samples_per_reason=0)
            self.assertTrue(report['partial'])
            for row, (_, calls) in zip(report['records'], tasks):
                self.assertEqual(row['slots'], 25)
                self.assertEqual(row['unique_inference_frames'], 9)
                self.assertEqual(len(calls), 9)
                self.assertEqual(row['source_pts_mismatches'], 0)
            self.assertEqual(len({r['experiment_model_id'] for r in report['records']}), 5)

    def test_strict_config_and_no_guessed_crop(self):
        self.runner()
        from training.dmd.face_experiment_config import validate_config
        self.assertEqual(validate_config(config()), config())
        for change in ('unknown', 'two_changes', 'baseline', 'crop', 'bool', 'duplicate'):
            data = deepcopy(config())
            if change == 'unknown': data['variants'][1]['auto_crop'] = True
            if change == 'two_changes': data['variants'][1]['detection'] = .3
            if change == 'baseline': data['variants'][0]['tracking'] = .2
            if change == 'crop': data['variants'][1] = dict(data['variants'][0], name='crop', crop_xyxy=[.5, 0, .2, 1])
            if change == 'bool': data['variants'][2]['detection'] = True
            if change == 'duplicate': data['variants'][1]['name'] = 'baseline'
            with self.subTest(change=change), self.assertRaises(ValueError): validate_config(data)

    def test_crop_remaps_original_pixels_and_marks_pose_experimental(self):
        run = self.runner()
        data = config(); data['variants'] = [data['variants'][0], dict(data['variants'][0], name='crop', crop_xyxy=[.1, .2, .9, .8])]
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, _, _):
            with capture_tasks() as tasks:
                report = run(**kwargs, experiment_config=data, output_dir=Path(base)/'report', max_seconds=.04, samples_per_reason=0)
            self.assertEqual(tasks[1][1][0][2], (300, 800, 3))
            crop = report['records'][1]
            self.assertEqual(crop['pose_coordinate_frame'], 'crop-experimental')
            self.assertFalse(crop['training_cache_compatible'])
            path = Path(base)/'report'/crop['directory']/'candidates.jsonl'
            detail = json.loads(path.read_text().splitlines()[0])
            bounds = detail['candidates'][0]['bbox']
            full_path = Path(base)/'report'/report['records'][0]['directory']/'candidates.jsonl'
            full_bounds = json.loads(full_path.read_text().splitlines()[0])['candidates'][0]['bbox']
            np.testing.assert_allclose(bounds, np.asarray(full_bounds)*[.8, .6, .8, .6]+100)
            self.assertEqual(detail['crop_pixels'], [100, 100, 900, 400])

    def test_runtime_and_cleanup_failure_never_publish_success(self):
        run = self.runner()
        from inference.experts.face_experiment_adapter import FaceExperimentAdapter
        for operation in ('predict', 'close'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, closed, _):
                original = getattr(FaceExperimentAdapter, operation)
                def fail(adapter, *args):
                    original(adapter, *args)
                    raise RuntimeError('experiment failed')
                with capture_tasks(), patch.object(FaceExperimentAdapter, operation, fail):
                    with self.assertRaisesRegex(RuntimeError, 'experiment failed'):
                        run(**kwargs, experiment_config=config(), output_dir=Path(base)/'report', samples_per_reason=0)
                self.assertFalse((Path(base)/'report/summary.json').exists())
                self.assertEqual(closed.count('task'), 3)

    def test_corrupt_video_or_cache_and_overlapping_output_rejected(self):
        run = self.runner()
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, _, source):
            with self.assertRaises(ValueError):
                run(**kwargs, experiment_config=config(), output_dir=kwargs['data_root']/'report')
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                run(**kwargs, experiment_config=config(), output_dir=Path(base)/'report')
            self.assertFalse((Path(base)/'report').exists())

    def test_positive_negative_unknown_denominators_and_no_torch_yolo_import(self):
        run = self.runner()
        data = config(); data['variants'] = data['variants'][:1]
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, _, _):
            attempt = kwargs['data_root']/'records/job1/attempt1'
            targets = np.load(attempt/'targets.npy'); targets[:20, 3] = 1; np.save(attempt/'targets.npy', targets)
            labels = np.load(attempt/'label_masks.npy'); labels[-10:, 3] = 0; np.save(attempt/'label_masks.npy', labels)
            complete = json.loads((attempt/'complete.json').read_text())
            for name in ('targets', 'label_masks'):
                complete['array_sha256'][name] = file_sha256(attempt/complete['arrays'][name])
            (attempt/'complete.json').write_text(json.dumps(complete))
            original_import = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('torch', 'torch_npu', 'yolox', 'ultralytics'):
                    raise AssertionError('Forbidden runtime import: '+name)
                return original_import(name, *args, **kwargs)
            with capture_tasks(), patch('builtins.__import__', guarded):
                row = run(**kwargs, experiment_config=data, output_dir=Path(base)/'report', samples_per_reason=0)['records'][0]
            for label, expected in (('all', 100), ('known_positive', 20), ('known_negative', 70), ('unknown', 10)):
                self.assertEqual(row['coverage'][label]['slots'], expected)
                self.assertEqual(row['coverage'][label]['variant_head_available'], expected)

    def test_high_rate_sampling_and_angle_gap_resets(self):
        run = self.runner()
        data = config(); data['variants'] = data['variants'][:1]
        with tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base, source_count=200, source_step_ms=20) as (kwargs, _, _):
            from inference.experts.gaze_geometry import head_pose
            call = 0
            def pose(matrix):
                nonlocal call
                call += 1
                return None if call == 2 else head_pose(matrix)
            with capture_tasks(), patch('inference.experts.gaze_geometry.head_pose', pose):
                row = run(**kwargs, experiment_config=data, output_dir=Path(base)/'report', max_seconds=.2, samples_per_reason=0)['records'][0]
            self.assertEqual(row['unique_inference_frames'], 5)
            self.assertEqual(row['source_pts_mismatches'], 0)
            self.assertEqual(row['head_mask_mismatches'], 1)
            self.assertEqual(row['angle_jumps']['pairs'], 2)
            self.assertFalse(row['baseline_masks_reproduced'])

    def test_cache_corruption_and_mutation_during_run_fail_closed(self):
        run = self.runner()
        from training.dmd.face_experiments import _replay
        for during_run in (False, True):
            with self.subTest(during_run=during_run), tempfile.TemporaryDirectory(dir='runs') as base, replay_fixture(base) as (kwargs, _, _):
                path = kwargs['data_root']/'records/job1/attempt1/features/masks.npy'
                def corrupt():
                    masks = np.load(path); masks[0, 5] = 0; np.save(path, masks)
                def replay(*args):
                    result = _replay(*args)
                    corrupt()
                    return result
                if not during_run: corrupt()
                with capture_tasks(), patch('training.dmd.face_experiments._replay', replay):
                    with self.assertRaisesRegex(ValueError, 'cache|changed'):
                        run(**kwargs, experiment_config=config(), output_dir=Path(base)/'report', max_seconds=.04, samples_per_reason=0)
                self.assertFalse((Path(base)/'report/summary.json').exists())

    def test_cli_config_preflight_uses_standard_library_only(self):
        result = subprocess.run([sys.executable, '-B', '-S', '-m', 'training.dmd.face_experiments', '--check-config'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), dict(valid=True, variants=5, assets_verified=False))


if __name__ == '__main__':
    unittest.main()
