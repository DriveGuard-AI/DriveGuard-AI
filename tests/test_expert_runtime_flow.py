"""Real adapters and export with synthetic external decoder/model APIs only."""
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from inference.perception.runtime import load_runtime_config
from inference.perception.video import run_video
from tests.perception_helpers import packet


@contextmanager
def runtime_fixture(directory, *, close_error=False, inference_error=False, multiple_faces=False,
                    source_count=100, source_step_ms=40):
    closed = []
    source = Path(directory) / 'synthetic.mp4'
    source.write_bytes(b'not a real video')
    config = load_runtime_config('configs/perception/runtime.example.json')
    config.update(recording_id='synthetic', driver_roi=[0, 0, 800, 500], driver_region_confirmed_exclusive=True)
    reference = packet()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.head = SimpleNamespace(decode_in_inference=True)
        def forward(self, image):
            row = torch.zeros((1, 1, 85), dtype=torch.float32)
            row[0, 0, :5] = torch.tensor([253.76, 126.88, 16.64, 20.8, 1])
            row[0, 0, 72] = .9
            return row
    model = Model()
    weight = Path(directory) / 'synthetic.pth'
    torch.save({'model': model.state_dict()}, weight)
    task = Path(directory) / 'synthetic.task'
    task.write_bytes(b'not a real mediapipe model')
    for asset, path in ((config['objects']['asset'], weight), (config['face_asset'], task), (config['hands_asset'], task)):
        asset.update(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source='test-only', license='test-only')

    class Base(SimpleNamespace):
        Delegate = SimpleNamespace(CPU='CPU')
    class Task:
        def __init__(self, options): self.options = options
        def detect_for_video(self, image, time):
            if inference_error and time == 80:
                raise RuntimeError('synthetic inference failure')
            def normalized(points):
                return [SimpleNamespace(x=x/1000, y=y/500, z=0) for x, y in points]
            faces = [normalized(reference['face']['landmarks'])]
            if multiple_faces: faces *= 2
            return SimpleNamespace(face_landmarks=faces, face_blendshapes=[], facial_transformation_matrixes=[],
                    hand_landmarks=[normalized(reference['hands']['items'][0]['landmarks'])],
                    hand_world_landmarks=[], handedness=[])
        def close(self):
            closed.append('task')
            if close_error: raise RuntimeError('synthetic close failure')
    tasks = SimpleNamespace(create_from_options=Task)
    mp = SimpleNamespace(__version__='test-only', Image=SimpleNamespace, ImageFormat=SimpleNamespace(SRGB='SRGB'),
        tasks=SimpleNamespace(BaseOptions=Base, vision=SimpleNamespace(RunningMode=SimpleNamespace(VIDEO='VIDEO'),
        FaceLandmarkerOptions=SimpleNamespace, HandLandmarkerOptions=SimpleNamespace, FaceLandmarker=tasks, HandLandmarker=tasks)))
    class Container:
        streams = SimpleNamespace(video=[object()])
        def __enter__(self): return self
        def __exit__(self, *args): closed.append('decoder')
        def decode(self, stream):
            for index in range(source_count):
                yield SimpleNamespace(pts=1000+index*source_step_ms, time_base=Fraction(1, 1000), is_corrupt=False,
                    to_ndarray=lambda format: np.broadcast_to(np.array([1, 2, 3], dtype=np.uint8), (500, 1000, 3)))
    classes = [str(i) for i in range(80)]
    classes[39], classes[41], classes[67] = 'bottle', 'cup', 'cell phone'
    modules = {'mediapipe': mp, 'av': SimpleNamespace(__version__='test-only', open=lambda *a, **k: Container()),
        'yolox': SimpleNamespace(__version__='0.3.0'),
        'yolox.exp': SimpleNamespace(get_exp=lambda *a: SimpleNamespace(num_classes=80, get_model=lambda: model)),
        'yolox.data.datasets': SimpleNamespace(COCO_CLASSES=classes),
        'cv2': SimpleNamespace(__version__='test-only', INTER_LINEAR=1,
              resize=lambda frame, size, **k: np.broadcast_to(frame[0, 0], (size[1], size[0], 3)))}
    with patch.dict(sys.modules, modules):
        yield source, config, closed


class RuntimeFlowTests(unittest.TestCase):
    def test_observer_sees_every_source_frame_before_resampling(self):
        import inspect
        self.assertIn('frame_observer', inspect.signature(run_video).parameters)
        with tempfile.TemporaryDirectory(dir='runs') as directory, runtime_fixture(
                directory, source_count=200, source_step_ms=20) as (source, config, closed):
            times = []
            report = run_video(source, Path(directory) / 'output', config,
                               frame_observer=lambda frame: times.append(frame.timestamp_ms))
            self.assertEqual(times, list(range(0, 4000, 20)))
            self.assertEqual(report['slots'], 100)
            self.assertEqual(closed.count('decoder'), 1)

    def test_observer_failure_closes_resources_without_success(self):
        import inspect
        self.assertIn('frame_observer', inspect.signature(run_video).parameters)
        with tempfile.TemporaryDirectory(dir='runs') as directory, runtime_fixture(directory) as (source, config, closed):
            output = Path(directory) / 'output'
            def fail(frame):
                raise ValueError('invalid source timeline')
            with self.assertRaisesRegex(ValueError, 'invalid source timeline'):
                run_video(source, output, config, frame_observer=fail)
            self.assertFalse((output / 'record.json').exists())
            self.assertEqual(closed.count('decoder'), 1)
            self.assertEqual(closed.count('task'), 2)

    def test_real_adapters_to_unlabeled_arrays_and_provenance(self):
        with tempfile.TemporaryDirectory(dir='runs') as directory, runtime_fixture(directory) as (source, config, closed):
            output = Path(directory) / 'output'
            report = run_video(source, output, config)
            self.assertEqual(report['slots'], 100)
            self.assertEqual(report['provenance']['clock']['origin_seconds'], '1')
            self.assertEqual(report['provenance']['versions']['mediapipe'], 'test-only')
            values, masks = np.load(output / 'values.npy'), np.load(output / 'masks.npy')
            self.assertEqual(values.shape, (100, 19))
            self.assertTrue((values[:, 8] > .89).all())
            self.assertTrue((masks[:, [0, 1, 2, 8, 13, 14]] == 1).all())
            self.assertFalse(masks[:, 5:8].any())
            self.assertFalse(np.load(output / 'label_masks.npy').any())
            self.assertEqual(closed.count('task'), 2)
            self.assertEqual(closed.count('decoder'), 1)
            self.assertFalse((output / 'incomplete.json').exists())
            self.assertEqual(json.loads((output / 'record.json').read_text()), report)

    def test_inference_or_cleanup_failure_never_publishes_success(self):
        for options in ({'inference_error': True}, {'close_error': True}):
            with tempfile.TemporaryDirectory(dir='runs') as directory, runtime_fixture(directory, **options) as (source, config, closed):
                output = Path(directory) / 'output'
                with self.assertRaises(RuntimeError):
                    run_video(source, output, config)
                self.assertFalse((output / 'record.json').exists())
                self.assertTrue((output / 'incomplete.json').exists())
                self.assertEqual(closed.count('task'), 2)
                self.assertEqual(closed.count('decoder'), 1)

    def test_multiple_faces_invalidates_hand_ownership_too(self):
        with tempfile.TemporaryDirectory(dir='runs') as directory, runtime_fixture(directory, multiple_faces=True) as (source, config, closed):
            output = Path(directory) / 'output'
            run_video(source, output, config)
            masks = np.load(output / 'masks.npy')
            self.assertFalse(masks[:, [0, 1, 2, 13, 14, 15, 16, 17, 18]].any())

    def test_config_cli_runs_without_site_packages(self):
        result = subprocess.run([sys.executable, '-B', '-S', '-m', 'inference.perception.video',
                                 '--config', 'configs/perception/runtime.example.json', '--check-config'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['assets_verified'])
