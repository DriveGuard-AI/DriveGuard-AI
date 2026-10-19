from types import SimpleNamespace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inference.experts.mediapipe_adapter import face_observation, hand_observation, to_rgb
from inference.experts import mediapipe_adapter


def points(count, x=0.25, y=0.5):
    return [SimpleNamespace(x=x, y=y, z=0.1) for _ in range(count)]


class MediaPipeConversionTests(unittest.TestCase):
    def test_rgb_and_non_square_pixel_conversion(self):
        frame = np.array([[[1, 2, 3]]], dtype=np.uint8)
        np.testing.assert_array_equal(to_rgb(frame), [[[3, 2, 1]]])
        result = face_observation([points(478)], 1000, 500, [0, 0, 800, 500], True, 40, 'face')
        self.assertEqual(result['landmarks'][0], [250, 250])
        self.assertTrue(result['driver_confirmed'])
        self.assertIsNone(result['head_pose'])

    def test_roi_is_not_driver_identity_and_multiple_faces_are_ambiguous(self):
        result = face_observation([points(478)], 1000, 500, [0, 0, 800, 500], False, 0, 'face')
        self.assertFalse(result['driver_confirmed'])
        result = face_observation([points(478), points(478, x=0.4)], 1000, 500,
                                  [0, 0, 800, 500], True, 0, 'face')
        self.assertIsNone(result['landmarks'])
        self.assertFalse(result['driver_confirmed'])

    def test_invalid_points_are_missing_not_clipped(self):
        for bad in (float('nan'), 1.01, -0.01):
            result = face_observation([points(478, x=bad)], 1000, 500, [0, 0, 800, 500], True, 0, 'face')
            self.assertEqual(result['status'], 'missing')
            self.assertIsNone(result['landmarks'])
            hands = hand_observation([points(21, x=bad)], 1000, 500, [0, 0, 800, 500], True, 0, 'hand')
            self.assertEqual(hands['status'], 'missing')
            self.assertEqual(hands['items'], [])

    def test_hands_need_explicit_ownership_and_full_roi_containment(self):
        for confirmed in (False, True):
            result = hand_observation([points(21), points(21, x=0.9)], 1000, 500,
                                      [0, 0, 800, 500], confirmed, 0, 'hand')
            self.assertEqual([x['driver_confirmed'] for x in result['items']], [confirmed, False])
            self.assertTrue(all(x['track_id'] is None for x in result['items']))

    def test_bad_frame_dtype_and_shape_rejected(self):
        for frame in (np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=np.uint8), np.zeros((0, 2, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                to_rgb(frame)


class MediaPipeRuntimeTests(unittest.TestCase):
    def test_local_tasks_options_inference_and_resource_cleanup(self):
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            path = Path(directory) / 'synthetic.task'
            path.write_bytes(b'test-only')
            asset = dict(path=str(path), sha256=hashlib.sha256(b'test-only').hexdigest(), source='test', license='test')
            resources = []

            class Task:
                def __init__(self, options):
                    self.options, self.closed = options, False
                    resources.append(self)

                def detect_for_video(self, image, timestamp):
                    np.testing.assert_array_equal(image.data[0, 0], [3, 2, 1])
                    return SimpleNamespace(face_landmarks=[points(478)], face_blendshapes=[],
                                           facial_transformation_matrixes=[], hand_landmarks=[points(21)],
                                           hand_world_landmarks=[points(21)], handedness=[])

                def close(self):
                    self.closed = True

            class Base(SimpleNamespace):
                Delegate = SimpleNamespace(CPU='CPU')

            factory = SimpleNamespace(create_from_options=Task)
            fake = SimpleNamespace(__version__='test-only', Image=SimpleNamespace, ImageFormat=SimpleNamespace(SRGB='SRGB'),
                    tasks=SimpleNamespace(BaseOptions=Base, vision=SimpleNamespace(RunningMode=SimpleNamespace(VIDEO='VIDEO'),
                    FaceLandmarkerOptions=SimpleNamespace, HandLandmarkerOptions=SimpleNamespace,
                    FaceLandmarker=factory, HandLandmarker=factory)))
            with patch.dict(sys.modules, {'mediapipe': fake}):
                adapter = mediapipe_adapter.MediaPipeAdapter(asset, asset, driver_roi=[0, 0, 4, 2],
                                                             driver_region_confirmed_exclusive=True)
                frame = np.tile(np.array([1, 2, 3], dtype=np.uint8), (2, 4, 1))
                face, hands = adapter.predict(frame, 40)
                self.assertEqual(face['landmarks'][0], [1, 1])
                self.assertTrue(hands['items'][0]['driver_confirmed'])
                self.assertEqual(resources[0].options.base_options.delegate, 'CPU')
                self.assertEqual(resources[0].options.running_mode, 'VIDEO')
                self.assertEqual(resources[0].options.num_faces, 2)
                for time in (40, 39, -1, True):
                    with self.assertRaises(ValueError):
                        adapter.predict(frame, time)
                adapter.close()
                adapter.close()
                self.assertTrue(all(task.closed for task in resources))
                with self.assertRaises(RuntimeError):
                    adapter.predict(frame, 80)

    def test_second_task_initialization_failure_closes_first(self):
        # Resource ownership is part of the public runtime contract.
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            path = Path(directory) / 'synthetic.task'
            path.write_bytes(b'test-only')
            asset = dict(path=str(path), sha256=hashlib.sha256(b'test-only').hexdigest(), source='test', license='test')
            closed = []
            class Base(SimpleNamespace):
                Delegate = SimpleNamespace(CPU='CPU')
            def fail(options):
                raise RuntimeError('synthetic initialization failure')
            fake = SimpleNamespace(__version__='test-only', tasks=SimpleNamespace(BaseOptions=Base,
                vision=SimpleNamespace(RunningMode=SimpleNamespace(VIDEO='VIDEO'),
                FaceLandmarkerOptions=SimpleNamespace, HandLandmarkerOptions=SimpleNamespace,
                FaceLandmarker=SimpleNamespace(create_from_options=lambda options: SimpleNamespace(close=lambda: closed.append(True))),
                HandLandmarker=SimpleNamespace(create_from_options=fail))))
            with patch.dict(sys.modules, {'mediapipe': fake}), self.assertRaises(RuntimeError):
                mediapipe_adapter.MediaPipeAdapter(asset, asset, driver_roi=[0, 0, 4, 2])
            self.assertEqual(closed, [True])
