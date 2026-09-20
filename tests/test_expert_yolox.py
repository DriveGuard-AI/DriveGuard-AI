"""Adapter contracts: real Torch/NumPy; optional YOLOX/OpenCV API doubles.

No detector accuracy or accelerator execution claim is made by these tests.
The resize double uses constant-color frames so it preserves the exact result
of real bilinear resizing; padding, geometry, filtering and NMS are real code.
"""

import hashlib
import importlib
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def prediction(category=67, box=(208, 104, 208, 104), score=.9):
    row = np.zeros(85, dtype=np.float32)
    row[:4], row[4], row[5 + category] = box, 1., score
    return row


class DetectorDouble(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(1)))
        self.head = types.SimpleNamespace(decode_in_inference=True)
        self.output = torch.zeros((1, 0, 85))
        self.received = None

    def forward(self, image):
        self.received = image.detach().clone()
        return self.output


class YOLOXAdapterTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('inference.experts.yolox_adapter')
        except ModuleNotFoundError:
            self.fail('YOLOX adapter module has not been implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / 'nano.pth'
        self.model = DetectorDouble()
        torch.save({'model': self.model.state_dict()}, path)
        self.asset = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'source': 'https://example.invalid/yolox-nano.pth', 'license': 'Apache-2.0'}
        self.exp = types.SimpleNamespace(num_classes=80, test_size=(640, 640),
                                         get_model=lambda: self.model)
        self.classes = ['class-' + str(i) for i in range(80)]
        self.classes[39], self.classes[41], self.classes[67] = 'bottle', 'cup', 'cell phone'
        self.exp_calls = []
        self.resize_calls = []

        def get_exp(path, name):
            self.exp_calls.append((path, name))
            return self.exp

        def resize(frame, size, *, interpolation):
            self.resize_calls.append((size, interpolation))
            return np.broadcast_to(frame[0, 0], (size[1], size[0], 3)).copy()

        self.yolox = types.ModuleType('yolox')
        self.yolox.__version__ = '0.3.0'
        modules = {'yolox': self.yolox, 'yolox.exp': types.SimpleNamespace(get_exp=get_exp),
                   'yolox.data.datasets': types.SimpleNamespace(COCO_CLASSES=tuple(self.classes)),
                   'cv2': types.SimpleNamespace(resize=resize, INTER_LINEAR=1, __version__='test-cv2')}
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def adapter(self, **options):
        return self.module.YOLOXAdapter(self.asset, **options)

    def frame(self):
        return np.broadcast_to(np.array([10, 20, 30], dtype=np.uint8), (200, 400, 3)).copy()

    def output(self, *rows):
        self.model.output = torch.from_numpy(np.asarray(rows, dtype=np.float32).reshape(1, -1, 85))

    def test_nonlegacy_bgr_top_left_padding_and_original_pixel_box(self):
        adapter = self.adapter()
        self.output(prediction())
        result = adapter.predict(self.frame(), 120)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['timestamp_ms'], 120)
        self.assertEqual(result['model_id'], adapter.model_id)
        self.assertEqual(len(result['items']), 1)
        item = result['items'][0]
        self.assertEqual(item['category'], 'phone')
        self.assertEqual(item['track_id'], None)
        np.testing.assert_allclose(item['box'], [100, 50, 300, 150])
        self.assertAlmostEqual(item['score'], .9)
        image = self.model.received.numpy()
        self.assertEqual(image.shape, (1, 3, 416, 416))
        self.assertEqual(image.dtype, np.float32)
        np.testing.assert_array_equal(image[0, :, 0, 0], [10, 20, 30])
        np.testing.assert_array_equal(image[0, :, 207, 415], [10, 20, 30])
        np.testing.assert_array_equal(image[0, :, 208, 0], [114, 114, 114])
        self.assertEqual(self.resize_calls, [((416, 208), 1)])
        self.assertEqual(self.exp_calls, [(None, 'yolox-nano')])
        self.assertEqual(self.exp.test_size, (416, 416))
        self.assertFalse(self.model.training)

    def test_class_aware_nms_and_all_class_argmax(self):
        unrelated = prediction(score=.8)
        unrelated[5] = .95
        self.output(prediction(score=.9), prediction(score=.8), prediction(39, score=.7),
                    prediction(41, score=.6), unrelated)
        items = self.adapter().predict(self.frame(), 0)['items']
        self.assertEqual([i['category'] for i in items], ['phone', 'bottle', 'cup'])

    def test_objectness_multiplies_class_probability_and_threshold_is_inclusive(self):
        row = prediction(score=.5)
        row[4] = .5
        self.output(row)
        self.assertEqual(len(self.adapter().predict(self.frame(), 0)['items']), 1)
        self.assertEqual(self.adapter(score_threshold=.251).predict(self.frame(), 0)['items'], [])

    def test_nms_ties_are_stable(self):
        self.output(prediction(box=(210, 104, 208, 104)), prediction())
        items = self.adapter().predict(self.frame(), 0)['items']
        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0]['box'][0], 106 / 1.04)

    def test_clips_and_discards_degenerate_boxes(self):
        self.output(prediction(box=(0, 0, 208, 104)), prediction(box=(600, 600, 20, 20)),
                    prediction(box=(100, 100, 0, 10)))
        items = self.adapter().predict(self.frame(), 0)['items']
        self.assertEqual(len(items), 1)
        np.testing.assert_allclose(items[0]['box'], [0, 0, 100, 50])

    def test_empty_predictions_are_successful(self):
        self.assertEqual(self.adapter().predict(self.frame(), 0)['items'], [])

    def test_invalid_frames_and_timestamps_rejected_before_forward(self):
        adapter = self.adapter()
        for frame in (None, np.zeros((10, 10)), np.zeros((10, 10, 3)),
                      np.zeros((0, 10, 3), dtype=np.uint8)):
            with self.subTest(shape=getattr(frame, 'shape', None)), self.assertRaises(ValueError):
                adapter.predict(frame, 0)
        for timestamp in (-1, True, 1.5, 2**63):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                adapter.predict(self.frame(), timestamp)
        self.assertIsNone(self.model.received)

    def test_invalid_settings_rejected(self):
        for options in ({'variant': 'yolox-s'}, {'input_size': 417}, {'input_size': True},
                        {'input_size': 4096}, {'score_threshold': float('nan')},
                        {'score_threshold': True}, {'nms_threshold': 1.1},
                        {'backend': 'auto'}, {'device_index': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.adapter(**options)

    def test_float64_global_default_does_not_change_float32_inference(self):
        class ConvDetectorDouble(DetectorDouble):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 1)

            def forward(self, image):
                self.conv(image)
                return super().forward(image)

        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            self.model = ConvDetectorDouble()
            torch.save({'model': self.model.state_dict()}, self.asset['path'])
            self.asset['sha256'] = hashlib.sha256(Path(self.asset['path']).read_bytes()).hexdigest()
            adapter = self.adapter()
            try:
                result = adapter.predict(self.frame(), 0)
            except RuntimeError as exc:
                self.fail(f'Float32 inference must work despite the global float64 default: {exc}')
            self.assertEqual(result['items'], [])
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in self.model.parameters()))
        finally:
            torch.set_default_dtype(previous)

    def test_broadcast_frame_bounds_and_extreme_aspect_ratio(self):
        adapter = self.adapter()
        pixel = np.array([10, 20, 30], dtype=np.uint8)
        for shape in ((1, 16385, 3), (16385, 1, 3), (8193, 8193, 3), (1, 16384, 3)):
            frame = np.broadcast_to(pixel, shape)
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                adapter.predict(frame, 0)
        self.assertIsNone(self.model.received)
        self.assertEqual(self.resize_calls, [])

    def test_wrong_version_or_coco_mapping_rejected(self):
        self.yolox.__version__ = '0.2.0'
        with self.assertRaisesRegex(ValueError, '0.3.0'):
            self.adapter()
        self.yolox.__version__ = '0.3.0'
        self.classes[67], self.classes[1] = self.classes[1], self.classes[67]
        sys.modules['yolox.data.datasets'].COCO_CLASSES = tuple(self.classes)
        with self.assertRaisesRegex(ValueError, 'COCO'):
            self.adapter()

    def test_wrong_number_of_classes_rejected(self):
        self.exp.num_classes = 3
        with self.assertRaisesRegex(ValueError, '80'):
            self.adapter()

    def test_safe_strict_checkpoint_and_hash_before_load(self):
        with patch.object(torch, 'load', wraps=torch.load) as loader:
            self.adapter()
        self.assertEqual(loader.call_args.kwargs, {'map_location': 'cpu', 'weights_only': True})
        self.asset['sha256'] = '0' * 64
        with patch.object(torch, 'load') as loader, self.assertRaises(ValueError):
            self.adapter()
        loader.assert_not_called()

    def test_checkpoint_requires_model_state_dictionary_and_strict_keys(self):
        for checkpoint in ({'weight': torch.zeros(1)}, {'model': {'wrong': torch.zeros(1)}}):
            torch.save(checkpoint, self.asset['path'])
            self.asset['sha256'] = hashlib.sha256(Path(self.asset['path']).read_bytes()).hexdigest()
            with self.subTest(keys=list(checkpoint)), self.assertRaises((ValueError, RuntimeError)):
                self.adapter()

    def test_safe_load_failure_never_retries_unsafe(self):
        with patch.object(torch, 'load', side_effect=TypeError('weights_only unsupported')) as loader:
            with self.assertRaises(TypeError):
                self.adapter()
        self.assertEqual(loader.call_count, 1)

    def test_selected_device_failure_propagates_without_fallback(self):
        with patch('training.common.device.select_device', side_effect=RuntimeError('NPU unavailable')) as selector:
            with self.assertRaisesRegex(RuntimeError, 'NPU unavailable'):
                self.adapter(backend='npu', device_index=2)
        selector.assert_called_once_with('npu', 2)
        self.assertIsNone(self.model.received)

    def test_nonfinite_and_invalid_output_rejected(self):
        adapter = self.adapter()
        invalid = [torch.zeros(1, 2, 84), torch.zeros(2, 1, 85), torch.zeros(1, 4000, 85)]
        for column, value in ((0, float('inf')), (4, -1), (5, 1.1), (2, -1)):
            row = prediction()
            row[column] = value
            invalid.append(torch.from_numpy(row.reshape(1, 1, 85)))
        for output in invalid:
            self.model.output = output
            with self.subTest(shape=output.shape), self.assertRaises(ValueError):
                adapter.predict(self.frame(), 0)

    def test_model_identity_changes_with_effective_settings(self):
        original = self.adapter().model_id
        self.assertEqual(original, self.adapter().model_id)
        for options in ({'variant': 'yolox-tiny'}, {'input_size': 320},
                        {'score_threshold': .4}, {'nms_threshold': .2}):
            with self.subTest(options=options):
                self.assertNotEqual(original, self.adapter(**options).model_id)

    def test_close_is_idempotent_and_rejects_reuse(self):
        adapter = self.adapter()
        adapter.close()
        adapter.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            adapter.predict(self.frame(), 0)

    def test_import_does_not_load_optional_inference_dependencies(self):
        code = ("import sys; import inference.experts.yolox_adapter; "
                "assert not {'torch','cv2','yolox'} & set(sys.modules)")
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
