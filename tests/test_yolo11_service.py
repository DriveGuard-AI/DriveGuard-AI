"""Real tensor/geometry and loopback transport tests; no model downloads.

Only vendor model construction and constant-color OpenCV resizing are doubled.
These tests do not certify real Ultralytics weights or Ascend execution.
"""
import hashlib
import http.client
import importlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


SOURCE = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt'


def load_module(test, name):
    try:
        return importlib.import_module('inference.experts.' + name)
    except ModuleNotFoundError:
        test.fail(name + ' is not implemented')


def row(category=67, box=(320, 320, 320, 160), score=.9):
    result = np.zeros(84, dtype=np.float32)
    result[:4], result[4 + category] = box, score
    return result


class ModelDouble(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.names = {i: 'class-' + str(i) for i in range(80)}
        self.names.update({39: 'bottle', 41: 'cup', 67: 'cell phone'})
        self.output = torch.zeros(1, 84, 0)
        self.received = None

    def forward(self, image):
        self.received = image.detach().clone()
        return self.output


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(self, 'yolo11_adapter')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / 'yolo11s.pt'
        path.write_bytes(b'test-only-weights')
        self.asset = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          source=SOURCE, license='AGPL-3.0')
        self.model = ModelDouble()
        self.loads = []

        def factory(path):
            self.loads.append(path)
            return types.SimpleNamespace(model=self.model)

        self.vendor = types.SimpleNamespace(__version__='8.3.199', YOLO=factory)
        cv2 = types.SimpleNamespace(__version__='test-cv2', INTER_LINEAR=1,
            resize=lambda frame, size, **kw: np.broadcast_to(frame[0, 0], (size[1], size[0], 3)).copy())
        patcher = patch.dict(sys.modules, {'ultralytics': self.vendor, 'cv2': cv2})
        patcher.start()
        self.addCleanup(patcher.stop)

    def adapter(self, **kwargs):
        return self.module.YOLO11Adapter(self.asset, backend='cpu', **kwargs)

    def output(self, *rows):
        self.model.output = torch.from_numpy(np.array(rows, dtype=np.float32).reshape(-1, 84).T[None])

    def frame(self):
        return np.broadcast_to(np.array([10, 20, 30], dtype=np.uint8), (200, 400, 3)).copy()

    def test_centered_rgb_normalization_and_original_coordinates(self):
        self.output(row())
        adapter = self.adapter()
        result = adapter.predict(self.frame(), 120)
        np.testing.assert_allclose(result['items'][0]['box'], [100, 50, 300, 150])
        self.assertEqual(result['items'][0]['category'], 'phone')
        self.assertIsNone(result['items'][0]['track_id'])
        self.assertEqual(result['timestamp_ms'], 120)
        self.assertEqual(result['model_id'], adapter.model_id)
        image = self.model.received.numpy()
        self.assertEqual(image.shape, (1, 3, 640, 640))
        np.testing.assert_allclose(image[0, :, 159, 0], np.array([114] * 3) / 255)
        np.testing.assert_allclose(image[0, :, 160, 0], np.array([30, 20, 10]) / 255)
        np.testing.assert_allclose(image[0, :, 480, 0], np.array([114] * 3) / 255)
        self.assertFalse(self.model.training)

    def test_stable_class_aware_nms_argmax_and_tuple_output(self):
        unrelated = row(score=.8)
        unrelated[4] = .95
        self.output(row(score=.9), row(score=.9), row(39, score=.7), row(41, score=.6), unrelated)
        self.model.output = (self.model.output, [])
        items = self.adapter().predict(self.frame(), 0)['items']
        self.assertEqual([item['category'] for item in items], ['phone', 'bottle', 'cup'])

    def test_cpu_forward_uses_channels_last_layout(self):
        # CPU convolutions measured 92 ms instead of 108 ms in this layout with a
        # bit-identical result; the input tensor must carry the same layout.
        self.output(row())
        adapter = self.adapter()
        result = adapter.predict(self.frame(), 120)
        received = self.model.received
        self.assertTrue(received.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(tuple(received.shape), (1, 3, 640, 640))
        # The logical values are unchanged by the memory format.
        np.testing.assert_allclose(received[0, :, 160, 0], np.array([30, 20, 10]) / 255)
        self.assertEqual(result['items'][0]['category'], 'phone')

    def test_clipping_and_degenerate_padding_box(self):
        self.output(row(box=(0, 160, 320, 160)), row(box=(100, 40, 20, 20)))
        items = self.adapter().predict(self.frame(), 0)['items']
        self.assertEqual(len(items), 1)
        np.testing.assert_allclose(items[0]['box'], [0, 0, 100, 50])

    def test_invalid_output_and_frame_are_rejected(self):
        adapter = self.adapter()
        for output in (torch.zeros(1, 85, 1), torch.zeros(2, 84, 1),
                       torch.zeros(1, 84, 8401), torch.full((1, 84, 1), float('nan'))):
            self.model.output = output
            with self.subTest(shape=output.shape), self.assertRaises(ValueError):
                adapter.predict(self.frame(), 0)
        for frame in (np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2, 3)), np.zeros((2, 2), np.uint8)):
            with self.assertRaises(ValueError):
                adapter.predict(frame, 0)
        with self.assertRaises(ValueError):
            adapter.predict(self.frame(), True)
        adapter.close()
        adapter.close()
        with self.assertRaises(RuntimeError):
            adapter.predict(self.frame(), 0)

    def test_verified_official_asset_and_pinned_vendor_before_loading(self):
        self.asset['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.adapter()
        self.assertFalse(self.loads)
        self.asset['sha256'] = hashlib.sha256(Path(self.asset['path']).read_bytes()).hexdigest()
        self.vendor.__version__ = '8.3.200'
        with self.assertRaises(ValueError):
            self.adapter()
        self.assertFalse(self.loads)
        self.vendor.__version__ = '8.3.199'
        self.asset['source'] = 'https://example.invalid/untrusted.pt'
        with self.assertRaises(ValueError):
            self.adapter()
        self.assertFalse(self.loads)

    def test_device_failure_never_falls_back_and_identity_binds_profile(self):
        with patch('training.common.device.select_device', side_effect=RuntimeError('NPU unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'NPU unavailable'):
                self.module.YOLO11Adapter(self.asset)
        one = self.adapter().model_id
        two = self.adapter(input_size=320).model_id
        self.assertNotEqual(one, two)
        self.model.names[67] = 'wrong'
        with self.assertRaises(ValueError):
            self.adapter()


class FakeDetector:
    model_id = 'yolo11:test-package-device-identity'
    identity = {'backend': 'cpu', 'ultralytics_version': '8.3.199'}

    def __init__(self):
        self.calls = []
        self.closed = False
        self.fail = False

    def predict(self, frame, timestamp_ms):
        if self.fail:
            raise RuntimeError('injected device failure')
        self.calls.append((frame.copy(), timestamp_ms))
        return dict(status='ok', timestamp_ms=timestamp_ms, model_id=self.model_id,
                    items=[dict(category='phone', box=[0, 0, 1, 1], score=.9, track_id=None)])

    def close(self):
        self.closed = True


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.server_module = load_module(self, 'detector_service')
        self.client_module = load_module(self, 'detector_client')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        token = Path(self.temp.name) / 'token'
        token.write_text('a' * 64)
        self.settings = dict(asset=dict(path=str(Path(self.temp.name) / 'yolo11s.pt'), sha256='a' * 64,
                            source=SOURCE, license='AGPL-3.0'), variant='yolo11s', backend='cpu',
                            device_index=0, input_size=640, score_threshold=.25, nms_threshold=.45)
        self.service = dict(host='127.0.0.1', port=18761, token_file=str(token))
        self.detector = FakeDetector()
        self.server = self.server_module.create_server(self.settings, self.service, self.detector, port=0)
        self.service['port'] = self.server.server_address[1]
        self.server_module.warmup_and_write_ready(self.server, Path(self.temp.name) / 'initial-ready.json')
        self.detector.calls.clear()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup_server)

    def cleanup_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def client(self, **options):
        return self.client_module.RemoteDetector(**dict(self.settings, **options), service=self.service)

    def request(self, method, path, body=b'', headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.service['port'], timeout=2)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        result = response.status, response.read()
        conn.close()
        return result

    def test_authenticated_roundtrip_and_identity(self):
        client = self.client()
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        result = client.predict(frame[:, ::-1], 40)
        self.assertEqual(result['model_id'], self.detector.model_id)
        self.assertEqual(client.model_id, self.detector.model_id)
        np.testing.assert_array_equal(self.detector.calls[-1][0], frame[:, ::-1])
        self.assertEqual(self.detector.calls[-1][1], 40)
        client.close()
        with self.assertRaises(RuntimeError):
            client.predict(frame, 80)

    def test_auth_identity_dimensions_and_length_rejected_before_inference(self):
        self.assertEqual(self.request('GET', '/health')[0], 401)
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            self.client(input_size=320)
        client = self.client()
        headers = {'Authorization': 'Bearer ' + 'a' * 64, 'X-Detector-Identity': client.request_identity,
                   'X-Model-Id': client.model_id, 'X-Frame-Width': '2', 'X-Frame-Height': '2',
                   'X-Timestamp-Ms': '0', 'Content-Type': 'application/octet-stream'}
        self.assertEqual(self.request('POST', '/infer', b'x', headers)[0], 400)
        headers['X-Frame-Width'] = '999999999'
        self.assertEqual(self.request('POST', '/infer', b'', headers)[0], 400)
        self.assertEqual(len(self.detector.calls), 0)

    def test_inference_error_is_not_a_success_or_fallback(self):
        client = self.client()
        self.detector.fail = True
        with self.assertLogs(self.server_module.LOG, level='ERROR'):
            with self.assertRaisesRegex(RuntimeError, '503'):
                client.predict(np.zeros((2, 2, 3), np.uint8), 0)

    def test_warmup_precedes_ready_and_failed_warmup_leaves_no_ready(self):
        ready = Path(self.temp.name) / 'ready.json'
        self.server_module.warmup_and_write_ready(self.server, ready)
        payload = json.loads(ready.read_text())
        self.assertEqual(payload['model_id'], self.detector.model_id)
        self.assertEqual(self.detector.calls[0][0].shape, (640, 640, 3))
        self.detector.fail = True
        with self.assertRaises(RuntimeError):
            self.server_module.warmup_and_write_ready(self.server, ready)
        self.assertFalse(ready.exists())
        self.assertNotEqual(self.server.health['status'], 'ready')

    def test_non_ascii_auth_is_rejected(self):
        self.assertEqual(self.request('GET', '/health', headers={'Authorization': 'Bearer é'})[0], 401)

    def test_server_health_starts_warming(self):
        server = self.server_module.create_server(self.settings, self.service, self.detector, port=0)
        try:
            self.assertEqual(server.health['status'], 'warming')
        finally:
            server.server_close()

    def test_restart_model_identity_and_bad_result_are_rejected(self):
        client = self.client()
        self.detector.model_id = 'yolo11:changed-package-or-device'
        with self.assertRaisesRegex(RuntimeError, '409'):
            client.predict(np.zeros((2, 2, 3), np.uint8), 0)
        self.detector.model_id = client.model_id
        with patch.object(self.detector, 'predict', return_value=dict(status='ok', model_id=client.model_id,
                timestamp_ms=0, items=[dict(category='phone', box=[0, 0, 9, 9], score=.9, track_id=None)])):
            with self.assertLogs(self.server_module.LOG, level='ERROR'):
                with self.assertRaisesRegex(RuntimeError, '503'):
                    client.predict(np.zeros((2, 2, 3), np.uint8), 0)

    def test_truncated_request_and_duplicate_headers_do_not_infer(self):
        client = self.client()
        headers = (f'POST /infer HTTP/1.0\r\nAuthorization: Bearer {"a" * 64}\r\n'
            f'X-Detector-Identity: {client.request_identity}\r\nX-Model-Id: {client.model_id}\r\n'
            'Content-Type: application/octet-stream\r\nX-Frame-Height: 2\r\nX-Frame-Width: 2\r\n'
            'X-Timestamp-Ms: 0\r\nContent-Length: 12\r\n')
        for suffix in ('\r\nx', 'Content-Length: 1\r\n\r\nx', 'Transfer-Encoding: chunked\r\n\r\nx'):
            with socket.create_connection(('127.0.0.1', self.service['port']), timeout=2) as connection:
                connection.sendall((headers + suffix).encode())
                connection.shutdown(socket.SHUT_WR)
                response = connection.recv(4096)
                self.assertIn(b'400', response.split(b'\r\n')[0])
        self.assertEqual(self.detector.calls, [])

    def test_main_cleans_detector_socket_and_ready_on_signal(self):
        ready = Path(self.temp.name) / 'cli-ready.json'
        config = dict(objects=dict(self.settings, service=self.service))
        self.server.shutdown()
        self.thread.join(2)
        with patch('inference.perception.runtime.load_runtime_config', return_value=config), \
             patch.object(self.server_module.logging, 'basicConfig'), \
             patch('inference.experts.yolo11_adapter.YOLO11Adapter', return_value=self.detector), \
             patch.object(self.server_module, 'create_server', return_value=self.server), \
             patch.object(self.server, 'serve_forever', side_effect=SystemExit(143)), \
             patch.object(self.server, 'server_close', wraps=self.server.server_close) as close:
            with self.assertRaises(SystemExit):
                self.server_module.main(['--config', 'test-runtime.json', '--ready-file', str(ready)])
            close.assert_called_once()
        self.assertTrue(self.detector.closed)
        self.assertFalse(ready.exists())

    def test_occupied_port_rejected_before_model_construction(self):
        ready = Path(self.temp.name) / 'second-service-ready.json'
        config = dict(objects=dict(self.settings, service=self.service))
        with patch('inference.perception.runtime.load_runtime_config', return_value=config), \
             patch.object(self.server_module.logging, 'basicConfig'), \
             patch.object(self.server_module.DetectorHTTPServer, 'serve_forever', side_effect=AssertionError('second service started')), \
             patch('inference.experts.yolo11_adapter.YOLO11Adapter', return_value=FakeDetector()) as construct:
            with self.assertRaises(OSError):
                self.server_module.main(['--config', 'test-runtime.json', '--ready-file', str(ready)])
            construct.assert_not_called()
        self.assertFalse(ready.exists())
        self.assertEqual(self.client().model_id, self.detector.model_id)

    def test_socket_reserved_before_model_and_released_on_constructor_failure(self):
        self.server.shutdown()
        self.thread.join(2)
        self.server.server_close()
        ready = Path(self.temp.name) / 'failed-ready.json'
        config = dict(objects=dict(self.settings, service=self.service))

        def fail_constructor(**kwargs):
            with socket.socket() as probe:
                # An already listening server must reject another listener.
                with self.assertRaises(OSError):
                    probe.bind(('127.0.0.1', self.service['port']))
                    probe.listen()
            raise RuntimeError('model load failed')

        with patch('inference.perception.runtime.load_runtime_config', return_value=config), \
             patch.object(self.server_module.logging, 'basicConfig'), \
             patch('inference.experts.yolo11_adapter.YOLO11Adapter', side_effect=fail_constructor):
            with self.assertRaisesRegex(RuntimeError, 'model load failed'):
                self.server_module.main(['--config', 'test-runtime.json', '--ready-file', str(ready)])
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', self.service['port']))
            probe.listen()
        self.assertFalse(ready.exists())

    def test_existing_ready_file_is_preserved_without_constructing_model(self):
        ready = Path(self.temp.name) / 'foreign-ready.json'
        ready.write_text('belongs to existing service', encoding='utf-8')
        config = dict(objects=dict(self.settings, service=self.service))
        with patch('inference.perception.runtime.load_runtime_config', return_value=config), \
             patch.object(self.server_module.logging, 'basicConfig'), \
             patch.object(self.server_module.DetectorHTTPServer, 'serve_forever', side_effect=AssertionError('second service started')), \
             patch('inference.experts.yolo11_adapter.YOLO11Adapter', return_value=FakeDetector()) as construct:
            with self.assertRaises(FileExistsError):
                self.server_module.main(['--config', 'test-runtime.json', '--ready-file', str(ready)])
            construct.assert_not_called()
        self.assertEqual(ready.read_text(encoding='utf-8'), 'belongs to existing service')

    def test_warmup_refuses_to_replace_another_ready_file(self):
        ready = Path(self.temp.name) / 'other-ready.json'
        ready.write_text('other service', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.server_module.warmup_and_write_ready(self.server, ready)
        self.assertEqual(ready.read_text(encoding='utf-8'), 'other service')

    def test_token_and_loopback_validation(self):
        for update in ({'host': '0.0.0.0'}, {'token_file': 'relative'}, {'port': True}):
            with self.assertRaises(ValueError):
                self.client_module.RemoteDetector(**self.settings, service=dict(self.service, **update))

    def test_client_import_does_not_load_tensor_vendor(self):
        code = "import sys; import inference.experts.detector_client; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
        result = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
