"""CPU-worker client: no Torch, Ultralytics or accelerator imports."""
import http.client
import json

from inference.experts.protocol import (MAX_RESPONSE, TIMEOUT_SECONDS, VERSION, read_token,
    settings_identity, validate_frame, validate_result, validate_service)


class RemoteDetector:
    def __init__(self, asset, *, variant='yolo11s', backend='npu', device_index=0,
                 input_size=640, score_threshold=.25, nms_threshold=.45, service):
        self.service = validate_service(service)
        self._token = read_token(service)
        self.request_identity = settings_identity(asset, variant=variant, backend=backend,
            device_index=device_index, input_size=input_size, score_threshold=score_threshold,
            nms_threshold=nms_threshold)
        self._closed = False
        self._max_predictions = sum((input_size // stride) ** 2 for stride in (8, 16, 32))
        health = self._request('GET', '/health')
        if (health.get('protocol') != VERSION or health.get('request_identity') != self.request_identity
                or health.get('status') != 'ready' or not isinstance(health.get('model_id'), str)
                or not health['model_id'].startswith('yolo11:') or not isinstance(health.get('identity'), dict)):
            raise RuntimeError('detector service identity/readiness mismatch')
        self.model_id = health['model_id']
        self.identity = health['identity']

    def _request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(self.service['host'], self.service['port'], timeout=TIMEOUT_SECONDS)
        request_headers = {'Authorization': 'Bearer ' + self._token,
                           'X-Detector-Identity': self.request_identity}
        request_headers.update(headers or {})
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise RuntimeError('detector response exceeds size limit')
            if response.status != 200:
                raise RuntimeError(f'detector service HTTP {response.status}: {raw[:300].decode("utf-8", errors="replace")}')
            result = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(result, dict):
                raise ValueError('response must be an object')
            return result
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise RuntimeError(f'detector service request failed: {exc}') from exc
        finally:
            connection.close()

    def predict(self, bgr_uint8_HWC, timestamp_ms):
        if self._closed:
            raise RuntimeError('remote detector is closed')
        validate_frame(bgr_uint8_HWC, timestamp_ms)
        height, width = bgr_uint8_HWC.shape[:2]
        result = self._request('POST', '/infer', bgr_uint8_HWC.tobytes(order='C'), {
            'Content-Type': 'application/octet-stream', 'X-Model-Id': self.model_id,
            'X-Frame-Width': str(width), 'X-Frame-Height': str(height), 'X-Timestamp-Ms': str(timestamp_ms)})
        try:
            return validate_result(result, self.model_id, timestamp_ms, width, height, self._max_predictions)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def close(self):
        self._closed = True
