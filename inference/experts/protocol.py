"""Versioned, bounded loopback detector contract (standard library only)."""
import hashlib
import math
from pathlib import Path
import re

from inference.experts.assets import model_identity, validate_asset_spec

VERSION = 'driveguard-yolo11-http-v1'
ULTRALYTICS_VERSION = '8.3.199'
TRUSTED_SOURCE = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt'
MAX_SIDE = 16384
MAX_PIXELS = 64 * 1024 * 1024
MAX_RESPONSE = 8 * 1024 * 1024
TIMEOUT_SECONDS = 120


def settings_identity(asset, *, variant='yolo11s', backend='npu', device_index=0,
                      input_size=640, score_threshold=.25, nms_threshold=.45):
    validate_asset_spec(asset)
    if asset['source'] != TRUSTED_SOURCE:
        raise ValueError('YOLO11 requires the explicitly trusted official yolo11s.pt source')
    if variant != 'yolo11s' or backend not in ('cpu', 'cuda', 'npu'):
        raise ValueError('Expected yolo11s and explicit cpu/cuda/npu backend')
    if type(device_index) is not int or device_index < 0 or (backend == 'cpu' and device_index != 0):
        raise ValueError('Invalid device_index')
    if type(input_size) is not int or not 32 <= input_size <= 1280 or input_size % 32:
        raise ValueError('input_size must be a multiple of 32 in [32, 1280]')
    for value in (score_threshold, nms_threshold):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('threshold must be a finite probability')
    directory = Path(__file__).resolve().parent
    # Both processes must run matching preprocessing, transport and device code.
    sources = [directory / name for name in
               ('protocol.py', 'yolo11_adapter.py', 'detector_client.py', 'detector_service.py', 'assets.py')]
    sources.append(directory.parents[1] / 'training/common/device.py')
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes().replace(b'\r\n', b'\n'))
    profile = dict(protocol=VERSION, code_sha256=digest.hexdigest(),
                   ultralytics_version=ULTRALYTICS_VERSION, variant=variant, backend=backend,
                   device_index=device_index, input_size=input_size,
                   score_threshold=float(score_threshold), nms_threshold=float(nms_threshold))
    return model_identity('yolo11-request', asset, profile)


def validate_service(service):
    if not isinstance(service, dict) or set(service) != {'host', 'port', 'token_file'}:
        raise ValueError('service requires exactly host, port, token_file')
    if service['host'] != '127.0.0.1':
        raise ValueError('detector service must bind literal loopback 127.0.0.1')
    if type(service['port']) is not int or not 1 <= service['port'] <= 65535:
        raise ValueError('port must be an integer in [1, 65535]')
    if not isinstance(service['token_file'], str) or not Path(service['token_file']).is_absolute():
        raise ValueError('token_file must be an absolute path')
    return dict(service)


def read_token(service):
    validate_service(service)
    with Path(service['token_file']).open('r', encoding='ascii') as handle:
        token = handle.read(258).strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{32,256}', token):
        raise ValueError('token_file must contain a 32 to 256 character random ASCII token')
    return token


def validate_dimensions(height, width, timestamp_ms):
    if (type(height) is not int or type(width) is not int
            or not 0 < height <= MAX_SIDE or not 0 < width <= MAX_SIDE
            or height * width > MAX_PIXELS):
        raise ValueError('frame dimensions exceed bounded HWC BGR contract')
    if type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 2**63 - 1:
        raise ValueError('timestamp_ms must be a nonnegative signed-64-bit integer')
    return height * width * 3


def validate_frame(frame, timestamp_ms):
    import numpy as np
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('frame must be a uint8 HWC BGR array')
    return validate_dimensions(*frame.shape[:2], timestamp_ms)


def validate_result(result, model_id, timestamp_ms, width, height, max_items):
    if (not isinstance(result, dict) or set(result) != {'status', 'model_id', 'timestamp_ms', 'items'}
            or result['status'] != 'ok' or result['model_id'] != model_id
            or type(result['timestamp_ms']) is not int or result['timestamp_ms'] != timestamp_ms
            or not isinstance(result['items'], list) or len(result['items']) > max_items):
        raise ValueError('invalid detector response identity or schema')
    for item in result['items']:
        if (not isinstance(item, dict) or set(item) != {'category', 'box', 'score', 'track_id'}
                or item['category'] not in ('phone', 'bottle', 'cup') or item['track_id'] is not None
                or type(item['score']) not in (int, float) or not math.isfinite(item['score'])
                or not 0 <= item['score'] <= 1 or not isinstance(item['box'], list)
                or len(item['box']) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in item['box'])):
            raise ValueError('invalid detector item')
        x1, y1, x2, y2 = item['box']
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('detector box outside frame')
    return result
