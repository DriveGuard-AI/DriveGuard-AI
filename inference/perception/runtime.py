"""Strict, standard-library-only runtime configuration preflight."""
from copy import deepcopy
import math
from pathlib import Path

from inference.experts.assets import validate_asset_spec
from training.common.config import canonical_hash, read_json


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f'{name} requires exactly: {keys}')


def load_runtime_config(path):
    config = read_json(path)
    _keys(config, 'schema_version recording_id view driver_roi driver_region_confirmed_exclusive objects face_asset hands_asset', 'runtime')
    if config['schema_version'] != 'driveguard-runtime-v1':
        raise ValueError('Unsupported runtime schema')
    for key in ('recording_id', 'view'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    if type(config['driver_region_confirmed_exclusive']) is not bool:
        raise ValueError('driver_region_confirmed_exclusive must be boolean, not automatic identity')
    roi = config['driver_roi']
    if not isinstance(roi, list) or len(roi) != 4 or not all(type(x) in (int, float) and math.isfinite(x) for x in roi):
        raise ValueError('driver_roi requires four finite pixel coordinates')
    if min(roi) < 0 or roi[2]-roi[0] < 1e-6 or roi[3]-roi[1] < 1e-6:
        raise ValueError('driver_roi must be nonnegative and nondegenerate')
    objects = config['objects']
    _keys(objects, 'asset variant backend device_index input_size score_threshold nms_threshold', 'objects')
    if objects['variant'] not in ('yolox-nano', 'yolox-tiny') or objects['backend'] not in ('cpu', 'cuda', 'npu'):
        raise ValueError('Expected built-in YOLOX nano/tiny and explicit cpu/cuda/npu backend')
    if type(objects['device_index']) is not int or objects['device_index'] < 0 or (objects['backend'] == 'cpu' and objects['device_index'] != 0):
        raise ValueError('Invalid device index')
    size = objects['input_size']
    if type(size) is not int or not 32 <= size <= 1280 or size % 32:
        raise ValueError('input_size must be a multiple of 32 in [32,1280]')
    for key in ('score_threshold', 'nms_threshold'):
        value = objects[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'{key} must be in (0,1]')
    for asset in (objects['asset'], config['face_asset'], config['hands_asset']):
        validate_asset_spec(asset)
        asset['path'] = str((Path(path).resolve().parent / Path(asset['path'])).resolve())
    return config


def runtime_signature(config):
    portable = deepcopy(config)
    for asset in (portable['objects']['asset'], portable['face_asset'], portable['hands_asset']):
        asset.pop('path')
    return canonical_hash({'implementation': 'driveguard-video-v1', 'config': portable,
                           'sampling': 'PTS-ceil-ms-causal-25Hz', 'tracking': 'unique-IoU0.3-age80ms-v1'})
