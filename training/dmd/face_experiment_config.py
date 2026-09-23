"""Strict standard-library-only face experiment configuration."""
from copy import deepcopy
import math
import re

from training.common.config import read_json

BASELINE = dict(name='baseline', mode='VIDEO', detection=.5, presence=.5, tracking=.5, crop_xyxy=None)


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {'schema_version', 'variants'} or config['schema_version'] != 'driveguard-face-experiments-v1':
        raise ValueError('Invalid face experiment configuration schema/keys')
    variants = config['variants']
    if not isinstance(variants, list) or not variants:
        raise ValueError('At least the explicit baseline variant is required')
    names = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict) or set(variant) != set(BASELINE):
            raise ValueError('Each variant requires exactly name, mode, detection, presence, tracking, crop_xyxy')
        name = variant['name']
        if not isinstance(name, str) or not re.fullmatch('[a-z][a-z0-9_-]{0,63}', name) or name in names:
            raise ValueError('Variant names must be unique safe lowercase identifiers')
        names.add(name)
        if variant['mode'] not in ('VIDEO', 'IMAGE'):
            raise ValueError('Variant mode must be VIDEO or IMAGE')
        for key in ('detection', 'presence', 'tracking'):
            value = variant[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{key} must be a finite confidence in [0, 1]')
        crop = variant['crop_xyxy']
        if crop is not None and (not isinstance(crop, list) or len(crop) != 4 or
                not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in crop) or
                crop[2] <= crop[0] or crop[3] <= crop[1] or crop == [0, 0, 1, 1]):
            raise ValueError('crop_xyxy must be an explicit nondegenerate normalized subrectangle, or null')
        changes = sum(variant[k] != BASELINE[k] for k in BASELINE if k != 'name')
        if index == 0 and variant != BASELINE:
            raise ValueError('First variant must be baseline: full-frame VIDEO with all thresholds .5')
        if index and changes != 1:
            raise ValueError('Every nonbaseline variant must change exactly one variable from baseline')
    return deepcopy(config)


def load_config(path):
    return validate_config(read_json(path))
