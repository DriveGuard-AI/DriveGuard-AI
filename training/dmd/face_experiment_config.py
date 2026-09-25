"""Strict standard-library-only face experiment configuration."""
from copy import deepcopy
import math
import re

from training.common.config import read_json

BASELINE = dict(name='baseline', mode='VIDEO', detection=.5, presence=.5, tracking=.5, crop_xyxy=None)


def validate_rectangle(rectangle):
    if (not isinstance(rectangle, list) or len(rectangle) != 4 or
            not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in rectangle) or
            rectangle[2] <= rectangle[0] or rectangle[3] <= rectangle[1] or rectangle == [0, 0, 1, 1]):
        raise ValueError('Region must be an explicit nondegenerate normalized subrectangle')


def validate_scope(scope):
    if not isinstance(scope, dict) or set(scope) != {'record_id', 'video_sha256', 'image_size'}:
        raise ValueError('Region scope requires record_id, video_sha256, image_size')
    if not isinstance(scope['record_id'], str) or not scope['record_id'].strip():
        raise ValueError('Region scope record_id must be nonempty')
    if not isinstance(scope['video_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', scope['video_sha256']):
        raise ValueError('Region scope video_sha256 must be a SHA256 digest')
    size = scope['image_size']
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        raise ValueError('Region scope image_size must be positive width,height')


def validate_variants(variants, *, regional=False):
    baseline = dict(BASELINE, head_region_mode='none') if regional else BASELINE
    if not isinstance(variants, list) or not variants:
        raise ValueError('At least the explicit baseline variant is required')
    names = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict) or set(variant) != set(baseline):
            raise ValueError('Invalid variant keys for experiment schema')
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
        if crop is not None:
            validate_rectangle(crop)
        if regional and (variant['head_region_mode'] not in ('none', 'gate', 'crop') or crop is not None):
            raise ValueError('Regional variants require none/gate/crop mode and null crop_xyxy; use per-record regions')
        changes = sum(variant[k] != baseline[k] for k in baseline if k != 'name')
        if index == 0 and variant != baseline:
            raise ValueError('First variant must be baseline: full-frame VIDEO with all thresholds .5')
        if index and changes != 1:
            raise ValueError('Every nonbaseline variant must change exactly one variable from baseline')


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('Invalid face experiment configuration')
    regional = config.get('schema_version') == 'driveguard-face-experiments-v2'
    allowed = ({'schema_version', 'variants', 'head_regions'},) if regional else (
        {'schema_version', 'variants'}, {'schema_version', 'variants', 'crop_scope'})
    if set(config) not in allowed or config.get('schema_version') not in (
            'driveguard-face-experiments-v1', 'driveguard-face-experiments-v2'):
        raise ValueError('Invalid face experiment configuration schema/keys')
    variants = config['variants']
    validate_variants(variants, regional=regional)
    if regional:
        regions = config['head_regions']
        if not isinstance(regions, list) or not regions:
            raise ValueError('At least one scoped head region is required')
        ids = set()
        for region in regions:
            if not isinstance(region, dict) or set(region) != {'record_id', 'video_sha256', 'image_size', 'head_roi_xyxy'}:
                raise ValueError('Invalid head region keys')
            validate_scope({k: v for k, v in region.items() if k != 'head_roi_xyxy'})
            validate_rectangle(region['head_roi_xyxy'])
            if region['record_id'] in ids:
                raise ValueError('Duplicate head region record scope')
            ids.add(region['record_id'])
        if not any(v['head_region_mode'] != 'none' for v in variants):
            raise ValueError('Head regions require a gate or crop variant')
    if 'crop_scope' in config:
        scope=config['crop_scope']
        validate_scope(scope)
        if not any(v['crop_xyxy'] is not None for v in variants):
            raise ValueError('crop_scope requires a crop variant')
    return deepcopy(config)


def load_config(path):
    return validate_config(read_json(path))
