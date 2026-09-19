"""Dependency-free, versioned evidence grouping; safe in python -S prechecks."""
from copy import deepcopy
import math

FEATURE_NAMES = ('ear_left', 'ear_right', 'mar', 'perclos_4s', 'closed_duration_s',
                 'head_yaw_deg', 'head_pitch_deg', 'head_roll_deg', 'phone_conf',
                 'drink_object_conf', 'phone_face_distance', 'drink_mouth_distance',
                 'blink_rate_per_min', 'phone_hand_distance', 'phone_hand_iou',
                 'phone_near_duration_s', 'drink_hand_distance', 'drink_hand_iou',
                 'drink_near_duration_s')
GROUP_INDICES = ((0,1,2,3,4,5,6,7,12), (8,9), (13,14,15,16,17,18), (10,11))
GROUP_DEPENDENCIES = ((1,), (0,), (0,2), (0,1))


def default_group_profile():
    return {'schema_version': 'driveguard-groups-v1',
            'groups': [{'name': name, 'features': [FEATURE_NAMES[i] for i in indices],
                        'experts': [ ('objects','face','hands')[i] for i in deps]}
                       for name, indices, deps in zip(('F','O','HO','FO'), GROUP_INDICES, GROUP_DEPENDENCIES)],
            'age_columns': ['objects','face','hands'], 'max_age_ms': 80,
            'history_steps': 100, 'min_valid_ratio': 0.5,
            'quality': ['valid_ratio','freshness'],
            'history_policy': 'conservative-invalid-until-gap-leaves-history-v1'}


def validate_group_profile(profile, feature_names):
    expected = default_group_profile()
    if not isinstance(profile, dict) or set(profile) != set(expected):
        raise ValueError('Invalid group_profile fields')
    if list(feature_names) != list(FEATURE_NAMES):
        raise ValueError('relation_gated_v1 requires the canonical 19 feature order')
    for key in set(expected) - {'min_valid_ratio'}:
        if profile[key] != expected[key] or (key in ('max_age_ms','history_steps') and type(profile[key]) is not int):
            raise ValueError(f'Unsupported group_profile {key}')
    ratio = profile['min_valid_ratio']
    if type(ratio) not in (int,float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('min_valid_ratio must be finite in [0,1]')
    return deepcopy(profile)
