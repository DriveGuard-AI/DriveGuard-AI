"""Canonical, dependency-free versioned joint evidence contract."""
from copy import deepcopy
from training.relation.profile import FEATURE_NAMES as LEGACY_FEATURE_NAMES
from training.relation.profile import GROUP_INDICES as LEGACY_GROUP_INDICES
from training.relation.profile import GROUP_DEPENDENCIES, default_group_profile as legacy_profile

FEATURE_NAMES = LEGACY_FEATURE_NAMES + ('iris_left_x','iris_left_y','iris_right_x','iris_right_y')
GAZE_CLASSES = ('left_mirror','left','front','center_mirror','front_right','right_mirror','right','infotainment','steering_wheel')
GROUP_INDICES = (LEGACY_GROUP_INDICES[0] + (19,20,21,22), *LEGACY_GROUP_INDICES[1:])
GAZE_HEAD_INDICES = (5,6,7)
GAZE_IRIS_PAIRS = ((19,20),(21,22))
GAZE_MIN_VALID_RATIO = 0.5


def default_group_profile():
    profile = legacy_profile()
    profile['schema_version'] = 'driveguard-groups-v2'
    profile['groups'][0]['features'] += list(FEATURE_NAMES[19:])
    profile['gaze_min_valid_ratio'] = GAZE_MIN_VALID_RATIO
    profile['gaze_validity'] = 'current-and-window-simultaneous-head-pose-and-one-iris-pair-v1'
    return profile


def validate_group_profile(profile, feature_names):
    if list(feature_names) != list(FEATURE_NAMES):
        raise ValueError('Joint model requires canonical 23 feature order')
    if profile != default_group_profile():
        raise ValueError('Unsupported joint group_profile')
    return deepcopy(profile)
