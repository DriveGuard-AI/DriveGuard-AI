"""Explicit new extraction contract; legacy profiles remain unchanged."""
from copy import deepcopy

from inference.perception.profile import load_profile, validate_profile, FEATURE_NAMES

IRIS_NAMES = ('iris_left_x','iris_left_y','iris_right_x','iris_right_y')
GEOMETRY = {'version':'mediapipe-matrix-iris-v1', 'matrix_basis':'C=diag(1,-1,-1); R=C*R_mp*C',
            'rotation':'Rz(roll)Ry(yaw)Rx(pitch)', 'iris_left':[362,263,473],
            'iris_right':[33,133,468], 'eye_open_ear_min':.2, 'eye_open_ear_max':.8,
            'iris_margin_eye_width':.05, 'gimbal_cosine_min':.0001}


def gaze_profile():
    profile = load_profile()
    profile.update(schema_version='driveguard-extractor-profile-v2',name='interaction-gaze-v2',
                   feature_schema_version='driveguard-interaction-v2',
                   feature_names=list(FEATURE_NAMES + IRIS_NAMES),gaze_geometry=deepcopy(GEOMETRY))
    for axis in ('yaw','pitch','roll'):
        profile['feature_definitions'][f'head_{axis}_deg'] = (
            f'MediaPipe canonical face {axis} estimate; validated matrix; '+ GEOMETRY['matrix_basis'] +
            '; '+ GEOMETRY['rotation'] + '; not vehicle-calibrated gaze; invalid matrix => unknown.')
    for side in ('left','right'):
        for axis in ('x','y'):
            profile['feature_definitions'][f'iris_{side}_{axis}'] = (
                f'Anatomical {side} iris center local {axis} / eye width; 478 points only; '
                'x along configured corners, y clockwise image perpendicular; closed/invalid => unknown.')
    return profile


def validate_gaze_profile(profile):
    if not isinstance(profile,dict) or set(profile) != set(gaze_profile()):
        raise ValueError('Invalid gaze extraction profile keys')
    if (profile['schema_version'] != 'driveguard-extractor-profile-v2' or
        profile['feature_schema_version'] != 'driveguard-interaction-v2' or
        profile['feature_names'] != list(FEATURE_NAMES + IRIS_NAMES) or profile['gaze_geometry'] != GEOMETRY):
        raise ValueError('Unsupported gaze feature geometry/order')
    definitions = profile['feature_definitions']
    if not isinstance(definitions,dict) or set(definitions) != set(profile['feature_names']) or any(
            not isinstance(value,str) or not value.strip() for value in definitions.values()):
        raise ValueError('All gaze feature definitions are required')
    legacy = deepcopy(profile)
    legacy.pop('gaze_geometry')
    legacy.update(schema_version='driveguard-extractor-profile-v1',feature_schema_version='driveguard-interaction-v1',
                  feature_names=list(FEATURE_NAMES),feature_definitions={k:definitions[k] for k in FEATURE_NAMES})
    validate_profile(legacy)
    return profile
