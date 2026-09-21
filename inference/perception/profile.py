"""Versioned candidate feature definition; importable without vision runtimes."""
from copy import deepcopy
import math
from pathlib import Path

from training.common.config import read_json

DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / 'configs/perception/interaction_v1.json'
FEATURE_NAMES = ('ear_left', 'ear_right', 'mar', 'perclos_4s', 'closed_duration_s',
                 'head_yaw_deg', 'head_pitch_deg', 'head_roll_deg', 'phone_conf',
                 'drink_object_conf', 'phone_face_distance', 'drink_mouth_distance',
                 'blink_rate_per_min', 'phone_hand_distance', 'phone_hand_iou',
                 'phone_near_duration_s', 'drink_hand_distance', 'drink_hand_iou',
                 'drink_near_duration_s')
COORDINATES = 'original_unmirrored_pixel_xy; x_right_y_down; head_pose_camera_z_forward_RzRyRx_degrees'
KEYS = {'schema_version', 'name', 'feature_schema_version', 'feature_names', 'sample_rate_hz',
        'max_age_ms', 'min_object_score', 'ear_closed_threshold', 'perclos_window_steps',
        'perclos_min_valid_ratio', 'blink_min_ms', 'blink_max_ms', 'hand_near_threshold',
        'eye_indices', 'mouth_indices', 'coordinate_system', 'feature_definitions'}


def validate_profile(profile):
    if isinstance(profile, dict) and profile.get('schema_version') == 'driveguard-extractor-profile-v2':
        from inference.perception.gaze_profile import validate_gaze_profile
        return validate_gaze_profile(profile)
    if not isinstance(profile, dict) or set(profile) != KEYS:
        raise ValueError('Invalid feature profile keys')
    if (profile['schema_version'] != 'driveguard-extractor-profile-v1' or
            profile['feature_schema_version'] != 'driveguard-interaction-v1' or
            profile['feature_names'] != list(FEATURE_NAMES) or
            profile['coordinate_system'] != COORDINATES):
        raise ValueError('Unsupported feature schema/order/coordinates')
    if not isinstance(profile['name'], str) or not profile['name'].strip():
        raise ValueError('Profile name required')
    for key in ('sample_rate_hz', 'perclos_window_steps', 'max_age_ms', 'blink_min_ms', 'blink_max_ms'):
        if type(profile[key]) is not int or profile[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if profile['sample_rate_hz'] != 25 or profile['perclos_window_steps'] != 100:
        raise ValueError('v1 fixes 25 Hz and a 100-slot statistics window')
    if not 0 < profile['blink_min_ms'] <= profile['blink_max_ms'] <= 4000:
        raise ValueError('Invalid blink duration bounds')
    if profile['max_age_ms'] > 1000:
        raise ValueError('Expert age must not exceed 1000 ms')
    for key in ('min_object_score', 'ear_closed_threshold', 'perclos_min_valid_ratio', 'hand_near_threshold'):
        value = profile[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'Invalid {key}')
    if (profile['eye_indices'] != {'left': [362, 385, 387, 263, 373, 380],
                                   'right': [33, 160, 158, 133, 153, 144]} or
            profile['mouth_indices'] != [13, 14, 78, 308]):
        raise ValueError('Landmark topology change requires a new supported profile')
    definitions = profile['feature_definitions']
    if (not isinstance(definitions, dict) or set(definitions) != set(FEATURE_NAMES) or
            any(not isinstance(v, str) or not v.strip() for v in definitions.values())):
        raise ValueError('Every feature requires a nonempty definition')
    return profile


def load_profile(path=None):
    return deepcopy(validate_profile(read_json(path or DEFAULT_PROFILE)))
