"""Dependency-free validation of opt-in training experiments."""
import math


def validate_experiment(settings):
    if not isinstance(settings, dict) or set(settings) != {'missing_head_positive_multiplier', 'face_dropout_probability'}:
        raise ValueError('Invalid body_dist_experiment keys')
    for key, value in settings.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite numeric')
    if not 1 <= settings['missing_head_positive_multiplier'] <= 10:
        raise ValueError('missing_head_positive_multiplier must be in [1,10]')
    if not 0 <= settings['face_dropout_probability'] <= 1:
        raise ValueError('face_dropout_probability must be in [0,1]')
    return settings
