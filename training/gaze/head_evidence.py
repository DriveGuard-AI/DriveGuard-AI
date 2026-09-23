"""Task availability for the opt-in head branch; inputs are already filtered masks."""
import numpy as np

HEAD_MODEL_KIND = 'relation_gaze_head_v1'
HEAD_MIN_VALID_RATIO = .5


def head_evidence(x):
    x = np.asarray(x)
    good = x[..., 28:31].astype(bool).all(axis=-1)
    ratio = good.mean(axis=-1)
    return good[..., -1] & (ratio >= HEAD_MIN_VALID_RATIO), ratio


def behavior_availability(inputs, model_kind):
    base = np.asarray(inputs['eligible'], dtype=bool).any(axis=-1)
    result = np.repeat(base[..., None], 4, axis=-1)
    if model_kind == HEAD_MODEL_KIND:
        result[..., 3] |= head_evidence(inputs['x'])[0]
    return result
