"""Task availability for the opt-in head branch; inputs are already filtered masks."""
import numpy as np

from training.gaze.head_policy import HEAD_MODEL_KIND, HEAD_MODEL_KINDS, HEAD_MIN_VALID_RATIO, resolve_head_policy
from training.gaze.head_policy import minimum_head_count


def head_statistics(x, policy=None):
    x = np.asarray(x)
    policy = resolve_head_policy(policy, x.shape[-2])
    good = x[..., 28:31].astype(bool).all(axis=-1)
    count = good.sum(axis=-1)
    ratio = good.mean(axis=-1)
    steps = good.shape[-1]
    last = np.where(good, np.arange(steps), -1).max(axis=-1)
    gap = steps - 1 - last
    eligible = (gap <= policy['max_gap_steps']) & (count >= minimum_head_count(policy, steps))
    return dict(eligible=eligible, valid_ratio=ratio, valid_count=count, gap_steps=gap, current_valid=good[..., -1])


def head_evidence(x, policy=None):
    stats = head_statistics(x, policy)
    return stats['eligible'], stats['valid_ratio']


def behavior_availability(inputs, model_kind, head_policy=None):
    base = np.asarray(inputs['eligible'], dtype=bool).any(axis=-1)
    result = np.repeat(base[..., None], 4, axis=-1)
    if model_kind in HEAD_MODEL_KINDS:
        result[..., 3] |= head_evidence(inputs['x'], head_policy)[0]
    return result
