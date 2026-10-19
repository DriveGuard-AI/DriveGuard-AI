"""Dependency-free, explicitly versioned head eligibility experiment settings."""
import math
from decimal import Decimal

HEAD_MODEL_KIND = 'relation_gaze_head_v1'
POLICY_MODEL_KIND = 'relation_gaze_head_policy_v1'
HEAD_MODEL_KINDS = (HEAD_MODEL_KIND, POLICY_MODEL_KIND)
HEAD_MIN_VALID_RATIO = .5


def validate_head_policy(policy, window_steps=100):
    if not isinstance(policy, dict) or set(policy) != {'min_valid_ratio', 'max_gap_steps'}:
        raise ValueError('head_policy requires min_valid_ratio and max_gap_steps')
    ratio, gap = policy['min_valid_ratio'], policy['max_gap_steps']
    if type(ratio) not in (float, int) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('head_policy min_valid_ratio must be finite and in (0, 1]')
    if type(gap) is not int or not 0 <= gap < window_steps:
        raise ValueError('head_policy max_gap_steps must be an integer in [0, window_steps)')
    return dict(policy)


def resolve_head_policy(policy=None, window_steps=100):
    if policy is None:
        policy = dict(min_valid_ratio=HEAD_MIN_VALID_RATIO, max_gap_steps=0)
    return validate_head_policy(policy, window_steps)


def minimum_head_count(policy, window_steps):
    """One integer boundary for NumPy, Torch and audits, including decimal ratios."""
    return math.ceil(Decimal(str(policy['min_valid_ratio'])) * window_steps)
