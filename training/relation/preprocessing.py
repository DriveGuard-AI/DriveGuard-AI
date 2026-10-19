"""Shared NumPy quality preprocessing. No labels or platform libraries are inputs."""
import numpy as np

from training.common.normalization import normalize
from training.relation.profile import FEATURE_NAMES, GROUP_INDICES, GROUP_DEPENDENCIES, validate_group_profile


def dependency_masks(masks, ages, profile):
    """Filter one contiguous interval; callers supply history_steps-1 preceding rows.

    Existing masks encode geometry/ownership and observed absence. Ages only remove
    evidence; they never create a valid measurement. Cached history after a gap is
    conservatively hidden until that gap leaves the full history window.
    """
    validate_group_profile(profile, FEATURE_NAMES)
    masks, ages = np.asarray(masks), np.asarray(ages)
    if masks.ndim != 2 or masks.shape[1] != 19 or ages.shape != (len(masks),3):
        raise ValueError('Expected masks[N,19] and ages[N,3]')
    if not np.isin(masks, [0,1]).all() or not np.isfinite(ages).all() or (ages < 0).any() or (ages > profile['max_age_ms']+1).any():
        raise ValueError('Invalid masks or expert ages')
    effective = masks.astype(bool,copy=True)
    fresh = ages <= profile['max_age_ms']
    for indices, deps in zip(GROUP_INDICES,GROUP_DEPENDENCIES):
        effective[:,indices] &= fresh[:,deps].all(axis=1)[:,None]
    # Statistics cannot retain information derived before an observation gap.
    history = profile['history_steps']
    for targets, parents in (((3,4,12),(0,1)), ((15,),(13,14)), ((18,),(16,17))):
        bad = (~effective[:,parents].all(axis=1)).astype(np.int64)
        prefix = np.concatenate(([0],np.cumsum(bad)))
        end = np.arange(1,len(bad)+1)
        valid_history = prefix[end] == prefix[np.maximum(0,end-history)]
        effective[:,targets] &= valid_history[:,None]
    return effective


def prepare_window(values, masks, ages, normalizer, profile):
    """Build model inputs from dependency-filtered masks and real ages.

    Dependency history is filtered before slicing this window, not reset at each
    training window boundary. The same function is available to inference clients.
    """
    validate_group_profile(profile, normalizer['feature_names'])
    values, masks, ages = np.asarray(values), np.asarray(masks), np.asarray(ages)
    if values.ndim != 2 or values.shape[1] != 19 or masks.shape != values.shape or ages.shape != (len(values),3) or not len(values):
        raise ValueError('Invalid relation window shape')
    if not np.isfinite(values).all() or not np.isfinite(ages).all() or not np.isin(masks,[0,1]).all() or (ages<0).any() or (ages>profile['max_age_ms']+1).any():
        raise ValueError('Invalid relation window values/ages/masks')
    # Defense in depth for stale current fields, without restarting history here.
    effective = masks.astype(bool,copy=True)
    for indices, deps in zip(GROUP_INDICES,GROUP_DEPENDENCIES):
        effective[:,indices] &= (ages[:,deps] <= profile['max_age_ms']).all(axis=1)[:,None]
    x = normalize(values,effective,normalizer)
    effective = x[:,19:].astype(bool)
    supported = np.asarray(normalizer['supported'],dtype=bool)
    quality = np.zeros((4,2),np.float32)
    eligible = np.zeros(4,dtype=bool)
    for i,(indices,deps) in enumerate(zip(GROUP_INDICES,GROUP_DEPENDENCIES)):
        count = supported[list(indices)].sum()
        ratio = effective[:,indices].sum()/(len(values)*count) if count else 0.
        age = ages[-1,list(deps)].max()
        present = bool(effective[-1,list(indices)].any())
        quality[i] = (ratio, max(0.,1.-float(age)/profile['max_age_ms']) if present else 0.)
        eligible[i] = bool(count and present and ratio >= profile['min_valid_ratio'] and age <= profile['max_age_ms'])
    return {'x': x, 'quality': quality, 'eligible': eligible}
