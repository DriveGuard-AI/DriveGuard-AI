"""Shared 23-feature NumPy preprocessing for training and inference."""
import numpy as np
from training.common.normalization import normalize
from training.relation.preprocessing import dependency_masks as legacy_dependency_masks
from training.relation.profile import default_group_profile as legacy_profile
from training.gaze.profile import FEATURE_NAMES, GROUP_INDICES, GROUP_DEPENDENCIES, validate_group_profile
from training.gaze.profile import GAZE_HEAD_INDICES, GAZE_IRIS_PAIRS, GAZE_MIN_VALID_RATIO


def dependency_masks(masks, ages, profile):
    validate_group_profile(profile,FEATURE_NAMES)
    masks,ages = np.asarray(masks),np.asarray(ages)
    if masks.ndim != 2 or masks.shape[1] != 23 or ages.shape != (len(masks),3):
        raise ValueError('Expected masks[N,23] and ages[N,3]')
    if not np.isin(masks,[0,1]).all():
        raise ValueError('Feature masks must be binary')
    original = legacy_dependency_masks(masks[:,:19],ages,legacy_profile())
    iris = masks[:,19:].astype(bool) & (ages[:,1:2] <= profile['max_age_ms'])
    return np.concatenate((original,iris),axis=1)


def gaze_available(masks, eligible):
    """Availability from input evidence only; labels never enter preprocessing."""
    masks = np.asarray(masks,dtype=bool)
    good = masks[:,GAZE_HEAD_INDICES].all(1) & (masks[:,GAZE_IRIS_PAIRS[0]].all(1) | masks[:,GAZE_IRIS_PAIRS[1]].all(1))
    return bool(eligible[0] and good[-1] and good.mean() >= GAZE_MIN_VALID_RATIO)


def prepare_window(values, masks, ages, normalizer, profile):
    validate_group_profile(profile,normalizer['feature_names'])
    values,masks,ages = np.asarray(values),np.asarray(masks),np.asarray(ages)
    if values.ndim != 2 or values.shape[1] != 23 or masks.shape != values.shape or ages.shape != (len(values),3) or not len(values):
        raise ValueError('Invalid joint window shape')
    if not np.isfinite(values).all() or not np.isfinite(ages).all() or not np.isin(masks,[0,1]).all() or (ages<0).any() or (ages>profile['max_age_ms']+1).any():
        raise ValueError('Invalid joint window values/ages/masks')
    effective=masks.astype(bool,copy=True)
    for indices,deps in zip(GROUP_INDICES,GROUP_DEPENDENCIES):
        effective[:,indices] &= (ages[:,deps] <= profile['max_age_ms']).all(1)[:,None]
    x=normalize(values,effective,normalizer)
    effective=x[:,23:].astype(bool)
    supported=np.asarray(normalizer['supported'],dtype=bool)
    quality=np.zeros((4,2),np.float32); eligible=np.zeros(4,dtype=bool)
    for i,(indices,deps) in enumerate(zip(GROUP_INDICES,GROUP_DEPENDENCIES)):
        count=supported[list(indices)].sum()
        ratio=effective[:,indices].sum()/(len(values)*count) if count else 0.
        age=ages[-1,list(deps)].max()
        present=bool(effective[-1,list(indices)].any())
        quality[i]=(ratio,max(0.,1.-float(age)/profile['max_age_ms']) if present else 0.)
        eligible[i]=bool(count and present and ratio >= profile['min_valid_ratio'] and age <= profile['max_age_ms'])
    return {'x':x,'quality':quality,'eligible':eligible}
