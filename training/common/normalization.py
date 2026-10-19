"""Training-only masked population statistics over unique recording slots."""
import numpy as np


def fit_normalizer(chunks, feature_names):
    size = len(feature_names)
    count = np.zeros(size, dtype=np.int64)
    mean = np.zeros(size, dtype=np.float64)
    m2 = np.zeros(size, dtype=np.float64)
    for values, masks in chunks:
        for c in range(size):
            selected = np.asarray(values[:, c][masks[:, c].astype(bool)], dtype=np.float64)
            n = len(selected)
            if not n:
                continue
            batch_mean = selected.mean()
            delta = batch_mean - mean[c]
            total = count[c] + n
            m2[c] += ((selected - batch_mean)**2).sum() + delta**2 * count[c] * n / total
            mean[c] += delta * n / total
            count[c] = total
    supported = count > 0
    std = np.maximum(np.sqrt(np.maximum(m2, 0) / np.maximum(count, 1)), 1e-6)
    std[~supported] = 1
    return {'feature_names': list(feature_names), 'count': count.tolist(),
            'mean': mean.tolist(), 'std': std.tolist(), 'supported': supported.tolist()}


def normalize(values, masks, state):
    size = values.shape[-1]
    mean = np.asarray(state['mean'], dtype=np.float64)
    std = np.asarray(state['std'], dtype=np.float64)
    supported = np.asarray(state['supported'], dtype=bool)
    if any(a.shape != (size,) for a in (mean, std, supported)):
        raise ValueError('Normalizer dimension mismatch')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid normalizer statistics')
    effective = masks.astype(bool) & supported
    scaled = np.clip((values.astype(np.float64) - mean) / std, -5, 5)
    scaled[~effective] = 0
    return np.concatenate((scaled.astype(np.float32), effective.astype(np.float32)), axis=-1)
