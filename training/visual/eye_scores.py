"""Tie-aware binary score diagnostics on unique, explicitly labelled frames."""
import math

import numpy as np


def score_summary(scores, targets, known):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets)
    known = np.asarray(known, dtype=bool)
    if scores.ndim != 1 or scores.shape != targets.shape or scores.shape != known.shape:
        raise ValueError('Expected aligned one-dimensional score/target/mask arrays')
    x, y = scores[known], targets[known]
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Known frames require finite scores and binary targets')
    positive, negative = int((y == 1).sum()), int((y == 0).sum())
    def counts(tp, fp):
        return dict(tp=tp, fp=fp, fn=positive-tp, tn=negative-fp,
                    recall=tp/positive if positive else None,
                    false_positive_rate=fp/negative if negative else None,
                    precision=tp/(tp+fp) if tp+fp else 0.)
    def distribution(values):
        if not len(values): return None
        return dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'),
                        map(float, np.quantile(values, [0, .05, .25, .5, .75, .95, 1]))))
    curve = [dict(threshold_logit=None, **counts(0, 0))]
    # None is the reject-all point; tied scores enter together, never by label.
    order = np.argsort(-x, kind='stable')
    x, y = x[order], y[order]
    tp = fp = previous_end = 0
    for end in np.r_[np.flatnonzero(np.diff(x)) + 1, len(x)] if len(x) else []:
        tp += int((y[previous_end:end] == 1).sum())
        fp = int(end) - tp
        curve.append(dict(threshold_logit=float(x[end-1]), **counts(tp, fp)))
        previous_end = int(end)
    auc = ap = None
    if positive:
        ap = sum((b['tp']-a['tp'])/positive*b['precision'] for a, b in zip(curve, curve[1:]))
    if positive and negative:
        auc = sum((b['fp']-a['fp'])/negative*(a['tp']+b['tp'])/(2*positive)
                  for a, b in zip(curve, curve[1:]))
    fpr_threshold = balanced_threshold = None
    if positive and negative:
        allowed = math.floor(.03*negative+1e-12)
        best_tp = 0
        best_balanced = .5
        for point in curve[1:]:
            if point['fp'] <= allowed and point['tp'] > best_tp:
                fpr_threshold, best_tp = point['threshold_logit'], point['tp']
            balanced = (point['recall']+1-point['false_positive_rate'])/2
            if balanced > best_balanced:
                balanced_threshold, best_balanced = point['threshold_logit'], balanced
    return dict(known=int(known.sum()), unknown_or_invalid=int((~known).sum()),
                positive=positive, negative=negative, roc_auc=auc, average_precision=ap,
                fixed_zero=counts(int(((x >= 0) & (y == 1)).sum()), int(((x >= 0) & (y == 0)).sum())),
                positive_scores=distribution(x[y == 1]), negative_scores=distribution(x[y == 0]),
                fpr03_threshold=fpr_threshold, balanced_accuracy_threshold=balanced_threshold,
                threshold_role='within_split_diagnostic_not_independent_calibration', curve=curve)
