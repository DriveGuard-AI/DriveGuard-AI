"""Train-only interventions and input audits; original cache arrays stay immutable."""
import numpy as np
import torch

from training.gaze.head_evidence import head_statistics, behavior_availability
from training.gaze.profile import GROUP_INDICES

FACE_INDICES = GROUP_INDICES[0] + GROUP_INDICES[3]


def drop_face_evidence(inputs):
    """Remove the whole face dependency, including head information duplicated in F."""
    result = {name: value.copy() for name, value in inputs.items()}
    result['x'][:, FACE_INDICES] = 0
    result['x'][:, [23+i for i in FACE_INDICES]] = 0
    result['quality'][[0,3]] = 0
    result['eligible'][[0,3]] = False
    return result


def _counts():
    return dict(windows=0, positive=0, negative=0, unknown=0,
                usable_positive=0, usable_negative=0, feature_ratio_sums=[0.]*23)


def audit_dataset(dataset, multiplier=1.):
    """Audit original inputs; sampling weights depend only on train labels/evidence."""
    if multiplier != 1 and any(record['split'] != 'train' for record in dataset.records):
        raise ValueError('Sampling weights may only be changed for train data')
    config = dataset.corpus.config
    weights = np.ones(len(dataset), np.float64)
    strata = {}; weighted = dropout_candidates = 0
    for index in range(len(dataset)):
        inputs, target, labels, *_ = dataset[index]
        view = dataset.metadata(index)['view']
        head = bool(head_statistics(inputs['x'],config['model']['head_policy'])['eligible'])
        available = bool(behavior_availability(inputs,config['model']['kind'],config['model']['head_policy'])[3])
        known = bool(labels[3]); positive = known and target[3] == 1
        pattern = ''.join(str(int(value)) for value in inputs['eligible'])
        key = f'{view}:H{int(head)}:F_O_HO_FO={pattern}'
        counts = strata.setdefault(key,_counts())
        counts['windows'] += 1
        counts['positive'] += int(positive)
        counts['negative'] += int(known and not positive)
        counts['unknown'] += int(not known)
        counts['usable_positive'] += int(positive and available)
        counts['usable_negative'] += int(known and not positive and available)
        counts['feature_ratio_sums'] = (np.asarray(counts['feature_ratio_sums'])+inputs['x'][:,23:].mean(0)).tolist()
        if view == 'body' and not head and positive and available:
            weights[index] = multiplier
            weighted += int(multiplier != 1)
        if _dropout_candidate(view, inputs): dropout_candidates += 1
    for counts in strata.values():
        counts['mean_effective_feature_valid_ratio'] = [v/counts['windows'] for v in counts.pop('feature_ratio_sums')]
    return dict(windows=len(dataset), weighted_windows=weighted,
                weight_sum=float(weights.sum()), dropout_candidates=dropout_candidates,
                feature_names=config['data']['feature_names'], strata=strata), weights


def _dropout_candidate(view, inputs):
    return (view == 'body' and bool(inputs['eligible'][1:3].any())
            and bool(inputs['x'][:,[23+i for i in FACE_INDICES]].any()))


class TrainingDataset:
    """Label-independent full-face dropout, using checkpointed CPU torch RNG."""
    def __init__(self, dataset, probability):
        if any(record['split'] != 'train' for record in dataset.records):
            raise ValueError('Face dropout is restricted to train data')
        if not 0 <= probability <= 1:
            raise ValueError('Invalid dropout probability')
        self.dataset, self.probability = dataset, probability
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(visited=0, candidates=0, applied=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs, *rest = self.dataset[index]
        self.stats['visited'] += 1
        if _dropout_candidate(self.dataset.metadata(index)['view'], inputs):
            self.stats['candidates'] += 1
            if self.probability > 0 and torch.rand(()).item() < self.probability:
                inputs = drop_face_evidence(inputs)
                self.stats['applied'] += 1
        return (inputs, *rest)
