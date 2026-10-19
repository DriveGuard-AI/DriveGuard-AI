"""Known-label multilabel metrics; no event-level or real-world claims."""
import torch
from training.common.config import TASKS


class ClassificationMetrics:
    def __init__(self, active_mask):
        self.active = active_mask.detach().cpu().bool()
        self.tp = torch.zeros(4, dtype=torch.int64)
        self.fp = torch.zeros(4, dtype=torch.int64)
        self.fn = torch.zeros(4, dtype=torch.int64)
        self.positive = torch.zeros(4, dtype=torch.int64)
        self.negative = torch.zeros(4, dtype=torch.int64)
        self.unknown = torch.zeros(4, dtype=torch.int64)

    def update_coverage(self, targets, masks):
        target = targets.detach().cpu().bool()
        labeled = masks.detach().cpu().bool()
        self.positive += (target & labeled).sum(0)
        self.negative += (~target & labeled).sum(0)
        self.unknown += (~labeled).sum(0)

    def update(self, logits, targets, masks):
        prediction = logits.detach().cpu() >= 0
        target = targets.detach().cpu().bool()
        labeled = masks.detach().cpu().bool()
        known = labeled & self.active
        self.tp += (prediction & target & known).sum(0)
        self.fp += (prediction & ~target & known).sum(0)
        self.fn += (~prediction & target & known).sum(0)
        self.update_coverage(targets, masks)

    def report(self):
        report, covered = {}, []
        for c, name in enumerate(TASKS):
            available = bool(self.active[c] and (self.positive[c] + self.negative[c]) > 0)
            precision = recall = f1 = None
            if available:
                tp, fp, fn = (int(value[c]) for value in (self.tp, self.fp, self.fn))
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else 0.0
                covered.append(name)
            report[name] = {'available': available, 'precision': precision, 'recall': recall, 'f1': f1,
                            'positive': int(self.positive[c]), 'negative': int(self.negative[c]),
                            'unknown': int(self.unknown[c])}
        macro = {key: sum(report[t][key] for t in covered) / len(covered) if covered else None
                 for key in ('precision', 'recall', 'f1')}
        return {'per_task': report, 'macro': macro, 'covered_tasks': covered, 'threshold': 0.5}
