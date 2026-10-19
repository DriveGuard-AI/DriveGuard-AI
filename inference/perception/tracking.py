"""Conservative one-step IoU identity, never equate a class with a track."""
from copy import deepcopy
import math


def _iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


class ObjectTracker:
    def __init__(self):
        self._previous = []
        self._time = None
        self._counter = 0

    def update(self, items, timestamp_ms):
        if type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 2**63-1 or (
                self._time is not None and timestamp_ms <= self._time):
            raise ValueError('Tracker timestamps must be strictly increasing nonnegative milliseconds')
        current = deepcopy(items)
        for item in current:
            box = item['box']
            if item['category'] not in ('phone', 'bottle', 'cup') or len(box) != 4 or not all(
                    type(x) in (int, float) and math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Invalid tracking candidate')
        previous = self._previous if self._time is not None and timestamp_ms-self._time <= 80 else []
        matches = [[j for j, old in enumerate(previous) if old['category'] == item['category']
                    and _iou(old['box'], item['box']) >= 0.3] for item in current]
        counter = self._counter
        for item, candidates in zip(current, matches):
            if len(candidates) == 1 and sum(candidates[0] in others for others in matches) == 1:
                item['track_id'] = previous[candidates[0]]['track_id']
            else:
                counter += 1
                item['track_id'] = f'object-{counter}'
        self._previous, self._time, self._counter = deepcopy(current), timestamp_ms, counter
        return current
