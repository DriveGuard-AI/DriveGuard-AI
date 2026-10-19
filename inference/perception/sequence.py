"""Causal, bounded-history feature extraction for one uniform recording stream."""
from collections import deque
from copy import deepcopy
import math

import numpy as np

from inference.perception.geometry import spatial_features
from inference.perception.profile import validate_profile


class _Duration:
    def __init__(self):
        self.previous = None
        self.start = None
        self.identity = None

    def update(self, active, time, identity='eye'):
        if active is None or (active and identity is None):
            self.previous, self.start, self.identity = None, None, None
            return None
        if not active:
            self.previous, self.start, self.identity = False, None, identity
            return 0.0
        if self.previous is False or (self.previous is True and identity != self.identity):
            self.start = time
        self.previous, self.identity = True, identity
        return (time - self.start) / 1000 if self.start is not None else None


class FeatureExtractor:
    """Each instance belongs to one recording/view/profile; no cross-stream reset magic."""
    def __init__(self, profile):
        self._profile = deepcopy(validate_profile(profile))
        self._identity = self._last_time = self._last_frame = None
        self._eyes = deque(maxlen=100)
        self._blinks = deque()
        self._eye_duration = _Duration()
        self._near = {'phone': _Duration(), 'drink': _Duration()}

    @property
    def profile(self):
        return deepcopy(self._profile)

    def step(self, packet):
        # All malformed input checks happen before changing temporal state.
        spatial = spatial_features(packet, self._profile)
        identity = (packet['recording_id'], packet['view'], packet['width'], packet['height'],
                    tuple(packet['driver_roi']), *(packet[key]['model_id'] for key in ('face', 'objects', 'hands')))
        time = packet['timestamp_ms']
        if self._identity is not None and identity != self._identity:
            raise ValueError('Recording/view/image/ROI/expert version changed; use a new extractor')
        if self._last_time is not None and (time != self._last_time + 40 or packet['frame_id'] == self._last_frame):
            raise ValueError('Expected a new frame ID and the next 40 ms slot; encode missing slots explicitly')
        values = dict(spatial['values'])
        # Advance a small bounded state snapshot; a failed numeric conversion must
        # not consume a slot or leave duration/history partly advanced.
        eyes, blinks = deque(self._eyes, maxlen=100), deque(self._blinks)
        eye_duration, near = deepcopy(self._eye_duration), deepcopy(self._near)
        eye = spatial['eye_closed']
        previous, start = eye_duration.previous, eye_duration.start
        if eye is False and previous is True and start is not None:
            duration = time - start
            if self._profile['blink_min_ms'] <= duration <= self._profile['blink_max_ms']:
                blinks.append((start - 40, time))  # Include the confirming open slot.
        values['closed_duration_s'] = eye_duration.update(eye, time)
        eyes.append(eye)
        window_start = time - 99 * 40
        while blinks and blinks[0][0] < window_start:
            blinks.popleft()
        known = [x for x in eyes if x is not None]
        full = len(eyes) == 100
        values['perclos_4s'] = (sum(known) / len(known) if full and
                                len(known) >= 100 * self._profile['perclos_min_valid_ratio'] else None)
        values['blink_rate_per_min'] = len(blinks) * 15.0 if full and len(known) == 100 else None
        for category in ('phone', 'drink'):
            values[f'{category}_near_duration_s'] = near[category].update(
                spatial[f'{category}_near'], time, spatial[f'{category}_track'])
        names = self._profile['feature_names']
        if any(v is not None and (not math.isfinite(v) or abs(v) > np.finfo(np.float32).max)
               for v in values.values()):
            raise ValueError('Derived feature is not representable as finite float32')
        output = np.array([values[n] if values[n] is not None else 0.0 for n in names], dtype=np.float32)
        masks = np.array([values[n] is not None for n in names], dtype=np.uint8)
        self._eyes, self._blinks = eyes, blinks
        self._eye_duration, self._near = eye_duration, near
        self._identity, self._last_time, self._last_frame = identity, time, packet['frame_id']
        return {'values': output, 'masks': masks}
