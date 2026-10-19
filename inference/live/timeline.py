"""Causal webcam resampling, matching the offline 25 Hz grid and RGB offsets."""

HEAD_OFFSETS = (2, 5, 8, 12, 15, 18, 21, 24)


class CausalClock:
    def __init__(self):
        self.last = None
        self.payload = None
        self.slot = 0

    def validate(self, timestamp_ms):
        if (type(timestamp_ms) is not int or timestamp_ms < 0 or
                (self.last is None and timestamp_ms != 0) or
                (self.last is not None and not 0 < timestamp_ms-self.last <= 10000)):
            raise ValueError('Capture timestamps must start at zero, strictly increase, and have gaps <=10s')

    def push(self, timestamp_ms, payload):
        self.validate(timestamp_ms)
        result = []
        # Retain the previous source until the next capture proves the interval.
        # A new frame must never fill slots preceding its capture timestamp.
        while self.last is not None and self.slot*40 < timestamp_ms:
            result.append(dict(slot=self.slot, source_time_ms=self.last,
                fresh=self.slot*40-self.last <= 80, payload=self.payload,
                rgb_sample=self.slot % 25 in (12,24),
                head_sample=self.slot % 25 in HEAD_OFFSETS))
            self.slot += 1
        self.last, self.payload = timestamp_ms, payload
        return result
