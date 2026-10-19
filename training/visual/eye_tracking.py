"""Causal, short-lived translation tracking of strictly confirmed native eyes.

Flow supplies candidate eye pixels only. It never establishes driver identity,
eye visibility, or an open/closed-eye label. Detection always takes precedence.
"""
import math

import numpy as np


POLICY = dict(
    version='confirmed-eye-translation-lk-v1',
    fallback_detector_reason='no_face_detected',
    max_age_ms=250, max_source_gap_ms=80,
    min_points=6, min_consistent_fraction=.6,
    max_corners=80, feature_quality=.01, feature_min_distance=3.,
    feature_block_size=3, neighborhood_side_per_eye_side=1.5,
    lk_window_size=21, lk_max_level=2, lk_max_iterations=30,
    lk_epsilon=.01, lk_min_eigenvalue=.0001,
    max_forward_backward_error_px=1., max_lk_error=20.,
    max_translation_residual_px=1.5,
    max_step_shift_px=20., max_step_shift_per_eye_span=.5,
    max_total_shift_px=40., max_total_shift_per_eye_span=1.,
    motion='median_translation_only_preserve_detected_square_and_span',
    containment='all_tracked_eye_pixels_inside_head_and_image_no_clipping',
    reseed='confirmed_detection_only',
    age='since_last_confirmed_detection_never_renewed_by_flow',
    label_access=False,
)


class EyeLocator:
    def __init__(self, adapter, head_roi, image_size, tracking=False):
        if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
                or any(type(v) is not int or v < 1 for v in image_size)):
            raise ValueError('Image size must be positive integer width,height')
        if (not isinstance(head_roi, (list, tuple)) or len(head_roi) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in head_roi)
                or not 0 <= head_roi[0] < head_roi[2] <= image_size[0]
                or not 0 <= head_roi[1] < head_roi[3] <= image_size[1]):
            raise ValueError('Head ROI must be inside the original image')
        if type(tracking) is not bool:
            raise ValueError('Tracking must be an explicit boolean')
        self.adapter = adapter
        self.head_roi = tuple(head_roi)
        self.image_size = tuple(image_size)
        self.tracking = tracking
        self._time = None
        self._cv2 = None
        self.last_trace = {}
        self._reset()

    def _reset(self):
        self._gray = self._points = self._roi = self._detected_roi = None
        self._detected_time = None
        self._width = 0.

    @staticmethod
    def _result(roi, width, reason, detector_reason, origin='missing', age=0):
        return dict(roi=roi, width=float(width), reason=reason,
                    detector_reason=detector_reason, origin=origin, track_age_ms=int(age))

    def _inside(self, points):
        x0, y0, x1, y1 = self.head_roi
        return ((points[:, 0] >= x0) & (points[:, 0] <= x1)
                & (points[:, 1] >= y0) & (points[:, 1] <= y1))

    def _seed(self, gray, roi, width, timestamp_ms):
        self._reset()
        center = (np.asarray(roi[:2]) + roi[2:]) / 2
        half = (roi[2] - roi[0]) * POLICY['neighborhood_side_per_eye_side'] / 2
        x0, y0, x1, y1 = self.head_roi
        # Seed only pixels inside the eye neighborhood and the confirmed head.
        left, top = math.ceil(max(x0, center[0]-half)), math.ceil(max(y0, center[1]-half))
        right, bottom = math.floor(min(x1, center[0]+half)), math.floor(min(y1, center[1]+half))
        mask = np.zeros(gray.shape, np.uint8)
        mask[top:bottom, left:right] = 255
        points = self._cv2.goodFeaturesToTrack(gray,
            maxCorners=POLICY['max_corners'], qualityLevel=POLICY['feature_quality'],
            minDistance=POLICY['feature_min_distance'], mask=mask,
            blockSize=POLICY['feature_block_size'])
        self.last_trace['seed_points'] = 0 if points is None else len(points)
        if points is None or len(points) < POLICY['min_points']:
            self.last_trace['tracking_reason'] = 'seed_insufficient_texture'
            return
        self._gray, self._points = gray, points
        self._roi, self._detected_roi = np.array(roi), np.array(roi)
        self._width, self._detected_time = width, timestamp_ms

    def _flow(self, gray):
        cv2 = self._cv2
        options = dict(winSize=(POLICY['lk_window_size'],)*2,
            maxLevel=POLICY['lk_max_level'],
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      POLICY['lk_max_iterations'], POLICY['lk_epsilon']),
            minEigThreshold=POLICY['lk_min_eigenvalue'])
        forward, status, error = cv2.calcOpticalFlowPyrLK(
            self._gray, gray, self._points, None, **options)
        if forward is None or status is None or error is None:
            self.last_trace['tracking_reason'] = 'forward_flow_failed'
            return None
        finite = np.isfinite(forward).all(axis=(1, 2))
        candidates = (finite & (status.ravel() == 1) & np.isfinite(error.ravel())
                      & (error.ravel() <= POLICY['max_lk_error']))
        self.last_trace['forward_points'] = int(candidates.sum())
        measured_error = error.ravel()[np.isfinite(error.ravel()) & (status.ravel() == 1)]
        self.last_trace['lk_error_max'] = float(measured_error.max()) if measured_error.size else None
        if candidates.sum() < POLICY['min_points']:
            self.last_trace['tracking_reason'] = 'forward_points_insufficient'
            return None
        prior = self._points[candidates]
        current = forward[candidates]
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._gray, current, None, **options)
        if backward is None or reverse_status is None:
            self.last_trace['tracking_reason'] = 'backward_flow_failed'
            return None
        a, b = prior[:, 0], current[:, 0]
        fb_error = np.linalg.norm(backward[:, 0]-a, axis=1)
        consistent = ((reverse_status.ravel() == 1) & np.isfinite(backward).all(axis=(1, 2))
            & (fb_error <= POLICY['max_forward_backward_error_px'])
            & self._inside(b))
        minimum = max(POLICY['min_points'], math.ceil(len(self._points)*POLICY['min_consistent_fraction']))
        measured_fb = fb_error[np.isfinite(fb_error) & (reverse_status.ravel() == 1)]
        self.last_trace.update(
            consistent_points=int(consistent.sum()), minimum_consistent_points=minimum,
            inside_head_points=int(self._inside(b).sum()),
            forward_backward_accepted_error_max_px=float(fb_error[consistent].max()) if consistent.any() else None,
            forward_backward_error_max_px=float(measured_fb.max()) if measured_fb.size else None)
        if consistent.sum() < minimum:
            self.last_trace['tracking_reason'] = 'forward_backward_inconsistent'
            return None
        shifts = b-a
        shift = np.median(shifts[consistent], axis=0)
        residual = np.linalg.norm(shifts-shift, axis=1)
        self.last_trace['translation_residual_max_px'] = float(residual[consistent].max())
        consistent &= residual <= POLICY['max_translation_residual_px']
        self.last_trace['translation_points'] = int(consistent.sum())
        if consistent.sum() < minimum:
            self.last_trace['tracking_reason'] = 'translation_inconsistent'
            return None
        shift = np.median(shifts[consistent], axis=0)
        self.last_trace.update(shift_dx_px=float(shift[0]), shift_dy_px=float(shift[1]),
                               step_shift_px=float(np.linalg.norm(shift)))
        if np.linalg.norm(shift) > min(POLICY['max_step_shift_px'], self._width*POLICY['max_step_shift_per_eye_span']):
            self.last_trace['tracking_reason'] = 'step_shift_exceeded'
            return None
        roi = self._roi + np.tile(shift, 2)
        if not self._inside(roi.reshape(2, 2)).all():
            self.last_trace['tracking_reason'] = 'outside_head'
            return None
        self.last_trace['total_shift_px'] = float(np.linalg.norm(roi[:2]-self._detected_roi[:2]))
        if np.linalg.norm(roi[:2]-self._detected_roi[:2]) > min(
                POLICY['max_total_shift_px'], self._width*POLICY['max_total_shift_per_eye_span']):
            self.last_trace['tracking_reason'] = 'total_shift_exceeded'
            return None
        self._gray, self._points, self._roi = gray, current[consistent], roi
        return roi.tolist()

    def step(self, bgr, timestamp_ms):
        """Consume one source image, exposing passive diagnostics in ``last_trace``.

        Each call replaces the trace; it is separate from the unchanged result
        and tracking policy. Metrics appear only for evaluated gates. A stopped
        trace records the first terminating gate; later seedless frames report
        unavailable/no_seed. Seed age is measured before any terminating reset.
        """
        self.last_trace = dict(diagnostic_version='eye-tracking-trace-v1',
            tracking_status='unavailable', tracking_reason='invalid_input',
            seed_age_ms=None, source_gap_ms=None)
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8
                or bgr.shape != (self.image_size[1], self.image_size[0], 3)
                or type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 2**63-1
                or (self._time is not None and timestamp_ms <= self._time)):
            self._reset()
            raise ValueError('Expected uint8 BGR source image and strictly increasing integer PTS')
        gap = None if self._time is None else timestamp_ms-self._time
        had_seed = self._detected_time is not None
        self.last_trace.update(
            seed_age_ms=timestamp_ms-self._detected_time if had_seed else None,
            source_gap_ms=gap, input_points=0 if self._points is None else len(self._points))
        self._time = timestamp_ms
        source_gap = gap is not None and gap > POLICY['max_source_gap_ms']
        if gap is not None and gap > POLICY['max_source_gap_ms']:
            self._reset()
        try:
            observation = self.adapter.predict(bgr, timestamp_ms)
        except BaseException:
            self.last_trace.update(tracking_status='stopped' if had_seed else 'unavailable',
                                   tracking_reason='adapter_error')
            self._reset()
            raise
        detail = getattr(self.adapter, 'last_face_diagnostics', None)
        detector_reason = detail.get('reason') if isinstance(detail, dict) else None
        if not isinstance(detector_reason, str) or not detector_reason:
            detector_reason = 'unclassified_face_rejection'
        # Lazy import also lets eye_frames import this module for policy identity.
        from training.visual.eye_frames import eye_geometry
        roi, width, reason = eye_geometry(observation, self.head_roi, self.image_size)
        if not self.tracking:
            self.last_trace.update(tracking_status='disabled', tracking_reason='disabled')
            return self._result(roi, width, reason, detector_reason,
                                'detected' if roi is not None else 'missing')
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        gray = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2GRAY)
        if roi is not None:
            self.last_trace.update(tracking_status='detected', tracking_reason='detected', seed_age_ms=0)
            self._seed(gray, roi, width, timestamp_ms)
            return self._result(roi, width, reason, detector_reason, 'detected')
        allows_fallback = (detector_reason == POLICY['fallback_detector_reason']
                and reason == 'unconfirmed_or_missing_face'
                and observation.get('status') in ('ok','missing')
                and observation.get('driver_confirmed') is False
                and observation.get('landmarks') is None
                and detail.get('detected_faces') == 0)
        self.last_trace['tracking_status'] = 'stopped' if had_seed else 'unavailable'
        if source_gap and had_seed:
            self.last_trace['tracking_reason'] = 'source_gap'
        elif not allows_fallback:
            self.last_trace['tracking_reason'] = 'detector_disallows_fallback'
        elif self._detected_time is None:
            self.last_trace['tracking_reason'] = 'no_seed'
        elif timestamp_ms-self._detected_time > POLICY['max_age_ms']:
            self.last_trace['tracking_reason'] = 'age_expired'
        else:
            roi = self._flow(gray)
            if roi is not None:
                self.last_trace.update(tracking_status='tracked', tracking_reason='tracked')
                return self._result(roi, self._width, 'accepted', detector_reason, 'tracked',
                                    timestamp_ms-self._detected_time)
        self._reset()
        return self._result(None, 0., reason, detector_reason)
