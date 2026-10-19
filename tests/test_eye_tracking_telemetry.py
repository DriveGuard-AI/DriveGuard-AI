"""Passive gate diagnostics must retain the bounded locator's public behavior."""
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_eye_tracking as fixtures
from training.visual.eye_tracking import POLICY


class EyeTrackingTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EyeFlowTests()
        self.fixture.setUp()

    def trace(self, locator, reason, status):
        self.assertTrue(hasattr(locator, 'last_trace'), 'passive tracking trace missing')
        trace = locator.last_trace
        self.assertEqual(trace['diagnostic_version'], 'eye-tracking-trace-v1')
        self.assertEqual(trace['tracking_reason'], reason)
        self.assertEqual(trace['tracking_status'], status)
        json.dumps(trace, allow_nan=False)
        return trace

    def test_policy_and_detected_return_are_unchanged(self):
        self.assertEqual(hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest(),
                         '566a6d429f6be9ee14b62d5f4fcea5862deeca79d34e52b3436778b1c0e3776e')
        locator = self.fixture.locator()
        self.assertEqual(locator.step(self.fixture.blank, 0), dict(
            roi=[44., 44., 76., 76.], width=16., reason='accepted',
            detector_reason='ok', origin='detected', track_age_ms=0))
        self.trace(locator, 'disabled', 'disabled')

    def test_real_translation_records_measured_metrics(self):
        locator, initial = self.fixture.seeded()
        seeded = self.trace(locator, 'detected', 'detected')
        self.assertGreaterEqual(seeded['seed_points'], 6)
        result = locator.step(self.fixture.translated(3, 2), 40)
        trace = self.trace(locator, 'tracked', 'tracked')
        self.assertEqual(set(result), {'roi', 'width', 'reason', 'detector_reason', 'origin', 'track_age_ms'})
        np.testing.assert_allclose(result['roi'], np.array(initial['roi']) + [3, 2, 3, 2], atol=.15)
        self.assertEqual(trace['seed_age_ms'], 40)
        self.assertEqual(trace['source_gap_ms'], 40)
        self.assertGreaterEqual(trace['translation_points'], 6)
        self.assertAlmostEqual(trace['step_shift_px'], np.sqrt(13), delta=.15)
        # Rejected candidate errors can exceed the gate even on accepted tracks.
        self.assertLessEqual(trace['forward_backward_accepted_error_max_px'], 1.)
        self.assertGreaterEqual(trace['forward_backward_error_max_px'],
                                trace['forward_backward_accepted_error_max_px'])
        self.assertEqual(seeded['tracking_reason'], 'detected', 'each step replaces the trace')

    def test_texture_failure_then_absent_seed(self):
        locator = self.fixture.locator(tracking=True)
        self.assertEqual(locator.step(self.fixture.blank, 0)['origin'], 'detected')
        trace = self.trace(locator, 'seed_insufficient_texture', 'detected')
        self.assertEqual(trace['seed_points'], 0)
        self.fixture.adapter.missing()
        locator.step(self.fixture.blank, 40)
        self.assertIsNone(self.trace(locator, 'no_seed', 'unavailable')['seed_age_ms'])

    def test_expiration_then_absent_seed_not_repeated_expiration(self):
        locator, _ = self.fixture.seeded()
        for time in (40, 80, 120, 160, 200, 240, 250):
            self.assertEqual(locator.step(self.fixture.translated(time / 80), time)['origin'], 'tracked')
        self.assertEqual(locator.step(self.fixture.translated(4), 251)['origin'], 'missing')
        self.assertEqual(self.trace(locator, 'age_expired', 'stopped')['seed_age_ms'], 251)
        locator.step(self.fixture.translated(4), 280)
        self.assertIsNone(self.trace(locator, 'no_seed', 'unavailable')['seed_age_ms'])

    def test_source_gap_is_first_stop_even_with_unsafe_detector(self):
        locator, _ = self.fixture.seeded()
        self.fixture.adapter.missing('multiple_faces')
        locator.step(self.fixture.translated(2), 81)
        self.assertEqual(self.trace(locator, 'source_gap', 'stopped')['seed_age_ms'], 81)
        self.fixture.adapter.missing()
        locator.step(self.fixture.translated(3), 121)
        self.trace(locator, 'no_seed', 'unavailable')

    def test_unsafe_detector_stops_before_age_gate(self):
        locator, _ = self.fixture.seeded()
        for time in (80, 160, 240):
            locator.step(self.fixture.texture, time)
        self.fixture.adapter.missing('multiple_faces')
        locator.step(self.fixture.texture, 251)
        self.trace(locator, 'detector_disallows_fallback', 'stopped')

    def test_new_detection_after_source_gap_still_seeds(self):
        locator, _ = self.fixture.seeded()
        self.fixture.adapter.observation = fixtures.face()
        self.fixture.adapter.reason = 'ok'
        self.assertEqual(locator.step(self.fixture.texture, 100)['origin'], 'detected')
        self.assertEqual(self.trace(locator, 'detected', 'detected')['seed_age_ms'], 0)

    def test_flow_failures_name_the_first_executed_gate(self):
        for reason in ('forward_flow_failed', 'forward_points_insufficient',
                       'backward_flow_failed', 'forward_backward_inconsistent',
                       'translation_inconsistent'):
            with self.subTest(reason=reason):
                self.fixture.adapter.observation = fixtures.face()
                locator, _ = self.fixture.seeded()
                original = self.fixture.cv2.calcOpticalFlowPyrLK
                calls = 0
                def flow(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    points, status, error = original(*args, **kwargs)
                    if reason == 'forward_flow_failed' and calls == 1:
                        return None, None, None
                    if reason == 'forward_points_insufficient' and calls == 1:
                        status[:] = 0
                    if reason == 'backward_flow_failed' and calls == 2:
                        return None, None, None
                    if reason == 'forward_backward_inconsistent' and calls == 2:
                        points += 3
                    if reason == 'translation_inconsistent':
                        # Supply consistent inverse motion with nonrigid forward shifts.
                        if calls == 1:
                            points = args[2].copy()
                            points[:, 0, 0] += np.arange(len(points)) % 3 * 5 - 5
                            status[:] = 1
                            error[:] = 0
                        else:
                            points = locator._points.copy()
                            status[:] = 1
                    return points, status, error
                with patch.object(self.fixture.cv2, 'calcOpticalFlowPyrLK', side_effect=flow):
                    self.assertEqual(locator.step(self.fixture.translated(2), 40)['origin'], 'missing')
                self.trace(locator, reason, 'stopped')
                locator.step(self.fixture.translated(3), 80)
                self.trace(locator, 'no_seed', 'unavailable')

    def test_displacement_and_containment_stops(self):
        locator, _ = self.fixture.seeded()
        locator.step(self.fixture.translated(12), 40)
        self.assertGreater(self.trace(locator, 'step_shift_exceeded', 'stopped')['step_shift_px'], 8)
        self.fixture.adapter.observation = fixtures.face()
        locator, _ = self.fixture.seeded()
        for time, shift in ((40, 7), (80, 14), (120, 21)):
            locator.step(self.fixture.translated(shift), time)
        self.assertGreater(self.trace(locator, 'total_shift_exceeded', 'stopped')['total_shift_px'], 16)
        self.fixture.adapter.observation = fixtures.face()
        locator = self.fixture.locator(tracking=True, head=(43, 10, 150, 110))
        locator.step(self.fixture.texture, 0)
        self.fixture.adapter.missing()
        locator.step(self.fixture.translated(-3), 40)
        self.trace(locator, 'outside_head', 'stopped')


if __name__ == '__main__':
    unittest.main()
