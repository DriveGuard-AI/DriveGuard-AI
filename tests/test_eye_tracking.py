"""Causal native-eye localization using real translated pixels."""
from copy import deepcopy
import importlib.util
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np


def face():
    points = np.tile([80., 60.], (478, 1))
    points[33], points[133] = [52., 60.], [68., 60.]
    points[362], points[263] = [88., 60.], [100., 60.]
    return dict(status='ok', driver_confirmed=True, landmarks=points.tolist())


MISSING = dict(status='missing', driver_confirmed=False, landmarks=None)


class Adapter:
    def __init__(self):
        self.observation = face()
        self.reason = 'ok'
        self.calls = []

    def predict(self, bgr, timestamp_ms):
        self.calls.append(timestamp_ms)
        self.last_face_diagnostics = {'reason': self.reason,'detected_faces':0 if self.reason=='no_face_detected' else 1}
        return deepcopy(self.observation)

    def missing(self, reason='no_face_detected'):
        self.observation = MISSING
        self.reason = reason


class EyeLocatorTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.eye_tracking'),
                             'bounded eye locator missing')
        from training.visual.eye_tracking import EyeLocator
        return EyeLocator

    def setUp(self):
        self.adapter = Adapter()
        self.blank = np.zeros((120, 160, 3), np.uint8)

    def locator(self, tracking=False, head=(10, 10, 150, 110)):
        return self.api()(self.adapter, head, (160, 120), tracking=tracking)

    def test_confirmed_detection_and_independent_detector_reason(self):
        locator = self.locator()
        self.adapter.reason = 'pose_matrix_invalid'
        result = locator.step(self.blank, 0)
        self.assertEqual(result, dict(roi=[44., 44., 76., 76.], width=16.,
            reason='accepted', detector_reason='pose_matrix_invalid', origin='detected', track_age_ms=0))
        self.adapter.missing()
        result = locator.step(self.blank, 40)
        self.assertIsNone(result['roi'])
        self.assertEqual(result['reason'], 'unconfirmed_or_missing_face')
        self.assertEqual(result['origin'], 'missing')
        self.assertEqual(self.adapter.calls, [0, 40])

    def test_missing_diagnostic_reason_has_explicit_fallback(self):
        self.adapter.reason = None
        result = self.locator().step(self.blank, 0)
        self.assertEqual(result['detector_reason'], 'unclassified_face_rejection')

    def test_all_face_and_eye_geometry_gates_are_preserved(self):
        for kind in ('unconfirmed', 'invalid', 'outside', 'tiny', 'padded'):
            self.adapter.observation = face()
            if kind == 'unconfirmed': self.adapter.observation['driver_confirmed'] = False
            elif kind == 'invalid': self.adapter.observation['landmarks'][0][0] = float('nan')
            elif kind == 'outside': self.adapter.observation['landmarks'][0][0] = 151
            elif kind == 'tiny': self.adapter.observation['landmarks'] = np.tile([60.,60.], (478,1)).tolist()
            else:
                self.adapter.observation['landmarks'][33] = [12.,60.]
                self.adapter.observation['landmarks'][133] = [28.,60.]
            with self.subTest(kind=kind):
                self.assertIsNone(self.locator().step(self.blank, 0)['roi'])

    def test_validates_frame_timestamp_and_constructor(self):
        for frame, timestamp in ((self.blank.astype(float),0), (self.blank[:,:,0],0),
                                 (self.blank[:100],0), (self.blank,-1), (self.blank,1.2),
                                 (self.blank,True), (self.blank,2**63)):
            with self.subTest(timestamp=timestamp, shape=frame.shape):
                with self.assertRaises(ValueError): self.locator().step(frame,timestamp)
        locator = self.locator()
        locator.step(self.blank,40)
        for timestamp in (40,39):
            with self.assertRaises(ValueError): locator.step(self.blank,timestamp)
        with self.assertRaises(ValueError): self.locator(head=(0,0,161,120))

    def test_detect_only_imports_neither_torch_nor_cv2(self):
        self.api()
        code = """import sys
import numpy as np
from training.visual.eye_tracking import EyeLocator
class Adapter:
    def predict(self, image, timestamp):
        return {'status': 'missing'}
EyeLocator(Adapter(), [0,0,20,20], [20,20]).step(np.zeros((20,20,3),np.uint8),0)
assert 'torch' not in sys.modules
assert 'cv2' not in sys.modules
"""
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'optical flow requires OpenCV')
class EyeFlowTests(EyeLocatorTests):
    def setUp(self):
        super().setUp()
        import cv2
        self.cv2 = cv2
        rng = np.random.default_rng(42)
        gray = rng.integers(0,256,(120,160), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray,(3,3),.6)
        self.texture = np.repeat(gray[:,:,None],3,axis=2)

    def translated(self, dx, dy=0):
        return self.cv2.warpAffine(self.texture,np.float32([[1,0,dx],[0,1,dy]]),(160,120))

    def seeded(self):
        locator = self.locator(tracking=True)
        initial = locator.step(self.texture,0)
        self.adapter.missing()
        return locator, initial

    def test_real_translation_uses_current_image_and_preserves_square_width(self):
        locator, initial = self.seeded()
        result = locator.step(self.translated(3,2),40)
        self.assertEqual(result['origin'],'tracked')
        np.testing.assert_allclose(result['roi'],np.array(initial['roi'])+[3,2,3,2],atol=.15)
        self.assertEqual(result['width'],initial['width'])
        self.assertEqual(result['track_age_ms'],40)
        self.assertEqual(result['detector_reason'],'no_face_detected')
        self.assertEqual(result['reason'],'accepted')
        self.assertEqual(self.adapter.calls,[0,40])

    def test_real_adapter_empty_face_status_can_recover(self):
        from inference.experts.mediapipe_adapter import face_observation
        from inference.experts.face_diagnostics import inspect_face
        locator,initial=self.seeded()
        observation=face_observation([],160,120,[10,10,150,110],True,40,'fixture',matrices=[])
        detail=inspect_face([],[],observation,160,120,[10,10,150,110],True)
        self.assertEqual(observation['status'],'ok')
        self.assertEqual(detail['reason'],'no_face_detected')
        def actual_empty(*args):
            self.adapter.last_face_diagnostics=detail
            return observation
        with patch.object(self.adapter,'predict',side_effect=actual_empty):
            result=locator.step(self.translated(3,2),40)
        self.assertEqual(result['origin'],'tracked')
        np.testing.assert_allclose(result['roi'],np.array(initial['roi'])+[3,2,3,2],atol=.15)

    def test_age_is_since_detection_and_does_not_renew(self):
        locator, _ = self.seeded()
        for timestamp in (40,80,120,160,200,240,250):
            result = locator.step(self.translated(timestamp/80),timestamp)
            self.assertEqual(result['origin'],'tracked')
            self.assertEqual(result['track_age_ms'],timestamp)
        self.assertEqual(locator.step(self.translated(4),251)['origin'],'missing')
        self.assertEqual(locator.step(self.translated(4),280)['origin'],'missing')

    def test_new_detection_reinitializes_age_and_location(self):
        locator,_ = self.seeded()
        locator.step(self.translated(2),40)
        self.adapter.observation = face()
        self.adapter.reason = 'ok'
        detected = locator.step(self.texture,80)
        self.assertEqual(detected['origin'],'detected')
        self.adapter.missing()
        tracked = locator.step(self.translated(2),120)
        self.assertEqual(tracked['track_age_ms'],40)
        np.testing.assert_allclose(tracked['roi'],np.array(detected['roi'])+[2,0,2,0],atol=.15)

    def test_source_gap_limit_resets_and_boundary_is_inclusive(self):
        locator,_ = self.seeded()
        self.assertEqual(locator.step(self.translated(2),80)['origin'],'tracked')
        self.assertEqual(locator.step(self.translated(3),161)['origin'],'missing')
        self.assertEqual(locator.step(self.translated(4),201)['origin'],'missing')

    def test_non_detection_rejections_reset_tracking(self):
        for reason in ('multiple_faces', 'invalid_landmarks', 'face_outside_head',
                       'driver_unconfirmed', None):
            with self.subTest(reason=reason):
                self.adapter.observation = face()
                locator,_ = self.seeded()
                self.adapter.missing(reason)
                self.assertEqual(locator.step(self.translated(2),40)['origin'],'missing')
                self.adapter.missing()
                self.assertEqual(locator.step(self.translated(3),80)['origin'],'missing')

    def test_blank_previous_or_current_texture_cannot_be_reused(self):
        locator = self.locator(tracking=True)
        locator.step(self.blank,0)
        self.adapter.missing()
        self.assertEqual(locator.step(self.blank,40)['origin'],'missing')
        self.adapter.observation = face()
        locator,_ = self.seeded()
        self.assertEqual(locator.step(self.blank,40)['origin'],'missing')

    def test_shift_and_head_containment_are_bounded(self):
        locator,_ = self.seeded()
        self.assertEqual(locator.step(self.translated(12),40)['origin'],'missing')
        self.adapter.observation = face()
        locator = self.locator(tracking=True,head=(43,10,150,110))
        locator.step(self.texture,0)
        self.adapter.missing()
        self.assertEqual(locator.step(self.translated(-3),40)['origin'],'missing')

    def test_invalid_input_clears_prior_tracking(self):
        locator,_ = self.seeded()
        with self.assertRaises(ValueError): locator.step(self.texture,0)
        self.assertEqual(locator.step(self.translated(2),40)['origin'],'missing')

    def test_unconfirmed_face_cannot_track_with_inconsistent_diagnostics(self):
        locator,_ = self.seeded()
        self.adapter.observation = face()
        self.adapter.observation['driver_confirmed'] = False
        self.assertEqual(locator.step(self.translated(2),40)['origin'],'missing')

    def test_forward_backward_disagreement_rejects_flow(self):
        locator,_ = self.seeded()
        original = self.cv2.calcOpticalFlowPyrLK
        calls = 0
        def inconsistent(*args,**kwargs):
            nonlocal calls
            points,status,error = original(*args,**kwargs)
            calls += 1
            if calls == 2: points += 3
            return points,status,error
        with patch.object(self.cv2,'calcOpticalFlowPyrLK',side_effect=inconsistent):
            self.assertEqual(locator.step(self.translated(2),40)['origin'],'missing')
        self.assertEqual(calls,2)

    def test_cumulative_translation_is_bounded_since_detection(self):
        locator,_ = self.seeded()
        self.assertEqual(locator.step(self.translated(7),40)['origin'],'tracked')
        self.assertEqual(locator.step(self.translated(14),80)['origin'],'tracked')
        self.assertEqual(locator.step(self.translated(21),120)['origin'],'missing')


if __name__ == '__main__': unittest.main()
