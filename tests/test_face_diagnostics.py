from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from inference.experts.mediapipe_adapter import face_observation, MediaPipeAdapter
from tests.test_expert_mediapipe import points
from tests.test_expert_runtime_flow import runtime_fixture
from tests.test_gaze_perception import gaze_packet


class FaceDiagnosticTests(unittest.TestCase):
    def inspect(self, faces, roi=None, exclusive=True, matrices=None):
        self.assertIsNotNone(importlib.util.find_spec('inference.experts.face_diagnostics'), 'face diagnostics missing')
        from inference.experts.face_diagnostics import inspect_face
        roi = roi or [0, 0, 800, 500]
        observation = face_observation(faces, 1000, 500, roi, exclusive, 0, 'test', matrices=matrices)
        before = deepcopy(observation)
        diagnostic = inspect_face(faces, matrices, observation, 1000, 500, roi, exclusive)
        self.assertEqual(before, observation)
        return diagnostic

    def test_reason_hierarchy_matches_original_selection(self):
        cases = [([], None, True, [], 'no_face_detected'),
                 ([points(478, x=.9)], None, True, [np.eye(4)], 'outside_roi'),
                 ([points(478), points(478)], None, True, [np.eye(4)]*2, 'ambiguous_faces'),
                 ([points(478)], None, False, [np.eye(4)], 'driver_unconfirmed'),
                 ([points(478)], None, True, None, 'pose_matrices_missing'),
                 ([points(478)], None, True, [], 'pose_matrix_count_mismatch'),
                 ([points(478)], None, True, [np.zeros((4,4))], 'pose_matrix_invalid'),
                 ([points(478)], None, True, [np.eye(4)], 'ok')]
        for faces, roi, exclusive, matrices, reason in cases:
            with self.subTest(reason=reason):
                result = self.inspect(faces, roi, exclusive, matrices)
                self.assertEqual(result['reason'], reason)
                self.assertEqual(result['head_usable'], reason == 'ok')

    def test_any_invalid_candidate_rejects_all_and_details_survive(self):
        for bad, detail in ((None, 'malformed'), (points(5), 'topology'), (points(478, x=1.01), 'out_of_bounds'),
                            (points(478, x=float('nan')), 'nonfinite')):
            with self.subTest(detail=detail):
                result = self.inspect([points(478), bad], matrices=[np.eye(4)]*2)
                self.assertEqual(result['reason'], 'invalid_landmarks')
                self.assertEqual(result['candidates'][1]['invalid_reason'], detail)
                self.assertFalse(result['head_usable'])

    def test_roi_boundary_pose_index_and_iris_are_independent(self):
        p = gaze_packet()['face']['landmarks']
        good = [SimpleNamespace(x=x/1000, y=y/500) for x,y in p]
        result = self.inspect([points(478, x=.9), good], matrices=[np.zeros((4,4)),np.eye(4)])
        self.assertEqual(result['selected_index'], 1)
        self.assertEqual(result['reason'], 'ok')
        self.assertTrue(result['iris_left_valid'])
        self.assertTrue(result['iris_right_valid'])
        result = self.inspect([points(478)], matrices=[np.eye(4)])
        self.assertEqual(result['reason'], 'ok')
        self.assertFalse(result['iris_left_valid'])
        self.assertEqual(result['iris_reason'], 'geometry_rejected')

    def test_adapter_opt_in_is_observational_only(self):
        import inspect
        self.assertIn('diagnostics', inspect.signature(MediaPipeAdapter).parameters)
        with tempfile.TemporaryDirectory(dir='runs') as directory, runtime_fixture(directory,gaze=True) as (_, config, _):
            options = dict(driver_roi=config['driver_roi'],driver_region_confirmed_exclusive=True,gaze_features=True)
            a=MediaPipeAdapter(config['face_asset'],config['hands_asset'],**options)
            b=MediaPipeAdapter(config['face_asset'],config['hands_asset'],**options,diagnostics=True)
            try:
                frame=np.broadcast_to(np.array([1,2,3],np.uint8),(500,1000,3))
                self.assertEqual(a.predict(frame,0),b.predict(frame,0))
                self.assertEqual(a.face_model_id,b.face_model_id)
                self.assertIsNone(a.last_face_diagnostics)
                self.assertEqual(b.last_face_diagnostics['reason'],'ok')
            finally:
                a.close(); b.close()


if __name__ == '__main__': unittest.main()
