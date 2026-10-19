import inspect
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.perception_helpers import packet


def gaze_packet(index=0):
    result = packet(index)
    points = result['face']['landmarks']
    points.extend([[320., 180.]] * 5 + [[470., 180.]] * 5)
    result['face']['head_pose'] = dict(yaw_deg=0., pitch_deg=0., roll_deg=0.)
    return result


class GazePerceptionTests(unittest.TestCase):
    def test_runtime_rejects_non_object_json(self):
        from inference.perception.runtime import load_runtime_config
        for value in (None, [], 1, 'runtime'):
            with self.subTest(value=value), patch('inference.perception.runtime.read_json', return_value=value):
                with self.assertRaisesRegex(ValueError, 'object'):
                    load_runtime_config('unused.json')

    def test_pose_known_axes_and_invalid_matrix(self):
        from inference.experts.gaze_geometry import head_pose
        for axis, field, sign in ((0, 'pitch_deg', 1), (1, 'yaw_deg', -1), (2, 'roll_deg', -1)):
            angle = math.radians(20)
            c, s = math.cos(angle), math.sin(angle)
            rotation = (np.array([[1,0,0],[0,c,-s],[0,s,c]]),
                        np.array([[c,0,s],[0,1,0],[-s,0,c]]),
                        np.array([[c,-s,0],[s,c,0],[0,0,1]]))[axis]
            transform = np.eye(4)
            transform[:3,:3] = rotation
            self.assertAlmostEqual(head_pose(transform)[field], 20 * sign, places=5)
        for matrix in (np.zeros((4,4)), np.full((4,4), np.nan), np.diag([-1,1,1,1]), np.eye(3)):
            self.assertIsNone(head_pose(matrix))

    def test_iris_local_coordinates_rotation_scale_and_missing(self):
        from inference.experts.gaze_geometry import iris_features
        points = gaze_packet()['face']['landmarks']
        result = iris_features(points)
        self.assertEqual(result, dict(iris_left_x=.5, iris_left_y=0., iris_right_x=.5, iris_right_y=0.))
        rotation = np.array([[0.,-1.],[1.,0.]])
        transformed = (np.array(points) @ rotation.T * 3 + 1000).tolist()
        self.assertEqual(iris_features(transformed), result)
        self.assertTrue(all(v is None for v in iris_features(points[:468]).values()))
        closed = gaze_packet()['face']['landmarks']
        for i in (385,387,373,380):
            closed[i][1] = 180.
        result = iris_features(closed)
        self.assertIsNone(result['iris_left_x'])
        self.assertEqual(result['iris_right_x'], .5)

    def test_pose_uses_same_face_index_as_roi_selection(self):
        from inference.experts.mediapipe_adapter import face_observation
        self.assertIn('matrices', inspect.signature(face_observation).parameters)
        points = gaze_packet()['face']['landmarks']
        def normalized(offset):
            return [SimpleNamespace(x=(x+offset)/1000, y=y/500) for x,y in points]
        invalid = np.zeros((4,4))
        result = face_observation([normalized(400), normalized(0)],1000,500,[0,0,800,500],True,0,'test',
                                  matrices=[invalid, np.eye(4)])
        self.assertTrue(result['driver_confirmed'])
        self.assertEqual(result['head_pose'], dict(yaw_deg=0., pitch_deg=0., roll_deg=0.))

    def test_new_profile_exports_23_columns_without_changing_old(self):
        from inference.perception.gaze_profile import gaze_profile
        from inference.perception.export import export_recording
        from inference.perception.profile import load_profile
        from training.gaze.profile import default_group_profile
        self.assertEqual(len(load_profile()['feature_names']),19)
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            output = Path(directory)/'features'
            report = export_recording((gaze_packet(i) for i in range(100)), output, gaze_profile(),
                                      group_profile=default_group_profile())
            self.assertEqual(report['schema_version'],'driveguard-feature-record-v3')
            values, masks = np.load(output/'values.npy'), np.load(output/'masks.npy')
            self.assertEqual(values.shape,(100,23))
            self.assertTrue(masks[:,19:].all())
            np.testing.assert_equal(values[:,19], .5)

    def test_runtime_requests_matrix_and_exports_new_features(self):
        from tests.test_expert_runtime_flow import runtime_fixture
        from inference.perception.video import run_video
        from inference.perception.gaze_profile import gaze_profile
        from training.gaze.profile import default_group_profile
        with tempfile.TemporaryDirectory(dir='runs') as directory, runtime_fixture(directory,gaze=True) as (source,config,closed):
            output=Path(directory)/'features'
            report=run_video(source,output,config,feature_profile=gaze_profile(),group_profile=default_group_profile())
            masks=np.load(output/'masks.npy')
            self.assertTrue(masks[:,5:8].all())
            self.assertTrue(masks[:,19:].all())
            self.assertEqual(report['schema_version'],'driveguard-feature-record-v3')
            self.assertEqual(closed.count('task'),2)

    def test_gaze_export_requires_matching_group_profile(self):
        from inference.perception.export import export_recording
        from inference.perception.gaze_profile import gaze_profile
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            with self.assertRaisesRegex(ValueError,'group'):
                export_recording((gaze_packet(i) for i in range(100)),Path(directory)/'features',gaze_profile())


if __name__ == '__main__':
    unittest.main()
