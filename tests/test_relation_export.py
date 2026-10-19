import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from inference.perception.export import export_recording
from inference.perception.profile import load_profile
from training.relation.profile import default_group_profile
from training.common.config import canonical_hash
from tests.perception_helpers import packet, missing_face

ROOT = Path(__file__).resolve().parents[1]


class RelationExportTests(unittest.TestCase):
    def test_opt_in_export_records_real_ages_and_unknown_labels(self):
        self.assertIn('group_profile', inspect.signature(export_recording).parameters, 'v2 export not implemented')
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as root:
            def stream():
                for i in range(105):
                    p = packet(i)
                    if i:
                        p['objects']['timestamp_ms'] -= 40
                    if i == 104:
                        p = missing_face(p)
                    yield p
            target = Path(root)/'export'
            report = export_recording(stream(),target,load_profile(),group_profile=default_group_profile())
            ages = np.load(target/'expert_age_ms.npy')
            self.assertEqual(ages.shape,(105,3))
            self.assertEqual(ages.dtype,np.float32)
            np.testing.assert_array_equal(ages[104],[40,81,0])
            self.assertEqual(report['schema_version'],'driveguard-feature-record-v2')
            self.assertEqual(report['group_profile_sha256'],canonical_hash(default_group_profile()))
            self.assertIn('expert_age_ms',report['array_sha256'])
            self.assertFalse(np.load(target/'label_masks.npy').any())

    def test_future_observation_never_publishes_success(self):
        self.assertIn('group_profile',inspect.signature(export_recording).parameters)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as root:
            p = packet(0)
            p['objects']['timestamp_ms'] = 1
            target = Path(root)/'bad'
            with self.assertRaises(ValueError):
                export_recording(iter([p]),target,load_profile(),group_profile=default_group_profile())
            self.assertFalse((target/'record.json').exists())


if __name__ == '__main__':
    unittest.main()
