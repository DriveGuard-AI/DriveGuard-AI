from copy import deepcopy
import unittest

import numpy as np

from tests.perception_helpers import packet, missing_face
from training.common.config import load_config, canonical_hash
from inference.perception.profile import load_profile, validate_profile
from inference.perception.sequence import FeatureExtractor


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile()

    def feature(self, result, name):
        i = self.profile['feature_names'].index(name)
        return float(result['values'][i]), int(result['masks'][i])

    def test_profile_and_training_configuration_agree(self):
        config = load_config('configs/training/temporal_interaction.json')
        self.assertEqual(config['data']['feature_names'], self.profile['feature_names'])
        self.assertEqual(config['data']['feature_schema_version'], self.profile['feature_schema_version'])
        self.assertEqual(config['model']['input_channels'], 38)
        self.assertEqual(len(self.profile['feature_definitions']), 19)
        self.assertEqual(load_config()['model']['input_channels'], 24)
        for field, bad in [('max_age_ms', -1), ('sample_rate_hz', True),
                           ('ear_closed_threshold', float('nan')), ('blink_max_ms', 1)]:
            value = deepcopy(self.profile)
            value[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_profile(value)
        value = deepcopy(self.profile)
        value['typo'] = 1
        with self.assertRaises(ValueError):
            validate_profile(value)

    def test_warmup_perclos_and_completed_blinks(self):
        extractor = FeatureExtractor(self.profile)
        for i in range(100):
            out = extractor.step(packet(i, closed=i in (1, 2)))
            if i < 99:
                self.assertEqual(self.feature(out, 'perclos_4s')[1], 0)
                self.assertEqual(self.feature(out, 'blink_rate_per_min')[1], 0)
        self.assertEqual(out['values'].dtype, np.float32)
        self.assertEqual(out['masks'].dtype, np.uint8)
        self.assertEqual(out['values'].shape, (19,))
        self.assertAlmostEqual(self.feature(out, 'perclos_4s')[0], 0.02)
        self.assertEqual(self.feature(out, 'blink_rate_per_min'), (15, 1))
        self.assertEqual(self.feature(out, 'head_yaw_deg'), (0, 0))

    def test_unknown_initial_closure_and_missing_reset(self):
        extractor = FeatureExtractor(self.profile)
        self.assertEqual(self.feature(extractor.step(packet(0, True)), 'closed_duration_s'), (0, 0))
        extractor.step(packet(1))
        self.assertEqual(self.feature(extractor.step(packet(2, True)), 'closed_duration_s'), (0, 1))
        out = extractor.step(packet(3, True))
        self.assertAlmostEqual(self.feature(out, 'closed_duration_s')[0], 0.04)
        extractor.step(missing_face(packet(4)))
        self.assertEqual(self.feature(extractor.step(packet(5, True)), 'closed_duration_s'), (0, 0))

    def test_perclos_coverage_boundary_and_blink_missing(self):
        for missing_count, expected_mask in [(20, 1), (21, 0)]:
            extractor = FeatureExtractor(self.profile)
            for i in range(100):
                value = missing_face(packet(i)) if i < missing_count else packet(i)
                out = extractor.step(value)
            self.assertEqual(self.feature(out, 'perclos_4s')[1], expected_mask)
            self.assertEqual(self.feature(out, 'blink_rate_per_min')[1], 0)

    def test_near_duration_tracks_and_missing_are_not_continuous(self):
        extractor = FeatureExtractor(self.profile)
        value = packet(0)
        value['objects']['items'] = []
        extractor.step(value)  # A known far slot establishes the onset boundary.
        self.assertEqual(self.feature(extractor.step(packet(1)), 'phone_near_duration_s'), (0, 1))
        self.assertAlmostEqual(self.feature(extractor.step(packet(2)), 'phone_near_duration_s')[0], .04)
        value = packet(3)
        value['objects']['items'][0]['track_id'] = 'phone-2'
        self.assertEqual(self.feature(extractor.step(value), 'phone_near_duration_s'), (0, 1))
        value = packet(4)
        value['hands']['items'] = []
        self.assertEqual(self.feature(extractor.step(value), 'phone_near_duration_s'), (0, 0))
        self.assertEqual(self.feature(extractor.step(packet(5)), 'phone_near_duration_s'), (0, 0))

    def test_untracked_near_is_not_duration_evidence(self):
        extractor = FeatureExtractor(self.profile)
        for i in range(5):
            value = packet(i)
            value['objects']['items'][0]['track_id'] = None
            out = extractor.step(value)
        self.assertEqual(self.feature(out, 'phone_hand_distance'), (0, 1))
        self.assertEqual(self.feature(out, 'phone_near_duration_s'), (0, 0))

    def test_invalid_packets_cannot_mutate_history(self):
        extractor = FeatureExtractor(self.profile)
        extractor.step(packet(0))
        invalid = []
        for field, value in [('recording_id', 'other'), ('view', 'face'), ('width', 1200),
                             ('driver_roi', [0, 0, 700, 500]), ('timestamp_ms', 80), ('frame_id', '0')]:
            p = packet(1)
            p[field] = value
            invalid.append(p)
        p = packet(1)
        p['face']['model_id'] = 'changed'
        invalid.append(p)
        p = packet(1)
        p['objects']['items'][0]['score'] = float('nan')
        invalid.append(p)
        for p in invalid:
            with self.assertRaises(ValueError):
                extractor.step(p)
        actual = extractor.step(packet(1, True))
        fresh = FeatureExtractor(self.profile)
        fresh.step(packet(0))
        expected = fresh.step(packet(1, True))
        np.testing.assert_array_equal(actual['values'], expected['values'])
        np.testing.assert_array_equal(actual['masks'], expected['masks'])

    def test_profile_snapshot_and_new_instance_reset(self):
        original = canonical_hash(self.profile)
        extractor = FeatureExtractor(self.profile)
        self.profile['ear_closed_threshold'] = 0.9
        extractor.step(packet(0))
        self.assertEqual(self.feature(extractor.step(packet(1)), 'closed_duration_s'), (0, 1))
        self.assertEqual(canonical_hash(extractor.profile), original)
        other = FeatureExtractor(load_profile())
        out = other.step(packet(10, True))
        self.assertEqual(self.feature(out, 'closed_duration_s'), (0, 0))

    def test_blink_requires_initial_open_evidence_inside_window(self):
        extractor = FeatureExtractor(self.profile)
        for i in range(101):
            out = extractor.step(packet(i, closed=i in (1, 2)))
        self.assertEqual(self.feature(out, 'blink_rate_per_min'), (0, 1))

    def test_extreme_valid_geometry_does_not_emit_infinity(self):
        extractor = FeatureExtractor(self.profile)
        value = packet(0)
        value['face']['landmarks'][362] = [0, 180]
        value['face']['landmarks'][263] = [1e-40, 180]
        with np.errstate(over='ignore'):
            out = extractor.step(value)
        self.assertTrue(np.isfinite(out['values']).all())
        self.assertEqual(self.feature(out, 'ear_left'), (0, 0))


if __name__ == '__main__':
    unittest.main()
