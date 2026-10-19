"""Authored OpenLABEL fixtures exercising causal, conservative supervision."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def document(*actions, count=8, shifts=(0, 0, 0), timeline=None):
    return {'openlabel': {
        'metadata': {'schema_version': '1.0.0', 'name': 'authored-fixture'},
        'frame_intervals': timeline or [{'frame_start': 0, 'frame_end': count - 1}],
        'streams': {view + '_camera': {
            'uri': view + '.mp4', 'stream_properties': {
                'total_frames': count, 'sync': {'frame_shift': shift}}}
            for view, shift in zip(('face', 'body', 'hands'), shifts)},
        'actions': {str(i): {'type': kind, 'frame_intervals': [
            {'frame_start': start, 'frame_end': end}]}
            for i, (kind, start, end) in enumerate(actions)},
    }}


class LabelTests(unittest.TestCase):
    def api(self):
        self.assertTrue((ROOT / 'training/dmd/labels.py').is_file(),
                        'DMD label compiler is not present')
        from training.dmd.labels import camera_offsets, compile_labels
        return camera_offsets, compile_labels

    def compile(self, doc, times=None, *, view='face', subset='distraction', threshold=80):
        return self.api()[1](doc, view, [0, 40, 80, 120] if times is None else times,
                             closed_eye_min_ms=threshold, subset=subset)

    def test_camera_offsets_normalize_to_earliest_camera(self):
        offsets = self.api()[0]
        self.assertEqual(offsets(document(shifts=(0, 2, 4))),
                         {'face': 0, 'body': 2, 'hands': 4})
        self.assertEqual(offsets(document(shifts=(0, -3, 2))),
                         {'face': 3, 'body': 0, 'hands': 5})

    def test_shift_maps_source_index_to_inclusive_annotation_frame(self):
        doc = document(('driver_actions/texting_left', 2, 3), shifts=(0, 2, 0))
        result = self.compile(doc, view='body')
        self.assertEqual(result['annotation_indices'], [2, 3, 4, 5])
        self.assertEqual([row[1] for row in result['label_masks']], [1, 1, 0, 0])
        self.assertEqual([row[1] for row in result['targets']], [1, 1, 0, 0])

    def test_duplicate_action_types_union_and_multilabel_coexist(self):
        result = self.compile(document(('driver_actions/texting_right', 0, 1),
                                       ('driver_actions/texting_right', 1, 3),
                                       ('driver_actions/drinking', 1, 2)))
        self.assertEqual(result['targets'], [[0, 1, 0, 0], [0, 1, 1, 0],
                                            [0, 1, 1, 0], [0, 1, 0, 0]])
        self.assertEqual(result['label_masks'][0], [0, 1, 0, 0])

    def test_only_safe_drive_supplies_behavior_negatives(self):
        result = self.compile(document(('driver_actions/safe_drive', 0, 0),
                                       ('driver_actions/radio', 1, 1)))
        self.assertEqual(result['label_masks'], [[0, 1, 1, 0]] + [[0]*4]*3)
        self.assertEqual(result['targets'], [[0]*4]*4)

    def test_safe_conflicts_affect_only_conflicting_behavior(self):
        result = self.compile(document(('driver_actions/safe_drive', 0, 3),
                                       ('driver_actions/phonecall_left', 0, 0),
                                       ('driver_actions/drinking', 1, 1)))
        self.assertEqual(result['label_masks'][:2], [[0, 0, 1, 0], [0, 1, 0, 0]])

    def test_unknown_driver_axis_does_not_erase_gaze_evidence(self):
        for unknown in ('unclassified', 'future_type'):
            result = self.compile(document(('driver_actions/' + unknown, 0, 3),
                                           ('driver_actions/drinking', 0, 3),
                                           ('gaze_on_road/not_looking_road', 0, 3)))
            self.assertEqual(result['targets'], [[0, 0, 0, 1]]*4)
            self.assertEqual(result['label_masks'], [[0, 0, 0, 1]]*4)

    def test_gaze_conflicts_unknown_and_independence(self):
        result = self.compile(document(('gaze_on_road/looking_road', 0, 3),
                                       ('gaze_on_road/not_looking_road', 1, 1),
                                       ('gaze_on_road/undefined', 2, 2),
                                       ('driver_actions/drinking', 0, 3)))
        self.assertEqual([row[3] for row in result['label_masks']], [1, 0, 0, 1])
        self.assertEqual([row[2] for row in result['targets']], [1]*4)

    def test_subset_boundaries_and_gaze_zone_are_unknown(self):
        doc = document(('driver_actions/drinking', 0, 3),
                       ('gaze_on_road/not_looking_road', 0, 3),
                       ('gaze_zone/left_mirror', 0, 3),
                       ('yawning/Yawning without hand', 0, 3))
        self.assertEqual(self.compile(doc, subset='gaze')['label_masks'], [[0]*4]*4)
        self.assertEqual(self.compile(doc, subset='drowsiness')['label_masks'], [[1, 0, 0, 0]]*4)
        self.assertEqual(self.compile(doc)['label_masks'], [[0, 0, 1, 1]]*4)
        self.assertEqual(self.compile(document(('gaze_zone/front', 0, 3)))['label_masks'], [[0]*4]*4)

    def test_past_sampling_irregular_pts_and_stale_frames(self):
        result = self.compile(document(('driver_actions/drinking', 0, 7)),
                              [0, 39, 81, 240])
        self.assertEqual(result['slots_ms'], [0, 40, 80, 120, 160, 200, 240])
        self.assertEqual(result['source_indices'], [0, 1, 1, 2, 2, 2, 3])
        self.assertEqual([row[2] for row in result['label_masks']], [1, 1, 1, 1, 1, 0, 1])
        self.assertEqual(result['summary']['stale_slots'], 1)

    def test_exact_eighty_ms_source_age_is_valid(self):
        result = self.compile(document(('driver_actions/drinking', 0, 2)), [0, 121])
        self.assertEqual([row[2] for row in result['label_masks']], [1, 1, 1, 0])

    def test_decode_length_controls_slots_not_declared_count(self):
        doc = document(('driver_actions/drinking', 0, 7))
        result = self.compile(doc, [0])
        self.assertEqual(result['slots_ms'], [0])
        self.assertEqual(result['summary']['source_frame_count'], 1)
        self.assertEqual(result['summary']['declared_frame_count'], 8)
        self.assertTrue(result['summary']['frame_count_mismatch'])
        doc['openlabel']['streams']['face_camera']['stream_properties']['total_frames'] = 1
        self.assertEqual(len(self.compile(doc)['slots_ms']), 4)

    def test_fatigue_closed_threshold_is_causal(self):
        doc = document(('eyes_state/close', 0, 7))
        result = self.compile(doc, subset='drowsiness')
        self.assertEqual([row[0] for row in result['label_masks']], [0, 0, 1, 1])
        self.assertEqual([row[0] for row in result['targets']], [0, 0, 1, 1])
        self.assertEqual(self.compile(doc, [0, 40], subset='drowsiness')['label_masks'], [[0]*4]*2)

    def test_fatigue_held_frame_cannot_advance_closed_duration(self):
        result = self.compile(document(('eyes_state/close', 0, 7)),
                              [0, 70, 150], subset='drowsiness')
        self.assertEqual(result['slots_ms'], [0, 40, 80, 120])
        self.assertEqual(result['label_masks'], [[0]*4]*4)

    def test_fatigue_scans_unsampled_source_frames(self):
        doc = document(('eyes_state/close', 0, 0), ('eyes_state/undefined', 1, 1),
                       ('eyes_state/close', 2, 5))
        result = self.compile(doc, [0, 10, 20, 40, 80, 120], subset='drowsiness')
        self.assertEqual([row[0] for row in result['label_masks']], [0, 0, 0, 1])

    def test_eye_unknown_transition_conflict_reset_closed_run(self):
        for value in ('undefined', 'opening', 'closing', 'open', 'future_type'):
            doc = document(('eyes_state/close', 0, 7), ('eyes_state/' + value, 2, 2))
            result = self.compile(doc, [0, 40, 80, 120, 160, 200], subset='drowsiness')
            self.assertEqual([row[0] for row in result['label_masks']], [0, 0, 0, 0, 0, 1])

    def test_pts_gap_resets_closed_run(self):
        result = self.compile(document(('eyes_state/close', 0, 7)),
                              [0, 40, 160, 200, 240], subset='drowsiness')
        self.assertEqual([row[0] for row in result['label_masks']], [0, 0, 0, 0, 0, 0, 1])

    def test_root_timeline_gap_and_decode_beyond_timeline_are_unknown(self):
        doc = document(('eyes_state/close', 0, 1), ('eyes_state/close', 3, 5),
                       timeline=[{'frame_start': 0, 'frame_end': 1},
                                 {'frame_start': 3, 'frame_end': 5}])
        result = self.compile(doc, [0, 40, 80, 120, 160, 200, 240], subset='drowsiness')
        self.assertEqual([row[0] for row in result['label_masks']], [0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(result['summary']['outside_timeline_slots'], 2)

    def test_yawn_independent_evidence_and_open_proxy_negative(self):
        for kind in ('Yawning with hand', 'Yawning without hand'):
            doc = document(('eyes_state/undefined', 0, 0), ('yawning/' + kind, 0, 0),
                           ('eyes_state/open', 1, 3), ('yawning/unknown', 2, 2))
            result = self.compile(doc, subset='drowsiness')
            self.assertEqual([row[0] for row in result['targets']], [1, 0, 0, 0])
            self.assertEqual([row[0] for row in result['label_masks']], [1, 1, 0, 1])

    def test_unknown_yawn_type_invalidates_only_yawn_evidence(self):
        doc = document(('yawning/Yawning with hand', 0, 3),
                       ('yawning/unreviewed', 0, 3),
                       ('eyes_state/undefined', 0, 0), ('eyes_state/close', 1, 3))
        result = self.compile(doc, subset='drowsiness')
        self.assertEqual([row[0] for row in result['targets']], [0, 0, 0, 1])
        self.assertEqual([row[0] for row in result['label_masks']], [0, 0, 0, 1])

    def test_source_timestamps_must_fit_signed_64_bit_range(self):
        with self.assertRaises(ValueError):
            self.compile(document(), [0, 2**63])

    def test_output_capacity_rejects_without_truncation(self):
        with self.assertRaisesRegex(ValueError, 'capacity.*truncat'):
            self.compile(document(), [0, 4_000_000])

    def test_output_capacity_boundary(self):
        self.api()
        with patch('training.dmd.labels.MAX_OUTPUT_SLOTS', 3):
            result = self.compile(document(), [0, 40, 80])
            self.assertEqual(result['slots_ms'], [0, 40, 80])
            with self.assertRaisesRegex(ValueError, 'capacity'):
                self.compile(document(), [0, 40, 80, 120])

    def test_huge_int64_timestamp_rejected_before_source_processing(self):
        self.api()
        with patch('training.dmd.labels._active_axes',
                   side_effect=AssertionError('source processing began before capacity validation')):
            with self.assertRaisesRegex(ValueError, 'capacity'):
                self.compile(document(), [0, 2**63 - 1])

    def test_summary_counts_and_json_compatibility(self):
        result = self.compile(document(('driver_actions/safe_drive', 0, 0),
                                       ('driver_actions/drinking', 1, 2)))
        self.assertEqual(result['schema_version'], 'driveguard-behavior-v1')
        self.assertEqual(result['tasks'], ['fatigue', 'phone_use', 'drinking', 'distraction'])
        self.assertEqual(result['summary']['class_counts']['drinking'],
                         {'positive': 2, 'negative': 1, 'unknown': 1})
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_valid_per_frame_references(self):
        doc = document(('driver_actions/drinking', 0, 2), count=4)
        doc['openlabel']['frames'] = {str(i): {'actions': {'0': {}} if i <= 2 else {}}
                                    for i in range(4)}
        self.assertEqual(self.compile(doc)['targets'][2][2], 1)

    def test_per_frame_mismatches_rejected(self):
        for frames in ({}, {'0': {'actions': {'missing': {}}}},
                       {str(i): {'actions': {}} for i in range(4)},
                       {'01': {'actions': {}}}, {'4': {'actions': {}}}):
            doc = document(('driver_actions/drinking', 0, 2), count=4)
            doc['openlabel']['frames'] = frames
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                self.compile(doc)

    def test_bad_arguments_rejected(self):
        for times in ([], [1], [0, 0], [0, -1], [0, 1.5], [False], [0, float('inf')], (0, 40)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.compile(document(), times)
        for threshold in (None, 0, -1, 1.5, True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.compile(document(), threshold=threshold)
        for kwargs in ({'view': 'mosaic'}, {'subset': 'other'}, {'subset': None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.compile(document(), **kwargs)
        with self.assertRaises(TypeError):
            self.api()[1](document(), 'face', [0], subset='drowsiness')
        with self.assertRaises(TypeError):
            self.api()[1](document(), 'face', [0], closed_eye_min_ms=80)

    def test_bad_annotation_and_stream_declarations_rejected(self):
        base = document(('driver_actions/drinking', 0, 2))
        edits = [
            (('metadata', 'schema_version'), '0.9'), (('metadata', 'name'), ''),
            (('actions', '0', 'type'), 42), (('actions', '0', 'type'), ''),
            (('actions', '0', 'frame_intervals'), [{'frame_start': 0, 'frame_end': 8}]),
            (('actions', '0', 'frame_intervals'), [{'frame_start': 0, 'frame_end': True}]),
            (('frame_intervals',), []), (('streams', 'face_camera', 'stream_properties', 'sync', 'frame_shift'), 1),
            (('streams', 'body_camera', 'stream_properties', 'sync', 'frame_shift'), True),
            (('streams', 'hands_camera', 'stream_properties', 'total_frames'), 0),
            (('streams', 'hands_camera'), None), (('extra',), float('nan')),
        ]
        for path, value in edits:
            doc = copy.deepcopy(base)
            target = doc['openlabel']
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                self.compile(doc)
        for view in ('face', 'body', 'hands'):
            doc = copy.deepcopy(base)
            del doc['openlabel']['streams'][view + '_camera']
            with self.assertRaises(ValueError):
                self.compile(doc)


if __name__ == '__main__':
    unittest.main()
