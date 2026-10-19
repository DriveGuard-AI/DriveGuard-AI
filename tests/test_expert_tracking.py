import unittest

from inference.perception.tracking import ObjectTracker


def item(x=0, category='phone'):
    return dict(category=category, box=[x, 0, x+10, 10], score=0.9, track_id=None)


class TrackingTests(unittest.TestCase):
    def test_unique_match_stays_stable_and_input_not_mutated(self):
        tracker = ObjectTracker()
        source = [item()]
        first = tracker.update(source, 0)
        second = tracker.update([item(1)], 40)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])
        self.assertIsNone(source[0]['track_id'])

    def test_missing_expiry_and_class_change_break_identity(self):
        for event in ('empty', 'expired', 'class'):
            tracker = ObjectTracker()
            first = tracker.update([item()], 0)[0]['track_id']
            if event == 'empty':
                tracker.update([], 40)
            after = tracker.update([item(category='cup' if event == 'class' else 'phone')],
                                   81 if event == 'expired' else 80)
            self.assertNotEqual(first, after[0]['track_id'])

    def test_ambiguous_matching_allocates_new_unique_ids(self):
        tracker = ObjectTracker()
        previous = {x['track_id'] for x in tracker.update([item(), item(2)], 0)}
        current = tracker.update([item(1), item(3)], 40)
        self.assertEqual(len({x['track_id'] for x in current}), 2)
        self.assertFalse(previous & {x['track_id'] for x in current})

    def test_bad_time_rejected_without_state_change(self):
        tracker = ObjectTracker()
        first = tracker.update([item()], 40)
        for time in (40, 39, True, -1):
            with self.assertRaises(ValueError):
                tracker.update([item(1)], time)
        self.assertEqual(first[0]['track_id'], tracker.update([item(1)], 80)[0]['track_id'])
