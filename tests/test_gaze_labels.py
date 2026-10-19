import unittest
from tests.test_dmd_labels import document


class GazeLabelTests(unittest.TestCase):
    def test_nine_classes_not_valid_conflicts_and_subset_masks(self):
        from training.dmd.gaze_labels import compile_gaze_labels
        from training.gaze.profile import GAZE_CLASSES
        doc = document(*[(f'gaze_zone/{name}', i, i) for i,name in enumerate(GAZE_CLASSES)],
                       ('gaze_zone/not_valid',9,9), ('gaze_zone/front',10,10),
                       ('gaze_zone/right',10,10), count=12)
        result = compile_gaze_labels(doc,'face',list(range(0,480,40)),closed_eye_min_ms=1000,subset='gaze')
        self.assertEqual(result['gaze_targets'][:9],list(range(9)))
        self.assertEqual(result['gaze_label_masks'], [1]*9+[0]*3)
        self.assertFalse(any(any(row) for row in result['label_masks']))
        self.assertEqual(result['gaze_targets'][9:],[0]*3)

    def test_gaze_is_face_only_and_stale_is_unknown(self):
        from training.dmd.gaze_labels import compile_gaze_labels
        doc = document(('gaze_zone/front',0,10), count=11)
        for view, subset in (('body','gaze'),('face','distraction')):
            result = compile_gaze_labels(doc,view,[0,40,80],closed_eye_min_ms=1000,subset=subset)
            self.assertFalse(any(result['gaze_label_masks']))
        result = compile_gaze_labels(doc,'face',[0,200],closed_eye_min_ms=1000,subset='gaze')
        self.assertEqual(result['gaze_label_masks'],[1,1,1,0,0,1])
