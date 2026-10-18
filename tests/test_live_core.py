import copy
import unittest

from inference.live.routing import compose_tasks
from inference.live.timeline import CausalClock


def base(scores=(1., 2., 3., 4.)):
    return dict(behavior_logits=list(scores), task_available=[True]*4,
                gaze_logits=list(range(9)), gaze_available=True)


class LiveRoutingTests(unittest.TestCase):
    def test_only_body_specialists_replace_base(self):
        original = base(); saved = copy.deepcopy(original)
        output = compose_tasks('body', original, base((5,6,7,8)),
                               dict(score=-.2, available=True), -.7,
                               dict(score=2., available=True))
        self.assertEqual(original, saved)
        self.assertEqual(output['tasks']['fatigue']['source'], 'yawn_eye_dense_track')
        self.assertTrue(output['tasks']['distraction']['detected'])
        self.assertEqual(output['tasks']['phone_use']['logit'], 2.)
        self.assertEqual(output['gaze']['validation_status'], 'unvalidated_body_view')

    def test_face_ignores_both_specialists(self):
        output = compose_tasks('face', base(), base(), None, -.7, None)
        self.assertEqual([v['source'] for v in output['tasks'].values()], ['coverage25']*4)
        self.assertEqual(output['gaze']['class_name'], 'steering_wheel')

    def test_missing_eye_event_does_not_become_safe(self):
        fallback=base();fallback['task_available'][3]=False
        output=compose_tasks('body',base(),fallback,None,-.7,None)
        self.assertIsNone(output['tasks']['fatigue']['detected'])
        self.assertIsNone(output['tasks']['fatigue']['score'])
        self.assertIsNone(output['tasks']['distraction']['detected'])
        self.assertNotIn('risk_level', output)

    def test_missing_rgb_uses_gap80_threshold_zero(self):
        output=compose_tasks('body',base(),base((0,0,0,-.1)),None,-.7,None)
        self.assertEqual(output['tasks']['distraction']['threshold_logit'],0.)
        self.assertFalse(output['tasks']['distraction']['detected'])

    def test_invalid_available_logits_rejected(self):
        with self.assertRaises(ValueError):
            compose_tasks('body',base(),base(),dict(score=float('nan'),available=True),0.,None)


class LiveClockTests(unittest.TestCase):
    def test_no_future_frame_and_exact_offline_offsets(self):
        clock=CausalClock(); samples=[]
        for time in range(0,4001,40):
            samples.extend(clock.push(time, time))
        self.assertEqual(len(samples),100)
        self.assertTrue(all(s['source_time_ms']<=s['slot']*40 for s in samples))
        self.assertEqual([s['slot'] for s in samples if s['rgb_sample']],
                         [12,24,37,49,62,74,87,99])
        self.assertEqual(sum(s['head_sample'] for s in samples),32)
        self.assertEqual(samples[-1]['slot'],99)

    def test_gap_is_not_fresh_evidence(self):
        clock=CausalClock();clock.push(0,'first')
        samples=clock.push(400,'next')
        self.assertEqual([s['fresh'] for s in samples],[True]*3+[False]*7)
        self.assertTrue(all(s['payload']=='first' for s in samples))

    def test_rejected_timestamps_do_not_advance(self):
        clock=CausalClock();clock.push(0,'first')
        for time in (0,-1,True,10001):
            with self.assertRaises(ValueError):clock.push(time,'bad')
        self.assertEqual(len(clock.push(40,'next')),1)
        self.assertEqual(CausalClock().push(0,'new'),[])

    def test_first_capture_must_be_zero(self):
        with self.assertRaises(ValueError):CausalClock().push(123,'bad')


if __name__=='__main__':unittest.main()
