import unittest
import numpy as np
from inference.live.models import LiveStream


class Models:
    threshold=-.7
    def encode_rgb(self,pixels):return np.zeros(512,np.float32)
    def action_scores(self,head,eye):return (1. if eye is not None else -1e6, -1.)
    def window(self,values,masks,ages,rgb,rgb_mask,view,fatigue):
        return dict(fatigue=fatigue,history=len(values),rgb_valid=bool(rgb_mask.all()))


def samples(start=0,end=100,fresh=True,eye=True):
    for slot in range(start,end):
        yield dict(slot=slot,source_time_ms=slot*40,values=[0.]*23,masks=[1]*23,ages=[0.]*3,
            fresh=fresh,rgb_pixels=np.zeros((3,224,224),np.uint8) if slot%25 in (12,24) and fresh else None,
            head_sample=slot%25 in (2,5,8,12,15,18,21,24),
            head_pixels=np.zeros((3,224,224),np.uint8) if fresh else None,
            eye_pixels=np.zeros((3,224,224),np.uint8) if eye else None,reset=False)


class StreamTests(unittest.TestCase):
    def test_warmup_then_event_and_full_history(self):
        stream=LiveStream(Models(),'body')
        self.assertIsNone(stream.consume(list(samples(end=99))))
        result=stream.consume(list(samples(99,100)))
        self.assertTrue(result['fatigue']['available'])
        self.assertEqual(result['fatigue']['score'],1.)
        self.assertEqual(result['history'],100)

    def test_missing_eye_resets_only_eye_not_yawn_availability(self):
        stream=LiveStream(Models(),'body')
        result=stream.consume(list(samples(eye=False)))
        self.assertTrue(result['fatigue']['available'])
        self.assertEqual(result['fatigue']['score'],-1.)

    def test_missing_source_disables_whole_fatigue_window(self):
        stream=LiveStream(Models(),'body')
        result=stream.consume(list(samples(fresh=False)))
        self.assertFalse(result['fatigue']['available'])
        self.assertFalse(result['rgb_valid'])

    def test_streams_do_not_share_history(self):
        first=LiveStream(Models(),'body');first.consume(list(samples()))
        second=LiveStream(Models(),'body')
        self.assertIsNone(second.consume(list(samples(end=50))))
        with self.assertRaises(ValueError):second.consume(list(samples(0,1)))


if __name__=='__main__':unittest.main()
