import base64
import io
import unittest
from copy import deepcopy
from types import SimpleNamespace
from PIL import Image
from inference.live.perception_worker import validate_camera,decode_jpeg,CameraPerception


class CameraInputTests(unittest.TestCase):
    def test_camera_identity_explicit(self):
        options=dict(view='body',image_size=[640,480],driver_roi=[0,0,1,1],
                     driver_region_confirmed_exclusive=False)
        result=validate_camera(options)
        self.assertEqual(result['driver_roi'],[0,0,640,480])
        self.assertFalse(result['driver_region_confirmed_exclusive'])
        for update in ({'view':'hands'},{'image_size':[999999,480]},
                       {'driver_roi':[-1,0,1,1]},{'driver_region_confirmed_exclusive':1}):
            with self.assertRaises(ValueError):validate_camera(dict(options,**update))

    def test_jpeg_shape_and_format_enforced(self):
        image=Image.new('RGB',(64,48));stream=io.BytesIO();image.save(stream,format='JPEG')
        encoded=base64.b64encode(stream.getvalue()).decode()
        self.assertEqual(decode_jpeg(encoded,[64,48]).shape,(48,64,3))
        with self.assertRaises(ValueError):decode_jpeg(encoded,[48,64])
        stream=io.BytesIO();image.save(stream,format='PNG')
        with self.assertRaises(ValueError):decode_jpeg(base64.b64encode(stream.getvalue()).decode(),[64,48])

    def test_real_feature_extraction_is_causal_and_gap_masks_are_retained(self):
        from tests.test_gaze_perception import gaze_packet
        from inference.live.timeline import CausalClock
        from inference.perception.sequence import FeatureExtractor
        from inference.perception.gaze_profile import gaze_profile
        from inference.perception.tracking import ObjectTracker
        worker=CameraPerception.__new__(CameraPerception)
        worker.camera=dict(view='body',image_size=[1000,500],driver_roi=[0,0,800,500],driver_region_confirmed_exclusive=True)
        worker.clock=CausalClock();worker.extractor=FeatureExtractor(gaze_profile());worker.tracker=ObjectTracker()
        worker.head_roi=[40,0,520,325];worker.gap_pending=False;worker.previous_capture=None
        def observation(name,time):
            value=deepcopy(gaze_packet()[name]);value['timestamp_ms']=time;return value
        worker.detector=SimpleNamespace(predict=lambda image,time:observation('objects',time))
        worker.landmarks=SimpleNamespace(predict=lambda image,time:(observation('face',time),observation('hands',time)))
        worker.locator=SimpleNamespace(step=lambda image,time:dict(roi=[300,160,340,200],origin='detected',reason='accepted'))
        stream=io.BytesIO();Image.new('RGB',(1000,500)).save(stream,format='JPEG')
        jpeg=base64.b64encode(stream.getvalue()).decode()
        self.assertEqual(worker.frame(0,jpeg)['samples'],[])
        samples=worker.frame(400,jpeg)['samples']
        self.assertEqual(len(samples),10)
        self.assertTrue(all(s['source_time_ms']==0 for s in samples))
        self.assertTrue(all(not any(s['masks']) for s in samples[3:]))
        self.assertIn('eye_pixels',samples[2])
        self.assertNotIn('head_pixels',samples[5])
        # New source gap survives until the first subsequent head sample.
        samples=worker.frame(560,jpeg)['samples']
        self.assertTrue(next(s for s in samples if s['head_sample'])['reset'])


if __name__=='__main__':unittest.main()
