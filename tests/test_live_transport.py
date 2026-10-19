import base64
import io
import unittest
from concurrent.futures import ThreadPoolExecutor
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
        # The detector now runs on a one-slot background pool. An instant reply
        # completes before the next frame, so this test still observes the same
        # per-frame evidence the serial implementation produced.
        worker.detector_pool=ThreadPoolExecutor(max_workers=1)
        worker.detector_pending=None;worker.detector_last=None
        worker.detector_submitted_at=0.0;worker.detector_deadline=30.0
        self.addCleanup(worker.detector_pool.shutdown)
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


class AsyncDetectorTests(unittest.TestCase):
    def make_worker(self,detector,updates=None):
        worker=CameraPerception.__new__(CameraPerception)
        worker.detector=detector
        worker.tracker=SimpleNamespace(update=lambda items,time:updates.append(time) or [])
        worker.detector_pool=ThreadPoolExecutor(max_workers=1)
        worker.detector_pending=None;worker.detector_last=None
        worker.detector_submitted_at=0.0;worker.detector_deadline=30.0
        self.addCleanup(worker.detector_pool.shutdown)
        return worker

    @staticmethod
    def reply(time):
        return dict(status='ok',timestamp_ms=time,model_id='m',items=[])

    def test_reuses_last_reply_while_a_request_is_in_flight(self):
        import threading
        started=threading.Event();gate=threading.Event()
        def predict(image,time):
            if time:
                started.set();gate.wait(5)
            return self.reply(time)
        worker=self.make_worker(SimpleNamespace(predict=predict),[])
        self.assertEqual(worker.objects_for(None,0)['timestamp_ms'],0)
        self.assertEqual(worker.objects_for(None,40)['timestamp_ms'],0)
        self.assertTrue(started.wait(5))
        # A slow detector must not stall the frame: the previous reply is
        # reused and nothing is queued behind the in-flight request.
        self.assertEqual(worker.objects_for(None,80)['timestamp_ms'],0)
        self.assertFalse(worker.detector_pending.done())
        gate.set()
        self.assertEqual(worker.detector_pending.result()['timestamp_ms'],40)
        # Once it lands, the next frame sees that capture time.
        self.assertEqual(worker.objects_for(None,120)['timestamp_ms'],40)

    def test_tracks_advance_once_per_detection(self):
        import threading
        started=threading.Event();gate=threading.Event()
        def predict(image,time):
            if time:
                started.set();gate.wait(5)
            return self.reply(time)
        updates=[]
        worker=self.make_worker(SimpleNamespace(predict=predict),updates)
        self.assertEqual(worker.objects_for(None,0)['timestamp_ms'],0)
        self.assertEqual(updates,[0])
        # Four frames pass while one detection is in flight.
        for time in (40,80,120):
            self.assertEqual(worker.objects_for(None,time)['timestamp_ms'],0)
        self.assertTrue(started.wait(5))
        self.assertEqual(updates,[0])
        gate.set()
        self.assertEqual(worker.detector_pending.result()['timestamp_ms'],40)
        self.assertEqual(worker.objects_for(None,160)['timestamp_ms'],40)
        self.assertEqual(updates,[0,40])

    def test_detector_failure_raises_on_the_collecting_frame(self):
        def predict(image,time):
            if time:raise RuntimeError('detector unavailable')
            return self.reply(time)
        worker=self.make_worker(SimpleNamespace(predict=predict),[])
        self.assertEqual(worker.objects_for(None,0)['timestamp_ms'],0)
        self.assertEqual(worker.objects_for(None,40)['timestamp_ms'],0)
        self.assertRaises(RuntimeError,worker.detector_pending.result)
        with self.assertRaises(RuntimeError):worker.objects_for(None,80)
    def test_stalled_request_fails_the_session_instead_of_freezing(self):
        import threading
        releaser=threading.Event()
        def predict(image,time):
            if time:releaser.wait(5)
            return self.reply(time)
        worker=self.make_worker(SimpleNamespace(predict=predict),[])
        worker.detector_deadline=0.05
        self.assertEqual(worker.objects_for(None,0)['timestamp_ms'],0)
        self.assertEqual(worker.objects_for(None,40)['timestamp_ms'],0)
        time_module=__import__('time');time_module.sleep(0.12)
        # Evidence would otherwise stay frozen for the transport timeout.
        with self.assertRaises(RuntimeError):worker.objects_for(None,80)
        releaser.set()


if __name__=='__main__':unittest.main()
