"""Torch-free JSONL camera worker, launched in the existing perception environment."""
import argparse
import base64
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import io
import json
import math
import sys

from inference.live.timeline import CausalClock


def validate_camera(options):
    if not isinstance(options,dict) or set(options)!={
            'view','image_size','driver_roi','driver_region_confirmed_exclusive'}:
        raise ValueError('Expected view, image_size, normalized driver_roi and explicit exclusivity')
    result=deepcopy(options);size=result['image_size'];roi=result['driver_roi']
    if (result['view'] not in ('body','face') or not isinstance(size,list) or len(size)!=2 or
            any(type(n) is not int or n<16 or n>1920 for n in size) or size[0]*size[1]>1920*1080):
        raise ValueError('Invalid camera view/resolution (maximum 1920x1080 pixels)')
    if (not isinstance(roi,list) or len(roi)!=4 or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in roi) or
            not (0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1) or
            type(result['driver_region_confirmed_exclusive']) is not bool):
        raise ValueError('Invalid driver ROI/ownership')
    result['driver_roi']=[v*size[i%2] for i,v in enumerate(roi)]
    return result


def decode_jpeg(encoded,size):
    import numpy as np
    from PIL import Image
    if not isinstance(encoded,str) or len(encoded)>4*1024*1024:
        raise ValueError('JPEG request exceeds 4 MiB base64 limit')
    raw=base64.b64decode(encoded,validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format!='JPEG' or list(image.size)!=size:
            raise ValueError('Expected JPEG at the session camera resolution')
        return np.ascontiguousarray(np.asarray(image.convert('RGB'))[:,:,::-1])


class CameraPerception:
    def __init__(self,runtime_config,options,head_crop):
        from inference.perception.runtime import load_runtime_config
        from inference.experts.mediapipe_adapter import MediaPipeAdapter
        from inference.experts.detector_client import RemoteDetector
        from inference.experts.face_experiment_adapter import FaceExperimentAdapter
        from inference.perception.tracking import ObjectTracker
        from inference.perception.sequence import FeatureExtractor
        from inference.perception.gaze_profile import gaze_profile
        from training.visual.eye_tracking import EyeLocator
        from training.visual.spatial import crop_roi
        from training.dmd.face_experiment_config import BASELINE
        self.resources=ExitStack();self.camera=validate_camera(options)
        self.config=load_runtime_config(runtime_config)
        self.config.update({key:self.camera[key] for key in ('view','driver_roi','driver_region_confirmed_exclusive')})
        self.config['recording_id']='live-camera'
        self.clock=CausalClock();self.extractor=FeatureExtractor(gaze_profile());self.tracker=ObjectTracker()
        self.gap_pending=False;self.previous_capture=None
        roi=self.camera['driver_roi'];size=self.camera['image_size']
        self.head_roi=crop_roi(roi,head_crop)
        try:
            if self.config['objects']['variant']!='yolo11s':raise ValueError('YOLO11 runtime required')
            self.detector=RemoteDetector(**self.config['objects']);self.resources.callback(self.detector.close)
            self.landmarks=MediaPipeAdapter(self.config['face_asset'],self.config['hands_asset'],
                driver_roi=roi,driver_region_confirmed_exclusive=self.camera['driver_region_confirmed_exclusive'],gaze_features=True)
            self.resources.callback(self.landmarks.close)
            self.locator=None
            if self.camera['view']=='body':
                adapter=FaceExperimentAdapter(self.config['face_asset'],driver_roi=roi,
                    driver_region_confirmed_exclusive=self.camera['driver_region_confirmed_exclusive'],
                    variant=dict(BASELINE,name='native_eye_head_crop',crop_xyxy=[v/size[i%2] for i,v in enumerate(self.head_roi)]))
                self.resources.callback(adapter.close)
                self.locator=EyeLocator(adapter,self.head_roi,size,tracking=True)
        except BaseException:self.close();raise

    def close(self):
        self.resources.close()

    def frame(self,timestamp_ms,jpeg):
        self.clock.validate(timestamp_ms)
        bgr=decode_jpeg(jpeg,self.camera['image_size'])
        objects=self.detector.predict(bgr,timestamp_ms)
        objects['items']=self.tracker.update(objects['items'],timestamp_ms)
        face,hands=self.landmarks.predict(bgr,timestamp_ms)
        eye=self.locator.step(bgr,timestamp_ms) if self.locator else None
        snapshot=dict(bgr=bgr,objects=objects,face=face,hands=hands,eye=eye)
        result=[]
        from training.visual.eye_frames import _pixels
        def encoded(image,roi):return base64.b64encode(_pixels(image,roi).tobytes()).decode('ascii')
        for sample in self.clock.push(timestamp_ms,snapshot):
            previous=sample['payload'];slot=sample['slot'];time=slot*40
            packet=dict(schema_version='driveguard-experts-v1',recording_id='live-camera',
                view=self.camera['view'],frame_id='slot-'+str(slot),timestamp_ms=time,
                width=self.camera['image_size'][0],height=self.camera['image_size'][1],
                driver_roi=self.camera['driver_roi'],**{k:previous[k] for k in ('objects','face','hands')})
            features=self.extractor.step(packet)
            row=dict(slot=slot,source_time_ms=sample['source_time_ms'],fresh=sample['fresh'],
                values=features['values'].tolist(),masks=features['masks'].tolist(),
                ages=[min(81,time-previous[name]['timestamp_ms']) if previous[name]['status']=='ok' else 81
                      for name in ('objects','face','hands')],head_sample=sample['head_sample'],reset=False)
            if self.camera['view']=='body':
                confirmed=self.camera['driver_region_confirmed_exclusive']
                if sample['rgb_sample'] and sample['fresh'] and confirmed:
                    row['rgb_pixels']=encoded(previous['bgr'],self.camera['driver_roi'])
                if sample['head_sample']:
                    row['reset']=self.gap_pending;self.gap_pending=False
                    if sample['fresh'] and confirmed:
                        row['head_pixels']=encoded(previous['bgr'],self.head_roi)
                        region=previous['eye']['roi']
                        if region is not None:row['eye_pixels']=encoded(previous['bgr'],region)
            result.append(row)
        # Set after emitting the previous-frame interval; the gap belongs to the
        # new source and must persist until the next sampled head frame.
        if self.previous_capture is not None and timestamp_ms-self.previous_capture>80:self.gap_pending=True
        self.previous_capture=timestamp_ms
        return dict(samples=result,perception=dict(face_status=face['status'],head_roi=self.head_roi,
            eye_roi=eye['roi'] if eye else None,eye_origin=eye['origin'] if eye else None,
            eye_reason=eye['reason'] if eye else None))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime-config',required=True)
    args=parser.parse_args();protocol=sys.stdout;camera=None
    try:
        for line in iter(lambda:sys.stdin.readline(5*1024*1024+1),''):
            try:
                if len(line)>5*1024*1024:raise ValueError('Worker input too large')
                request=json.loads(line)
                # Third-party runtime logs never enter the JSONL response stream.
                with redirect_stdout(sys.stderr):
                    if request['operation']=='open':
                        if camera is not None:raise ValueError('Session already open')
                        camera=CameraPerception(args.runtime_config,request['camera'],request['head_crop'])
                        result=dict(status='ready')
                    elif request['operation']=='frame' and camera is not None:
                        result=camera.frame(request['timestamp_ms'],request['jpeg'])
                    else:raise ValueError('Invalid worker operation/session')
                protocol.write(json.dumps(dict(ok=True,result=result),allow_nan=False)+'\n');protocol.flush()
            except Exception as error:
                protocol.write(json.dumps(dict(ok=False,error=str(error)))+'\n');protocol.flush()
                break  # Advanced tracker state cannot be retried safely.
    finally:
        if camera is not None:camera.close()


if __name__=='__main__':main()
