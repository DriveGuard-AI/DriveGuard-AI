"""Isolated face-only experiments; never a production feature-cache adapter."""
from contextlib import ExitStack
import importlib
import math
from types import SimpleNamespace

from inference.experts.assets import model_identity, verify_asset
from inference.experts.mediapipe_adapter import face_observation, to_rgb
from training.dmd.face_experiment_config import BASELINE, validate_rectangle, validate_variants


class FaceExperimentAdapter:
    def __init__(self, face_asset, *, driver_roi, driver_region_confirmed_exclusive, variant, head_region=None):
        regional = 'head_region_mode' in variant
        baseline = dict(BASELINE, head_region_mode='none') if regional else BASELINE
        validate_variants([variant] if variant == baseline else [baseline, variant], regional=regional)
        if head_region is not None:
            validate_rectangle(head_region)
        self._region_mode = variant.get('head_region_mode', 'none')
        if self._region_mode != 'none' and head_region is None:
            raise ValueError('Scoped head region is required for gate/crop')
        self._head_region = list(head_region) if head_region is not None else None
        if type(driver_region_confirmed_exclusive) is not bool:
            raise ValueError('Driver exclusivity must be an explicit boolean')
        if not isinstance(driver_roi, (tuple, list)) or len(driver_roi) != 4 or not all(
                type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in driver_roi) or (
                driver_roi[2] <= driver_roi[0] or driver_roi[3] <= driver_roi[1]):
            raise ValueError('driver_roi must be original-pixel xyxy')
        self.variant = dict(variant)
        self._roi, self._exclusive = list(driver_roi), driver_region_confirmed_exclusive
        self._time, self._shape, self._closed = None, None, False
        self._resources = ExitStack()
        self.last_face_diagnostics = None
        asset = verify_asset(face_asset)
        self._mp = importlib.import_module('mediapipe')
        base, vision = self._mp.tasks.BaseOptions, self._mp.tasks.vision
        # Exact original face identity is exposed only for the full-frame VIDEO baseline.
        original_settings = dict(implementation='driveguard-mediapipe-gaze-v2', version=self._mp.__version__,
            delegate='CPU', mode='VIDEO', num_faces=2, num_hands=4, confidence=.5,
            coordinates='full-frame-unmirrored-xy', driver_roi=self._roi,
            driver_region_confirmed_exclusive=self._exclusive, gaze_geometry='mediapipe-matrix-iris-v1')
        self.baseline_face_model_id = model_identity('mediapipe-face', asset, original_settings) if variant == baseline else None
        settings = dict(implementation='driveguard-face-experiment-v1', version=self._mp.__version__,
            delegate='CPU', num_faces=2, variant=self.variant, driver_roi=self._roi,
            driver_region_confirmed_exclusive=self._exclusive, gaze_geometry='mediapipe-matrix-iris-v1',
            landmark_coordinates='original-image-pixels',
            pose_coordinate_frame='crop-experimental' if variant['crop_xyxy'] or self._region_mode == 'crop' else 'full-frame')
        if regional:
            settings['head_region'] = self._head_region
        self.experiment_model_id = model_identity('mediapipe-face-experiment', asset, settings)
        try:
            self._face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=asset['path'], delegate=base.Delegate.CPU),
                running_mode=getattr(vision.RunningMode, variant['mode']), num_faces=2,
                min_face_detection_confidence=variant['detection'], min_face_presence_confidence=variant['presence'],
                min_tracking_confidence=variant['tracking'], output_face_blendshapes=False,
                output_facial_transformation_matrixes=True))
            self._resources.callback(self._face.close)
        except BaseException:
            self.close()
            raise

    def predict(self, bgr, timestamp_ms):
        if self._closed:
            raise RuntimeError('Face experiment adapter is closed')
        if type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 2**63-1 or (
                self._time is not None and timestamp_ms <= self._time):
            raise ValueError('Experiment source timestamps must strictly increase')
        rgb = to_rgb(bgr)
        height, width = rgb.shape[:2]
        if self._roi[2] > width or self._roi[3] > height or self._shape not in (None, (height, width)):
            raise ValueError('Image size changed or ROI is outside original image')
        crop = self.variant['crop_xyxy']
        if self._region_mode == 'crop':
            crop = self._head_region
        self.head_region_pixels = ([self._head_region[i]*v for i, v in enumerate((width, height, width, height))]
                                   if self._head_region is not None else None)
        self.has_crop = crop is not None
        box = [0, 0, width, height] if crop is None else [
            math.floor(crop[0]*width), math.floor(crop[1]*height),
            math.ceil(crop[2]*width), math.ceil(crop[3]*height)]
        x1, y1, x2, y2 = box
        self.crop_pixels = box
        # MediaPipe requires contiguous storage, including for an explicit crop.
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb[y1:y2, x1:x2].copy())
        try:
            raw = self._face.detect(image) if self.variant['mode'] == 'IMAGE' else self._face.detect_for_video(image, timestamp_ms)
            faces = raw.face_landmarks
            if crop is not None:
                # Reject invalid crop-normalized coordinates BEFORE mapping them into the image.
                def mapped(point):
                    x, y = float(point.x), float(point.y)
                    if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
                        return SimpleNamespace(x=float('nan'), y=float('nan'))
                    return SimpleNamespace(x=(x1+x*(x2-x1))/width, y=(y1+y*(y2-y1))/height)
                faces = [[mapped(point) for point in face] for face in faces]
            matrices = raw.facial_transformation_matrixes
            observation = face_observation(faces, width, height, self._roi, self._exclusive,
                timestamp_ms, self.experiment_model_id, matrices=matrices,
                head_roi=self.head_region_pixels if self._region_mode != 'none' else None)
            from inference.experts.face_diagnostics import inspect_face
            self.last_face_diagnostics = inspect_face(faces, matrices, observation, width, height, self._roi, self._exclusive,
                head_roi=self.head_region_pixels, enforce_head_roi=self._region_mode != 'none')
        except BaseException:
            self.close()
            raise
        self._time, self._shape = timestamp_ms, (height, width)
        return observation

    def close(self):
        if not self._closed:
            self._closed = True
            self._resources.close()
