"""Full-frame MediaPipe Tasks conversion. ROI containment is not identity proof."""
import math
from contextlib import ExitStack
import importlib

import numpy as np

from inference.experts.assets import model_identity, verify_asset


def to_rgb(bgr):
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 1:
        raise ValueError('Expected a nonempty uint8 HWC BGR frame')
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _pixels(landmarks, count, width, height):
    if len(landmarks) not in count:
        raise ValueError('Unexpected landmark topology')
    result = []
    for point in landmarks:
        x, y = float(point.x), float(point.y)
        if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError('Landmark outside normalized image; do not clip into a valid observation')
        result.append([x*width, y*height])
    return result


def _inside(points, roi):
    return all(roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3] for x, y in points)


def face_observation(faces, width, height, roi, exclusive, timestamp_ms, model_id, *, matrices=None):
    result = dict(status='ok', timestamp_ms=timestamp_ms, model_id=model_id,
                  driver_confirmed=False, landmarks=None, head_pose=None)
    try:
        candidates = [_pixels(face, (468, 478), width, height) for face in faces]
    except (ValueError, TypeError, AttributeError, OverflowError):
        result['status'] = 'missing'
        return result
    selected = [(index,points) for index,points in enumerate(candidates) if _inside(points, roi)]
    candidates = [points for _,points in selected]
    if len(candidates) == 1:
        result.update(landmarks=candidates[0], driver_confirmed=exclusive)
        if matrices is not None and len(matrices) == len(faces):
            from inference.experts.gaze_geometry import head_pose
            result['head_pose'] = head_pose(matrices[selected[0][0]])
    elif len(candidates) > 1:
        result['status'] = 'missing'
    return result


def hand_observation(hands, width, height, roi, exclusive, timestamp_ms, model_id):
    result = dict(status='ok', timestamp_ms=timestamp_ms, model_id=model_id, items=[])
    try:
        candidates = [_pixels(hand, (21,), width, height) for hand in hands]
    except (ValueError, TypeError, AttributeError, OverflowError):
        result['status'] = 'missing'
        return result
    result['items'] = [dict(landmarks=points, driver_confirmed=exclusive and _inside(points, roi),
                            track_id=None) for points in candidates]
    return result


class MediaPipeAdapter:
    """Owns independent CPU Face/Hand VIDEO tasks for one recording only."""
    def __init__(self, face_asset, hands_asset, *, driver_roi, driver_region_confirmed_exclusive=False, gaze_features=False,
                 diagnostics=False):
        if type(diagnostics) is not bool:
            raise ValueError('diagnostics must be boolean')
        self._diagnostics, self.last_face_diagnostics = diagnostics, None
        if type(gaze_features) is not bool:
            raise ValueError('gaze_features must be boolean')
        if type(driver_region_confirmed_exclusive) is not bool:
            raise ValueError('Driver exclusivity must be an explicit boolean annotation')
        if not isinstance(driver_roi, (list, tuple)) or len(driver_roi) != 4 or not all(
                type(x) in (int, float) and math.isfinite(x) for x in driver_roi) or min(driver_roi) < 0 or (
                driver_roi[2]-driver_roi[0] < 1e-6 or driver_roi[3]-driver_roi[1] < 1e-6):
            raise ValueError('driver_roi must be a nondegenerate original-pixel xyxy rectangle')
        face_asset, hands_asset = verify_asset(face_asset), verify_asset(hands_asset)
        try:
            self._mp = importlib.import_module('mediapipe')
        except ImportError as error:
            raise RuntimeError('MediaPipe Tasks is required in the perception environment; no automatic install') from error
        self._roi, self._exclusive = list(driver_roi), driver_region_confirmed_exclusive
        self._gaze = gaze_features
        self._time, self._shape, self._closed = None, None, False
        self._resources = ExitStack()
        base, vision = self._mp.tasks.BaseOptions, self._mp.tasks.vision
        common = dict(running_mode=vision.RunningMode.VIDEO, min_tracking_confidence=0.5)
        settings = dict(implementation='driveguard-mediapipe-v1', version=self._mp.__version__,
                        delegate='CPU', mode='VIDEO', num_faces=2, num_hands=4, confidence=0.5,
                        coordinates='full-frame-unmirrored-xy', driver_roi=self._roi,
                        driver_region_confirmed_exclusive=self._exclusive)
        if gaze_features:
            settings.update(implementation='driveguard-mediapipe-gaze-v2',gaze_geometry='mediapipe-matrix-iris-v1')
        self.face_model_id = model_identity('mediapipe-face', face_asset, settings)
        self.hands_model_id = model_identity('mediapipe-hands', hands_asset, settings)
        try:
            self._face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=face_asset['path'], delegate=base.Delegate.CPU),
                num_faces=2, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
                output_face_blendshapes=False, output_facial_transformation_matrixes=gaze_features, **common))
            self._resources.callback(self._face.close)
            self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=hands_asset['path'], delegate=base.Delegate.CPU),
                num_hands=4, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, **common))
            self._resources.callback(self._hands.close)
        except BaseException:
            self.close()
            raise

    def predict(self, bgr, timestamp_ms):
        if self._closed:
            raise RuntimeError('MediaPipe adapter is closed')
        if type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 2**63-1 or (
                self._time is not None and timestamp_ms <= self._time):
            raise ValueError('MediaPipe VIDEO requires strictly increasing nonnegative milliseconds')
        rgb = to_rgb(bgr)
        height, width = rgb.shape[:2]
        if self._roi[2] > width or self._roi[3] > height or (self._shape is not None and self._shape != (height, width)):
            raise ValueError('Frame dimensions changed or driver ROI is outside image')
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        try:
            face = self._face.detect_for_video(image, timestamp_ms)
            hands = self._hands.detect_for_video(image, timestamp_ms)
            pose_options = {'matrices': face.facial_transformation_matrixes} if self._gaze else {}
            result = (face_observation(face.face_landmarks, width, height, self._roi, self._exclusive, timestamp_ms, self.face_model_id, **pose_options),
                      hand_observation(hands.hand_landmarks, width, height, self._roi, self._exclusive, timestamp_ms, self.hands_model_id))
            if result[0]['status'] == 'missing':
                # Ambiguous/invalid face evidence cannot uphold the exclusive-driver assertion.
                for hand in result[1]['items']:
                    hand['driver_confirmed'] = False
            if self._diagnostics:
                from inference.experts.face_diagnostics import inspect_face
                self.last_face_diagnostics = inspect_face(face.face_landmarks, pose_options.get('matrices'),
                    result[0], width, height, self._roi, self._exclusive)
        except BaseException:
            self.close()  # A partially advanced internal tracker cannot safely retry the same timestamp.
            raise
        self._time, self._shape = timestamp_ms, (height, width)
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            self._resources.close()
