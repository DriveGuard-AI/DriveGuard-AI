"""Image-derived gaze cues, not a calibrated gaze ray or driver identity model."""
import math
import numpy as np

IRIS_NAMES = ('iris_left_x', 'iris_left_y', 'iris_right_x', 'iris_right_y')


def head_pose(matrix):
    """MediaPipe canonical->observed to image-camera Rz(roll)Ry(yaw)Rx(pitch).

    MediaPipe metric axes are x-right/y-up/z-back. Change BOTH bases using
    C=diag(1,-1,-1). Allow uniform scale then reject shear/reflection/degeneracy.
    Angles describe the canonical face orientation, not vehicle-calibrated gaze.
    """
    try:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return None
        if not np.allclose(matrix[3], [0,0,0,1], atol=1e-5, rtol=0):
            return None
        block = matrix[:3,:3]
        scales = np.linalg.norm(block, axis=0)
        scale = scales.mean()
        if scale < 1e-6 or not np.allclose(scales, scale, rtol=.03, atol=1e-6):
            return None
        rotation = block / scale
        if np.linalg.det(rotation) <= 0 or not np.allclose(rotation.T @ rotation,np.eye(3),atol=.03,rtol=0):
            return None
        u, _, vt = np.linalg.svd(rotation)
        conversion = np.diag([1.,-1.,-1.])
        rotation = conversion @ (u @ vt) @ conversion
        cosine = math.hypot(rotation[0,0], rotation[1,0])
        if cosine < 1e-4:  # No arbitrary Euler branch near gimbal lock.
            return None
        return dict(yaw_deg=math.degrees(math.atan2(-rotation[2,0],cosine)),
                    pitch_deg=math.degrees(math.atan2(rotation[2,1],rotation[2,2])),
                    roll_deg=math.degrees(math.atan2(rotation[1,0],rotation[0,0])))
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError):
        return None


def iris_features(points):
    """478-point iris centers in each eye's image-local basis / corner width.

    Anatomical left uses 362->263, center473; right33->133, center468.
    y axis is a 90 degree clockwise image rotation of x. Closed/degenerate
    eyes are unknown; bounds are conservative geometry checks, not confidence.
    """
    result = dict.fromkeys(IRIS_NAMES)
    try:
        points = np.asarray(points,dtype=np.float64)
    except (ValueError,TypeError,OverflowError):
        return result
    if points.shape != (478,2) or not np.isfinite(points).all():
        return result
    for side, indices, center in (('left',(362,385,387,263,373,380),473),
                                  ('right',(33,160,158,133,153,144),468)):
        eye = points[list(indices)]
        axis = eye[3]-eye[0]
        width = np.linalg.norm(axis)
        if width < 1e-6:
            continue
        ear = (np.linalg.norm(eye[1]-eye[5])+np.linalg.norm(eye[2]-eye[4])) / (2*width)
        if not .2 < ear < .8:
            continue
        axis /= width
        vertical = np.array([-axis[1],axis[0]])
        iris = points[center]-eye[0]
        x, y = float(iris @ axis / width), float(iris @ vertical / width)
        lids = (eye[1:]-eye[0]) @ vertical / width
        if not (0 <= x <= 1 and max(-.4,float(lids.min())-.05) <= y <= min(.4,float(lids.max())+.05)):
            continue
        result[f'iris_{side}_x'], result[f'iris_{side}_y'] = x,y
    return result
