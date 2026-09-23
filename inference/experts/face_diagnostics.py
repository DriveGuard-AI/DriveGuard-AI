"""Explain the existing face conversion without accepting or repairing evidence."""
import math

from inference.experts.gaze_geometry import iris_features
from inference.experts.mediapipe_adapter import _pixels, _inside


def _candidate(face, index, width, height, roi):
    detail = dict(index=index, landmark_count=None, invalid_reason=None, bbox=None,
                  inside_roi=False, outside_roi_points=None)
    try:
        detail['landmark_count'] = len(face)
        xy = [(float(p.x), float(p.y)) for p in face]
        finite = [(x, y) for x, y in xy if math.isfinite(x) and math.isfinite(y)]
        if finite:
            # These boxes describe rejected candidates too; only the preview clips them.
            bounds = [min(x for x,y in finite)*width, min(y for x,y in finite)*height,
                      max(x for x,y in finite)*width, max(y for x,y in finite)*height]
            detail['bbox'] = bounds if all(math.isfinite(v) for v in bounds) else None
        if len(face) not in (468, 478):
            detail['invalid_reason'] = 'topology'
        elif len(finite) != len(xy):
            detail['invalid_reason'] = 'nonfinite'
        elif any(not 0 <= x <= 1 or not 0 <= y <= 1 for x,y in xy):
            detail['invalid_reason'] = 'out_of_bounds'
        else:
            points = _pixels(face, (468,478), width, height)
            detail['inside_roi'] = _inside(points, roi)
            detail['outside_roi_points'] = sum(not (roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]) for x,y in points)
    except (ValueError, TypeError, AttributeError, OverflowError):
        detail['invalid_reason'] = 'malformed'
    return detail


def inspect_face(faces, matrices, observation, width, height, roi, exclusive):
    """Reason precedence mirrors face_observation; raw candidates never become evidence."""
    candidates = [_candidate(face, i, width, height, roi) for i,face in enumerate(faces)]
    selected = [c['index'] for c in candidates if c['invalid_reason'] is None and c['inside_roi']]
    if not faces:
        reason = 'no_face_detected'
    elif any(c['invalid_reason'] for c in candidates):
        reason = 'invalid_landmarks'  # Existing adapter rejects ALL if any candidate is invalid.
    elif not selected:
        reason = 'outside_roi'
    elif len(selected) > 1:
        reason = 'ambiguous_faces'
    elif not exclusive:
        reason = 'driver_unconfirmed'
    elif matrices is None:
        reason = 'pose_matrices_missing'
    elif len(matrices) != len(faces):
        reason = 'pose_matrix_count_mismatch'
    elif observation['head_pose'] is None:
        reason = 'pose_matrix_invalid'
    else:
        reason = 'ok'
    face_usable = bool(observation['status'] == 'ok' and observation['driver_confirmed'] and observation['landmarks'] is not None)
    iris = iris_features(observation['landmarks']) if face_usable else {}
    left = iris.get('iris_left_x') is not None and iris.get('iris_left_y') is not None
    right = iris.get('iris_right_x') is not None and iris.get('iris_right_y') is not None
    return dict(reason=reason, detected_faces=len(faces), roi_candidates=len(selected),
                selected_index=selected[0] if len(selected)==1 and not any(c['invalid_reason'] for c in candidates) else None,
                candidates=candidates, face_usable=face_usable,
                head_usable=face_usable and observation['head_pose'] is not None,
                iris_left_valid=left, iris_right_valid=right,
                iris_reason='ok' if left or right else ('face_unavailable' if not face_usable else
                            'topology_without_iris' if len(observation['landmarks']) != 478 else 'geometry_rejected'))
