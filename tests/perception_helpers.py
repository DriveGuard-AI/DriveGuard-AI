"""Explicitly synthetic pixel-space expert observations, not model predictions."""
from copy import deepcopy


def packet(index=0, closed=False):
    points = [[400.0, 200.0] for _ in range(468)]
    points[0], points[1] = [250, 100], [550, 350]
    height = 4 if closed else 12
    for indices, x in [([362, 385, 387, 263, 373, 380], 450),
                       ([33, 160, 158, 133, 153, 144], 300)]:
        for i, xy in zip(indices, [[x, 180], [x+10, 180-height/2],
                                  [x+30, 180-height/2], [x+40, 180],
                                  [x+30, 180+height/2], [x+10, 180+height/2]]):
            points[i] = xy
    for i, xy in zip([13, 14, 78, 308], [[390, 250], [390, 270], [350, 260], [430, 260]]):
        points[i] = xy
    time = index * 40
    hand = [[600, 290] for _ in range(21)]
    hand[0], hand[1] = [580, 260], [620, 320]
    return {'schema_version': 'driveguard-experts-v1', 'recording_id': 'synthetic-video',
            'view': 'body', 'frame_id': str(index), 'timestamp_ms': time,
            'width': 1000, 'height': 500, 'driver_roi': [0, 0, 800, 500],
            'face': {'status': 'ok', 'timestamp_ms': time, 'model_id': 'synthetic-face-v1',
                     'driver_confirmed': True, 'landmarks': points, 'head_pose': None},
            'objects': {'status': 'ok', 'timestamp_ms': time, 'model_id': 'synthetic-detector-v1',
                        'items': [{'category': 'phone', 'box': [590, 280, 630, 330],
                                   'score': 0.9, 'track_id': 'phone-1'}]},
            'hands': {'status': 'ok', 'timestamp_ms': time, 'model_id': 'synthetic-hand-v1',
                      'items': [{'landmarks': hand, 'driver_confirmed': True, 'track_id': 'hand-1'}]}}


def missing_face(value):
    result = deepcopy(value)
    result['face'].update(status='missing', driver_confirmed=False, landmarks=None, head_pose=None)
    return result
