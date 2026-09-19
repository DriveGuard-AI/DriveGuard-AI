"""Stateless geometry for the strict ``driveguard-experts-v1`` packet schema.

Coordinates are image pixels, including on non-square images. Expert payloads
are validated even when stale or not attributed to the driver. A confirmed
face or hand is usable only when *all* its points lie inside the driver ROI.
Objects use their box center for ROI attribution. No models are imported.

Sub-micropixel lengths are treated as numerically degenerate using a 1e-6
pixel tolerance. Frame dimensions are bounded by signed 32-bit integers and
millisecond timestamps/maximum age by signed 64-bit integers. These are
engineering limits that prevent arithmetic and downstream float32 overflow,
not sensor accuracy guarantees or calibrated physiological thresholds.
"""

import math


_EPS = 1e-6
_MAX_DIMENSION = 2**31 - 1
_MAX_TIMESTAMP = 2**63 - 1


_STATIC_NAMES = (
    "ear_left", "ear_right", "mar", "head_yaw_deg", "head_pitch_deg",
    "head_roll_deg", "phone_conf", "drink_object_conf", "phone_face_distance",
    "drink_mouth_distance", "phone_hand_distance", "phone_hand_iou",
    "drink_hand_distance", "drink_hand_iou",
)


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} must have exactly these fields: {sorted(keys)}")


def _number(value, label, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{label} is outside its finite allowed range")
    return value


def _integer(value, label, minimum=0, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{label} must be an integer within [{minimum}, {maximum}]")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _track(value, label):
    if value is not None:
        _text(value, label)


def _bool(value, label):
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")


def _point(value, width, height, label):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a pixel [x, y] list")
    _number(value[0], label + '.x', 0, width)
    _number(value[1], label + '.y', 0, height)


def _box(value, width, height, label):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be a pixel [x1, y1, x2, y2] list")
    _point(value[:2], width, height, label)
    _point(value[2:], width, height, label)
    if not _usable_box(value):
        raise ValueError(f"{label} must have width and height >= {_EPS} pixels")


def _landmarks(value, counts, width, height, label):
    if not isinstance(value, list) or len(value) not in counts:
        raise ValueError(f"{label} has an unsupported landmark count")
    for i, point in enumerate(value):
        _point(point, width, height, f"{label}[{i}]")


def _validate_packet(packet):
    _exact(packet, ("schema_version", "recording_id", "view", "frame_id", "timestamp_ms",
                    "width", "height", "driver_roi", "face", "objects", "hands"), "packet")
    if packet["schema_version"] != "driveguard-experts-v1":
        raise ValueError("unsupported expert packet schema_version")
    for field in ("recording_id", "view", "frame_id"):
        _text(packet[field], field)
    _integer(packet["timestamp_ms"], "timestamp_ms", maximum=_MAX_TIMESTAMP)
    for field in ("width", "height"):
        _integer(packet[field], field, 1, _MAX_DIMENSION)
    width, height = packet["width"], packet["height"]
    _box(packet["driver_roi"], width, height, "driver_roi")
    for name in ("face", "objects", "hands"):
        expert = packet[name]
        payload = ("driver_confirmed", "landmarks", "head_pose") if name == "face" else ("items",)
        _exact(expert, ("status", "timestamp_ms", "model_id") + payload, name)
        if expert["status"] not in ("ok", "missing", "error"):
            raise ValueError(f"{name}.status is invalid")
        _integer(expert["timestamp_ms"], name + ".timestamp_ms", maximum=_MAX_TIMESTAMP)
        if expert["timestamp_ms"] > packet["timestamp_ms"]:
            raise ValueError(f"{name}.timestamp_ms is in the future")
        _text(expert["model_id"], name + ".model_id")
        if name == "face":
            _bool(expert["driver_confirmed"], "face.driver_confirmed")
            if expert["status"] != "ok" and (expert["driver_confirmed"] or
                    expert["landmarks"] is not None or expert["head_pose"] is not None):
                raise ValueError("failed face expert must have no payload")
            if expert["landmarks"] is not None:
                _landmarks(expert["landmarks"], (468, 478), width, height, "face.landmarks")
            elif expert["head_pose"] is not None:
                raise ValueError("head_pose requires face landmarks")
            if expert["head_pose"] is not None:
                _exact(expert["head_pose"], ("yaw_deg", "pitch_deg", "roll_deg"), "face.head_pose")
                for key, value in expert["head_pose"].items():
                    _number(value, "head_pose." + key, -180, 180)
        else:
            if not isinstance(expert["items"], list):
                raise ValueError(f"{name}.items must be a list")
            if expert["status"] != "ok" and expert["items"]:
                raise ValueError(f"failed {name} expert must have no payload")
            tracks = set()
            for item in expert["items"]:
                if name == "objects":
                    _exact(item, ("category", "box", "score", "track_id"), "object")
                    if item["category"] not in ("phone", "bottle", "cup"):
                        raise ValueError("unsupported object category")
                    _box(item["box"], width, height, "object.box")
                    _number(item["score"], "object.score", 0, 1)
                else:
                    _exact(item, ("landmarks", "driver_confirmed", "track_id"), "hand")
                    _bool(item["driver_confirmed"], "hand.driver_confirmed")
                    _landmarks(item["landmarks"], (21,), width, height, "hand.landmarks")
                _track(item["track_id"], name + ".track_id")
                if name == "objects" and item["track_id"] is not None:
                    if item["track_id"] in tracks:
                        raise ValueError("duplicate object track_id")
                    tracks.add(item["track_id"])


def _validate_geometry_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("profile must be a dictionary")
    required = ("max_age_ms", "min_object_score", "ear_closed_threshold", "hand_near_threshold",
                "eye_indices", "mouth_indices")
    if not set(required).issubset(profile):
        raise ValueError("profile is missing required geometry fields")
    _integer(profile["max_age_ms"], "max_age_ms", maximum=_MAX_TIMESTAMP)
    _number(profile["min_object_score"], "min_object_score", 0, 1)
    for field in ("ear_closed_threshold", "hand_near_threshold"):
        _number(profile[field], field, 0)
    _exact(profile["eye_indices"], ("left", "right"), "eye_indices")
    for indices, count in [(profile["eye_indices"]["left"], 6),
                           (profile["eye_indices"]["right"], 6), (profile["mouth_indices"], 4)]:
        if not isinstance(indices, list) or len(indices) != count:
            raise ValueError("profile landmark indices have an invalid count")
        for index in indices:
            _integer(index, "landmark index")
            if index >= 468:
                raise ValueError("profile landmark index must be below 468")
        if len(set(indices)) != count:
            raise ValueError("profile landmark indices must be distinct")


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _usable_box(box):
    # Requiring both sides also bounds the diagonal away from zero, and
    # prevents underflow of areas used in object/hand IoU calculations.
    return box[2] - box[0] >= _EPS and box[3] - box[1] >= _EPS


def _bounds(points):
    return (min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points))


def _center(box):
    return ((box[0] + box[2])/2, (box[1] + box[3])/2)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ear(points, indices):
    p = [points[i] for i in indices]
    width = _distance(p[0], p[3])
    return (_distance(p[1], p[5]) + _distance(p[2], p[4])) / (2 * width) if width >= _EPS else None


def _gap(a, b):
    return math.hypot(max(0, a[0] - b[2], b[0] - a[2]),
                      max(0, a[1] - b[3], b[1] - a[3]))


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
    return intersection / union


def spatial_features(packet, profile):
    """Validate and extract 14 static values plus nullable temporal signals.

    ``None`` means unavailable; a fresh, successful object expert with no
    eligible detection produces confidence 0. Missing driver hands never
    establish a negative interaction. Sequence-derived values are omitted.
    The caller validates full profile identity; this function validates the
    geometry fields it consumes. Inputs are never mutated.
    """
    _validate_packet(packet)
    _validate_geometry_profile(profile)
    values = dict.fromkeys(_STATIC_NAMES)
    result = {"values": values, "eye_closed": None, "phone_near": None,
              "drink_near": None, "phone_track": None, "drink_track": None}
    roi = packet["driver_roi"]

    def fresh(expert):
        return expert["status"] == "ok" and packet["timestamp_ms"] - expert["timestamp_ms"] <= profile["max_age_ms"]

    face = packet["face"]
    points = face["landmarks"]
    face_center = mouth_center = face_diagonal = None
    if fresh(face) and face["driver_confirmed"] and points is not None and all(_inside(p, roi) for p in points):
        box = _bounds(points)
        if _usable_box(box):
            face_center, face_diagonal = _center(box), _distance(box[:2], box[2:])
            for side in ("left", "right"):
                values["ear_" + side] = _ear(points, profile["eye_indices"][side])
            upper, lower, left, right = [points[i] for i in profile["mouth_indices"]]
            mouth_width = _distance(left, right)
            values["mar"] = _distance(upper, lower) / mouth_width if mouth_width >= _EPS else None
            mouth_center = ((upper[0]+lower[0])/2, (upper[1]+lower[1])/2)
            if values["ear_left"] is not None and values["ear_right"] is not None:
                result["eye_closed"] = (values["ear_left"] + values["ear_right"])/2 < profile["ear_closed_threshold"]
        if face["head_pose"] is not None:
            for axis in ("yaw", "pitch", "roll"):
                values[f"head_{axis}_deg"] = float(face["head_pose"][f"{axis}_deg"])

    hands = []
    if fresh(packet["hands"]):
        for hand in packet["hands"]["items"]:
            if hand["driver_confirmed"] and all(_inside(p, roi) for p in hand["landmarks"]):
                box = _bounds(hand["landmarks"])
                if _usable_box(box):
                    hands.append((box, hand["track_id"] or ""))
    if fresh(packet["objects"]):
        eligible = [o for o in packet["objects"]["items"] if o["score"] >= profile["min_object_score"] and _inside(_center(o["box"]), roi)]
        eligible.sort(key=lambda o: (-o["score"], tuple(o["box"]), o["category"], o["track_id"] or ""))
        for kind, categories, conf, face_key, anchor in (
            ("phone", ("phone",), "phone_conf", "phone_face_distance", face_center),
            ("drink", ("bottle", "cup"), "drink_object_conf", "drink_mouth_distance", mouth_center),
        ):
            candidate = next((o for o in eligible if o["category"] in categories), None)
            values[conf] = float(candidate["score"]) if candidate else 0.
            if candidate is not None and anchor is not None:
                values[face_key] = _distance(_center(candidate["box"]), anchor) / face_diagonal
            if hands:
                if candidate is None:
                    result[kind + "_near"] = False
                else:
                    box = candidate["box"]
                    hand_box, _ = min(hands, key=lambda h: (_gap(box, h[0]), h[0], h[1]))
                    distance = _gap(box, hand_box) / _distance(roi[:2], roi[2:])
                    values[kind + "_hand_distance"] = distance
                    values[kind + "_hand_iou"] = _iou(box, hand_box)
                    result[kind + "_near"] = distance <= profile["hand_near_threshold"]
                    result[kind + "_track"] = candidate["track_id"]
    return result
