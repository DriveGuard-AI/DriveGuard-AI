"""Synthetic pixel-space contract tests; no model weights or camera required."""

import copy
import importlib
import math
import unittest


def profile():
    return {
        "max_age_ms": 80, "min_object_score": .25,
        "ear_closed_threshold": .2, "hand_near_threshold": .03,
        "eye_indices": {"left": [362, 385, 387, 263, 373, 380],
                        "right": [33, 160, 158, 133, 153, 144]},
        "mouth_indices": [13, 14, 78, 308],
    }


def packet():
    points = [[300., 150.] for _ in range(468)]
    points[0], points[1] = [200., 100.], [400., 200.]
    for indices, x in ((profile()["eye_indices"]["left"], 220),
                       (profile()["eye_indices"]["right"], 320)):
        for index, point in zip(indices, [[x, 130], [x+10, 124], [x+30, 124],
                                          [x+40, 130], [x+30, 136], [x+10, 136]]):
            points[index] = point
    for index, point in zip(profile()["mouth_indices"],
                            [[300, 155], [300, 175], [280, 165], [320, 165]]):
        points[index] = point
    return {
        "schema_version": "driveguard-experts-v1", "recording_id": "r1",
        "view": "front", "frame_id": "f1", "timestamp_ms": 1000,
        "width": 1000, "height": 500, "driver_roi": [100, 50, 600, 450],
        "face": {"status": "ok", "timestamp_ms": 1000, "model_id": "face-v1",
                 "driver_confirmed": True, "landmarks": points,
                 "head_pose": {"yaw_deg": 10, "pitch_deg": -20, "roll_deg": 30}},
        "objects": {"status": "ok", "timestamp_ms": 1000, "model_id": "objects-v1", "items": []},
        "hands": {"status": "ok", "timestamp_ms": 1000, "model_id": "hands-v1", "items": []},
    }


def object_item(box, category="phone", score=.8, track="p1"):
    return {"category": category, "box": box, "score": score, "track_id": track}


def hand_item(box, confirmed=True, track="h1"):
    x1, y1, x2, y2 = box
    return {"landmarks": [[x1, y1], [x2, y2]] + [[(x1+x2)/2, (y1+y2)/2]]*19,
            "driver_confirmed": confirmed, "track_id": track}


class PerceptionGeometryTests(unittest.TestCase):
    def extract(self, value=None, settings=None):
        try:
            extract = importlib.import_module("inference.perception.geometry").spatial_features
        except ModuleNotFoundError:
            self.fail("spatial_features implementation is missing")
        return extract(packet() if value is None else value, profile() if settings is None else settings)

    def test_non_square_pixel_ear_mar_canonical_pose_and_no_mutation(self):
        p = packet()
        original = copy.deepcopy(p)
        result = self.extract(p)
        values = result["values"]
        self.assertAlmostEqual(values["ear_left"], .3)
        self.assertAlmostEqual(values["ear_right"], .3)
        self.assertAlmostEqual(values["mar"], .5)
        self.assertEqual([values[k] for k in ("head_yaw_deg", "head_pitch_deg", "head_roll_deg")], [10, -20, 30])
        self.assertFalse(result["eye_closed"])
        self.assertEqual(len(values), 14)
        self.assertEqual(p, original)

    def test_degenerate_landmarks_invalidate_dependent_features(self):
        p = packet()
        p["face"]["landmarks"] = [[300, 150] for _ in range(478)]
        p["objects"]["items"] = [object_item([400, 100, 450, 200])]
        result = self.extract(p)
        for key in ("ear_left", "ear_right", "mar", "phone_face_distance", "drink_mouth_distance"):
            self.assertIsNone(result["values"][key])
        self.assertIsNone(result["eye_closed"])

    def test_one_degenerate_eye_does_not_report_closed(self):
        p = packet()
        points = p["face"]["landmarks"]
        points[263] = points[362]
        result = self.extract(p)
        self.assertIsNone(result["values"]["ear_left"])
        self.assertAlmostEqual(result["values"]["ear_right"], .3)
        self.assertIsNone(result["eye_closed"])

    def test_sub_micropixel_eye_and_mouth_widths_are_unavailable(self):
        p = packet()
        points = p["face"]["landmarks"]
        points[263] = [points[362][0] + 1e-7, points[362][1]]
        points[308] = [points[78][0] + 1e-7, points[78][1]]
        result = self.extract(p)
        self.assertIsNone(result["values"]["ear_left"])
        self.assertIsNone(result["values"]["mar"])
        self.assertIsNone(result["eye_closed"])
        self.assertAlmostEqual(result["values"]["ear_right"], .3)

    def test_sub_micropixel_face_and_hand_bounds_are_unavailable(self):
        p = packet()
        p["face"]["landmarks"] = [[300, 150] for _ in range(468)]
        p["face"]["landmarks"][1] = [300 + 1e-7, 150 + 1e-7]
        p["objects"]["items"] = [object_item([300, 220, 350, 260])]
        p["hands"]["items"] = [hand_item([300, 220, 300 + 1e-7, 260])]
        result = self.extract(p)
        for key in ("ear_left", "ear_right", "mar", "phone_face_distance",
                    "phone_hand_distance", "phone_hand_iou"):
            self.assertIsNone(result["values"][key])
        self.assertIsNone(result["phone_near"])

    def test_sub_micropixel_roi_and_object_boxes_are_rejected(self):
        for target in ("roi", "object"):
            p = packet()
            tiny = [300, 220, 300 + 1e-7, 260]
            if target == "roi":
                p["driver_roi"] = tiny
            else:
                p["objects"]["items"] = [object_item(tiny)]
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.extract(p)

    def test_integer_engineering_bounds_reject_overflow_and_accept_limits(self):
        for key, maximum in (("width", 2**31 - 1), ("height", 2**31 - 1),
                             ("timestamp_ms", 2**63 - 1)):
            p = packet()
            p[key] = maximum
            self.extract(p)
            for bad in (maximum + 1, 10**400):
                p[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    self.extract(p)
        for name in ("face", "objects", "hands"):
            p = packet()
            p["timestamp_ms"] = 2**63 - 1
            p[name]["timestamp_ms"] = 2**63 - 1
            self.extract(p)
            p[name]["timestamp_ms"] = 2**63
            with self.subTest(expert=name), self.assertRaises(ValueError):
                self.extract(p)
        settings = profile()
        settings["max_age_ms"] = 2**63 - 1
        self.extract(settings=settings)
        settings["max_age_ms"] = 2**63
        with self.assertRaises(ValueError):
            self.extract(settings=settings)

    def test_no_detection_and_expert_failure_have_distinct_confidence(self):
        p = packet()
        p["hands"]["items"] = [hand_item([300, 220, 350, 260])]
        result = self.extract(p)
        self.assertEqual(result["values"]["phone_conf"], 0.)
        self.assertFalse(result["phone_near"])
        self.assertIsNone(result["values"]["phone_hand_distance"])
        for status in ("missing", "error"):
            p["objects"]["status"] = status
            result = self.extract(p)
            self.assertIsNone(result["values"]["phone_conf"])
            self.assertIsNone(result["phone_near"])

    def test_stale_experts_are_ignored_inclusive_freshness_boundary(self):
        p = packet()
        p["objects"]["items"] = [object_item([300, 230, 330, 250])]
        p["hands"]["items"] = [hand_item([300, 220, 350, 260])]
        for expert in ("face", "objects", "hands"):
            p[expert]["timestamp_ms"] = 920
        self.assertTrue(self.extract(p)["phone_near"])
        for expert in ("face", "objects", "hands"):
            p[expert]["timestamp_ms"] = 919
        result = self.extract(p)
        self.assertTrue(all(value is None for value in result["values"].values()))

    def test_background_and_low_score_objects_do_not_win(self):
        p = packet()
        p["objects"]["items"] = [object_item([700, 100, 800, 200], score=.99, track="background"),
                                  object_item([300, 220, 330, 250], score=.24, track="weak"),
                                  object_item([350, 220, 380, 250], score=.6, track="driver")]
        p["hands"]["items"] = [hand_item([350, 220, 380, 250])]
        result = self.extract(p)
        self.assertEqual(result["phone_track"], "driver")
        self.assertEqual(result["values"]["phone_conf"], .6)
        self.assertEqual(result["values"]["phone_hand_iou"], 1.)

    def test_unknown_passenger_and_degenerate_hands_have_no_interaction(self):
        for hands in ([], [hand_item([300, 220, 350, 260], confirmed=False)],
                      [hand_item([650, 220, 700, 260])], [hand_item([300, 220, 300, 260])]):
            p = packet()
            p["objects"]["items"] = [object_item([300, 220, 350, 260])]
            p["hands"]["items"] = hands
            result = self.extract(p)
            self.assertIsNone(result["phone_near"])
            self.assertIsNone(result["values"]["phone_hand_distance"])
            self.assertIsNone(result["values"]["phone_hand_iou"])

    def test_candidates_and_hand_ties_are_deterministic_and_consistent(self):
        p = packet()
        p["objects"]["items"] = [object_item([400, 100, 440, 140], track="winner"),
                                  object_item([410, 100, 450, 140], track="other"),
                                  object_item([250, 160, 270, 180], category="cup", track="cup"),
                                  object_item([240, 160, 260, 180], category="bottle", track="bottle")]
        p["hands"]["items"] = [hand_item([390, 100, 430, 140], track="h2"),
                                hand_item([400, 110, 440, 150], track="h1")]
        result = self.extract(p)
        self.assertEqual(result["phone_track"], "winner")
        self.assertEqual(result["drink_track"], "bottle")
        self.assertAlmostEqual(result["values"]["phone_face_distance"], math.hypot(120, -30)/math.hypot(200, 100))
        self.assertAlmostEqual(result["values"]["drink_mouth_distance"], math.hypot(50, 5)/math.hypot(200, 100))
        self.assertAlmostEqual(result["values"]["phone_hand_iou"], 1200/2000)
        p["objects"]["items"].reverse()
        p["hands"]["items"].reverse()
        self.assertEqual(result, self.extract(p))

    def test_rectangle_gap_uses_roi_diagonal_and_inclusive_near_threshold(self):
        p = packet()
        p["objects"]["items"] = [object_item([400, 300, 440, 340], track=None)]
        p["hands"]["items"] = [hand_item([300, 200, 350, 250])]
        settings = profile()
        settings["hand_near_threshold"] = math.hypot(50, 50)/math.hypot(500, 400)
        result = self.extract(p, settings)
        self.assertEqual(result["values"]["phone_hand_distance"], settings["hand_near_threshold"])
        self.assertTrue(result["phone_near"])
        self.assertEqual(result["values"]["phone_hand_iou"], 0.)
        self.assertIsNone(result["phone_track"])

    def test_face_requires_confirmation_all_points_inside_roi_and_presence(self):
        for variant in ("unconfirmed", "outside", "missing"):
            p = packet()
            if variant == "unconfirmed":
                p["face"]["driver_confirmed"] = False
            elif variant == "outside":
                p["face"]["landmarks"][0] = [700, 150]
            else:
                p["face"]["landmarks"] = None
                p["face"]["head_pose"] = None
            result = self.extract(p)
            self.assertIsNone(result["values"]["ear_left"])
            self.assertIsNone(result["values"]["head_yaw_deg"])

    def test_invalid_packet_fields_rejected_before_computing(self):
        edits = [
            (("schema_version",), "wrong"), (("recording_id",), ""), (("frame_id",), 1),
            (("view",), "  "), (("timestamp_ms",), True), (("width",), 0), (("height",), 500.),
            (("driver_roi",), [100, 50, 100, 450]), (("driver_roi",), [0, 0, 1001, 500]),
            (("face", "status"), "unknown"), (("face", "model_id"), ""),
            (("face", "timestamp_ms"), 1001), (("objects", "timestamp_ms"), -1),
            (("hands", "timestamp_ms"), True), (("face", "driver_confirmed"), 1),
            (("face", "landmarks"), [[1, 2]]), (("face", "landmarks", 0), [math.nan, 0]),
            (("face", "landmarks", 0), [True, 0]), (("face", "landmarks", 0), [1001, 0]),
            (("face", "head_pose", "yaw_deg"), 181), (("face", "head_pose", "pitch_deg"), math.inf),
        ]
        for path, bad in edits:
            p = packet()
            target = p
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = bad
            with self.subTest(path=path, bad=bad), self.assertRaises((ValueError, TypeError)):
                self.extract(p)
        for path in ((), ("face",), ("objects",), ("hands",), ("face", "head_pose")):
            p = packet()
            target = p
            for key in path:
                target = target[key]
            target["unexpected"] = 1
            with self.subTest(unknown=path), self.assertRaises((ValueError, TypeError)):
                self.extract(p)

    def test_invalid_objects_hands_and_failed_payloads_rejected_even_when_stale(self):
        for key, bad in (("category", "food"), ("box", [20, 10, 10, 20]),
                         ("box", [0, 0, 1001, 400]), ("score", True), ("score", 1.01),
                         ("score", math.nan), ("track_id", ""), ("unknown", 1)):
            p = packet()
            item = object_item([300, 100, 330, 200])
            item[key] = bad
            p["objects"]["items"] = [item]
            p["objects"]["timestamp_ms"] = 0
            with self.subTest(object_field=key), self.assertRaises((ValueError, TypeError)):
                self.extract(p)
        for key, bad in (("landmarks", [[1, 2]]), ("driver_confirmed", 1), ("track_id", 4), ("unknown", 1)):
            p = packet()
            item = hand_item([300, 100, 330, 200])
            item[key] = bad
            p["hands"]["items"] = [item]
            with self.subTest(hand_field=key), self.assertRaises((ValueError, TypeError)):
                self.extract(p)
        p = packet()
        item = object_item([300, 100, 330, 200])
        p["objects"]["items"] = [item, item]
        with self.assertRaises((ValueError, TypeError)):
            self.extract(p)
        for expert in ("face", "objects", "hands"):
            p = packet()
            p["objects"]["items"] = [object_item([300, 100, 330, 200])]
            p["hands"]["items"] = [hand_item([300, 100, 330, 200])]
            p[expert]["status"] = "missing"
            with self.subTest(failed=expert), self.assertRaises((ValueError, TypeError)):
                self.extract(p)
        p = packet()
        p["face"]["landmarks"] = None
        with self.assertRaises((ValueError, TypeError)):
            self.extract(p)
        p = packet()
        p["objects"].update(status="missing", timestamp_ms=1001)
        with self.assertRaises((ValueError, TypeError)):
            self.extract(p)


if __name__ == "__main__":
    unittest.main()
