from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from inference.perception.video_source import VideoFrame, decode_video, sample_frames
from inference.perception.video import expert_packets
from inference.perception.runtime import load_runtime_config
from inference.perception.export import export_recording
from inference.perception.profile import load_profile
from tests.perception_helpers import packet


class VideoTests(unittest.TestCase):
    def frame(self, time, shape=(500, 1000, 3)):
        return VideoFrame(time, np.zeros(shape, dtype=np.uint8))

    def test_causal_sampling_no_future_no_tail_and_reuses_last_source(self):
        result = [(slot, frame.timestamp_ms) for slot, frame in sample_frames(map(self.frame, [0, 33, 67, 200]))]
        self.assertEqual(result, [(0, 0), (40, 33), (80, 67), (120, 67), (160, 67), (200, 200)])
        self.assertEqual([(t, f.timestamp_ms) for t, f in sample_frames(map(self.frame, [0, 10, 20, 39]))], [(0, 0)])

    def test_invalid_timing_dimensions_and_empty_input_fail(self):
        for frames in ([], [self.frame(1)], list(map(self.frame, [0, 0])), list(map(self.frame, [0, -1])),
                       [self.frame(0), self.frame(40, (400, 1000, 3))]):
            with self.subTest(frames=len(frames)), self.assertRaises(ValueError):
                list(sample_frames(frames))

    def test_pts_decode_origin_ceiling_and_cleanup(self):
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            path = Path(directory) / 'fixture.mp4'
            path.write_bytes(b'not a real video; decoder boundary fixture')
            closed = []
            class Container:
                streams = SimpleNamespace(video=[object()])
                def __enter__(self): return self
                def __exit__(self, *args): closed.append(True)
                def decode(self, stream):
                    for pts in (9000, 12003, 16200):
                        yield SimpleNamespace(pts=pts, time_base=Fraction(1, 90000), is_corrupt=False,
                                              to_ndarray=lambda format: np.zeros((2, 4, 3), dtype=np.uint8))
            with patch.dict(sys.modules, {'av': SimpleNamespace(open=lambda *a, **k: Container())}):
                frames = list(decode_video(path))
            self.assertEqual([frame.timestamp_ms for frame in frames], [0, 34, 80])
            self.assertEqual(closed, [True])

    def test_decode_restricts_demuxers_and_protocols_before_open(self):
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            path = Path(directory) / 'untrusted.mp4'
            path.write_bytes(b'ffconcat version 1.0\nfile other.mp4\n')
            def restricted_open(path, *, mode, options):
                self.assertEqual(mode, 'r')
                self.assertEqual(options['protocol_whitelist'], 'file')
                self.assertEqual(set(options['format_whitelist'].split(',')), {'mov', 'matroska', 'webm', 'avi'})
                self.assertEqual(options['enable_drefs'], '0')
                self.assertEqual(options['use_absolute_path'], '0')
                raise ValueError('synthetic demuxer rejects ffconcat before secondary I/O')
            with patch.dict(sys.modules, {'av': SimpleNamespace(open=restricted_open)}):
                with self.assertRaisesRegex(ValueError, 'rejects ffconcat'):
                    list(decode_video(path))

    def test_expert_cache_retains_source_time_stale_masks_and_unique_slots(self):
        calls = []
        class Detector:
            def predict(self, image, time):
                calls.append(time)
                result = deepcopy(packet()['objects'])
                result['timestamp_ms'] = time
                result['items'][0]['track_id'] = None
                return result
        class Landmarks:
            def predict(self, image, time):
                result = packet()
                for key in ('face', 'hands'):
                    result[key]['timestamp_ms'] = time
                return result['face'], result['hands']
        config = load_runtime_config('configs/perception/runtime.example.json')
        config.update(recording_id='synthetic', driver_roi=[0, 0, 800, 500])
        frames = [self.frame(0), self.frame(200)] + [self.frame(time) for time in range(240, 4001, 40)]
        packets = list(expert_packets(frames, Detector(), Landmarks(), config))
        self.assertEqual(calls[:3], [0, 200, 240])
        self.assertEqual(packets[3]['objects']['timestamp_ms'], 0)
        self.assertEqual(len({p['frame_id'] for p in packets}), 101)
        self.assertNotEqual(packets[0]['objects']['items'][0]['track_id'], packets[5]['objects']['items'][0]['track_id'])
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            out = Path(directory) / 'record'
            report = export_recording(iter(packets), out, load_profile(), provenance={'kind': 'synthetic'})
            masks = np.load(out / 'masks.npy')
            self.assertTrue(masks[0].any())
            self.assertFalse(masks[3].any())
            self.assertEqual(report['provenance'], {'kind': 'synthetic'})
            self.assertFalse(np.load(out / 'label_masks.npy').any())
            self.assertEqual(np.load(out / 'values.npy').shape, (101, 19))

    def test_runtime_strict_validation_without_opening_assets(self):
        config = load_runtime_config('configs/perception/runtime.example.json')
        self.assertFalse(config['driver_region_confirmed_exclusive'])
        for mutate in (lambda c: c.update(unknown=True), lambda c: c.update(driver_region_confirmed_exclusive=1),
                       lambda c: c['objects'].update(backend='auto'), lambda c: c.update(driver_roi=[0, 0, float('inf'), 500])):
            with tempfile.TemporaryDirectory(dir='runs') as directory:
                value = deepcopy(config)
                mutate(value)
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_runtime_config(path)
