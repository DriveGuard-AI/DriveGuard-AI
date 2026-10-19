"""Local YOLOX + MediaPipe video extraction; no downloads and no inferred labels."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import importlib
import json
from pathlib import Path
import platform
import sys

from inference.experts.assets import file_sha256, verify_asset
from inference.perception.runtime import load_runtime_config, runtime_signature


def expert_packets(frames, detector, landmarks, config):
    from inference.perception.tracking import ObjectTracker
    from inference.perception.video_source import sample_frames
    tracker, last_time, cached = ObjectTracker(), None, None
    signature = runtime_signature(config)
    for slot, frame in sample_frames(frames):
        height, width = frame.bgr.shape[:2]
        roi = config['driver_roi']
        if roi[2] > width or roi[3] > height:
            raise ValueError('Configured driver ROI is outside the original decoded image')
        if last_time != frame.timestamp_ms:
            objects = detector.predict(frame.bgr, frame.timestamp_ms)
            face, hands = landmarks.predict(frame.bgr, frame.timestamp_ms)
            objects = deepcopy(objects)
            objects['items'] = tracker.update(objects['items'], frame.timestamp_ms)
            cached = {'objects': objects, 'face': deepcopy(face), 'hands': deepcopy(hands)}
            for expert in cached.values():
                expert['model_id'] += f':runtime-{signature}'
            last_time = frame.timestamp_ms
        yield dict(schema_version='driveguard-experts-v1', recording_id=config['recording_id'],
                   view=config['view'], frame_id=f'slot-{slot//40}', timestamp_ms=slot,
                   width=width, height=height, driver_roi=list(roi), **deepcopy(cached))


def run_video(input_path, output_dir, config, *, group_profile=None, frame_observer=None, feature_profile=None):
    config = deepcopy(config)
    gaze = config.get('schema_version') == 'driveguard-runtime-v2'
    if gaze != bool(feature_profile and feature_profile.get('feature_schema_version') == 'driveguard-interaction-v2'):
        raise ValueError('Gaze runtime and explicit new feature profile must be selected together')
    # Preflight all three local assets before importing either model runtime.
    for location, key in ((config['objects'], 'asset'), (config, 'face_asset'), (config, 'hands_asset')):
        location[key] = verify_asset(location[key])
    source = Path(input_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError('Input must be a local video file')
    if Path(output_dir).exists():
        raise FileExistsError('Output directory already exists; choose a new directory')
    initial_stat = source.stat()
    provenance = dict(kind='local-video-runtime', source_video=dict(path=str(source), sha256=file_sha256(source),
                      size_bytes=initial_stat.st_size), config=config, runtime_signature=runtime_signature(config),
                      validation='Runtime execution is not a model accuracy or accelerator qualification',
                      clock={}, versions={'python': platform.python_version()})
    from inference.experts.yolox_adapter import YOLOXAdapter
    from inference.experts.mediapipe_adapter import MediaPipeAdapter
    from inference.perception.video_source import decode_video
    from inference.perception.export import export_recording
    from inference.perception.profile import load_profile
    with ExitStack() as stack:
        remote = config['objects']['variant'] == 'yolo11s'
        if remote:
            from inference.experts.detector_client import RemoteDetector
            detector = RemoteDetector(**config['objects'])
        else:
            detector = YOLOXAdapter(**config['objects'])
        stack.callback(detector.close)
        gaze_options = {'gaze_features': True} if gaze else {}
        landmarks = MediaPipeAdapter(config['face_asset'], config['hands_asset'], driver_roi=config['driver_roi'],
                                     driver_region_confirmed_exclusive=config['driver_region_confirmed_exclusive'], **gaze_options)
        stack.callback(landmarks.close)
        if remote:
            provenance['detector_service_model_id'] = detector.model_id
        for name in ('numpy', 'torch', 'cv2', 'mediapipe', 'av') + (() if remote else ('yolox',)):
            provenance['versions'][name] = str(importlib.import_module(name).__version__)
        if config['objects']['backend'] == 'npu' and not remote:
            provenance['versions']['torch_npu'] = str(importlib.import_module('torch_npu').__version__)
        frames = decode_video(source, provenance['clock'])
        stack.callback(frames.close)
        def observed_frames():
            for frame in frames:
                if frame_observer is not None:
                    frame_observer(frame)
                yield frame
        def checked_packets():
            yield from expert_packets(observed_frames(), detector, landmarks, config)
            final_stat = source.stat()
            if (final_stat.st_size, final_stat.st_mtime_ns) != (initial_stat.st_size, initial_stat.st_mtime_ns):
                raise ValueError('Source video changed during extraction; do not publish this cache')
            # Resource cleanup is part of success, so finish it before export publishes record.json.
            stack.close()
        options = {'group_profile':group_profile} if group_profile is not None else {}
        return export_recording(checked_packets(), output_dir, feature_profile or load_profile(), provenance=provenance, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check-config', action='store_true', help='Structure only: no dependencies, weights, video or device access')
    parser.add_argument('--input', help='Single-stream local video; raw decoded orientation, not mirrored')
    parser.add_argument('--output-dir', help='New directory only')
    parser.add_argument('--group-profile',help='Explicit v2 groups JSON; absent preserves v1 export')
    args = parser.parse_args(argv)
    try:
        config = load_runtime_config(args.config)
        group_profile = None
        if args.group_profile:
            from training.common.config import read_json
            from training.relation.profile import FEATURE_NAMES, validate_group_profile
            group_profile = validate_group_profile(read_json(args.group_profile),FEATURE_NAMES)
        if args.check_config:
            print(json.dumps({'status': 'config_structure_valid', 'assets_verified': False,
                              'runtime_signature': runtime_signature(config)}))
            return 0
        if not args.input or not args.output_dir:
            parser.error('--input and --output-dir are required unless --check-config is used')
        options = {'group_profile':group_profile} if group_profile is not None else {}
        report = run_video(args.input, args.output_dir, config, **options)
        print(json.dumps({'status': report['status'], 'slots': report['slots'], 'output_dir': str(Path(args.output_dir).resolve())}))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        print(f'Video extraction failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
