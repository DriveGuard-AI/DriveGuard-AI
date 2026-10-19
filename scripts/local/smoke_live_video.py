"""Feed a recorded video through the loopback live service on the 25 Hz grid.

The live service expects JPEG frames whose timestamps start at zero and increase
strictly. A recorded video is fed with a fixed step (default 40 ms = 25 Hz), so
the causal grid stays dense and every task can be available. This is offline
analysis, not real time: the wall clock is slower than the video timeline.

    python scripts/local/smoke_live_video.py --video D:\\data\\clip.mp4 --max-frames 400

Use --output to keep one JSON line per produced result.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_live import request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8765')
    parser.add_argument('--token-file', default='runs/live-local/api.token')
    parser.add_argument('--size', default='1280x720', help='session image_size, must stay <= 1920x1080')
    parser.add_argument('--roi', default='0.15,0.0,0.75,1.0', help='normalized driver_roi')
    parser.add_argument('--view', choices=('body', 'face'), default='body')
    parser.add_argument('--step-ms', type=int, default=40)
    parser.add_argument('--stride', type=int, default=1,
                        help='feed every Nth source frame; use with --step-ms to model a '
                             'lower capture rate without turning the clip into slow motion')
    parser.add_argument('--max-frames', type=int, default=0, help='0 means the whole video')
    parser.add_argument('--quality', type=int, default=85)
    parser.add_argument('--output', default='', help='optional JSONL file for produced results')
    parser.add_argument('--retries', type=int, default=2,
                        help='reopen the session and continue after a failed segment')
    args = parser.parse_args(argv)

    import cv2

    if args.step_ms <= 0 or args.stride <= 0:
        raise SystemExit('error: --step-ms and --stride must be positive')
    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f'error: video not found: {video}')
    width, height = (int(part) for part in args.size.lower().split('x'))
    roi = [float(part) for part in args.roi.split(',')]
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise SystemExit('error: --roi must be four normalized values x0,y0,x1,y1')
    token = Path(args.token_file).read_text(encoding='utf-8').strip()

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f'error: cannot open video: {video}')
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    camera = {'view': args.view, 'image_size': [width, height], 'driver_roi': roi,
              'driver_region_confirmed_exclusive': True}

    limit = args.max_frames if args.max_frames > 0 else source_frames
    stream = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        stream = Path(args.output).open('w', encoding='utf-8')

    # The service closes a session after any inference error and requires
    # timestamps to restart at zero, so a failed run continues in a new segment
    # instead of discarding the whole video.
    position = 0
    segment = 0
    failures = 0
    fed = 0
    results = []
    latencies = []
    print(json.dumps({'event': 'source', 'fps': round(source_fps, 3), 'frames': source_frames,
                      'size': source_size, 'session_size': [width, height], 'roi': roi,
                      'step_ms': args.step_ms}), flush=True)
    started = time.perf_counter()
    try:
        while position < limit:
            segment += 1
            segment_start = position
            capture.set(cv2.CAP_PROP_POS_FRAMES, segment_start)
            code, opened = request(args.base_url + '/api/live/sessions', token, 'POST', camera)
            if code not in (200, 201):
                raise SystemExit(f'error: session open failed: {code} {opened}')
            session = opened['session_id']
            emitted = 0
            print(json.dumps({'event': 'opened', 'segment': segment, 'session': session,
                              'start_frame': segment_start,
                              'elapsed_s': round(time.perf_counter() - started, 1)}), flush=True)
            try:
                while position < limit:
                    ok, frame = capture.read()
                    if not ok:
                        limit = position
                        break
                    position += 1
                    if (position - 1) % args.stride:
                        if position % 100 == 0:
                            print(json.dumps({'event': 'progress', 'frames': position,
                                              'elapsed_s': round(time.perf_counter() - started, 1),
                                              'results': len(results)}), flush=True)
                        continue
                    if (frame.shape[1], frame.shape[0]) != (width, height):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                    if not ok:
                        raise SystemExit('error: JPEG encode failed')
                    payload = {'timestamp_ms': emitted * args.step_ms,
                               'jpeg': base64.b64encode(encoded.tobytes()).decode('ascii')}
                    code, reply = request(f'{args.base_url}/api/live/sessions/{session}/frames', token,
                                          'POST', payload)
                    if code != 200:
                        raise RuntimeError(f'frame {position} failed: {code} {reply}')
                    latencies.append(reply['processing_ms'])
                    if reply['result'] is not None:
                        results.append(reply['result'])
                        if stream is not None:
                            stream.write(json.dumps({'segment': segment, 'frame': position - 1,
                                                     'result': reply['result']},
                                                    ensure_ascii=False) + '\n')
                    emitted += 1
                    fed += 1
                    if position % 100 == 0:
                        print(json.dumps({'event': 'progress', 'frames': position,
                                          'elapsed_s': round(time.perf_counter() - started, 1),
                                          'results': len(results)}), flush=True)
            except Exception as error:
                failures += 1
                print(json.dumps({'event': 'segment_failed', 'frame': position,
                                  'failures': failures, 'error': str(error)[:200]}), flush=True)
                if failures > args.retries:
                    print(json.dumps({'event': 'giving_up', 'frame': position}), flush=True)
                    break
                time.sleep(2)
            finally:
                request(f'{args.base_url}/api/live/sessions/{session}', token, 'DELETE')
    finally:
        if stream is not None:
            stream.close()
        capture.release()

    wall = time.perf_counter() - started
    timeline_ms = fed * args.step_ms
    summary = {'frames': position, 'fed': fed, 'results': len(results), 'segments': segment,
               'failed_segments': failures,
               'wall_s': round(wall, 1), 'timeline_s': round(timeline_ms / 1000, 1),
               'realtime_factor': round(timeline_ms / 1000 / wall, 2) if wall else None,
               'processing_ms': {'min': round(min(latencies)), 'p50': round(sorted(latencies)[len(latencies) // 2]),
                                 'max': round(max(latencies))} if latencies else None}
    if results:
        tasks = {}
        for name in results[-1]['tasks']:
            available = sum(1 for item in results if item['tasks'][name]['available'])
            detected = sum(1 for item in results if item['tasks'][name]['detected'])
            tasks[name] = {'available': available, 'detected': detected}
        gaze = {}
        for item in results:
            key = item['gaze'].get('class_name') or 'unavailable'
            gaze[key] = gaze.get(key, 0) + 1
        summary['tasks'] = tasks
        summary['gaze'] = gaze
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
