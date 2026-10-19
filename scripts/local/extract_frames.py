"""Extract a video into a JPEG sequence on a fixed grid, for offline app runs.

The application deliberately does not depend on a video decoder: it consumes a
directory of JPEGs produced here and posts them to the live service with
timestamps `index * step_ms`. Output naming is zero-padded so that a plain
lexicographic sort equals presentation order.

    python scripts/local/extract_frames.py --video D:\\data\\clip.mp4 --output runs/frames/clip
"""
import argparse
from pathlib import Path

import cv2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', default='1280x720', help='output frame size WxH')
    parser.add_argument('--step-ms', type=int, default=40, help='grid step, 40 ms = 25 Hz')
    parser.add_argument('--quality', type=int, default=85)
    parser.add_argument('--max-seconds', type=float, default=0.0, help='0 means the whole video')
    args = parser.parse_args(argv)

    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f'error: video not found: {video}')
    width, height = (int(part) for part in args.size.lower().split('x'))
    step_seconds = args.step_ms / 1000.0
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'error: output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f'error: cannot open video: {video}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    limit_seconds = args.max_seconds if args.max_seconds > 0 else float('inf')

    written = 0
    frame_index = 0
    next_emit = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            frame_index += 1
            if timestamp > limit_seconds:
                break
            if timestamp + 1e-9 < next_emit:
                continue
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            path = output / f'{written:06d}.jpg'
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]):
                raise SystemExit(f'error: failed to write {path}')
            written += 1
            next_emit = written * step_seconds
    finally:
        capture.release()

    print(f'frames={written} size={width}x{height} step_ms={args.step_ms} '
          f'source_fps={fps:.3f} timeline_s={written * step_seconds:.1f} output={output}')


if __name__ == '__main__':
    main()
