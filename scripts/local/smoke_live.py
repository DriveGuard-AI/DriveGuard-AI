"""Drive a synthetic camera session against the loopback live service.

The camera page needs a real webcam, so this script exercises the same HTTP
contract with generated JPEG frames. It proves the transport, perception
subprocess, detector service and model window all run together; it does NOT
prove recognition quality, because the frames contain no face.

Frames are posted at a fixed wall-clock interval so the server sees roughly the
frame rate a browser would produce. A synthetic frame is a gradient plus a
rectangle, which MediaPipe cannot turn into landmarks, so unavailable tasks are
the expected outcome.
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def request(url, token, method='GET', payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    http = urllib.request.Request(url, data=data, method=method)
    http.add_header('Authorization', 'Bearer ' + token)
    if data is not None:
        http.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(http, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def synthetic_jpeg(width, height, index):
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(0, height, 8):
        level = 40 + (y * 60) // max(1, height)
        draw.rectangle([0, y, width, y + 7], fill=(level, level + 20, level + 40))
    offset = (index * 7) % max(1, width // 3)
    draw.rectangle([offset, height // 4, offset + width // 3, height // 2], fill=(198, 154, 128))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8765')
    parser.add_argument('--token-file', default='runs/live-local/api.token')
    parser.add_argument('--frames', type=int, default=160)
    parser.add_argument('--interval-ms', type=int, default=40)
    parser.add_argument('--size', default='640x480')
    parser.add_argument('--view', choices=('body', 'face'), default='body')
    args = parser.parse_args(argv)

    token = Path(args.token_file).read_text(encoding='utf-8').strip()
    width, height = (int(part) for part in args.size.lower().split('x'))
    camera = {'view': args.view, 'image_size': [width, height],
              'driver_roi': [0.15, 0.05, 0.85, 0.95],
              'driver_region_confirmed_exclusive': True}

    code, opened = request(args.base_url + '/api/live/sessions', token, 'POST', camera)
    if code not in (200, 201):
        raise SystemExit(f'error: session open failed: {code} {opened}')
    session = opened['session_id']
    print(json.dumps({'event': 'opened', **opened}, sort_keys=True))

    seen = []
    result = None
    try:
        for index in range(args.frames):
            frame = {'timestamp_ms': index * args.interval_ms,
                     'jpeg': synthetic_jpeg(width, height, index)}
            code, reply = request(f'{args.base_url}/api/live/sessions/{session}/frames', token,
                                  'POST', frame)
            if code != 200:
                raise SystemExit(f'error: frame {index} failed: {code} {reply}')
            state = reply['status']
            if not seen or seen[-1] != state:
                seen.append(state)
                print(json.dumps({'event': 'status', 'frame': index, 'status': state,
                                  'slots': reply['feature_slots'],
                                  'perception': reply['perception'],
                                  'processing_ms': round(reply['processing_ms'], 1)},
                                 sort_keys=True))
            if reply['result'] is not None:
                result = reply['result']
            time.sleep(args.interval_ms / 1000)
    finally:
        code, stopped = request(f'{args.base_url}/api/live/sessions/{session}', token, 'DELETE')
        print(json.dumps({'event': 'stopped', 'code': code, 'reply': stopped}, sort_keys=True))

    summary = {'statuses': seen, 'produced_result': result is not None}
    if result is not None:
        summary['tasks'] = {name: {'available': value['available'], 'detected': value['detected']}
                            for name, value in result['tasks'].items()}
        summary['window'] = [result.get('window_start_ms'), result.get('window_end_ms')]
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
