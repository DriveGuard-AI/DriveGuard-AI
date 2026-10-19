"""Break down one live perception frame inside the perception environment.

Instantiates the real ``CameraPerception`` (remote detector + MediaPipe + eye
locator + feature extractor) and times every collaborator without touching
product code. The detector figure is exactly what the asynchronous detector
removes from the critical path, so this is the measurement that justifies (or
refutes) pipelining it.

    .venv-perception\\Scripts\\python.exe scripts/local/profile_perception.py 60

Requires a running detector service and a prepared development bundle.
"""
import base64
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import inference.live.perception_worker as module
from inference.live.perception_worker import CameraPerception

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 60
FRAMES = Path(__import__('os').environ.get('DGB_PROFILE_FRAMES', ROOT / 'runs/frames/electr1'))
BUNDLE = Path(__import__('os').environ.get('DGB_PROFILE_BUNDLE', ROOT / 'runs/live-local/bundle'))

OPTIONS = {"view": "body", "image_size": [1280, 720],
           "driver_roi": [0.15, 0.0, 0.75, 1.0],
           "driver_region_confirmed_exclusive": True}
HEAD_CROP = {"version": "driver-relative-rect-v1", "xyxy": [0.05, 0.0, 0.65, 0.65]}


def main():
    payloads = [base64.b64encode(p.read_bytes()).decode("ascii")
                for p in sorted(FRAMES.glob("*.jpg"))[:COUNT]]
    if not payloads:
        raise SystemExit(f"no JPEG frames under {FRAMES}")
    print(f"frames={len(payloads)} bundle={BUNDLE}")

    phases = {name: [] for name in
              ("decode", "detector", "landmarks", "eye", "features", "tracker")}

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                phases[name].append((time.perf_counter() - started) * 1000)
        return wrapper

    worker = CameraPerception(str(BUNDLE / "runtime.json"), OPTIONS, HEAD_CROP)
    original_decode = module.decode_jpeg
    module.decode_jpeg = timed("decode", original_decode)
    worker.detector.predict = timed("detector", worker.detector.predict)
    worker.landmarks.predict = timed("landmarks", worker.landmarks.predict)
    if worker.locator is not None:
        worker.locator.step = timed("eye", worker.locator.step)
    worker.extractor.step = timed("features", worker.extractor.step)
    worker.tracker.update = timed("tracker", worker.tracker.update)
    try:
        totals = []
        for index, payload in enumerate(payloads):
            started = time.perf_counter()
            worker.frame(index * 40, payload)
            if index:
                totals.append((time.perf_counter() - started) * 1000)

        print(f"\nfull frame()  n={len(totals)}  median={statistics.median(totals):7.1f}ms  "
              f"min={min(totals):7.1f}  max={max(totals):7.1f}")
        for name, values in phases.items():
            if values:
                ordered = sorted(values)
                print(f"  {name:10s} n={len(values):3d}  median={statistics.median(values):7.2f}ms  "
                      f"p90={ordered[int(len(ordered) * 0.9)]:7.2f}  max={max(values):7.2f}")
        accounted = sum(statistics.median(v) for v in phases.values() if v)
        full = statistics.median(totals)
        print(f"  {'accounted':10s}      median={accounted:7.2f}ms   "
              f"unaccounted={full - accounted:6.2f}ms")
        detector = statistics.median(phases["detector"]) if phases["detector"] else 0.0
        print(f"\nan asynchronous detector leaves a {full - detector:.1f}ms critical path "
              f"({1000 / (full - detector):.1f} Hz) instead of {full:.1f}ms ({1000 / full:.1f} Hz)")
    finally:
        module.decode_jpeg = original_decode
        worker.close()


if __name__ == '__main__':
    main()
