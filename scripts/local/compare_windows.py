"""Compare two live-video runs window by window.

The live service republishes the most recent window result on every frame, so a
one second window appears 12-25 times in the JSONL. The distinct comparison unit
is therefore ``window_end_ms``, not the row. This is how a pipeline change is
judged: availability, detection agreement and score deltas per task.

    python scripts/local/compare_windows.py runs/live-local/a.jsonl runs/live-local/b.jsonl
"""
import json
import sys
from pathlib import Path

TASKS = ("fatigue", "phone_use", "drinking", "distraction")


def load(path):
    """Deduplicate to one entry per window; the last occurrence wins."""
    windows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        windows[row["result"]["window_end_ms"]] = row["result"]
    return windows


def show(value):
    return value if value is None else round(value, 3)


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: compare_windows.py BASE.jsonl CANDIDATE.jsonl")
    baseline, candidate = load(argv[0]), load(argv[1])
    common = sorted(set(baseline) & set(candidate))
    print(f"baseline  {Path(argv[0]).name}: {len(baseline)} distinct windows")
    print(f"candidate {Path(argv[1]).name}: {len(candidate)} distinct windows")
    print(f"common windows: {len(common)}\n")
    if not common:
        raise SystemExit("no overlapping windows")

    def rate(rows, task, key):
        return sum(1 for f in common if rows[f]["tasks"][task][key]) / len(common) * 100

    print(f"{'task':13s} {'avail B':>8s} {'avail C':>8s} {'detect B':>9s} {'detect C':>9s} {'agree':>9s}")
    for task in TASKS:
        agree = sum(1 for f in common
                    if baseline[f]["tasks"][task]["detected"] == candidate[f]["tasks"][task]["detected"])
        print(f"{task:13s} {rate(baseline, task, 'available'):7.1f}% "
              f"{rate(candidate, task, 'available'):7.1f}% "
              f"{rate(baseline, task, 'detected'):8.1f}% {rate(candidate, task, 'detected'):8.1f}% "
              f"{f'{agree}/{len(common)}':>9s}")

    print(f"\n{'task':13s} {'score d med':>12s} {'score d max':>12s} "
          f"{'mean B':>9s} {'mean C':>9s} {'n':>4s}")
    for task in TASKS:
        pairs = [(baseline[f]["tasks"][task]["score"], candidate[f]["tasks"][task]["score"])
                 for f in common
                 if baseline[f]["tasks"][task]["score"] is not None
                 and candidate[f]["tasks"][task]["score"] is not None]
        if not pairs:
            print(f"{task:13s} {'-':>12s} {'-':>12s} {'-':>9s} {'-':>9s} {0:4d}")
            continue
        diffs = sorted(abs(b - c) for b, c in pairs)
        mean_b = sum(b for b, _ in pairs) / len(pairs)
        mean_c = sum(c for _, c in pairs) / len(pairs)
        print(f"{task:13s} {diffs[len(diffs) // 2]:12.4f} {diffs[-1]:12.4f} "
              f"{mean_b:9.4f} {mean_c:9.4f} {len(pairs):4d}")

    gaze_agree = sum(1 for f in common
                     if baseline[f]["gaze"].get("class_name") == candidate[f]["gaze"].get("class_name"))
    gaze_b = sum(1 for f in common if baseline[f]["gaze"]["available"]) / len(common) * 100
    gaze_c = sum(1 for f in common if candidate[f]["gaze"]["available"]) / len(common) * 100
    print(f"\ngaze available {gaze_b:.1f}% -> {gaze_c:.1f}%, "
          f"class agreement {gaze_agree}/{len(common)} ({gaze_agree / len(common) * 100:.1f}%)")

    print("\nwindows where the detected task set differs:")
    shown = 0
    for f in common:
        changed = [t for t in TASKS
                   if baseline[f]["tasks"][t]["detected"] != candidate[f]["tasks"][t]["detected"]]
        if changed:
            shown += 1
            detail = ", ".join(
                f"{t} {show(baseline[f]['tasks'][t]['score'])}->{show(candidate[f]['tasks'][t]['score'])}"
                for t in changed)
            print(f"  {f:6d} ms  {detail}")
    print(f"  total {shown}/{len(common)} windows differ")


if __name__ == '__main__':
    main(sys.argv[1:])
