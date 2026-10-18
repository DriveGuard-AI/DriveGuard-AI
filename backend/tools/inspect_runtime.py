"""Inspect the running backend instance.

Prints: active monitoring session, snapshot sequence, effective snapshot rate,
current risk state, and the position inside the scripted scenario loop.

Usage (from backend/):
    .venv\\Scripts\\python.exe tools\\inspect_runtime.py --token dev-token
    .venv\\Scripts\\python.exe tools\\inspect_runtime.py --scenario fatigue
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.inference.scenarios import get_scenario  # noqa: E402

DEFAULT_DB = BACKEND_DIR / "var" / "driveguard.db"


def fetch_json(url: str, token: str | None) -> dict:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect DriveGuard AI runtime state")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="dev-token")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--scenario", default=None, help="override scenario name for timeline display")
    args = parser.parse_args()

    print(f"== service {args.base_url} ==")
    try:
        health = fetch_json(f"{args.base_url}/api/v1/health", None)
        status = fetch_json(f"{args.base_url}/api/v1/system/status", args.token)
        snapshot = fetch_json(f"{args.base_url}/api/v1/monitoring/snapshot", args.token)
    except Exception as exc:  # noqa: BLE001
        print(f"  request failed: {exc}")
        print("  hint: start the backend, or pass --token with the configured DGB_ADMIN_TOKEN")
        return 1

    pipeline = status["pipeline"]
    fps = float(pipeline.get("processed_fps") or 0.0)
    sequence = int(snapshot["sequence"])
    # `sequence` counts snapshots produced by THIS process, so it resets on restart.
    # Deriving uptime from it is the only in-band way without adding API fields.
    process_uptime = (sequence / fps) if fps > 0 else 0.0

    print(f"  service_version : {health['service_version']}")
    print(f"  inference_mode  : {pipeline['inference_mode']}  (degraded={pipeline['degraded']})")
    print(f"  pipeline state  : {pipeline['state']}")
    print(f"  measured fps    : {fps}   <- from /system/status, this drives how fast `sequence` grows")
    print(f"  ws clients      : {status['websocket']['clients']}")

    print("\n== risk snapshot counter ==")
    print(f"  session_id      : {snapshot['session_id']}")
    print(f"  snapshot seq    : {sequence}")
    print(f"  process uptime  : ~{process_uptime:.0f} s  (estimated as sequence / fps)")
    if process_uptime > 0:
        print(f"  => sequence +{fps:.0f} per second, i.e. one increment every {1000 / fps:.0f} ms")

    try:
        connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        row = connection.execute(
            "SELECT started_at, model_bundle_version FROM monitoring_sessions "
            "WHERE status = 'active' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        connection.close()
        if row:
            print(f"\n  session started : {row[0]}  bundle={row[1]}")
            print("                    NOTE: the active session is recovered across restarts (NFR-002),")
            print("                    so it can be much older than this process.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  (cannot read {args.db}: {exc})")

    scenario_name = args.scenario or "default"
    scenario = get_scenario(scenario_name)
    print(f"\n== scripted scenario '{scenario.name}' (loop {scenario.loop_seconds:.0f} s) ==")
    print(f"  {scenario.description}")
    for index, segment in enumerate(scenario.segments, start=1):
        print(
            f"    {index}. {segment.name:<10} {segment.duration:>4.1f}s  "
            f"level={segment.risk_level:<7} risk={str(segment.primary_risk):<10} quality={segment.quality}"
        )

    if process_uptime > 0:
        offset = process_uptime % scenario.loop_seconds
        segment, progress = scenario.segment_at(process_uptime)
        print(
            f"\n  loop position   : {offset:.1f}s / {scenario.loop_seconds:.0f}s "
            f"-> segment '{segment.name}' at {progress * 100:.0f}%"
        )

    print("\n== current snapshot ==")
    print(f"  captured_at     : {snapshot['captured_at']}")
    print(f"  risk_level      : {snapshot['risk_level']}")
    print(f"  primary_risk    : {snapshot['primary_risk']}")
    print(f"  confidence      : {snapshot['confidence']}")
    print(f"  quality         : {snapshot['quality']}")
    print(f"  reason_codes    : {snapshot['reason_codes']}")
    print("  NOTE: values come from app/inference/fake.py (scripted replay), not a real model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
