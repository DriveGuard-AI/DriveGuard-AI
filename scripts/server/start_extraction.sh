#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run on the Linux server only.' >&2; exit 2; }
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -x .venv-yolo-npu/bin/python && -x .venv-perception-arm64/bin/python ]] || { echo 'Run setup_yolo11.sh first; preserve the existing perception environment.' >&2; exit 2; }
mkdir -p runs
SESSION_DIR=$(mktemp -d runs/extraction-session-XXXXXXXX)
nohup /opt/buildtools/Python-3.11.4/bin/python3 -u -m training.dmd.server_runner \
  --session-dir "$SESSION_DIR" "$@" \
  > "$SESSION_DIR/supervisor.log" 2>&1 < /dev/null &
RUNNER_PID=$!
printf '%s\n' "$RUNNER_PID" > "$SESSION_DIR/launcher.pid"
printf 'Session: %s\nPID: %s\n' "$SESSION_DIR" "$RUNNER_PID"
printf 'Status: bash scripts/server/status_extraction.sh %q\n' "$SESSION_DIR"
printf 'Logs: tail -f %q/supervisor.log %q/extraction.log\n' "$SESSION_DIR" "$SESSION_DIR"
echo 'Started in background, not yet a readiness/success confirmation. Inspect status/logs.'
