#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ $# -eq 1 && "$1" == --help ]]; then
  echo 'usage: bash scripts/server/start_live_composite.sh'
  echo 'Environment: PYTHON_BIN DETECTOR_PYTHON PERCEPTION_PYTHON BUNDLE_DIR SESSION_DIR PORT DETECTOR_PORT BACKEND DEVICE_INDEX'
  echo 'Loopback only. Use an SSH tunnel to open the camera page from your own computer.'
  exit 0
fi
[[ $# -eq 0 ]] || { echo 'error: configure through environment variables' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
DETECTOR_PYTHON="${DETECTOR_PYTHON:-.venv-yolo-npu/bin/python}"
PERCEPTION_PYTHON="${PERCEPTION_PYTHON:-.venv-perception-arm64/bin/python}"
BUNDLE_DIR="${BUNDLE_DIR:-models/releases/live-composite-v1}"
SESSION_DIR="${SESSION_DIR:-runs/live-composite-session-001}"
PORT="${PORT:-8765}"
DETECTOR_PORT="${DETECTOR_PORT:-8766}"
BACKEND="${BACKEND:-npu}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
for executable in "$PYTHON_BIN" "$DETECTOR_PYTHON" "$PERCEPTION_PYTHON"; do
  [[ -x "$executable" ]] || { echo "error: Missing interpreter: $executable" >&2; exit 2; }
done
[[ ! -e "$SESSION_DIR" ]] || { echo 'error: Use a NEW SESSION_DIR' >&2; exit 2; }
"$PYTHON_BIN" -m inference.live.bundle --check "$BUNDLE_DIR/bundle.json"
"$PERCEPTION_PYTHON" -c 'import cv2,mediapipe,numpy; from PIL import Image; import inference.live.perception_worker'
"$DETECTOR_PYTHON" -c 'import torch,ultralytics; import inference.experts.detector_service'
umask 077
"$PYTHON_BIN" - "$BUNDLE_DIR" "$SESSION_DIR" "$PORT" "$DETECTOR_PORT" <<'PY'
import json,secrets,sys
from pathlib import Path
from inference.perception.runtime import load_runtime_config
bundle,session=map(lambda p:Path(p).resolve(),sys.argv[1:3])
port,detector_port=map(int,sys.argv[3:])
if not (1024<=port<=65535 and 1024<=detector_port<=65535 and port!=detector_port):
    raise ValueError('Distinct ports in 1024..65535 required')
runtime=load_runtime_config(bundle/'runtime.json')
session.mkdir(parents=True,exist_ok=False)
token=secrets.token_urlsafe(32)
(session/'api.token').write_text(token,encoding='ascii')
(session/'detector.token').write_text(secrets.token_urlsafe(32),encoding='ascii')
runtime['objects']['service']=dict(host='127.0.0.1',port=detector_port,token_file=str(session/'detector.token'))
(session/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
(session/'open-url.txt').write_text(f'http://localhost:{port}/#token={token}\n',encoding='ascii')
PY
detector_pid=''
model_pid=''
cleanup() {
  if [[ -n "$model_pid" ]]; then kill "$model_pid" 2>/dev/null || true; wait "$model_pid" 2>/dev/null || true; fi
  if [[ -n "$detector_pid" ]]; then kill "$detector_pid" 2>/dev/null || true; wait "$detector_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$DETECTOR_PYTHON" -u -m inference.experts.detector_service \
  --config "$SESSION_DIR/runtime.json" --ready-file "$SESSION_DIR/detector-ready.json" \
  >"$SESSION_DIR/detector.log" 2>&1 &
detector_pid=$!
ready=0
for ((attempt=0; attempt<120; attempt++)); do
  kill -0 "$detector_pid" 2>/dev/null || { echo "error: Detector exited; see $SESSION_DIR/detector.log" >&2; exit 2; }
  if [[ -f "$SESSION_DIR/detector-ready.json" ]]; then ready=1; break; fi
  sleep 1
done
[[ "$ready" -eq 1 ]] || { echo 'error: Detector startup timed out' >&2; exit 2; }
echo "Detector ready. Loading composite models; model log: $SESSION_DIR/model.log"
echo "After model status=ready, open the URL in $SESSION_DIR/open-url.txt through your SSH tunnel."
"$PYTHON_BIN" -u -m inference.live.server --bundle "$BUNDLE_DIR/bundle.json" \
  --runtime-config "$SESSION_DIR/runtime.json" --perception-python "$PERCEPTION_PYTHON" \
  --token-file "$SESSION_DIR/api.token" --port "$PORT" --backend "$BACKEND" --device-index "$DEVICE_INDEX" \
  >"$SESSION_DIR/model.log" 2>&1 &
model_pid=$!
echo "launcher_pid=$$ model_pid=$model_pid detector_pid=$detector_pid"
wait "$model_pid"
