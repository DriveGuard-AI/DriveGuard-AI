#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-/opt/buildtools/Python-3.11.4/bin/python3}"

exec "$PYTHON_BIN" -m training.gaze.body_dist_suite \
  --backend npu \
  --manifest "${MANIFEST:-runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json}" \
  --data-root "${DATA_ROOT:-runs/dmd-full-yolo11-features-001}" \
  --epochs "${EPOCHS:-20}" \
  --output-dir "${OUTPUT_DIR:-runs/body-dist-comparison-001}" \
  "$@"
