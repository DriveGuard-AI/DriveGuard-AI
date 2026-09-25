#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
exec "$PYTHON_BIN" -u -m training.visual.fatigue_head_stability \
  --spatial-report "${SPATIAL_REPORT:-runs/body-fatigue-spatial-001/comparison/comparison.json}" \
  --output-dir "${OUTPUT_DIR:-runs/body-fatigue-head-stability-001}" \
  --backend npu --device-index "${DEVICE_INDEX:-0}" --seeds 42 43 "$@"
