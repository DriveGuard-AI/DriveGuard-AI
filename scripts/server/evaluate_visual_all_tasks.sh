#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
exec "$PYTHON_BIN" -u -m training.visual.all_tasks \
  --prepared-dir "${PREPARED_DIR:-runs/body-visual-comparison-001/prepared-001}" \
  --stability-report "${STABILITY_REPORT:-runs/body-visual-stability-001/comparison.json}" \
  --output-dir "${OUTPUT_DIR:-runs/visual-all-tasks-001}" \
  --backend "${BACKEND:-npu}" --device-index "${DEVICE_INDEX:-0}" \
  --batch-size "${BATCH_SIZE:-64}" \
  "$@"
