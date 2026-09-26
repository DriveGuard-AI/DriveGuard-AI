#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
exec "$PYTHON_BIN" -u -m training.visual.eye_diagnostics \
  --action-report "${ACTION_REPORT:-runs/body-fatigue-actions-001/comparison.json}" \
  --variant "${VARIANT:-B_action_supervision-seed42-best-event}" \
  --dataset-root "${DATASET_ROOT:-Dataset}" \
  --output-dir "${OUTPUT_DIR:-runs/body-fatigue-eye-diagnostics-001}" \
  --backend npu --device-index "${DEVICE_INDEX:-0}" "$@"
