#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
# Six matched epochs by default; patience exceeds budget so auxiliary and
# control see the same number of passes. Both best and last remain reportable.
exec "$PYTHON_BIN" -u -m training.visual.fatigue_actions \
  --adaptation-report "${ADAPTATION_REPORT:-runs/body-fatigue-head-adaptation-001/comparison.json}" \
  --variant "${VARIANT:-D_tune32-seed43-best}" \
  --dataset-root "${DATASET_ROOT:-Dataset}" \
  --output-dir "${OUTPUT_DIR:-runs/body-fatigue-actions-001}" \
  --backend npu --device-index "${DEVICE_INDEX:-0}" \
  --batch-size "${BATCH_SIZE:-2}" --accumulation-steps "${ACCUMULATION_STEPS:-8}" \
  --frame-chunk "${FRAME_CHUNK:-16}" --epochs "${EPOCHS:-6}" \
  --early-stopping-patience "${PATIENCE:-7}" --seeds 42 43 "$@"
