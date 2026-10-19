#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
exec "$PYTHON_BIN" -u -m training.visual.head_adaptation \
  --stability-report "${STABILITY_REPORT:-runs/body-fatigue-head-stability-001/comparison.json}" \
  --variant "${VARIANT:-C_regularized-seed42-best}" \
  --dataset-root "${DATASET_ROOT:-Dataset}" \
  --weights "${WEIGHTS:-models/local/resnet18-f37072fd.pth}" \
  --output-dir "${OUTPUT_DIR:-runs/body-fatigue-head-adaptation-001}" \
  --backend npu --device-index "${DEVICE_INDEX:-0}" \
  --batch-size "${BATCH_SIZE:-2}" --accumulation-steps "${ACCUMULATION_STEPS:-8}" \
  --frame-chunk "${FRAME_CHUNK:-16}" --epochs "${EPOCHS:-10}" \
  --seeds 42 43 "$@"
