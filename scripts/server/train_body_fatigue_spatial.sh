#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
exec "$PYTHON_BIN" -u -m training.visual.fatigue_spatial_pipeline \
  --fatigue-report "${FATIGUE_REPORT:-runs/body-fatigue-visual-001/comparison.json}" \
  --dataset-root "${DATASET_ROOT:-Dataset}" \
  --weights "${WEIGHTS:-models/local/resnet18-f37072fd.pth}" \
  --output-dir "${OUTPUT_DIR:-runs/body-fatigue-spatial-001}" \
  --backend npu --device-index "${DEVICE_INDEX:-0}" \
  --rgb-backend "${RGB_BACKEND:-npu}" --rgb-device-index "${RGB_DEVICE_INDEX:-0}" \
  --rgb-batch-size "${RGB_BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-64}" \
  --patience "${PATIENCE:-4}" --seeds 42 43 "$@"
