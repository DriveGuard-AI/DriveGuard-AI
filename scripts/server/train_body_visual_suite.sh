#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
exec "$PYTHON_BIN" -u -m training.visual.pipeline \
  --manifest "${MANIFEST:-runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json}" \
  --data-root "${DATA_ROOT:-runs/dmd-full-yolo11-features-001}" \
  --dataset-root "${DATASET_ROOT:-Dataset}" \
  --checkpoint "${CHECKPOINT:-runs/body-dist-comparison-001/gap80/train-001/best.pt}" \
  --weights "${WEIGHTS:-models/local/resnet18-f37072fd.pth}" \
  --output-dir "${OUTPUT_DIR:-runs/body-visual-comparison-001}" \
  --backend npu --device-index "${DEVICE_INDEX:-0}" \
  --rgb-backend "${RGB_BACKEND:-cpu}" --rgb-device-index "${RGB_DEVICE_INDEX:-0}" \
  --rgb-batch-size "${RGB_BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-64}" --seed "${SEED:-42}" \
  "$@"
