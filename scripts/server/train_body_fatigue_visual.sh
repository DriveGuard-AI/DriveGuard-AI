#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
exec "$PYTHON_BIN" -u -m training.visual.fatigue \
  --reference-prepared-dir "${PREPARED_DIR:-runs/body-visual-comparison-001/prepared-001}" \
  --output-dir "${OUTPUT_DIR:-runs/body-fatigue-visual-001}" \
  --backend npu --device-index "${DEVICE_INDEX:-0}" \
  --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-64}" \
  --patience "${PATIENCE:-4}" --seeds 42 43 "$@"
