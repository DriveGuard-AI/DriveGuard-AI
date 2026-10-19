#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-perception-arm64/bin/python}"
DIAGNOSTIC_SUMMARY="${DIAGNOSTIC_SUMMARY:-runs/coverage25-composite-001/fatigue-coverage25/summary.json}"
DATASET_ROOT="${DATASET_ROOT:-Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/body-fatigue-recovery-001}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/perception/face_experiments.json}"
# Existing gap80/coverage25 training configs use 100 slots with stride 25.
# Older diagnostic summaries omit these; the CLI deliberately requires them.
WINDOW_STEPS="${WINDOW_STEPS:-100}"
STRIDE_STEPS="${STRIDE_STEPS:-25}"
if [[ ! -x "$PYTHON_BIN" || ! -f "$DIAGNOSTIC_SUMMARY" || ! -d "$DATASET_ROOT" ]]; then
  echo 'error: Check PYTHON_BIN (perception environment), DIAGNOSTIC_SUMMARY and DATASET_ROOT.' >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "error: Use a NEW OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 2
fi
echo "Fatigue evidence recovery: $EXPERIMENT_CONFIG"
echo "Body validation, all subsets; face inference on CPU; no training. Output: $OUTPUT_DIR"
exec env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$PYTHON_BIN" -u -m training.dmd.fatigue_recovery \
  --diagnostic-summary "$DIAGNOSTIC_SUMMARY" --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" --experiment-config "$EXPERIMENT_CONFIG" \
  --window-steps "$WINDOW_STEPS" --stride-steps "$STRIDE_STEPS" "$@"
