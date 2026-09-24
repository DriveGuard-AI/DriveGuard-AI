#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
PREPARED_DIR="${PREPARED_DIR:-runs/body-visual-comparison-001/prepared-001}"
STABILITY_REPORT="${STABILITY_REPORT:-runs/body-visual-stability-001/comparison.json}"
OUTPUT_BASE_CHECKPOINT="${OUTPUT_BASE_CHECKPOINT:-runs/relation-gaze-head-coverage25-npu-001/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/coverage25-composite-001}"
BACKEND="${BACKEND:-npu}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
if [[ $# -ne 0 ]]; then
  echo 'Use environment variables to configure this script; no positional arguments.' >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "error: Use a NEW OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 2
fi
read_prepared_field() {
  "$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; print(json.loads((Path(sys.argv[1])/"prepared.json").read_text(encoding="utf-8"))[sys.argv[2]])' "$PREPARED_DIR" "$1"
}
GAP80_CHECKPOINT="$(read_prepared_field baseline_checkpoint)"
EVALUATION_MANIFEST="$(read_prepared_field manifest)"
EVALUATION_DATA_ROOT="$(read_prepared_field data_root)"
if [[ ! -f "$OUTPUT_BASE_CHECKPOINT" || ! -f "$GAP80_CHECKPOINT" || ! -f "$STABILITY_REPORT" ]]; then
  echo 'error: Missing coverage25/gap80 checkpoint or stability report.' >&2
  exit 2
fi
mkdir -p -- "$OUTPUT_DIR"
echo '[1/3] Full-task comparison: two output bases, frozen C_fusion context'
"$PYTHON_BIN" -u -m training.visual.all_tasks \
  --prepared-dir "$PREPARED_DIR" --stability-report "$STABILITY_REPORT" \
  --output-base-checkpoint "$OUTPUT_BASE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/all-tasks" \
  --backend "$BACKEND" --device-index "$DEVICE_INDEX" --batch-size "$BATCH_SIZE"
echo '[2/3] Gap80 fatigue evidence audit'
"$PYTHON_BIN" -u -m training.gaze.fatigue_diagnostics \
  --checkpoint "$GAP80_CHECKPOINT" --output-dir "$OUTPUT_DIR/fatigue-gap80" \
  --manifest "$EVALUATION_MANIFEST" --data-root "$EVALUATION_DATA_ROOT" \
  --backend "$BACKEND" --device-index "$DEVICE_INDEX" --batch-size "$BATCH_SIZE"
echo '[3/3] Coverage25 fatigue evidence audit'
"$PYTHON_BIN" -u -m training.gaze.fatigue_diagnostics \
  --checkpoint "$OUTPUT_BASE_CHECKPOINT" --output-dir "$OUTPUT_DIR/fatigue-coverage25" \
  --manifest "$EVALUATION_MANIFEST" --data-root "$EVALUATION_DATA_ROOT" \
  --backend "$BACKEND" --device-index "$DEVICE_INDEX" --batch-size "$BATCH_SIZE"
echo "Complete: $OUTPUT_DIR"
