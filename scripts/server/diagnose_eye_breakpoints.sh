#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 1 && "$1" == --help ]]; then
  echo 'usage: bash scripts/server/diagnose_eye_breakpoints.sh [--continue-run]'
  echo 'Reproduce fixed dense_track inference, then verify source locator stop traces. Zero training.'
  exit 0
fi
resume=()
if [[ $# -eq 1 && "$1" == --continue-run ]]; then
  resume=(--continue-run)
elif [[ $# -ne 0 ]]; then
  echo 'usage: bash scripts/server/diagnose_eye_breakpoints.sh [--continue-run]' >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PERCEPTION_PYTHON="${PERCEPTION_PYTHON:-.venv-perception-arm64/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
RECOVERY_REPORT="${RECOVERY_REPORT:-runs/body-fatigue-eye-recovery-001/comparison/comparison.json}"
DATASET_ROOT="${DATASET_ROOT:-Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/body-fatigue-eye-breakpoints-001}"
for executable in "$PERCEPTION_PYTHON" "$PYTHON_BIN"; do
  [[ -x "$executable" ]] || { echo "error: Missing Python executable: $executable" >&2; exit 2; }
done
[[ -f "$RECOVERY_REPORT" && -d "$DATASET_ROOT" ]] || { echo 'error: RECOVERY_REPORT or DATASET_ROOT missing' >&2; exit 2; }
if [[ -e "$OUTPUT_DIR" && ${#resume[@]} -eq 0 ]]; then
  echo 'error: Use a NEW OUTPUT_DIR or --continue-run with unchanged settings' >&2
  exit 2
fi
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PERCEPTION_PYTHON" -c \
  'import av, cv2, mediapipe, numpy; from PIL import Image; import training.visual.eye_tracking_replay'
"$PYTHON_BIN" -c 'import torch, torch_npu; import training.visual.eye_breakpoints; assert torch.npu.is_available(), "NPU unavailable"'
echo 'Stage 1/2: reproduce all validation windows and export eye breakpoint playback'
"$PYTHON_BIN" -u -m training.visual.eye_breakpoints \
  --recovery-report "$RECOVERY_REPORT" --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR/analysis" --backend npu --device-index "${DEVICE_INDEX:-0}" "${resume[@]}"
echo 'Stage 2/2: replay requested recordings from the start and verify old cache pixels'
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PERCEPTION_PYTHON" -u -m training.visual.eye_tracking_replay \
  --request "$OUTPUT_DIR/analysis/replay-request.json" --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR/source-replay" "${resume[@]}"
echo "Done: $OUTPUT_DIR/analysis/summary.json and $OUTPUT_DIR/source-replay/summary.json"
