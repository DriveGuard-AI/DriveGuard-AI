#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# Extraction uses the existing CPU perception environment; training uses NPU.
PERCEPTION_PYTHON="${PERCEPTION_PYTHON:-.venv-perception-arm64/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
ACTION_REPORT="${ACTION_REPORT:-runs/body-fatigue-actions-001/comparison.json}"
VARIANT="${VARIANT:-B_action_supervision-seed42-best-event}"
DATASET_ROOT="${DATASET_ROOT:-Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/body-fatigue-eyes-001}"
resume=()
if [[ $# -eq 1 && "$1" == --continue-run ]]; then
  resume=(--continue-run)
elif [[ $# -ne 0 ]]; then
  echo 'usage: bash scripts/server/train_body_fatigue_eyes.sh [--continue-run]' >&2
  exit 2
fi
for executable in "$PERCEPTION_PYTHON" "$PYTHON_BIN"; do
  [[ -x "$executable" ]] || { echo "error: Missing Python executable: $executable" >&2; exit 2; }
done
[[ -f "$ACTION_REPORT" && -d "$DATASET_ROOT" ]] || {
  echo 'error: ACTION_REPORT or DATASET_ROOT is missing' >&2; exit 2;
}
if [[ -e "$OUTPUT_DIR" && ${#resume[@]} -eq 0 ]]; then
  echo 'error: Use a NEW OUTPUT_DIR, or --continue-run with unchanged settings' >&2
  exit 2
fi

# Fail before video decoding when either environment is incomplete.
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PERCEPTION_PYTHON" -c \
  'import av, mediapipe, numpy; from PIL import Image; import training.visual.eye_frames'
"$PYTHON_BIN" -c 'import torch, torch_npu; import training.visual.fatigue_eyes; assert torch.npu.is_available(), "NPU unavailable"'
if [[ -z "${HEAD_CACHE:-}" ]]; then
  HEAD_CACHE="$(env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PERCEPTION_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["plan"]["provenance"]["backbone_provenance"]["frame_cache"])' "$ACTION_REPORT")"
fi
[[ -f "$HEAD_CACHE/manifest.json" ]] || { echo "error: Head cache missing: $HEAD_CACHE" >&2; exit 2; }

env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PERCEPTION_PYTHON" -u -m training.visual.eye_frames \
  --head-cache "$HEAD_CACHE" --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR/native-eyes" "${resume[@]}"

exec "$PYTHON_BIN" -u -m training.visual.fatigue_eyes \
  --action-report "$ACTION_REPORT" --variant "$VARIANT" --dataset-root "$DATASET_ROOT" \
  --eye-cache "$OUTPUT_DIR/native-eyes" --output-dir "$OUTPUT_DIR/comparison" \
  --backend npu --device-index "${DEVICE_INDEX:-0}" \
  --batch-size "${BATCH_SIZE:-32}" --accumulation-steps "${ACCUMULATION_STEPS:-1}" \
  --frame-chunk "${FRAME_CHUNK:-16}" --epochs "${EPOCHS:-6}" \
  --early-stopping-patience "${PATIENCE:-7}" --seeds 42 43 "${resume[@]}"
