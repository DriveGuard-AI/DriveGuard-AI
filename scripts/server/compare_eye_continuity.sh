#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PERCEPTION_PYTHON="${PERCEPTION_PYTHON:-.venv-perception-arm64/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
EYE_REPORT="${EYE_REPORT:-runs/body-fatigue-eyes-001/comparison/comparison.json}"
VARIANT="${VARIANT:-B_eye_native-seed42-best-combined}"
DATASET_ROOT="${DATASET_ROOT:-Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/body-fatigue-eye-recovery-001}"
resume=()
if [[ $# -eq 1 && "$1" == --continue-run ]]; then
  resume=(--continue-run)
elif [[ $# -eq 1 && "$1" == --help ]]; then
  echo 'usage: bash scripts/server/compare_eye_continuity.sh [--continue-run]'
  echo 'Build three validation eye caches, then compare fixed eye/yawn weights. Zero training.'
  exit 0
elif [[ $# -ne 0 ]]; then
  echo 'usage: bash scripts/server/compare_eye_continuity.sh [--continue-run]' >&2
  exit 2
fi
for executable in "$PERCEPTION_PYTHON" "$PYTHON_BIN"; do
  [[ -x "$executable" ]] || { echo "error: Missing Python executable: $executable" >&2; exit 2; }
done
[[ "$VARIANT" == B_eye_native-seed42-best-combined ]] || {
  echo 'error: This comparison fixes B_eye_native-seed42-best-combined' >&2; exit 2;
}
[[ -f "$EYE_REPORT" && -d "$DATASET_ROOT" ]] || { echo 'error: EYE_REPORT or DATASET_ROOT missing' >&2; exit 2; }
if [[ -e "$OUTPUT_DIR" && ${#resume[@]} -eq 0 ]]; then
  echo 'error: Use a NEW OUTPUT_DIR or --continue-run with unchanged settings' >&2
  exit 2
fi
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PERCEPTION_PYTHON" -c \
  'import av, cv2, mediapipe, numpy; from PIL import Image; import training.visual.eye_frames'
"$PYTHON_BIN" -c 'import torch, torch_npu; import training.visual.eye_recovery; assert torch.npu.is_available(), "NPU unavailable"'
HEAD_CACHE="$(env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PERCEPTION_PYTHON" -c \
  'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r["schema_version"]=="driveguard-independent-eye-comparison-v1" and r["status"]=="complete"; assert len([c for c in r["comparisons"] if c["variant"]==sys.argv[2]])==1; print(r["plan"]["provenance"]["head_cache"])' "$EYE_REPORT" "$VARIANT")"
for mode in sampled8 dense dense_track; do
  echo "Eye continuity cache: $mode (validation records only)"
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PERCEPTION_PYTHON" -u -m training.visual.eye_frames \
    --head-cache "$HEAD_CACHE" --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR/eyes/$mode" --recovery "$mode" --split val "${resume[@]}"
done
exec "$PYTHON_BIN" -u -m training.visual.eye_recovery \
  --eye-report "$EYE_REPORT" --variant "$VARIANT" --dataset-root "$DATASET_ROOT" \
  --recovery-root "$OUTPUT_DIR/eyes" --output-dir "$OUTPUT_DIR/comparison" \
  --backend npu --device-index "${DEVICE_INDEX:-0}" "${resume[@]}"
