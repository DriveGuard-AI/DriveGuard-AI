#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ $# -eq 1 && "$1" == --help ]]; then
  echo 'usage: bash scripts/server/export_live_composite.sh'
  echo 'Environment: PYTHON_BIN ALL_TASKS_REPORT RECOVERY_REPORT RGB_WEIGHTS RUNTIME_CONFIG BUNDLE_DIR'
  exit 0
fi
[[ $# -eq 0 ]] || { echo 'error: configure through environment variables' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-.venv-visual-npu/bin/python}"
ALL_TASKS_REPORT="${ALL_TASKS_REPORT:-runs/coverage25-composite-001/all-tasks/report.json}"
RECOVERY_REPORT="${RECOVERY_REPORT:-runs/body-fatigue-eye-recovery-001/comparison/comparison.json}"
RGB_WEIGHTS="${RGB_WEIGHTS:-models/local/resnet18-f37072fd.pth}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-runs/dmd-full-yolo11.local.runtime.json}"
BUNDLE_DIR="${BUNDLE_DIR:-models/releases/live-composite-v1}"
exec "$PYTHON_BIN" -u -m inference.live.bundle \
  --all-tasks-report "$ALL_TASKS_REPORT" --recovery-report "$RECOVERY_REPORT" \
  --rgb-weights "$RGB_WEIGHTS" --runtime-config "$RUNTIME_CONFIG" --output-dir "$BUNDLE_DIR"
