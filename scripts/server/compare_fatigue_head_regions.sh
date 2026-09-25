#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 || ! -f "$1" ]]; then
  echo 'usage: bash scripts/server/compare_fatigue_head_regions.sh <head-region-config.json> [replay arguments]' >&2
  exit 2
fi
export EXPERIMENT_CONFIG="$1"
shift
export OUTPUT_DIR="${OUTPUT_DIR:-runs/body-fatigue-head-regions-001}"
echo 'Scoped head-region comparison: baseline, full-frame gate, and head crop. No training.'
exec bash scripts/server/recover_body_fatigue.sh "$@"
