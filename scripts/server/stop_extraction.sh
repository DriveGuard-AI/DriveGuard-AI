#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
SESSION_DIR=${1:?Provide the exact session directory printed by start_extraction.sh}
[[ -d "$SESSION_DIR" && -f "$SESSION_DIR/owner.json" ]] || { echo 'Not an owned extraction session.' >&2; exit 2; }
# No PID kill, pattern kill, lock removal, or cache deletion. The supervisor owns shutdown.
touch -- "$SESSION_DIR/STOP"
echo 'Stop requested: no new videos will be scheduled; active videos drain before exit.'
echo 'Wait for stopped/failed/complete status and process exit before restarting with --resume.'
