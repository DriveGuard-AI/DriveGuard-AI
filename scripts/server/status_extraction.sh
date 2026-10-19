#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
SESSION_DIR=${1:?Provide the session directory printed by start_extraction.sh}
if [[ -f "$SESSION_DIR/owner.json" ]]; then
  /opt/buildtools/Python-3.11.4/bin/python3 -S -m training.dmd.server_runner status "$SESSION_DIR"
else
  echo 'No status yet; inspect supervisor.log (startup may have failed).'
fi
[[ ! -f "$SESSION_DIR/supervisor.log" ]] || tail -n 15 -- "$SESSION_DIR/supervisor.log"
