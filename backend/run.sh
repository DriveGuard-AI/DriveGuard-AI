#!/usr/bin/env bash
# DriveGuard AI backend launcher (macOS / Linux / Git Bash).
#
#   ./run.sh                     # default port 8000
#   DGB_PORT=8123 ./run.sh       # custom port

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if [ ! -x ".venv/bin/python" ]; then
  echo "[setup] creating .venv with ${PYTHON_BIN} ..."
  "${PYTHON_BIN}" -m venv .venv
fi

if [ ! -f ".venv/.deps-installed" ]; then
  echo "[setup] installing dependencies (first run may take a while) ..."
  ./.venv/bin/python -m pip install --upgrade pip --quiet
  ./.venv/bin/python -m pip install -r requirements-dev.txt --quiet
  touch .venv/.deps-installed
fi

export DGB_ADMIN_TOKEN="${DGB_ADMIN_TOKEN:-dev-token}"
PORT="${DGB_PORT:-8000}"

echo "[run] dashboard http://127.0.0.1:${PORT}/    openapi http://127.0.0.1:${PORT}/docs"
exec ./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"
