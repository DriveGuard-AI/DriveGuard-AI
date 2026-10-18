#!/usr/bin/env bash
# DriveGuard AI backend launcher (macOS / Linux / Git Bash).
#
#   ./run.sh                     # default port 8000
#   DGB_PORT=8123 ./run.sh       # custom port
#
# Interpreter version comes from .python-version (override with PYTHON_BIN).
# Dependencies come from requirements-dev.lock.txt, which pins the versions the
# backend test suite passes with; requirements.txt keeps the readable ranges.

set -euo pipefail
cd "$(dirname "$0")"

VERSION="$(tr -d '[:space:]' < .python-version 2>/dev/null || true)"
VERSION="${VERSION:-3.11}"
PYTHON_BIN="${PYTHON_BIN:-python${VERSION}}"

LOCK="requirements-dev.lock.txt"
if [ ! -f "${LOCK}" ]; then
  echo "[error] ${LOCK} is missing." >&2
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "[setup] creating .venv with ${PYTHON_BIN} ..."
  "${PYTHON_BIN}" -m venv .venv
fi

ACTUAL="$(./.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "${ACTUAL}" != "${VERSION}" ]; then
  echo "[error] .venv runs Python ${ACTUAL} but .python-version requires ${VERSION}." >&2
  echo "        Remove .venv and re-run, or set PYTHON_BIN to a matching interpreter." >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  HASH="$(sha256sum "${LOCK}" | cut -d' ' -f1)"
else
  HASH="$(shasum -a 256 "${LOCK}" | cut -d' ' -f1)"
fi
STAMP="${VERSION}|${HASH}"

if [ ! -f ".venv/.deps-installed" ] || [ "$(cat .venv/.deps-installed)" != "${STAMP}" ]; then
  echo "[setup] installing pinned dependencies (first run may take a while) ..."
  ./.venv/bin/python -m pip install --upgrade pip --quiet
  ./.venv/bin/python -m pip install -r "${LOCK}" --quiet
  printf '%s\n' "${STAMP}" > .venv/.deps-installed
fi

export DGB_ADMIN_TOKEN="${DGB_ADMIN_TOKEN:-dev-token}"
PORT="${DGB_PORT:-8000}"

echo "[run] dashboard http://127.0.0.1:${PORT}/    openapi http://127.0.0.1:${PORT}/docs"
exec ./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"
