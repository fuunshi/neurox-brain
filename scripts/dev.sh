#!/usr/bin/env bash
# Runs the HTTP service for local development.
#
#   ./scripts/dev.sh
#
# The model is loaded during start-up, so the first `/health` may report
# `loading` for a second or two — that is intentional, not an error. See the
# lifespan handler in `src/neurox_brain/main.py`.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "No virtualenv. Run:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  echo "  .venv/bin/python -m spacy download en_core_web_sm" >&2
  exit 1
fi

exec env PYTHONPATH=src .venv/bin/uvicorn neurox_brain.main:app \
  --host "${BRAIN_HOST:-127.0.0.1}" \
  --port "${BRAIN_PORT:-8000}" \
  --reload
