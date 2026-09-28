#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.lock
  .venv/bin/pip install --no-deps -e .
fi
if [ ! -f web/dist/index.html ]; then
  npm --prefix web ci
  npm --prefix web run build
fi
if [ ! -f artifacts/models/access.json ] && [ ! -f vercel_runtime/artifacts/models/access.json ]; then
  echo 'Missing models: neither artifacts/ nor vercel_runtime/ is present. See README.md.' >&2
  exit 1
fi
exec .venv/bin/uvicorn moscollector.api:app --host 127.0.0.1 --port "${CONTOUR_PORT:-8000}"
