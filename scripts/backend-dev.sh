#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PYTHON="${BACKEND_PYTHON:-/home/feilvvl/TSPilot/tspilot_env/bin/python}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-5680}"

if [[ ! -x "$BACKEND_PYTHON" ]]; then
  echo "Backend Python not found or not executable: $BACKEND_PYTHON" >&2
  echo "Override with: BACKEND_PYTHON=/path/to/python scripts/backend-dev.sh" >&2
  exit 1
fi

cd "$ROOT_DIR"
exec "$BACKEND_PYTHON" -m uvicorn app.server:app \
  --host "$BACKEND_HOST" \
  --port "$BACKEND_PORT" \
  --reload \
  --reload-dir "$ROOT_DIR"
