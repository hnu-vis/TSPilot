#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"

BACKEND_PYTHON="${BACKEND_PYTHON:-/home/feilvvl/TSPilot/tspilot_env/bin/python}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-5680}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5670}"

backend_pid=""
frontend_pid=""

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "$frontend_pid" ]] && kill -0 "$frontend_pid" 2>/dev/null; then
    kill "$frontend_pid" 2>/dev/null || true
  fi
  if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null; then
    kill "$backend_pid" 2>/dev/null || true
  fi
  wait "$frontend_pid" "$backend_pid" 2>/dev/null || true
  exit "$code"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

if [[ ! -x "$BACKEND_PYTHON" ]]; then
  echo "Backend Python not found or not executable: $BACKEND_PYTHON" >&2
  echo "Override with: BACKEND_PYTHON=/path/to/python scripts/dev.sh" >&2
  exit 1
fi

require_command npm

trap cleanup EXIT INT TERM

echo "Starting TSPilot backend: http://$BACKEND_HOST:$BACKEND_PORT"
(
  cd "$ROOT_DIR"
  "$BACKEND_PYTHON" -m uvicorn app.server:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
) &
backend_pid=$!

echo "Starting TSPilot frontend: http://$FRONTEND_HOST:$FRONTEND_PORT"
(
  cd "$FRONTEND_DIR"
  npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
frontend_pid=$!

echo
echo "Services are starting."
echo "Frontend: http://$FRONTEND_HOST:$FRONTEND_PORT"
echo "Backend:  http://$BACKEND_HOST:$BACKEND_PORT"
echo "Press Ctrl+C to stop both."
echo

wait -n "$backend_pid" "$frontend_pid"
