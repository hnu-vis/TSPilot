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

port_pids() {
  local port="$1"
  {
    if command -v lsof >/dev/null 2>&1; then
      lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
    elif command -v fuser >/dev/null 2>&1; then
      fuser -n tcp "$port" 2>/dev/null || true
    elif command -v ss >/dev/null 2>&1; then
      ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p'
    else
      echo "No port inspection command found. Install lsof, fuser, or ss." >&2
      return 1
    fi
  } | tr ' ' '\n' | sed '/^$/d' | sort -u
}

kill_port() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(port_pids "$port" || true)"
  if [[ -z "$pids" ]]; then
    return
  fi

  echo "Stopping existing $label process(es) on port $port: $pids"
  kill $pids 2>/dev/null || true
  sleep 1

  local remaining=""
  local pid
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      remaining="$remaining $pid"
    fi
  done
  if [[ -n "$remaining" ]]; then
    echo "Force stopping $label process(es) on port $port:$remaining"
    kill -9 $remaining 2>/dev/null || true
  fi
}

if [[ ! -x "$BACKEND_PYTHON" ]]; then
  echo "Backend Python not found or not executable: $BACKEND_PYTHON" >&2
  echo "Override with: BACKEND_PYTHON=/path/to/python scripts/dev.sh" >&2
  exit 1
fi

require_command npm

kill_port "$FRONTEND_PORT" "frontend"
kill_port "$BACKEND_PORT" "backend"

trap cleanup EXIT INT TERM

echo "Starting TSPilot backend: http://$BACKEND_HOST:$BACKEND_PORT"
(
  cd "$ROOT_DIR"
  "$BACKEND_PYTHON" -m uvicorn app.server:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload --reload-dir "$ROOT_DIR"
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
