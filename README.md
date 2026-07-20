# TSPilot v0.2 Startup Guide

This repository contains the TSPilot v0.2 backend and React frontend.

## Prerequisites

- Python environment: `/home/feilvvl/TSPilot/tspilot_env`
- Node.js and npm for the frontend
- A configured `.env` file at the repository root

Create `.env` from the example if needed:

```bash
cp .env.example .env
```

Set at least:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://aihubmix.com/v1
OPENAI_MODEL=gpt-5.4-mini
TSPILOT_ROOT=.
```

## Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

Backend dependencies should already be installed in `/home/feilvvl/TSPilot/tspilot_env`. If the environment is missing packages, install the project dependencies there before starting.

## Start

From the repository root:

```bash
scripts/dev.sh
```

Defaults:

- Backend: `http://127.0.0.1:5680`
- Frontend: `http://127.0.0.1:5173`
- LAN access: `http://10.110.1.71:5173`

The script starts both services and stops both when you press `Ctrl+C`. The frontend listens on `0.0.0.0`; the backend stays on `127.0.0.1` and is reached through the Vite proxy.

## Start Separately

Backend:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 5680
```

Frontend:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

## Useful Checks

Backend health:

```bash
curl --noproxy '*' http://127.0.0.1:5680/health
```

List configured database resources:

```bash
curl --noproxy '*' http://127.0.0.1:5680/api/v1/resources/databases
```

Run backend tests:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m pytest tests -q
```

## Troubleshooting

- If `scripts/dev.sh` cannot find Python, set `BACKEND_PYTHON=/path/to/python`.
- If ports are occupied, set `BACKEND_PORT` or `FRONTEND_PORT` before running the script.
- If external access fails, confirm the frontend is listening on `0.0.0.0:5173` and add `10.110.1.71` to your proxy bypass list if needed.
- If model calls fail, verify `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL` in `.env`.
