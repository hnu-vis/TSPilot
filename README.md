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

## Start Backend and Frontend Together

From the repository root:

```bash
scripts/dev.sh
```

Defaults:

- Backend: `http://127.0.0.1:5680`
- Frontend listens on `0.0.0.0:5670`
- Local frontend URL: `http://127.0.0.1:5670`
- LAN frontend URL on this machine: `http://10.110.1.71:5670`

The script starts both services and stops both when you press `Ctrl+C`.

The backend intentionally defaults to `127.0.0.1` because the Vite dev server proxies `/api` and `/health` to it. External browsers only need to reach the frontend port.

## Start Services Separately

Backend:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 5680
```

Frontend:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5670
```

Do not start the frontend with `--host 127.0.0.1` if you need LAN/external access. In Vite, the last `--host` argument wins, so a command such as `vite --host 0.0.0.0 --host 127.0.0.1` will only listen on localhost.

## Access From Another Machine

Use the host's LAN IP and frontend port:

```text
http://10.110.1.71:5670/
```

If you use another port, replace `5670` with `FRONTEND_PORT`.

To start on a custom externally reachable port:

```bash
FRONTEND_HOST=0.0.0.0 FRONTEND_PORT=5174 scripts/dev.sh
```

Check whether the frontend is really listening externally:

```bash
ss -ltnp | grep ':5670'
curl --noproxy '*' -I http://10.110.1.71:5670/
```

Expected listener:

```text
0.0.0.0:5670
```

If you see `127.0.0.1:5670`, the frontend is only available locally.

## Useful Checks

Backend health:

```bash
curl --noproxy '*' http://127.0.0.1:5680/health
```

Frontend from the host LAN IP:

```bash
curl --noproxy '*' -I http://10.110.1.71:5670/
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
- If the frontend cannot reach the API, confirm the backend health endpoint returns `{"status":"ok"}`.
- If external access fails, confirm the frontend listener is `0.0.0.0:5670`, not `127.0.0.1:5670`.
- If your browser or shell uses an HTTP proxy, add `10.110.1.71` or your LAN CIDR to the proxy bypass list. For shell checks:

  ```bash
  export NO_PROXY="$NO_PROXY,10.110.1.71"
  export no_proxy="$no_proxy,10.110.1.71"
  ```

- If another machine still cannot connect after the service listens on `0.0.0.0`, check firewall/security-group rules for the frontend port.
- If model calls fail, verify `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL` in `.env`.
