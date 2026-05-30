#!/usr/bin/env bash
# Start the Paramesh AI web UI (FastAPI + Three.js viewer).
# Default parameters: host 127.0.0.1, port 8080.
# Usage:  ./start-server.sh   (Ctrl+C to stop)
#
# Runs python in the FOREGROUND via exec: the shell process is replaced
# by python, so Ctrl+C goes straight to uvicorn (clean shutdown) and the
# script ends exactly when the server does. Do NOT background + wait here
# — under git-bash/MSYS, `wait` on a native python.exe can return early
# and leave the server running orphaned.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
echo "Starting Paramesh AI server at http://127.0.0.1:8080 ..."
echo "Press Ctrl+C to stop."
exec python -m mesh2param.viz_server.app
