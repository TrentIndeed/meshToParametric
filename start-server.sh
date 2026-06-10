#!/usr/bin/env bash
# Start the Paramesh AI web UI (FastAPI + Three.js viewer).
# Default parameters: host 127.0.0.1, port 8080.
# Usage:  ./start-server.sh   (Ctrl+C to stop)
#
# Port handling (mirrors start-server.ps1):
#   1. If something listens on 8080, try to kill it (assumed stale
#      previous instance) and wait until the port actually frees.
#   2. If it is still held (another application owns it), fall back to
#      the first free port in 8081..8090 via PARAMESH_PORT.
#
# Runs python in the FOREGROUND via exec: the shell process is replaced
# by python, so Ctrl+C goes straight to uvicorn (clean shutdown) and the
# script ends exactly when the server does. Do NOT background + wait here
# — under git-bash/MSYS, `wait` on a native python.exe can return early
# and leave the server running orphaned.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

port_listeners() {
    # PIDs of LISTENING sockets on the given port (Windows netstat).
    netstat -ano 2>/dev/null | awk -v p=":$1" \
        '$1 ~ /TCP/ && $2 ~ p"$" && $4 == "LISTENING" {print $5}' | sort -u
}

PORT=8080
pids=$(port_listeners "$PORT")
if [ -n "$pids" ]; then
    for pid in $pids; do
        echo "Stopping existing process on port $PORT (PID $pid)..."
        taskkill //F //PID "$pid" >/dev/null 2>&1
    done
    # Wait until the LISTEN socket is actually gone (up to 10s)
    for _ in $(seq 1 20); do
        sleep 0.5
        [ -z "$(port_listeners "$PORT")" ] && break
    done
fi

# Still busy (another application owns it)? Pick the next free port.
if [ -n "$(port_listeners "$PORT")" ]; then
    for candidate in $(seq 8081 8090); do
        if [ -z "$(port_listeners "$candidate")" ]; then
            echo "Port 8080 is in use by another application; using port $candidate instead."
            PORT=$candidate
            break
        fi
    done
    if [ -n "$(port_listeners "$PORT")" ]; then
        echo "ERROR: no free port found in 8080-8090." >&2
        exit 1
    fi
fi

export PARAMESH_PORT=$PORT
echo "Starting Paramesh AI server at http://127.0.0.1:$PORT ..."
echo "Press Ctrl+C to stop."
exec python -m mesh2param.viz_server.app
