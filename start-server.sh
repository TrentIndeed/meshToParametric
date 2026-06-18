#!/usr/bin/env bash
# Start the Paramesh AI web UI inside the Linux Docker container.
#
# Why the container: the alpha-wrap watertight repair (ingest) and the exact-IoU
# residual arbiter only work on Linux — pymeshlab's alpha-wrap plugin won't load
# on Windows, where they silently no-op and AI/scan meshes come out far worse.
# This builds + runs the same image that deploys to Hetzner.
#
# Usage:
#   ./start-server.sh             # build (if needed) + run on http://127.0.0.1:8080
#   ./start-server.sh --no-build  # skip the build, just (re)start the container
#   ./start-server.sh --native    # old behaviour: native python (NO alpha-wrap)
# Ctrl+C stops it (the container is removed on exit).
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
IMAGE=paramesh-server
CONTAINER=paramesh-server
NO_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --no-build) NO_BUILD=1 ;;
        --native) NATIVE=1 ;;
    esac
done

if [ -n "$NATIVE" ]; then
    export PARAMESH_PORT=8080
    echo "Starting Paramesh AI (native, NO alpha-wrap) at http://127.0.0.1:8080 ..."
    exec python -m mesh2param.viz_server.app
fi

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running. Start Docker (Desktop), or use --native." >&2
    exit 1
fi

if [ "$NO_BUILD" -eq 0 ]; then
    echo "Building $IMAGE (Linux image with the alpha-wrap / exact-IoU stack)..."
    docker build -t "$IMAGE" . || { echo "ERROR: image build failed." >&2; exit 1; }
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1

# Host path for -v: git-bash on Windows needs the Windows form (pwd -W) and
# MSYS_NO_PATHCONV so /app/... isn't mangled; plain Linux uses pwd as-is.
if HOSTROOT=$(pwd -W 2>/dev/null); then export MSYS_NO_PATHCONV=1; else HOSTROOT=$(pwd); fi
ENVFILE=()
[ -f .env ] && ENVFILE=(--env-file .env)

# DB in a NAMED volume, not a host bind-mount: SQLite locking deadlocks over
# Docker Desktop's Windows filesystem mount and hangs startup.
echo "Starting $CONTAINER at http://127.0.0.1:8080  (Ctrl+C to stop)"
echo "Note: the Claude CLI (AI chat) is unavailable inside the container; it degrades gracefully."
exec docker run --rm --name "$CONTAINER" \
    -p 8080:8080 \
    -v paramesh-data:/app/data \
    -v "${HOSTROOT}/mesh2param:/app/mesh2param" \
    -v "${HOSTROOT}/models:/app/models" \
    "${ENVFILE[@]}" \
    -e DATABASE_URL="sqlite:////app/data/mesh2param.db" \
    -e PYTHONUTF8=1 \
    "$IMAGE"
