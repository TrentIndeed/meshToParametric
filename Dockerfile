# Paramesh AI server image — the full web UI (FastAPI + Three.js + PyVista
# thumbnails) PLUS the alpha-wrap / exact-IoU stack (pymeshlab + manifold3d).
# Running the server in THIS Linux image is what activates the watertight
# alpha-wrap repair (ingest) and the exact-IoU residual arbiter — both no-ops
# on native Windows, where pymeshlab's alpha-wrap plugin won't load. The same
# image deploys to Hetzner (docker-compose) and runs locally (start-server.ps1).
#
# Python 3.11 (not 3.13): matches the validated _docker stack — pymeshlab,
# manifold3d and cadquery==2.7.0 all have known-good wheels here.
FROM python:3.11-slim

# System libs: server/PyVista (libgl1, libglib2.0-0, libxrender1, libsm6) +
# pymeshlab's alpha-wrap plugin (libcom-err2/krb5/opengl/glu — without these the
# .so won't load and the watertight repair silently no-ops, the Windows failure).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libgomp1 libegl1 \
        libcom-err2 libkrb5-3 libgssapi-krb5-2 libk5crypto3 libxcb1 \
        libxkbcommon0 libdbus-1-3 libfontconfig1 libfreetype6 \
        libopengl0 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Server + pipeline deps come from pyproject (pyvista, fastapi, uvicorn, trimesh,
# fast-simplification, ...). Then the build/IoU extras NOT in pyproject that the
# CadQuery export + alpha-wrap repair + exact IoU need: cadquery, manifold3d,
# pymeshlab, lxml (trimesh's 3MF loader). trimesh pinned to the validated 4.11.5.
RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir \
        trimesh==4.11.5 cadquery==2.7.0 manifold3d pymeshlab lxml

# SQLite DB lives here; mounted from the host at run time so it persists.
RUN mkdir -p /app/data
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "mesh2param.viz_server.app:app", \
     "--host", "0.0.0.0", "--port", "8080"]
