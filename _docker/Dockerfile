# This repo's pipeline on Linux — where pymeshlab's alpha-wrap plugin loads, so
# we get RELIABLE exact-boolean IoU (manifold3d on a watertight-wrapped target),
# the metric needed to pursue shape-IoU-primary optimization (Pillars 2/3).
# Lean: pipeline + IoU deps only (no pyvista/VTK, no server). Build context is
# the repo root: docker build -f _docker/Dockerfile -t mesh2param .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libgomp1 libegl1 \
        libcom-err2 libkrb5-3 libgssapi-krb5-2 libk5crypto3 libxcb1 \
        libxkbcommon0 libdbus-1-3 libfontconfig1 libfreetype6 \
        libopengl0 libglu1-mesa ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pipeline + IoU deps (cadquery for build, manifold3d + pymeshlab[alpha-wrap]
# for exact IoU, fast-simplification for deterministic decimation). No pyvista.
RUN pip install --no-cache-dir \
        "numpy>=2,<3" scipy scikit-learn scikit-image networkx shapely \
        pydantic click pymeshfix matplotlib \
        trimesh==4.11.5 fast-simplification \
        cadquery==2.7.0 manifold3d pymeshlab

# trimesh's 3MF loader needs lxml (separate layer keeps the heavy install cached).
RUN pip install --no-cache-dir lxml

WORKDIR /app
COPY mesh2param/ /app/mesh2param/
COPY scripts/ /app/scripts/
COPY pyproject.toml /app/
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
# models/ mounted at /app/models at run time.
