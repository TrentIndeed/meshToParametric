FROM python:3.13-slim

# System deps for cadquery (OCC) and trimesh
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir cadquery shapely

# Data directory for SQLite
RUN mkdir -p /app/data

EXPOSE 8080

CMD ["uvicorn", "mesh2param.viz_server.app:app", "--host", "0.0.0.0", "--port", "8080"]
