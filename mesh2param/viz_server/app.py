"""mesh2param web UI — FastAPI backend serving Three.js CAD viewer.

Endpoints:
  GET  /                  → main viewer page
  POST /api/upload        → upload mesh file, run pipeline, return feature tree
  GET  /api/step/{index}  → get tessellated geometry for a construction step
  GET  /api/status        → pipeline status and token count
  POST /auth/signup       → create account
  POST /auth/login        → login
  GET  /auth/verify       → verify token
  POST /settings/*        → change password, delete account
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mesh2param.db.database import init_db
from mesh2param.api.auth import router as auth_router
from mesh2param.api.chat import router as chat_router
from mesh2param.api.settings import router as settings_router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="mesh2param", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(settings_router)

# In-memory state for the current session
_state = {
    "features": None,  # FeatureTree
    "cleaned": None,  # CleanedMesh
    "steps": [],  # list of ConstructionStep
    "status": "idle",  # idle | processing | done | error
    "tokens_used": 0,
    "error": None,
    "mesh_volume": 0.0,
    "cad_volume": 0.0,
    "progress": 0,
    "progress_stage": "",
}


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/landing", response_class=HTMLResponse)
async def landing():
    return FileResponse(str(STATIC_DIR / "landing.html"))


@app.get("/api/status")
async def status():
    steps_info = []
    for i, step in enumerate(_state["steps"]):
        steps_info.append({
            "index": i,
            "name": step.name,
            "volume": round(step.volume, 1),
            "has_geometry": step.tess_vertices is not None and len(step.tess_vertices) > 0,
        })
    return {
        "status": _state["status"],
        "tokens_used": _state["tokens_used"],
        "num_steps": len(_state["steps"]),
        "steps": steps_info,
        "mesh_volume": round(_state["mesh_volume"], 1),
        "cad_volume": round(_state["cad_volume"], 1),
        "error": _state["error"],
        "bbox_min": _state.get("bbox_min", [0, 0, 0]),
        "bbox_max": _state.get("bbox_max", [0, 0, 0]),
    }


@app.post("/api/waitlist")
async def waitlist(body: dict):
    """Save a waitlist email to the database."""
    from mesh2param.db.database import get_db, SessionLocal, Base, engine
    from sqlalchemy import Column, Integer, String, DateTime
    from datetime import datetime

    email = body.get("email", "").strip().lower()
    if not email or "@" not in email:
        return JSONResponse({"error": "Valid email required"}, status_code=400)

    db = SessionLocal()
    try:
        # Check if already on waitlist (use raw SQL since model may not exist yet)
        from sqlalchemy import text
        engine.execute(text(
            "CREATE TABLE IF NOT EXISTS waitlist (id INTEGER PRIMARY KEY, email TEXT UNIQUE, created_at TEXT)"
        ))
        try:
            engine.execute(
                text("INSERT INTO waitlist (email, created_at) VALUES (:e, :t)"),
                {"e": email, "t": datetime.utcnow().isoformat()},
            )
        except Exception:
            pass  # duplicate email, ignore
        return {"status": "ok", "message": "You're on the list"}
    finally:
        db.close()


@app.get("/api/progress")
async def progress():
    return {
        "progress": _state["progress"],
        "stage": _state["progress_stage"],
        "status": _state["status"],
    }


@app.get("/api/step/{index}")
async def get_step(index: int):
    """Return tessellated geometry for a construction step as JSON arrays."""
    if index < 0 or index >= len(_state["steps"]):
        return JSONResponse({"error": "Invalid step index"}, status_code=404)

    step = _state["steps"][index]
    if step.tess_vertices is None or step.tess_faces is None:
        return JSONResponse({"error": "No geometry for this step"}, status_code=404)

    is_sketch = "sketch" in step.name.lower()
    return {
        "name": step.name,
        "volume": round(step.volume, 1),
        "is_sketch": is_sketch,
        "vertices": step.tess_vertices.flatten().tolist(),
        "faces": step.tess_faces.flatten().tolist(),
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a mesh file and run the full pipeline."""
    _state["status"] = "processing"
    _state["error"] = None
    _state["steps"] = []

    # Save uploaded file to temp
    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    try:
        result = _run_pipeline(tmp.name)
        _state["status"] = "done"
        return {"status": "done", "num_steps": len(_state["steps"])}
    except Exception as e:
        _state["status"] = "error"
        _state["error"] = str(e)
        logger.exception("Pipeline failed")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _run_pipeline(mesh_path: str):
    """Run the full mesh→parametric pipeline."""
    import subprocess
    import sys
    import pickle

    from mesh2param.config import (
        FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
    )
    from mesh2param.pipeline.ingest import ingest_mesh
    from mesh2param.pipeline.segment import segment_mesh
    from mesh2param.pipeline.fit import fit_surfaces
    from mesh2param.pipeline.topology import build_topology
    from mesh2param.pipeline.recognize import recognize_features

    # Stages 1-5 run in the main process (no CadQuery VTK conflict)
    _state["progress"] = 10
    _state["progress_stage"] = "Ingesting mesh..."
    cleaned = ingest_mesh(mesh_path, IngestConfig(remesh=False))

    _state["progress"] = 25
    _state["progress_stage"] = "Segmenting surfaces..."
    regions = segment_mesh(cleaned, SegmentConfig())

    _state["progress"] = 40
    _state["progress_stage"] = "Fitting primitives..."
    fitted = fit_surfaces(regions, cleaned, FitConfig())

    _state["progress"] = 55
    _state["progress_stage"] = "Building topology..."
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())

    _state["progress"] = 65
    _state["progress_stage"] = "Recognizing features..."
    tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())

    _state["features"] = tree
    _state["cleaned"] = cleaned
    _state["mesh_volume"] = _estimate_volume(cleaned)
    _state["bbox_min"] = cleaned.bbox[0].tolist()
    _state["bbox_max"] = cleaned.bbox[1].tolist()

    _state["progress"] = 75
    _state["progress_stage"] = "Building CAD model..."

    # Stage 6: CadQuery export in subprocess (VTK isolation)
    tmp_in = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    pickle.dump((tree, cleaned), tmp_in)
    tmp_in.close()

    tmp_out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    tmp_out.close()

    script = f"""
import pickle, sys
sys.path.insert(0, '.')
with open(r'{tmp_in.name}', 'rb') as f:
    tree, cleaned = pickle.load(f)
from mesh2param.pipeline.export_cadquery import export_cadquery
result = export_cadquery(tree, cleaned, use_llm=False)
result.solid = None
with open(r'{tmp_out.name}', 'wb') as f:
    pickle.dump(result, f)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=180,
    )
    Path(tmp_in.name).unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"CadQuery failed: {proc.stderr[-300:]}")

    with open(tmp_out.name, "rb") as f:
        result = pickle.load(f)
    Path(tmp_out.name).unlink(missing_ok=True)

    _state["steps"] = result.construction_steps
    _state["cad_volume"] = result.cad_volume
    _state["tokens_used"] = 0
    _state["progress"] = 100
    _state["progress_stage"] = "Done"


def _estimate_volume(cleaned) -> float:
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=cleaned.vertices, faces=cleaned.faces, process=False,
    )
    if mesh.is_watertight:
        return abs(float(mesh.volume))
    return abs(float(mesh.convex_hull.volume))


def run(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
