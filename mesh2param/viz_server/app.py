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


app = FastAPI(title="Paramesh AI", lifespan=lifespan)
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
    "tokens_spent": 0,
}

# Token budgets per plan
PLAN_TOKENS = {
    "free": 100,
    "starter": 1000,
    "pro": 5000,
    "unlimited": float("inf"),
}

# Max upload file size per plan (MB)
PLAN_UPLOAD_MB = {
    "free": 10,
    "starter": 50,
    "pro": 200,
    "unlimited": 500,
}


def _compute_token_cost(cleaned, tree) -> int:
    """Compute token cost based on mesh complexity + features."""
    cost = 0
    # Base: face count / 100
    cost += max(1, cleaned.num_faces // 100)
    # Features: holes=2, chamfers/fillets=3, pockets/bosses=3
    from mesh2param.models.feature import FeatureType
    for f in tree.features:
        if f.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
            cost += 2
        elif f.type in (FeatureType.CHAMFER, FeatureType.FILLET):
            cost += 3
        elif f.type in (FeatureType.POCKET, FeatureType.BOSS, FeatureType.SLOT):
            cost += 3
    return cost


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/landing", response_class=HTMLResponse)
async def landing():
    return FileResponse(str(STATIC_DIR / "landing.html"))


@app.get("/clean", response_class=HTMLResponse)
async def cleaner():
    return FileResponse(str(STATIC_DIR / "cleaner.html"))


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


@app.post("/api/clean")
async def clean_mesh(file: UploadFile = File(...)):
    """Free mesh cleaner — repair and return cleaned mesh as STL."""
    import trimesh

    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    try:
        from mesh2param.pipeline.ingest import ingest_mesh
        from mesh2param.config import IngestConfig

        cleaned = ingest_mesh(tmp.name, IngestConfig(remesh=False))

        # Export cleaned mesh as STL
        mesh = trimesh.Trimesh(
            vertices=cleaned.vertices, faces=cleaned.faces, process=False,
        )
        out_tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        mesh.export(out_tmp.name)
        out_tmp.close()

        # Build stats
        original = trimesh.load(tmp.name)
        orig_faces = len(original.faces) if hasattr(original, 'faces') else 0
        orig_verts = len(original.vertices) if hasattr(original, 'vertices') else 0

        return JSONResponse({
            "status": "ok",
            "original": {
                "faces": orig_faces,
                "vertices": orig_verts,
                "watertight": bool(getattr(original, 'is_watertight', False)),
            },
            "cleaned": {
                "faces": cleaned.num_faces,
                "vertices": cleaned.num_vertices,
                "watertight": cleaned.is_watertight,
                "bbox": cleaned.bbox.tolist(),
            },
            "download_url": f"/api/clean/download/{Path(out_tmp.name).name}",
            "vertices": cleaned.vertices.flatten().tolist(),
            "faces": cleaned.faces.flatten().tolist(),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/clean/resize")
async def resize_mesh(body: dict):
    """Resize a cleaned mesh by scale factor or target dimensions."""
    import trimesh

    vertices = body.get("vertices", [])
    faces = body.get("faces", [])
    scale = body.get("scale", 1.0)
    target_width = body.get("target_width")
    target_height = body.get("target_height")
    target_depth = body.get("target_depth")

    if not vertices or not faces:
        return JSONResponse({"error": "No geometry"}, status_code=400)

    import numpy as np
    verts = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    fcs = np.array(faces, dtype=np.int64).reshape(-1, 3)

    # Compute current dimensions
    bbox_size = verts.max(axis=0) - verts.min(axis=0)

    # If target dimensions given, compute scale per axis
    if target_width or target_height or target_depth:
        sx = target_width / bbox_size[0] if target_width and bbox_size[0] > 0 else 1.0
        sy = target_height / bbox_size[1] if target_height and bbox_size[1] > 0 else 1.0
        sz = target_depth / bbox_size[2] if target_depth and bbox_size[2] > 0 else 1.0
        center = (verts.max(axis=0) + verts.min(axis=0)) / 2
        verts = (verts - center) * np.array([sx, sy, sz]) + center
    elif scale != 1.0:
        center = (verts.max(axis=0) + verts.min(axis=0)) / 2
        verts = (verts - center) * scale + center

    new_size = verts.max(axis=0) - verts.min(axis=0)

    # Export resized STL
    mesh = trimesh.Trimesh(vertices=verts, faces=fcs, process=False)
    out_tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    mesh.export(out_tmp.name)
    out_tmp.close()

    return {
        "status": "ok",
        "dimensions": new_size.tolist(),
        "vertices": verts.flatten().tolist(),
        "faces": fcs.flatten().tolist(),
        "download_url": f"/api/clean/download/{Path(out_tmp.name).name}",
    }


@app.get("/api/clean/download/{filename}")
async def download_cleaned(filename: str):
    """Download a cleaned mesh file."""
    tmp_dir = tempfile.gettempdir()
    path = Path(tmp_dir) / filename
    if not path.exists() or not filename.endswith(".stl"):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(str(path), media_type="application/octet-stream",
                        filename="cleaned_mesh.stl")


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


@app.get("/api/usage")
async def usage_status():
    """Check token usage and plan limits."""
    plan = "free"  # TODO: read from user session
    limit = PLAN_TOKENS.get(plan, 100)
    upload_mb = PLAN_UPLOAD_MB.get(plan, 10)
    return {
        "plan": plan,
        "tokens_spent": _state["tokens_spent"],
        "tokens_limit": limit if limit != float("inf") else -1,
        "tokens_remaining": max(0, limit - _state["tokens_spent"]) if limit != float("inf") else -1,
        "upload_limit_mb": upload_mb,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a mesh file and run the full pipeline (costs tokens)."""
    plan = "free"  # TODO: read from user session
    limit = PLAN_TOKENS.get(plan, 100)
    if _state["tokens_spent"] >= limit:
        return JSONResponse(
            {"status": "error", "error": f"Token limit reached ({int(limit)} tokens on {plan} plan). Upgrade for more."},
            status_code=403,
        )

    # Check file size
    content = await file.read()
    max_bytes = PLAN_UPLOAD_MB.get(plan, 10) * 1024 * 1024
    if len(content) > max_bytes:
        return JSONResponse(
            {"status": "error", "error": f"File too large ({len(content)//1024//1024}MB). {plan} plan allows {PLAN_UPLOAD_MB[plan]}MB."},
            status_code=413,
        )
    _state["status"] = "processing"
    _state["error"] = None
    _state["steps"] = []

    # Save uploaded file to temp (already read above for size check)
    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
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
    _state["progress"] = 100
    _state["progress_stage"] = "Done"

    # Charge tokens based on complexity
    token_cost = _compute_token_cost(cleaned, tree)
    _state["tokens_spent"] += token_cost
    _state["tokens_used"] = _state["tokens_spent"]


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
