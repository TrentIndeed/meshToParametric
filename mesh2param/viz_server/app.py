"""Paramesh AI — FastAPI backend with job queue for concurrent users.

Each upload creates a job with a unique ID. A background worker thread
processes jobs one at a time. Frontend polls by job ID.

Endpoints:
  GET  /                       → app viewer
  GET  /landing                → marketing page
  GET  /clean                  → free mesh cleaner
  POST /api/upload             → submit pipeline job, returns {job_id}
  GET  /api/job/{id}/status    → poll job progress
  GET  /api/job/{id}/step/{i}  → get construction step geometry
  POST /api/clean              → free mesh repair
  POST /api/clean/resize       → resize mesh
  POST /api/chat               → AI chat
  POST /auth/*                 → auth endpoints
"""

from __future__ import annotations

import logging
import tempfile
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
from mesh2param.viz_server.job_queue import queue

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

PLAN_TOKENS = {"free": 100, "starter": 1000, "pro": 5000, "unlimited": float("inf")}
PLAN_UPLOAD_MB = {"free": 10, "starter": 50, "pro": 200, "unlimited": 500}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Paramesh AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(settings_router)


# ── Pages ──

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/landing", response_class=HTMLResponse)
async def landing():
    return FileResponse(str(STATIC_DIR / "landing.html"))

@app.get("/clean", response_class=HTMLResponse)
async def cleaner():
    return FileResponse(str(STATIC_DIR / "cleaner.html"))

@app.get("/robots.txt")
async def robots():
    return FileResponse(str(STATIC_DIR / "robots.txt"), media_type="text/plain")

@app.get("/sitemap.xml")
async def sitemap():
    return FileResponse(str(STATIC_DIR / "sitemap.xml"), media_type="application/xml")

@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    return FileResponse(str(STATIC_DIR / "privacy.html"))

@app.get("/terms", response_class=HTMLResponse)
async def terms():
    return FileResponse(str(STATIC_DIR / "terms.html"))


# ── Job-based pipeline ──

@app.post("/api/preview")
async def preview_mesh(file: UploadFile = File(...)):
    """Upload a mesh and return raw geometry for 3D preview (no pipeline).

    Also returns complexity estimate and token cost prediction.
    """
    import trimesh

    content = await file.read()
    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()

    try:
        from mesh2param.pipeline.ingest import ingest_mesh
        from mesh2param.config import IngestConfig

        cleaned = ingest_mesh(tmp.name, IngestConfig(remesh=False))

        # Estimate complexity and token cost
        face_count = cleaned.num_faces
        bbox = cleaned.bbox
        dims = (bbox[1] - bbox[0]).tolist()
        volume = 0.0
        try:
            m = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces, process=False)
            if m.is_watertight:
                volume = abs(float(m.volume))
        except Exception:
            pass

        # Rough token estimate (before feature recognition)
        estimated_tokens = max(5, round((face_count // 150 + face_count // 200) / 5) * 5)
        complexity = "Simple" if face_count < 500 else "Medium" if face_count < 2000 else "Complex"

        return {
            "status": "ok",
            "filename": file.filename,
            "temp_path": tmp.name,  # kept for pipeline run
            "faces": face_count,
            "vertices": cleaned.num_vertices,
            "watertight": cleaned.is_watertight,
            "dimensions": [round(d, 1) for d in dims],
            "volume": round(volume, 1),
            "complexity": complexity,
            "estimated_tokens": estimated_tokens,
            "mesh_vertices": cleaned.vertices.flatten().tolist(),
            "mesh_faces": cleaned.faces.flatten().tolist(),
        }
    except Exception as e:
        Path(tmp.name).unlink(missing_ok=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.post("/api/run-pipeline")
async def run_pipeline(body: dict):
    """Run the full pipeline on a previously previewed mesh. Takes temp_path from /api/preview."""
    plan = "free"
    limit = PLAN_TOKENS.get(plan, 100)
    if queue.tokens_spent >= limit:
        return JSONResponse(
            {"status": "error", "error": f"Token limit reached ({int(limit)} on {plan} plan)."},
            status_code=403,
        )

    temp_path = body.get("temp_path", "")
    filename = body.get("filename", "mesh.stl")
    if not temp_path or not Path(temp_path).exists():
        return JSONResponse({"status": "error", "error": "No mesh to process. Upload first."}, status_code=400)

    job_id = queue.submit(filename, temp_path)
    return {"status": "queued", "job_id": job_id}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Submit a mesh for pipeline processing. Returns a job ID to poll."""
    plan = "free"
    limit = PLAN_TOKENS.get(plan, 100)
    if queue.tokens_spent >= limit:
        return JSONResponse(
            {"status": "error", "error": f"Token limit reached ({int(limit)} on {plan} plan)."},
            status_code=403,
        )

    content = await file.read()
    max_bytes = PLAN_UPLOAD_MB.get(plan, 10) * 1024 * 1024
    if len(content) > max_bytes:
        return JSONResponse(
            {"status": "error", "error": f"File too large. {plan} plan allows {PLAN_UPLOAD_MB[plan]}MB."},
            status_code=413,
        )

    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()

    job_id = queue.submit(file.filename, tmp.name)
    return {"status": "queued", "job_id": job_id}


@app.get("/api/job/{job_id}/status")
async def job_status(job_id: str):
    """Poll a job's progress."""
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    steps_info = []
    for i, step in enumerate(job.steps):
        steps_info.append({
            "index": i,
            "name": step.name,
            "volume": round(step.volume, 1),
            "has_geometry": step.tess_vertices is not None and len(step.tess_vertices) > 0,
        })

    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.progress_stage,
        "error": job.error,
        "num_steps": len(job.steps),
        "steps": steps_info,
        "mesh_volume": round(job.mesh_volume, 1),
        "cad_volume": round(job.cad_volume, 1),
        "bbox_min": job.bbox_min,
        "bbox_max": job.bbox_max,
        "token_cost": job.token_cost,
        "tokens_used": queue.tokens_spent,
    }


@app.get("/api/job/{job_id}/step/{index}")
async def job_step(job_id: str, index: int):
    """Get tessellated geometry for a construction step."""
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if index < 0 or index >= len(job.steps):
        return JSONResponse({"error": "Invalid step index"}, status_code=404)

    step = job.steps[index]
    if step.tess_vertices is None or step.tess_faces is None:
        return JSONResponse({"error": "No geometry"}, status_code=404)

    return {
        "name": step.name,
        "volume": round(step.volume, 1),
        "is_sketch": "sketch" in step.name.lower(),
        "vertices": step.tess_vertices.flatten().tolist(),
        "faces": step.tess_faces.flatten().tolist(),
    }


# ── Legacy endpoints (redirect to latest job) ──

@app.get("/api/status")
async def status_legacy():
    """Legacy: return status of the most recent job."""
    with queue._lock:
        if not queue._jobs:
            return {"status": "idle", "tokens_used": 0, "num_steps": 0, "steps": [],
                    "mesh_volume": 0, "cad_volume": 0, "error": None,
                    "bbox_min": [0,0,0], "bbox_max": [0,0,0]}
        job = list(queue._jobs.values())[-1]

    steps_info = [{"index": i, "name": s.name, "volume": round(s.volume, 1),
                   "has_geometry": s.tess_vertices is not None and len(s.tess_vertices) > 0}
                  for i, s in enumerate(job.steps)]
    return {
        "job_id": job.id,
        "status": job.status,
        "tokens_used": queue.tokens_spent,
        "num_steps": len(job.steps),
        "steps": steps_info,
        "mesh_volume": round(job.mesh_volume, 1),
        "cad_volume": round(job.cad_volume, 1),
        "error": job.error,
        "bbox_min": job.bbox_min,
        "bbox_max": job.bbox_max,
    }


@app.get("/api/step/{index}")
async def step_legacy(index: int):
    """Legacy: return step from most recent job."""
    with queue._lock:
        if not queue._jobs:
            return JSONResponse({"error": "No jobs"}, status_code=404)
        job = list(queue._jobs.values())[-1]
    if index < 0 or index >= len(job.steps):
        return JSONResponse({"error": "Invalid step"}, status_code=404)
    step = job.steps[index]
    if step.tess_vertices is None:
        return JSONResponse({"error": "No geometry"}, status_code=404)
    return {
        "name": step.name, "volume": round(step.volume, 1),
        "is_sketch": "sketch" in step.name.lower(),
        "vertices": step.tess_vertices.flatten().tolist(),
        "faces": step.tess_faces.flatten().tolist(),
    }


@app.get("/api/progress")
async def progress_legacy():
    """Legacy: progress of most recent job."""
    with queue._lock:
        if not queue._jobs:
            return {"progress": 0, "stage": "", "status": "idle"}
        job = list(queue._jobs.values())[-1]
    return {"progress": job.progress, "stage": job.progress_stage, "status": job.status}


@app.get("/api/usage")
async def usage_status():
    import time as _time
    plan = "free"
    limit = PLAN_TOKENS.get(plan, 100)
    now = _time.time()

    # Compute daily/weekly/monthly usage from job history
    day_ago = now - 86400
    week_ago = now - 604800
    month_ago = now - 2592000
    daily = weekly = monthly = 0
    total_jobs = 0

    with queue._lock:
        for job in queue._jobs.values():
            if job.status == "done":
                total_jobs += 1
                if job.created_at >= month_ago:
                    monthly += job.token_cost
                if job.created_at >= week_ago:
                    weekly += job.token_cost
                if job.created_at >= day_ago:
                    daily += job.token_cost

    return {
        "plan": plan,
        "tokens_spent": queue.tokens_spent,
        "tokens_limit": limit if limit != float("inf") else -1,
        "tokens_remaining": max(0, limit - queue.tokens_spent) if limit != float("inf") else -1,
        "upload_limit_mb": PLAN_UPLOAD_MB.get(plan, 10),
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "total_jobs": total_jobs,
    }


# ── Free mesh cleaner (unchanged, no queue needed) ──

@app.post("/api/clean")
async def clean_mesh(file: UploadFile = File(...)):
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
        mesh = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces, process=False)
        out_tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        mesh.export(out_tmp.name)
        out_tmp.close()
        original = trimesh.load(tmp.name)
        return JSONResponse({
            "status": "ok",
            "original": {"faces": len(getattr(original,'faces',[])), "vertices": len(getattr(original,'vertices',[])),
                         "watertight": bool(getattr(original,'is_watertight',False))},
            "cleaned": {"faces": cleaned.num_faces, "vertices": cleaned.num_vertices,
                        "watertight": cleaned.is_watertight, "bbox": cleaned.bbox.tolist()},
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
    import trimesh
    vertices = body.get("vertices", [])
    faces = body.get("faces", [])
    scale = body.get("scale", 1.0)
    tw, th, td = body.get("target_width"), body.get("target_height"), body.get("target_depth")
    if not vertices or not faces:
        return JSONResponse({"error": "No geometry"}, status_code=400)
    verts = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    fcs = np.array(faces, dtype=np.int64).reshape(-1, 3)
    bbox_size = verts.max(0) - verts.min(0)
    if tw or th or td:
        sx = tw / bbox_size[0] if tw and bbox_size[0] > 0 else 1.0
        sy = th / bbox_size[1] if th and bbox_size[1] > 0 else 1.0
        sz = td / bbox_size[2] if td and bbox_size[2] > 0 else 1.0
        c = (verts.max(0) + verts.min(0)) / 2
        verts = (verts - c) * np.array([sx, sy, sz]) + c
    elif scale != 1.0:
        c = (verts.max(0) + verts.min(0)) / 2
        verts = (verts - c) * scale + c
    mesh = trimesh.Trimesh(vertices=verts, faces=fcs, process=False)
    out_tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    mesh.export(out_tmp.name); out_tmp.close()
    return {"status": "ok", "dimensions": (verts.max(0)-verts.min(0)).tolist(),
            "vertices": verts.flatten().tolist(), "faces": fcs.flatten().tolist(),
            "download_url": f"/api/clean/download/{Path(out_tmp.name).name}"}


@app.get("/api/clean/download/{filename}")
async def download_cleaned(filename: str):
    path = Path(tempfile.gettempdir()) / filename
    if not path.exists() or not filename.endswith(".stl"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(path), media_type="application/octet-stream", filename="cleaned_mesh.stl")


@app.post("/api/waitlist")
async def waitlist(body: dict):
    from mesh2param.db.database import SessionLocal
    from sqlalchemy import text
    from datetime import datetime
    email = body.get("email", "").strip().lower()
    if not email or "@" not in email:
        return JSONResponse({"error": "Valid email required"}, status_code=400)
    db = SessionLocal()
    try:
        with db.bind.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS waitlist (id INTEGER PRIMARY KEY, email TEXT UNIQUE, created_at TEXT)"))
            conn.execute(text("INSERT OR IGNORE INTO waitlist (email, created_at) VALUES (:e, :t)"),
                         {"e": email, "t": datetime.utcnow().isoformat()})
            conn.commit()
        return {"status": "ok", "message": "You're on the list"}
    finally:
        db.close()


def run(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
