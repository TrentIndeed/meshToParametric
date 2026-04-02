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
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env for AUTH_SECRET and other config
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
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

PLAN_TOKENS = {"free": 100, "starter": 1000, "pro": 5000, "unlimited": float("inf"), "developer": float("inf")}
PLAN_UPLOAD_MB = {"free": 10, "starter": 50, "pro": 200, "unlimited": 500, "developer": 999}

# ── Per-IP token tracking (prevents free trial abuse) ──
_ip_tokens: dict[str, int] = {}  # ip → tokens spent
_ip_jobs: dict[str, int] = {}    # ip → job count

def _get_client_ip(request) -> str:
    """Get real client IP, checking X-Forwarded-For for reverse proxy (Caddy)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def _check_ip_limit(ip: str, plan: str = "free") -> tuple[bool, str | None]:
    """Check if IP has exceeded free tier limits."""
    limit = PLAN_TOKENS.get(plan, 100)
    spent = _ip_tokens.get(ip, 0)
    if spent >= limit:
        return False, f"Token limit reached for this IP ({spent}/{int(limit)}). Sign up for more."
    jobs = _ip_jobs.get(ip, 0)
    if plan == "free" and jobs >= 10:
        return False, f"Free trial: max 10 jobs per IP. You've used {jobs}. Sign up for unlimited."
    return True, None

def _charge_ip(ip: str, cost: int):
    """Record token usage for an IP."""
    _ip_tokens[ip] = _ip_tokens.get(ip, 0) + cost


def _get_user_plan(request) -> str:
    """Get the user's plan from their auth cookie."""
    token = request.cookies.get("paramesh_token", "")
    if not token:
        return "free"
    try:
        from mesh2param.api.auth import get_user_plan
        from mesh2param.db.database import SessionLocal
        db = SessionLocal()
        try:
            return get_user_plan(token, db)
        finally:
            db.close()
    except Exception:
        return "free"
    _ip_jobs[ip] = _ip_jobs.get(ip, 0) + 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Charge IP when jobs complete
    queue.set_on_complete(lambda job: _charge_ip(job.client_ip, job.token_cost) if job.client_ip else None)
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
    return FileResponse(str(STATIC_DIR / "landing.html"))

@app.get("/home", response_class=HTMLResponse)
async def home():
    return FileResponse(str(STATIC_DIR / "home.html"))

@app.get("/app", response_class=HTMLResponse)
async def viewer():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/clean", response_class=HTMLResponse)
async def cleaner():
    return FileResponse(str(STATIC_DIR / "cleaner.html"))

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return FileResponse(str(STATIC_DIR / "login.html"))

@app.get("/changelog", response_class=HTMLResponse)
async def changelog_page():
    return FileResponse(str(STATIC_DIR / "changelog.html"))

@app.get("/contact", response_class=HTMLResponse)
async def contact_page():
    return FileResponse(str(STATIC_DIR / "contact.html"))

# ── Learn pages (programmatic SEO) ──

def _load_learn_data():
    import json
    path = STATIC_DIR / "learn-data.json"
    with open(path) as f:
        return json.load(f)

def _render_learn_page(entry, all_entries):
    """Server-render a learn page by replacing template placeholders."""
    template = (STATIC_DIR / "learn-template.html").read_text(encoding="utf-8")

    # CTA HTML based on type
    cta_map = {
        "trial": '<p>Convert your mesh to parametric CAD</p><a href="/app" class="btn btn-primary" style="width:100%;justify-content:center;">Try Free — 3 Generations</a>',
        "cleaner": '<p>Fix your mesh for free</p><a href="/clean" class="btn btn-primary" style="width:100%;justify-content:center;">Open Mesh Cleaner</a>',
        "waitlist": '<p>Get notified when this launches</p><a href="/#waitlist" class="btn btn-primary" style="width:100%;justify-content:center;">Join Waitlist</a>',
    }

    # Related links HTML
    related_html = ""
    for slug in entry.get("related", []):
        rel = next((e for e in all_entries if e["slug"] == slug), None)
        if rel:
            related_html += f'<a href="/learn/{rel["slug"]}">{rel["title"]}</a>\n'

    html = template.replace("{{TITLE}}", entry["title"])
    html = html.replace("{{SLUG}}", entry["slug"])
    html = html.replace("{{QUESTION}}", entry["question"])
    html = html.replace("{{TLDR}}", entry["tldr"])
    html = html.replace("{{BODY}}", entry["body"])
    html = html.replace("{{RELATED_LINKS}}", related_html)
    html = html.replace("{{CTA_HTML}}", cta_map.get(entry.get("cta_type", "trial"), cta_map["trial"]))
    return html

def _render_learn_index(entries):
    """Server-render the learn index page."""
    template = (STATIC_DIR / "learn-index.html").read_text(encoding="utf-8")
    cards = ""
    for e in entries:
        cards += f'<a href="/learn/{e["slug"]}" class="card"><h3>{e["title"]}</h3><p>{e["tldr"][:120]}...</p></a>\n'
    return template.replace("{{CARDS}}", cards)


@app.get("/learn", response_class=HTMLResponse)
async def learn_index():
    entries = _load_learn_data()
    return HTMLResponse(_render_learn_index(entries))

@app.get("/learn/{slug}", response_class=HTMLResponse)
async def learn_page(slug: str):
    entries = _load_learn_data()
    entry = next((e for e in entries if e["slug"] == slug), None)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return HTMLResponse(_render_learn_page(entry, entries))


@app.get("/robots.txt")
async def robots():
    return FileResponse(str(STATIC_DIR / "robots.txt"), media_type="text/plain")

@app.get("/sitemap.xml")
async def sitemap():
    """Dynamic sitemap including all learn pages."""
    entries = _load_learn_data()
    urls = [
        ("https://parameshai.com/", "1.0"),
        ("https://parameshai.com/clean", "0.9"),
        ("https://parameshai.com/app", "0.8"),
        ("https://parameshai.com/learn", "0.8"),
        ("https://parameshai.com/changelog", "0.5"),
        ("https://parameshai.com/contact", "0.4"),
        ("https://parameshai.com/privacy", "0.2"),
        ("https://parameshai.com/terms", "0.2"),
    ]
    for e in entries:
        urls.append((f"https://parameshai.com/learn/{e['slug']}", "0.7"))

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for url, priority in urls:
        xml += f"  <url><loc>{url}</loc><priority>{priority}</priority></url>\n"
    xml += "</urlset>"
    return HTMLResponse(content=xml, media_type="application/xml")

@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    return FileResponse(str(STATIC_DIR / "privacy.html"))

@app.get("/terms", response_class=HTMLResponse)
async def terms():
    return FileResponse(str(STATIC_DIR / "terms.html"))


# ── Job-based pipeline ──

@app.post("/api/preview")
async def preview_mesh(request: Request, file: UploadFile = File(...)):
    """Upload a mesh and return raw geometry for 3D preview (no pipeline).

    Also returns complexity estimate and token cost prediction.
    """
    import trimesh

    # Check IP limits before processing
    plan = _get_user_plan(request)
    ip = _get_client_ip(request)
    ok, reason = _check_ip_limit(ip, plan)
    if not ok:
        return JSONResponse({"status": "error", "error": reason}, status_code=403)

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

        # Complexity analysis
        import numpy as np
        normals = cleaned.normals
        avg_n = normals.mean(axis=0)
        avg_norm = np.linalg.norm(avg_n)
        if avg_norm > 1e-15:
            avg_n /= avg_norm
        normal_spread = float(1.0 - np.mean(np.abs(normals @ avg_n)))

        # Count approximate surface types by normal clustering
        unique_dirs = set()
        for n in normals[::max(1, len(normals)//200)]:
            snapped = tuple(round(x, 1) for x in n)
            unique_dirs.add(snapped)
        approx_surfaces = len(unique_dirs)

        # Aspect ratio (how box-like vs thin/long)
        sorted_dims = sorted(dims, reverse=True)
        aspect_ratio = sorted_dims[0] / (sorted_dims[2] + 1e-10)

        # Complexity score 0-100
        score = 0
        score += min(30, face_count / 100)        # faces: 0-30
        score += min(20, approx_surfaces)          # surface variety: 0-20
        score += min(15, normal_spread * 50)       # normal spread: 0-15
        score += min(15, aspect_ratio * 2)         # aspect ratio: 0-15
        score += 0 if cleaned.is_watertight else 10  # non-watertight penalty
        score += min(10, len(dims) and max(dims) / 50)  # size: 0-10
        score = min(100, round(score))

        # Classification and limits
        if score <= 25:
            complexity = "Simple"
            difficulty = "low"
            can_process = True
            reason = None
        elif score <= 50:
            complexity = "Medium"
            difficulty = "medium"
            can_process = True
            reason = None
        elif score <= 75:
            complexity = "Complex"
            difficulty = "high"
            can_process = True
            reason = None
        else:
            complexity = "Too Complex"
            difficulty = "extreme"
            can_process = face_count <= 10000
            reason = "This mesh has too many surfaces, high curvature variation, or non-prismatic geometry. The tool works best on prismatic mechanical parts (brackets, plates, mounts) with under 5000 faces." if not can_process else None

        # Hard limits
        if face_count > 10000:
            can_process = False
            reason = f"Mesh has {face_count:,} faces (limit: 10,000). Simplify the mesh before uploading."
        if not cleaned.is_watertight and face_count > 5000:
            can_process = False
            reason = f"Non-watertight mesh with {face_count:,} faces. Repair the mesh first using the free cleaner at /clean."

        estimated_tokens = max(5, round((face_count // 150 + face_count // 200) / 5) * 5)

        return {
            "status": "ok",
            "filename": file.filename,
            "temp_path": tmp.name,
            "faces": face_count,
            "vertices": cleaned.num_vertices,
            "watertight": cleaned.is_watertight,
            "dimensions": [round(d, 1) for d in dims],
            "volume": round(volume, 1),
            "complexity": complexity,
            "complexity_score": score,
            "difficulty": difficulty,
            "can_process": can_process,
            "reject_reason": reason,
            "estimated_tokens": estimated_tokens,
            "approx_surfaces": approx_surfaces,
            "normal_spread": round(normal_spread, 3),
            "aspect_ratio": round(aspect_ratio, 1),
            "mesh_vertices": cleaned.vertices.flatten().tolist(),
            "mesh_faces": cleaned.faces.flatten().tolist(),
        }
    except Exception as e:
        Path(tmp.name).unlink(missing_ok=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.post("/api/run-pipeline")
async def run_pipeline(request: Request, body: dict):
    """Run the full pipeline on a previously previewed mesh. Takes temp_path from /api/preview."""
    plan = _get_user_plan(request)
    ip = _get_client_ip(request)
    ok, reason = _check_ip_limit(ip, plan)
    if not ok:
        return JSONResponse({"status": "error", "error": reason}, status_code=403)
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

    job_id = queue.submit(filename, temp_path, client_ip=ip)
    return {"status": "queued", "job_id": job_id}


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    """Submit a mesh for pipeline processing. Returns a job ID to poll."""
    plan = _get_user_plan(request)
    ip = _get_client_ip(request)
    ok, reason = _check_ip_limit(ip, plan)
    if not ok:
        return JSONResponse({"status": "error", "error": reason}, status_code=403)
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

    job_id = queue.submit(file.filename, tmp.name, client_ip=ip)
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
async def usage_status(request: Request):
    import time as _time
    plan = _get_user_plan(request)
    limit = PLAN_TOKENS.get(plan, 100)
    ip = _get_client_ip(request)
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
        "ip_tokens": _ip_tokens.get(ip, 0),
        "ip_jobs": _ip_jobs.get(ip, 0),
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
    from mesh2param.db.database import SessionLocal, WaitlistEntry
    email = body.get("email", "").strip().lower()
    if not email or "@" not in email:
        return JSONResponse({"error": "Valid email required"}, status_code=400)
    db = SessionLocal()
    try:
        existing = db.query(WaitlistEntry).filter(WaitlistEntry.email == email).first()
        if not existing:
            db.add(WaitlistEntry(email=email))
            db.commit()
        return {"status": "ok", "message": "You're on the list"}
    finally:
        db.close()


@app.get("/api/changelog")
async def changelog_api():
    """Return recent git commits for the changelog page."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--no-merges", "-30", "--format=%H|%ad|%s", "--date=short"],
            capture_output=True, text=True, timeout=10, cwd=str(Path(__file__).parent.parent.parent),
        )
        commits = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append({"hash": parts[0][:8], "date": parts[1], "message": parts[2]})
        return {"commits": commits}
    except Exception:
        return {"commits": []}


@app.post("/api/contact")
async def contact_form(body: dict):
    """Save a support ticket and send email notification."""
    from mesh2param.db.database import SessionLocal, SupportTicket
    import smtplib
    from email.mime.text import MIMEText

    name = body.get("name", "").strip()
    email = body.get("email", "").strip()
    message = body.get("message", "").strip()

    if not name or not email or not message:
        return JSONResponse({"error": "All fields required"}, status_code=400)

    # Save to database
    db = SessionLocal()
    try:
        ticket = SupportTicket(name=name, email=email, message=message)
        db.add(ticket)
        db.commit()
    finally:
        db.close()

    # Try to send email notification (fail silently if SMTP not configured)
    try:
        import os
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_pass = os.getenv("SMTP_PASS", "")
        notify_email = "trentondragon02@gmail.com"

        if smtp_user and smtp_pass:
            msg = MIMEText(f"From: {name} <{email}>\n\n{message}")
            msg["Subject"] = f"[Paramesh AI] Support: {name}"
            msg["From"] = smtp_user
            msg["To"] = notify_email

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, notify_email, msg.as_string())
    except Exception as e:
        logger.warning("Email notification failed (SMTP not configured?): %s", e)

    return {"status": "ok", "message": "Message sent"}


@app.post("/api/support-chat")
async def support_chat(body: dict):
    """AI support chatbot — answers questions about Paramesh AI."""
    from mesh2param.ai import is_available, call_ai

    message = body.get("message", "").strip()
    if not message:
        return {"reply": "Please type a question."}

    if not is_available():
        # Fallback: simple FAQ responses
        msg_lower = message.lower()
        if "price" in msg_lower or "cost" in msg_lower or "token" in msg_lower:
            return {"reply": "Paramesh AI uses token-based pricing. Free tier gets 100 tokens. Starter is $9/mo (1000 tokens), Pro is $29/mo (5000), Unlimited is $79/mo. Simple parts cost ~20 tokens, complex parts ~55."}
        elif "format" in msg_lower or "stl" in msg_lower or "file" in msg_lower:
            return {"reply": "We support STL, OBJ, PLY, and 3MF files. The pipeline repairs topology and fixes normals automatically."}
        elif "clean" in msg_lower or "repair" in msg_lower:
            return {"reply": "The free mesh cleaner at /clean repairs normals, removes degenerate faces, merges vertices, and fills holes. No signup needed for cleaning — it's totally free."}
        elif "onshape" in msg_lower:
            return {"reply": "Onshape live export is coming soon. Currently you can export as STEP files via CadQuery. The Onshape integration will use the REST API to push your feature tree directly."}
        else:
            return {"reply": "I can help with questions about pricing, file formats, features, or troubleshooting. What would you like to know?"}

    prompt = f"""You are the support assistant for Paramesh AI (parameshai.com), a tool that converts
triangulated mesh files (STL/OBJ/3MF) into parametric CAD models.

Key facts:
- Free tier: 100 tokens, mesh cleaner unlimited, 10MB upload
- Starter $9/mo: 1000 tokens, 50MB, STEP export
- Pro $29/mo: 5000 tokens, 200MB, Onshape + AI chat
- Unlimited $79/mo: unlimited tokens, 500MB
- Supported formats: STL, OBJ, PLY, 3MF
- Pipeline: ingest → segment → fit → topology → recognize → export
- Features detected: holes, chamfers, fillets, pockets, bosses
- Export: CadQuery STEP files, Onshape coming soon
- Free mesh cleaner at /clean (no login for cleaning only)
- AI chatbot in the app for editing features

Be concise (2-3 sentences). Be helpful and friendly. If you don't know, suggest contacting support at /contact.

User question: {message}"""

    try:
        reply = call_ai(prompt, timeout=20)
        # Strip any JSON wrapping
        if reply.startswith("{"):
            import json
            try:
                data = json.loads(reply)
                reply = data.get("reply", reply)
            except Exception:
                pass
        return {"reply": reply}
    except Exception as e:
        return {"reply": "I'm having trouble connecting right now. Please try /contact to reach us directly."}


def run(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
