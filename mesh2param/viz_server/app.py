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

import faulthandler
import logging
import os
import sys
import tempfile

# Dump a Python stack trace if the process segfaults. Writes to stderr
# which is merged into out.txt. Invaluable for diagnosing native-code
# crashes (VTK / OCP / trimesh C extensions).
faulthandler.enable(file=sys.stderr, all_threads=True)
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
PLAN_PARTS = {"free": 3, "payg": 9999, "pro": 30, "team": 100, "developer": 9999}
PLAN_UPLOAD_MB = {"free": 10, "payg": 50, "starter": 50, "pro": 200, "team": 500, "unlimited": 500, "developer": 999}

# ── Free trial tracking ──

def _get_client_ip(request) -> str:
    """Get real client IP, checking X-Forwarded-For for reverse proxy (Caddy)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def _is_local_request(ip: str) -> bool:
    """Direct localhost access = the developer running the server.

    Production sits behind Caddy, which always supplies X-Forwarded-For
    (so _get_client_ip returns the real public IP there); a loopback
    client IP only happens when hitting uvicorn directly in local dev.
    """
    return ip in ("127.0.0.1", "::1", "localhost", "testclient")


def _check_free_trial(request) -> tuple[bool, str | None]:
    """Check if guest user (no auth cookie) has used their 1 free generation."""
    token = request.cookies.get("paramesh_token", "")
    if token:
        return True, None  # logged in, skip free trial check
    free_used = request.cookies.get("paramesh_free_used", "")
    if free_used:
        return False, "Free trial used. Sign up for unlimited conversions — it's free."
    return True, None


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
            plan = get_user_plan(token, db)
            logger.debug("Resolved user plan=%s from auth cookie", plan)
            return plan
        finally:
            db.close()
    except Exception:
        logger.exception("Failed to resolve user plan from auth cookie")
        return "free"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Paramesh AI server")
    try:
        init_db()
    except Exception:
        logger.exception("Failed to initialize database on startup")
        raise
    # Load completed jobs from DB so they survive restarts
    queue.load_completed_from_db()
    # Job completion callback (logging only now — free trial tracked via cookie)
    queue.set_on_complete(lambda job: logger.info("Job %s completed, tokens=%d", job.id, job.token_cost))
    logger.info("Server startup complete")
    yield
    logger.info("Server shutting down")


app = FastAPI(title="Paramesh AI", lifespan=lifespan)

# Same-origin by default; extra origins (e.g. staging) via ALLOWED_ORIGINS env var.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://parameshai.com,https://www.parameshai.com,"
        "http://127.0.0.1:8080,http://localhost:8080",
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Prevent Chrome from caching JS/CSS files
from starlette.middleware.base import BaseHTTPMiddleware
class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/") and request.url.path.endswith((".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
app.add_middleware(NoCacheStaticMiddleware)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Mesh formats we accept for upload; user-supplied filenames are never trusted
# beyond this whitelisted suffix.
ALLOWED_MESH_SUFFIXES = {".stl", ".obj", ".ply", ".3mf"}


def _safe_suffix(filename: str | None) -> str | None:
    """Return the lowercased mesh suffix if whitelisted, else None."""
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in ALLOWED_MESH_SUFFIXES else None

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(settings_router)
from mesh2param.api.onshape_oauth import router as onshape_oauth_router  # noqa: E402
app.include_router(onshape_oauth_router)


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
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Learn data file not found: %s", path)
        return []
    except json.JSONDecodeError:
        logger.exception("Failed to parse learn data JSON: %s", path)
        return []

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

    ip = _get_client_ip(request)
    plan = _get_user_plan(request)
    logger.debug("Preview request: filename=%r ip=%s plan=%s", file.filename, ip, plan)

    # Preview is always free (no limit check — the pipeline run is what counts)

    content = await file.read()
    logger.debug("Preview file received: %r size=%d bytes", file.filename, len(content))
    suffix = _safe_suffix(file.filename)
    if suffix is None:
        return JSONResponse(
            {"status": "error", "error": "Unsupported file type. Use STL, OBJ, PLY, or 3MF."},
            status_code=400,
        )
    # Store in data/uploads/ instead of system temp (survives server restarts)
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=upload_dir)
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

        # Complexity analysis — based on geometry, not face count.
        # A scanned part can have millions of faces but still be simple
        # (6 planes, 4 holes). Score is based on:
        # 1. Number of distinct normal directions (surface count estimate)
        # 2. Whether the part is prismatic (mostly axis-aligned normals)
        # 3. Watertight status
        import numpy as np

        normals = cleaned.normals
        abs_normals = np.abs(normals)

        # Count how many faces have axis-aligned normals (prismatic indicator)
        # A face is "axis-aligned" if its largest normal component > 0.85
        max_component = abs_normals.max(axis=1)
        prismatic_ratio = float((max_component > 0.85).sum()) / max(len(normals), 1)

        # Estimate number of distinct surface directions via clustering
        # Round normals to 1 decimal and count unique directions
        rounded = np.round(normals, 1)
        unique_dirs = len(set(map(tuple, rounded)))

        # Score: more unique directions = more complex geometry
        # Prismatic parts: ~10-30 unique dirs. Organic: 100+
        score = 0
        if unique_dirs <= 20:
            score += 5
        elif unique_dirs <= 50:
            score += 15
        elif unique_dirs <= 100:
            score += 35
        elif unique_dirs <= 200:
            score += 55
        else:
            score += 75

        # Prismatic bonus — mostly axis-aligned = simple
        if prismatic_ratio > 0.7:
            score = max(5, score - 20)
        elif prismatic_ratio < 0.3:
            score += 15  # organic/freeform penalty

        # Non-watertight penalty
        if not cleaned.is_watertight:
            score += 10

        score = min(100, round(score))

        normal_spread = round(1.0 - prismatic_ratio, 3)

        # Classification
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
            can_process = prismatic_ratio > 0.3
            reason = "This mesh appears to be organic or freeform geometry. The tool works best on prismatic mechanical parts (brackets, plates, mounts)." if not can_process else None

        # No hard face count limit — high-poly scans of simple parts are fine
        if not cleaned.is_watertight and face_count > 50000:
            can_process = False
            reason = f"Non-watertight mesh with {face_count:,} faces. Repair the mesh first using the free cleaner at /clean."
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
            "face_count": face_count,
            "normal_spread": round(normal_spread, 3),
            "aspect_ratio": round(sorted(dims, reverse=True)[0] / (sorted(dims, reverse=True)[2] + 1e-10), 1),
            "mesh_vertices": cleaned.vertices.flatten().tolist(),
            "mesh_faces": cleaned.faces.flatten().tolist(),
        }
    except Exception:
        logger.exception("Preview failed for file %r", file.filename)
        Path(tmp.name).unlink(missing_ok=True)
        return JSONResponse(
            {"status": "error", "error": "Could not process this mesh. Check the file and try again."},
            status_code=500,
        )


@app.post("/api/check-file")
async def check_file(body: dict):
    """Check if a previously uploaded temp file still exists."""
    temp_path = body.get("temp_path", "")
    return {"exists": bool(temp_path and Path(temp_path).exists())}


@app.post("/api/run-pipeline")
async def run_pipeline(request: Request, body: dict):
    """Run the full pipeline on a previously previewed mesh. Takes temp_path from /api/preview."""
    plan = _get_user_plan(request)
    ip = _get_client_ip(request)
    logger.debug("Run-pipeline request: filename=%r ip=%s plan=%s",
                 body.get("filename"), ip, plan)
    token = request.cookies.get("paramesh_token", "")
    if _is_local_request(ip):
        logger.debug("Local dev request — skipping trial/plan limits")
    else:
        # Check free trial for guest users (1 free generation)
        ok, reason = _check_free_trial(request)
        if not ok:
            return JSONResponse({"status": "error", "error": reason}, status_code=403)

        # Check plan limits for logged-in users. Count only THIS
        # client's completed jobs — the old global count billed every
        # job in the local DB (any user, any session) against the
        # requester's monthly limit.
        if token:
            parts_limit = PLAN_PARTS.get(plan, 3)
            import time as _time
            month_ago = _time.time() - 2592000
            with queue._lock:
                monthly_jobs = sum(
                    1 for job in queue._jobs.values()
                    if job.status == "done" and job.created_at >= month_ago
                    and job.client_ip == ip
                )
            if parts_limit < 9999 and monthly_jobs >= parts_limit:
                return JSONResponse(
                    {"status": "error", "error": f"Monthly limit reached ({monthly_jobs}/{parts_limit} parts on {plan} plan)."},
                    status_code=403,
                )

    temp_path = body.get("temp_path", "")
    filename = body.get("filename", "mesh.stl")
    if not temp_path or not Path(temp_path).exists():
        return JSONResponse({"status": "error", "error": "No mesh to process. Upload first."}, status_code=400)

    num_variants = max(1, min(4, int(body.get("num_variants", 1))))
    target_dims = body.get("target_dims")  # {width, height, thickness} or null
    job_id = queue.submit(filename, temp_path, client_ip=ip, num_variants=num_variants)
    # Pass target dimensions to job for post-cleaning resize
    job = queue.get(job_id)
    if job and target_dims:
        job._target_dims = target_dims
    # Step-through preference applied AT SUBMIT (the separate
    # per-job endpoint remains for mid-run toggling).
    step_through = body.get("step_through")
    if job is not None and step_through is not None:
        job.step_through = bool(step_through)
    logger.info("Pipeline queued: job_id=%s filename=%r ip=%s plan=%s variants=%d", job_id, filename, ip, plan, num_variants)

    # Set free trial cookie for guest users (marks the 1 free generation as used)
    response = JSONResponse({"status": "queued", "job_id": job_id})
    if not token:
        response.set_cookie("paramesh_free_used", "1", max_age=365*86400, samesite="lax", path="/")
    return response


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    """Submit a mesh for pipeline processing. Returns a job ID to poll."""
    plan = _get_user_plan(request)
    ip = _get_client_ip(request)
    logger.debug("Upload request: filename=%r ip=%s plan=%s", file.filename, ip, plan)
    if not _is_local_request(ip):
        ok, reason = _check_free_trial(request)
        if not ok:
            return JSONResponse({"status": "error", "error": reason}, status_code=403)
        parts_limit = PLAN_PARTS.get(plan, 3)
        if parts_limit < 9999 and queue.tokens_spent >= PLAN_TOKENS.get(plan, 100):
            return JSONResponse(
                {"status": "error", "error": f"Monthly limit reached on {plan} plan."},
                status_code=403,
            )

    content = await file.read()
    max_bytes = PLAN_UPLOAD_MB.get(plan, 10) * 1024 * 1024
    if len(content) > max_bytes:
        return JSONResponse(
            {"status": "error", "error": f"File too large. {plan} plan allows {PLAN_UPLOAD_MB[plan]}MB."},
            status_code=413,
        )

    suffix = _safe_suffix(file.filename)
    if suffix is None:
        return JSONResponse(
            {"status": "error", "error": "Unsupported file type. Use STL, OBJ, PLY, or 3MF."},
            status_code=400,
        )
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=upload_dir)
    tmp.write(content)
    tmp.close()

    job_id = queue.submit(file.filename, tmp.name, client_ip=ip)
    logger.info("Upload queued: job_id=%s filename=%r size=%d ip=%s",
                job_id, file.filename, len(content), ip)
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

    # Variant info
    variants_info = []
    for i, v in enumerate(getattr(job, 'variants', [])):
        v_steps = v.get('steps', [])
        variants_info.append({
            "index": i,
            "name": v.get('name', f'Variant {i+1}'),
            "strategy": v.get('strategy', ''),
            "num_steps": len(v_steps),
            "cad_volume": round(v.get('cad_volume', 0), 1),
        })

    result = {
        "job_id": job.id,
        "filename": job.filename,
        "status": job.status,
        "progress": job.progress,
        "stage": job.progress_stage,
        "error": job.error,
        "num_steps": len(job.steps),
        "steps": steps_info,
        # Live CAD-build streaming: the right pane refetches the newest step
        # whenever steps_version changes; building_live is True while the CAD
        # subprocess is still emitting steps.
        "steps_version": getattr(job, 'steps_version', 0),
        "building_live": getattr(job, 'building_live', False),
        "mesh_volume": round(job.mesh_volume, 1),
        "cad_volume": round(job.cad_volume, 1),
        "bbox_min": job.bbox_min,
        "bbox_max": job.bbox_max,
        "token_cost": job.token_cost,
        "tokens_used": queue.tokens_spent,
        "paused": job.paused,
        "variants": variants_info,
        "active_variant": getattr(job, 'active_variant', 0),
        "debug_summary": list(getattr(job, 'debug_summary', [])),
        # Feature-tree editor: recognized features (indices match the rebuild
        # bundle) + whether this build can be edited (has a saved bundle).
        "editable": bool(getattr(job, "cad_bundle_path", None)),
        "tree_features": [
            {"index": i, "type": f.type.value, "depth": round(float(f.depth), 3)}
            for i, f in enumerate(job.features.features)
        ] if (job.features is not None and hasattr(job.features, "features")) else [],
    }

    # Include progress geometry if pipeline is still running
    pv = getattr(job, 'progress_vertices', None)
    if job.status == "processing" and pv:
        result["progress_geometry"] = {
            "vertices": pv,
            "faces": job.progress_faces,
            "stage": job.progress_stage,
            "version": getattr(job, 'progress_version', 0),
        }
        if job.progress_colors:
            result["progress_geometry"]["colors"] = job.progress_colors
            result["progress_geometry"]["colors_per_face"] = getattr(job, 'progress_colors_per_face', False)
        if getattr(job, '_decomp_pieces', None):
            result["progress_geometry"]["pieces"] = job._decomp_pieces
        if job.progress_edges:
            result["progress_geometry"]["edges"] = job.progress_edges
        if job.progress_edge_colors:
            result["progress_geometry"]["edge_colors"] = job.progress_edge_colors

    return result


@app.get("/api/job/{job_id}/mesh")
async def job_mesh(job_id: str):
    """The pipeline's FINAL (decimated + aligned) mesh — the exact frame the CAD
    was built in. The Compare view uses this instead of the raw upload so the
    mesh and the model show in the same orientation (no flip)."""
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    cm = getattr(job, "cleaned", None)
    if cm is None:
        return JSONResponse({"error": "No cleaned mesh yet"}, status_code=404)
    return {
        "vertices": cm.vertices.flatten().tolist(),
        "faces": cm.faces.flatten().tolist(),
    }


@app.post("/api/job/{job_id}/edit")
async def edit_features(job_id: str, request: Request):
    """Rebuild the model from an EDITED feature tree (StudioCAD-style editing).

    Body: {deletes:[idx], suppress:[idx], finishes:[{type:'fillet'|'chamfer',
    value:mm}]}. The full edit state is sent each time and applied to the
    ORIGINAL build bundle (idempotent), so the result replaces a single "Edited"
    construction step rather than stacking. Returns the new step index; the
    frontend then loads /step/{index} as usual.
    """
    import asyncio

    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    bundle = getattr(job, "cad_bundle_path", None)
    if not bundle:
        return JSONResponse(
            {"error": "This model can't be edited yet (no saved build)."},
            status_code=400)
    body = await request.json()

    def _mid(f):
        m = f.get("edge_mid")
        if isinstance(m, (list, tuple)) and len(m) == 3:
            return [float(m[0]), float(m[1]), float(m[2])]
        return None

    edits = {
        "deletes": [int(i) for i in body.get("deletes", [])],
        "suppress": [int(i) for i in body.get("suppress", [])],
        "finishes": [
            {"type": str(f.get("type")), "value": float(f.get("value", 0)),
             "edge_mid": _mid(f)}
            for f in body.get("finishes", [])
            if f.get("type") in ("fillet", "chamfer") and float(f.get("value", 0)) > 0
        ],
    }
    from mesh2param.viz_server.rebuild import rebuild_from_bundle
    # Refresh the model STEP so edge-picking + download reflect the edited model.
    out_step = getattr(job, "cad_step_path", None)
    # Fast path: when the DELETE set is unchanged, reuse the cached base solid
    # (no export). The build seeds the base with no deletes, so a first
    # finish-only edit already hits it. Rebuilding the base only on delete change.
    cur_deletes = frozenset(edits["deletes"]) | frozenset(edits["suppress"])
    base_step = getattr(job, "cad_editbase_path", None)
    base_deletes = getattr(job, "_edit_base_deletes", frozenset())
    base_valid = bool(base_step and os.path.exists(base_step) and cur_deletes == base_deletes)
    res = await asyncio.to_thread(
        rebuild_from_bundle, bundle, edits, out_step, base_step, base_valid)
    if res.get("ok") and not base_valid:
        job._edit_base_deletes = cur_deletes  # base was rebuilt for this set
    if not res.get("ok"):
        return JSONResponse(
            {"error": res.get("error", "rebuild failed"),
             "detail": res.get("detail")}, status_code=422)

    from mesh2param.pipeline.export_cadquery import ConstructionStep
    step = ConstructionStep(
        name="Edited", tess_vertices=np.asarray(res["verts"]),
        tess_faces=np.asarray(res["faces"]), volume=float(res.get("cad_volume", 0.0)))
    idx = getattr(job, "_edit_step_index", None)
    if idx is not None and 0 <= idx < len(job.steps):
        job.steps[idx] = step
    else:
        job.steps.append(step)
        idx = len(job.steps) - 1
        job._edit_step_index = idx
    job.steps_version += 1
    job.cad_volume = float(res.get("cad_volume", job.cad_volume))
    return {
        "ok": True, "step_index": idx, "num_steps": len(job.steps),
        "faces": int(np.asarray(res["faces"]).shape[0]),
        "applied": res.get("applied", []), "warnings": res.get("warnings", []),
        "cad_volume": res.get("cad_volume"),
    }


@app.get("/api/job/{job_id}/edges")
async def job_edges(job_id: str):
    """Pickable CAD edges of the current built model (polylines + midpoints), for
    the hover-highlight / per-edge fillet picking in the viewer."""
    import asyncio
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    step_path = getattr(job, "cad_step_path", None)
    if not step_path or not os.path.exists(step_path):
        return JSONResponse({"error": "No model edges available yet."}, status_code=400)
    from mesh2param.viz_server.rebuild import extract_edges
    res = await asyncio.to_thread(extract_edges, step_path)
    if not res.get("ok"):
        return JSONResponse({"error": res.get("error", "edge extraction failed")},
                            status_code=422)
    return {"ok": True, "edges": res.get("edges", []),
            "bbox_min": res.get("bbox_min"), "bbox_max": res.get("bbox_max")}


@app.post("/api/thumbnail")
async def thumbnail(file: UploadFile = File(...)):
    """Lightweight geometry for a folder-browser thumbnail. Parsed with trimesh
    (handles 3MF scene transforms that the browser's 3MF loader can drop), so
    AI/.3mf previews always render. The client only calls this when its own
    client-side parse fails. Returns decimated {vertices, faces}."""
    import io
    import trimesh
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ("stl", "obj", "ply", "3mf"):
        return JSONResponse({"error": "unsupported"}, status_code=400)
    data = await file.read()
    try:
        loaded = trimesh.load(io.BytesIO(data), file_type=ext, force="mesh")
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    if loaded is None or not hasattr(loaded, "faces") or len(loaded.faces) == 0:
        return JSONResponse({"error": "no geometry"}, status_code=422)
    # Decimate to keep the thumbnail payload small (a raw AI mesh is 100k+ faces
    # = ~20MB of JSON). fast-simplification is deterministic and installed.
    verts = np.asarray(loaded.vertices, np.float64)
    faces = np.asarray(loaded.faces, np.int64)
    if len(faces) > 4000:
        try:
            import fast_simplification
            verts, faces = fast_simplification.simplify(
                verts, faces, target_count=4000)
        except Exception:
            # Last resort: keep payload bounded by dropping to the first 4000 tris.
            faces = faces[:4000]
    return {
        "vertices": np.asarray(verts, np.float32).flatten().tolist(),
        "faces": np.asarray(faces, np.int32).flatten().tolist(),
    }


@app.get("/api/job/{job_id}/download-step")
async def download_step(job_id: str):
    """Download the current (possibly edited) model as a STEP file."""
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    step_path = getattr(job, "cad_step_path", None)
    if not step_path or not os.path.exists(step_path):
        return JSONResponse({"error": "No STEP available."}, status_code=404)
    fname = (job.filename or "model").rsplit(".", 1)[0] + ".step"
    return FileResponse(step_path, media_type="application/step", filename=fname)


@app.post("/api/queue/reset")
async def queue_reset():
    """Last-resort recovery: kill the in-flight CadQuery subprocess,
    mark the currently processing job as errored, and replace the
    worker thread with a fresh one. Useful when a job is hung and the
    automatic watchdog hasn't caught it yet."""
    result = queue.force_reset()
    return result


@app.post("/api/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job. Unpauses and marks as error so the worker
    moves on, AND kills the CadQuery subprocess if one is mid-flight
    (otherwise the worker stays blocked inside subprocess.wait for up
    to 10 min and the next queued job can't start)."""
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job.paused = False
    job.status = "error"
    job.error = "Cancelled by user"
    proc = getattr(job, "_cad_subprocess", None)
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
    return {"status": "cancelled"}


@app.post("/api/job/{job_id}/continue")
async def job_continue(job_id: str):
    """Resume a paused job (e.g. after body decomposition review)."""
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job.paused = False
    return {"status": "resumed"}


@app.post("/api/job/{job_id}/step-through/{enabled}")
async def set_step_through(job_id: str, enabled: str):
    """Toggle step-through mode (pause after each visual stage)."""
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job.step_through = enabled == "1"
    # If turning off while paused, unpause immediately
    if not job.step_through and job.paused:
        job.paused = False
    return {"step_through": job.step_through}


@app.get("/api/pause-stages")
async def get_pause_stages():
    """List the pipeline stages that support a one-shot breakpoint, as
    [{key, name}, ...] in pipeline order. Frontend uses this to populate
    the 'pause at stage' dropdown."""
    from mesh2param.viz_server.job_queue import PAUSE_STAGES
    return {"stages": [{"key": k, "name": n} for k, n in PAUSE_STAGES]}


@app.post("/api/job/{job_id}/pause-at/{stage}")
async def set_pause_at_stage(job_id: str, stage: str):
    """Set a one-shot breakpoint: in non-step mode, pause at exactly this
    stage. Pass 'none' to clear. The pipeline resumes (and runs to
    completion without further pauses) when the user presses Continue."""
    from mesh2param.viz_server.job_queue import PAUSE_STAGE_KEYS
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if stage in ("none", "", "off"):
        job.pause_at_stage = None
        # If the job is currently stopped at the (now-cleared)
        # breakpoint and step mode is off, resume it.
        if job.paused and not job.step_through:
            job.paused = False
    elif stage in PAUSE_STAGE_KEYS:
        job.pause_at_stage = stage
    else:
        return JSONResponse({"error": f"Unknown stage: {stage}"}, status_code=400)
    return {"pause_at_stage": job.pause_at_stage}


@app.post("/api/job/{job_id}/variant/{variant_index}")
async def switch_variant(job_id: str, variant_index: int):
    """Switch the active build variant. Updates steps for all subsequent requests."""
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    variants = getattr(job, 'variants', [])
    if variant_index < 0 or variant_index >= len(variants):
        return JSONResponse({"error": "Invalid variant index"}, status_code=404)

    job.active_variant = variant_index
    job.steps = variants[variant_index]['steps']
    job.cad_volume = variants[variant_index].get('cad_volume', 0)

    steps_info = []
    for i, step in enumerate(job.steps):
        steps_info.append({
            "index": i, "name": step.name,
            "volume": round(step.volume, 1),
            "has_geometry": step.tess_vertices is not None and len(step.tess_vertices) > 0,
        })

    return {
        "active_variant": variant_index,
        "name": variants[variant_index].get('name', ''),
        "strategy": variants[variant_index].get('strategy', ''),
        "num_steps": len(job.steps),
        "steps": steps_info,
        "cad_volume": round(job.cad_volume, 1),
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

    result = {
        "name": step.name,
        "volume": round(step.volume, 1),
        "is_sketch": "sketch" in step.name.lower(),
        "vertices": step.tess_vertices.flatten().tolist(),
        "faces": step.tess_faces.flatten().tolist(),
    }
    if step.sketch_overlay is not None:
        result["sketch_overlay"] = step.sketch_overlay.flatten().tolist()
    if getattr(step, "overlay_arcs", None):
        result["overlay_arcs"] = step.overlay_arcs
    return result


@app.get("/api/job/{job_id}/compare")
async def job_compare(job_id: str, n_slices: int = 7):
    """Compare original mesh to final CAD solid via cross-section slicing.

    Returns accuracy score (0-100), per-slice IoU data, and a comparison figure.
    """
    job = queue.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job.status != "done":
        return JSONResponse({"error": f"Job not done (status={job.status})"}, status_code=400)
    if not job.steps:
        return JSONResponse({"error": "No construction steps"}, status_code=400)
    if job.cleaned is None:
        return JSONResponse({"error": "No cleaned mesh data"}, status_code=400)

    # Use the final construction step as the CAD solid
    final_step = job.steps[-1]
    if final_step.tess_vertices is None or final_step.tess_faces is None:
        return JSONResponse({"error": "Final step has no tessellation"}, status_code=400)

    # Determine extrude direction from the feature tree
    extrude_dir = np.array([0.0, 0.0, 1.0])
    if job.features is not None and hasattr(job.features, "features") and job.features.features:
        extrude_dir = np.array(job.features.features[0].direction, dtype=np.float64)

    from mesh2param.pipeline.compare import compare_slices

    try:
        result = compare_slices(
            cleaned=job.cleaned,
            cad_vertices=final_step.tess_vertices,
            cad_faces=final_step.tess_faces,
            extrude_dir=extrude_dir,
            n_slices=n_slices,
        )
    except Exception as e:
        logger.exception("Compare failed for job %s", job_id)
        return JSONResponse({"error": f"Comparison failed: {e}"}, status_code=500)

    # Copy figure to static dir so it can be served
    figure_url = ""
    if result.get("figure_path") and Path(result["figure_path"]).exists():
        import shutil
        compare_dir = STATIC_DIR / "compare"
        compare_dir.mkdir(exist_ok=True)
        dest_name = f"compare_{job_id}.png"
        dest = compare_dir / dest_name
        shutil.copy2(result["figure_path"], dest)
        figure_url = f"/static/compare/{dest_name}"
        # Clean up temp file
        try:
            Path(result["figure_path"]).unlink()
        except Exception:
            pass

    return {
        "score": result["score"],
        "per_slice": result["per_slice"],
        "figure_url": figure_url,
    }


# ── Legacy endpoints (redirect to latest job) ──

@app.get("/api/status")
async def status_legacy():
    """Legacy: return status of the most recent completed job."""
    with queue._lock:
        if not queue._jobs:
            return {"status": "idle", "tokens_used": 0, "num_steps": 0, "steps": [],
                    "mesh_volume": 0, "cad_volume": 0, "error": None,
                    "bbox_min": [0,0,0], "bbox_max": [0,0,0]}
        # Find the most recent "done" job (skip errored/cancelled ones)
        done_jobs = [j for j in queue._jobs.values() if j.status == "done"]
        if not done_jobs:
            return {"status": "idle", "tokens_used": 0, "num_steps": 0, "steps": [],
                    "mesh_volume": 0, "cad_volume": 0, "error": None,
                    "bbox_min": [0,0,0], "bbox_max": [0,0,0]}
        job = done_jobs[-1]

    steps_info = [{"index": i, "name": s.name, "volume": round(s.volume, 1),
                   "has_geometry": s.tess_vertices is not None and len(s.tess_vertices) > 0}
                  for i, s in enumerate(job.steps)]
    # Recognized features (for the feature-tree editor) — indices match the
    # rebuild bundle's tree.features, so deleting index i here deletes feature i
    # on rebuild. Only meaningful once a build with a saved bundle exists.
    tree_features = []
    if job.features is not None and hasattr(job.features, "features"):
        for i, f in enumerate(job.features.features):
            tree_features.append({
                "index": i, "type": f.type.value,
                "depth": round(float(f.depth), 3),
            })
    return {
        "job_id": job.id,
        "status": job.status,
        "tokens_used": queue.tokens_spent,
        "num_steps": len(job.steps),
        "steps": steps_info,
        "tree_features": tree_features,
        "editable": bool(getattr(job, "cad_bundle_path", None)),
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
    legacy_result = {
        "name": step.name, "volume": round(step.volume, 1),
        "is_sketch": "sketch" in step.name.lower(),
        "vertices": step.tess_vertices.flatten().tolist(),
        "faces": step.tess_faces.flatten().tolist(),
    }
    if getattr(step, "sketch_overlay", None) is not None:
        legacy_result["sketch_overlay"] = step.sketch_overlay.flatten().tolist()
    if getattr(step, "overlay_arcs", None):
        legacy_result["overlay_arcs"] = step.overlay_arcs
    return legacy_result


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

    # Parts-based usage: count completed jobs this month
    parts_limit = PLAN_PARTS.get(plan, 3)
    monthly_jobs = sum(1 for job in queue._jobs.values() if job.status == "done" and job.created_at >= month_ago)

    return {
        "plan": plan,
        "parts_used": monthly_jobs,
        "parts_limit": parts_limit if parts_limit < 9999 else -1,
        "tokens_spent": queue.tokens_spent,
        "tokens_limit": limit if limit != float("inf") else -1,
        "upload_limit_mb": PLAN_UPLOAD_MB.get(plan, 10),
        "total_jobs": total_jobs,
    }


# ── Free mesh cleaner (unchanged, no queue needed) ──

@app.post("/api/clean")
async def clean_mesh(file: UploadFile = File(...)):
    import trimesh
    logger.debug("Clean request: filename=%r", file.filename)
    suffix = _safe_suffix(file.filename)
    if suffix is None:
        return JSONResponse(
            {"status": "error", "error": "Unsupported file type. Use STL, OBJ, PLY, or 3MF."},
            status_code=400,
        )
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=upload_dir)
    content = await file.read()
    logger.debug("Clean file received: %r size=%d bytes", file.filename, len(content))
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
    except Exception:
        logger.exception("Clean failed for file %r", file.filename)
        return JSONResponse(
            {"status": "error", "error": "Could not clean this mesh. Check the file and try again."},
            status_code=500,
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/resize-mesh")
async def resize_mesh_file(body: dict):
    """Resize a previously uploaded mesh file in-place by target dimensions."""
    import trimesh
    temp_path = body.get("temp_path", "")
    if not temp_path or not Path(temp_path).exists():
        return JSONResponse({"error": "Mesh file not found"}, status_code=400)

    tw = body.get("width")
    th = body.get("height")
    tt = body.get("thickness")
    if not any([tw, th, tt]):
        return JSONResponse({"error": "No target dimensions"}, status_code=400)

    try:
        loaded = trimesh.load(temp_path)
        mesh = loaded.dump(concatenate=True) if hasattr(loaded, 'dump') else loaded
        bbox_size = mesh.vertices.max(0) - mesh.vertices.min(0)
        # Sort both target and actual dims largest→smallest to match them
        target = sorted([tw or 0, th or 0, tt or 0], reverse=True)
        actual = sorted(bbox_size.tolist(), reverse=True)
        scales = [t / a if t and t > 0 and a > 0 else 1.0 for t, a in zip(target, actual)]
        # Map scales back to XYZ by sorting actual dims' axis indices
        axis_order = sorted(range(3), key=lambda i: -bbox_size[i])
        scale_xyz = [1.0, 1.0, 1.0]
        for i, ax in enumerate(axis_order):
            scale_xyz[ax] = scales[i]

        center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
        mesh.vertices = (mesh.vertices - center) * scale_xyz + center
        mesh.export(temp_path)

        new_bbox = mesh.vertices.max(0) - mesh.vertices.min(0)
        logger.info("Resized mesh: %s → scales=%s new_dims=%s", temp_path, scale_xyz, new_bbox.tolist())
        return {
            "status": "ok",
            "dimensions": [round(d, 1) for d in sorted(new_bbox.tolist(), reverse=True)],
            "scale_xyz": [round(s, 4) for s in scale_xyz],
        }
    except Exception:
        logger.exception("Resize failed")
        return JSONResponse({"error": "Resize failed. Check the mesh and try again."}, status_code=500)


@app.post("/api/clean/resize")
async def resize_mesh(body: dict):
    import trimesh
    logger.debug("Resize request: scale=%s target_dims=%s/%s/%s",
                 body.get("scale"), body.get("target_width"),
                 body.get("target_height"), body.get("target_depth"))
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
    mesh.export(out_tmp.name)
    out_tmp.close()
    return {"status": "ok", "dimensions": (verts.max(0)-verts.min(0)).tolist(),
            "vertices": verts.flatten().tolist(), "faces": fcs.flatten().tolist(),
            "download_url": f"/api/clean/download/{Path(out_tmp.name).name}"}


@app.get("/api/clean/download/{filename}")
async def download_cleaned(filename: str):
    # Only serve plain .stl filenames that resolve directly inside the temp dir
    # (no separators, no "..", no symlink escapes).
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if Path(filename).name != filename or not filename.endswith(".stl"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    path = (temp_dir / filename).resolve()
    if path.parent != temp_dir or not path.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(path), media_type="application/octet-stream", filename="cleaned_mesh.stl")


@app.post("/api/waitlist")
async def waitlist(body: dict):
    from mesh2param.db.database import SessionLocal, WaitlistEntry
    email = body.get("email", "").strip().lower()
    logger.debug("Waitlist request: email=%r", email)
    if not email or "@" not in email:
        return JSONResponse({"error": "Valid email required"}, status_code=400)
    db = SessionLocal()
    try:
        existing = db.query(WaitlistEntry).filter(WaitlistEntry.email == email).first()
        if not existing:
            db.add(WaitlistEntry(email=email))
            db.commit()
            logger.info("New waitlist signup: %s", email)
        return {"status": "ok", "message": "You're on the list"}
    except Exception:
        logger.exception("Database error adding to waitlist: %s", email)
        return JSONResponse({"status": "error", "error": "Failed to join waitlist"}, status_code=500)
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
        logger.debug("Changelog: returned %d commits", len(commits))
        return {"commits": commits}
    except Exception:
        logger.exception("Failed to fetch git changelog")
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
    logger.debug("Contact form submission: name=%r email=%r msg_len=%d",
                 name, email, len(message))

    if not name or not email or not message:
        return JSONResponse({"error": "All fields required"}, status_code=400)

    # Save to database
    db = SessionLocal()
    try:
        ticket = SupportTicket(name=name, email=email, message=message)
        db.add(ticket)
        db.commit()
        logger.info("Support ticket saved: name=%r email=%r", name, email)
    except Exception:
        logger.exception("Failed to save support ticket from %r", email)
        return JSONResponse({"status": "error", "error": "Failed to save message"}, status_code=500)
    finally:
        db.close()

    # Try to send email notification (fail silently if SMTP not configured)
    try:
        import os
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_pass = os.getenv("SMTP_PASS", "")
        notify_email = os.getenv("NOTIFY_EMAIL", "trentondragon02@gmail.com")

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
    logger.debug("Support chat: message_len=%d", len(message))
    if not message:
        return {"reply": "Please type a question."}

    # Deflect casual/non-problem messages
    msg_lower = message.lower().strip()
    casual_patterns = ["hi", "hello", "hey", "sup", "yo", "what's up", "how are you",
                       "thanks", "thank you", "bye", "goodbye", "lol", "haha", "ok",
                       "cool", "nice", "test", "testing", "asdf"]
    if msg_lower in casual_patterns or len(message) < 3:
        return {"reply": "I'm here to help with specific questions about Paramesh AI — pricing, file formats, or technical issues. What do you need help with?"}

    if not is_available():
        # Fallback: simple FAQ responses
        if "price" in msg_lower or "cost" in msg_lower or "plan" in msg_lower:
            return {"reply": "Paramesh AI is currently in private beta. Plans will start from free (3 parts/month) up to Team ($149/mo, unlimited). Join the waitlist for early access pricing."}
        elif "format" in msg_lower or "stl" in msg_lower or "file" in msg_lower:
            return {"reply": "We support STL, OBJ, PLY, and 3MF files. The pipeline repairs topology and fixes normals automatically."}
        elif "onshape" in msg_lower:
            return {"reply": "Pro users can connect their Onshape account and push a converted part straight into their own Onshape document — use Export → Push to Onshape after a conversion (the menu's Connect Onshape links your account)."}
        elif "waitlist" in msg_lower or "launch" in msg_lower or "when" in msg_lower or "beta" in msg_lower:
            return {"reply": "Paramesh AI is in private beta. Join the waitlist on our homepage and we'll notify you when access opens up."}
        else:
            return {"reply": "I can help with questions about pricing, file formats, features, or technical issues. If you need more help, reach us at /contact."}

    prompt = f"""You are the support assistant for Paramesh AI (parameshai.com), a tool that converts
triangulated mesh files (STL/OBJ/3MF) into parametric CAD models. The product is currently in private beta (waitlist only).

Key facts:
- Currently in private beta — users join the waitlist for access
- Plans at launch: Free (3 parts/mo), Maker $19/mo (30 parts), Pro $49/mo (150 parts), Team $149/mo (unlimited)
- Supported formats: STL, OBJ, PLY, 3MF
- Pipeline: ingest → segment → fit → topology → recognize → export
- Features detected: holes, chamfers, fillets, pockets, bosses, slots
- Export: CadQuery STEP files, Onshape integration planned
- AI-powered dimension planning and feature recognition

Rules:
- Only answer questions about Paramesh AI — pricing, formats, features, or technical issues.
- If the user is just chatting or doesn't have a real question, tell them you're here for support questions and to reach out at /contact for anything else.
- Be concise (1-2 sentences max). Don't over-explain.
- If you don't know, suggest /contact.

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
    except Exception:
        logger.exception("Support chat AI call failed")
        return {"reply": "I'm having trouble connecting right now. Please try /contact to reach us directly."}


# ── Custom error pages ──
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    # API routes return JSON errors
    if request.url.path.startswith("/api/") or request.url.path.startswith("/auth/"):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    # HTML pages get the 404 page
    if exc.status_code == 404:
        return HTMLResponse(
            (STATIC_DIR / "404.html").read_text(encoding="utf-8"),
            status_code=404,
        )
    # Other errors (500, etc.) get the maintenance page
    return HTMLResponse(
        (STATIC_DIR / "maintenance.html").read_text(encoding="utf-8"),
        status_code=exc.status_code,
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error: %s", request.url.path)
    if request.url.path.startswith("/api/") or request.url.path.startswith("/auth/"):
        return JSONResponse({"error": "Internal server error"}, status_code=500)
    return HTMLResponse(
        (STATIC_DIR / "maintenance.html").read_text(encoding="utf-8"),
        status_code=500,
    )


def run(host: str | None = None, port: int | None = None):
    import uvicorn
    if host is None:
        host = os.getenv("PARAMESH_HOST", "127.0.0.1")
    if port is None:
        port = int(os.getenv("PARAMESH_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
