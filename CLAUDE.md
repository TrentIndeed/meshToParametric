# Paramesh AI

## Project Overview
Python pipeline that converts triangulated mesh files (STL/OBJ/PLY/3MF) of prismatic mechanical parts into fully parametric CAD models with a reconstructed feature tree. Two export paths: Onshape REST API + FeatureScript, or CadQuery STEP files (offline). Web UI with Three.js viewer and AI chatbot.

Target parts: brackets, mounts, plates — prismatic components with planar faces, cylindrical holes, pockets, bosses, fillets, chamfers. Not freeform/organic geometry.

Target inflows (all the same "dirty mesh" problem): clean CAD STLs, **physical-part scans**, and **AI-generated meshes** (Meshy/Tripo). Market & roadmap (user base, the AI-gen→CAD difficulty ladder, why scans + AI meshes are one market): **`docs/ROADMAP.md`**. Scan/AI front-end technical plan + status: `docs/SCAN_RECOVERY_PLAN.md`.

## Active handoffs (read before changing the corresponding stage)
- **`docs/FILLET_AND_HOLE_HANDOFF.md`** — historical notes on Stages 4.5–6. The collinear-merge / reconciliation path it documents has been replaced by the stitched-fillets stage and the per-edge measured-radius matcher in `export_cadquery.py`. Read for background; current code is in `viz_server/job_queue.py` (stitched stage block) and `pipeline/export_cadquery.py` (`_apply_recognized_fillets_to_solid`).
- **`docs/SEGMENTATION_HANDOFF.md`** — open issue with corner-triangle classification at fillet-fillet junctions in Stage 2.

## Claude Operating Protocol (follow every session)
Designed for the $200 Max plan: maximize repo improvement per unit of allowance,
no blind rework. Full rationale in `docs/CLAUDE_USAGE_PLAN.md`.
- **Classify the task**: GEOMETRY/recon vs WEB/infra/docs. The visual rules below
  apply to GEOMETRY only — don't pay the render tax on a CSS/infra/doc fix.
- **See before you fix** (GEOMETRY): before diagnosing, run
  `scripts/diag_render_sheet.py <part>` (in the `mesh2param` container; see plan
  doc for the mount line) and READ the PNG. Diagnose from the heatmap
  (red=extra, blue=inside/missing) + overlay, NOT from IoU/volume alone — volume
  error hides shape failures (AI13: vol_err 1% but a hollow box modeled as a
  solid-with-pockets, IoU 0.55).
- **Verify before you claim**: re-render before/after. Clean-CAD changes →
  `pytest tests/test_step_comparison.py` (must stay 18/7). AI changes → container
  `diag_ai_build.py` on the AFFECTED parts for exact IoU. Report the delta; never
  say "works" without it.
- **Escalate only on a trigger**: "audit/all parts" → one diagnosis subagent per
  part + synthesis; change to a load-bearing stage (ingest/segment/fit/recognize)
  → an adversarial-verify subagent that tries to refute the fix; any pipeline
  behavior change → exactly ONE full AI-set container run (not per tweak).
- **Budget**: one composited sheet per part (≤1568px); recall memory before
  re-investigating, record after; product-facing AI features use the metered API,
  never this dev subscription.

## Setup
```bash
pip install -e ".[dev]"
pip install cadquery  # for CadQuery/STEP export
```

Onshape credentials via env vars: `ONSHAPE_ACCESS_KEY`, `ONSHAPE_SECRET_KEY`, or `ONSHAPE_OAUTH_TOKEN`.

AI features: `npm install -g @anthropic-ai/claude-code && claude "hello"`

Auth: set `AUTH_SECRET` in `.env` (changing it breaks existing password hashes).

## Running
```bash
# CLI
mesh2param convert input.stl --onshape-doc "My Document" --onshape-workspace main
mesh2param convert input.stl --dry-run --output feature_tree.json

# Web UI — native (no alpha-wrap; AI meshes degrade)
python -m mesh2param.viz_server.app    # http://127.0.0.1:8080
```

### Dev runs in DOCKER (Linux) — alpha-wrap + reliable exact IoU are Linux-only
pymeshlab's alpha-wrap plugin loads on Linux but NOT Windows, so the watertight
repair (ingest) and the exact-IoU arbiter are no-ops natively. Run the server +
all geometry diagnostics in containers:
```bash
./start-server.sh             # build+run the server image on http://127.0.0.1:8080
./start-server.sh --no-build  # just (re)start (seconds); --native for the no-Docker path
```
- Server image = top-level `Dockerfile` (`paramesh-server`); container stays up,
  source is bind-mounted (frontend edits are live on refresh; Python edits need a
  `docker restart paramesh-server`, NOT a rebuild). DB in a NAMED volume
  (`paramesh-data`) — a host bind-mount deadlocks SQLite on Windows.
- Lean diagnosis image = `_docker/Dockerfile` (`mesh2param`, pipeline + IoU only).
  Run a diag against mounted source/models:
  `MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)/mesh2param":/app/mesh2param -v "$(pwd -W)/models":/app/models -v "$(pwd -W)/scripts":/app/scripts mesh2param python scripts/<diag>.py`

## Testing & diagnostics
```bash
pytest tests/test_step_comparison.py -v   # clean-CAD guardrail — MUST stay 18/7
pytest tests/test_all_meshes.py -v        # all 5 reference meshes
```
Geometry diagnosis (run in the `mesh2param` container — see the mount line above):
- `scripts/diag_render_sheet.py <part>` — final mesh-vs-recon + signed-distance heatmap (SEE before you fix; per the Operating Protocol)
- `scripts/diag_stage_sheet.py <part>` — early stages: raw | cleaned+axes | segmentation | topology
- `scripts/diag_ai_build.py AIn ...` — per-part exact IoU + cad_vol_err (sequential)
- `scripts/diag_ai_parallel.py [--workers N]` — same, in a process pool with live per-stage logging (spawn; pymeshlab can't fork). LOCAL parallelism is limited by the pymeshlab-concurrency wall — for a full fast sweep use Modal (below).

## Package Structure
```
mesh2param/
  pipeline/      — stage modules: ingest.py, segment.py, fit.py, topology.py, recognize.py, export_onshape.py, export_cadquery.py
  models/        — dataclasses: mesh.py, region.py, surface.py, feature.py, tree.py
  viz/           — visualization per stage (PyVista 3D + matplotlib 2D)
  viz_server/    — FastAPI + Three.js web UI (app.py, job_queue.py, static/)
  db/            — SQLite database models (database.py: User, MeshJob, JobStep, WaitlistEntry, SupportTicket)
  api/           — REST endpoints (auth.py, chat.py, settings.py)
  onshape/       — REST client (client.py: API-key Basic auth, no HMAC) + FeatureTree→BTM builder (feature_builder.py); live export wired in pipeline/export_onshape.py, BTM conventions documented in feature_builder docstring
  ai.py          — Claude CLI wrapper with injection protection + budget caps
  config.py      — Pydantic settings (IngestConfig, SegmentConfig, FitConfig, TopologyConfig, RecognizeConfig, ExportConfig)
  viz_server/modal_runner.py — optional Modal dispatch (USE_MODAL); spawn job → isolated cloud container, poller finalizes
tests/           — pytest suite with fixtures per stage + STEP comparison + all-mesh parametric tests
models/          — mesh + reference STEP pairs (models/Mesh1/ through models/Mesh5/); AI test parts in models/AI1..AI15/
scripts/         — diag_render_sheet.py, diag_stage_sheet.py, diag_ai_build.py, diag_ai_parallel.py (geometry diagnosis; run in Docker)
modal_app.py     — Modal app (top-level): build_mesh = full pipeline + exact IoU in an isolated container; `modal run` for parallel dev, `modal deploy` for prod
Dockerfile / _docker/Dockerfile — server image (paramesh-server) / lean diagnosis image (mesh2param)
start-server.sh — build/run the server container (--no-build, --native flags); Ctrl+C stops + removes it, restarting gives a fresh container
```

## Pipeline Stages
1. **Mesh Ingestion & Cleanup** — load, repair, remesh → `CleanedMesh`. Repair order is fill_holes → poisson → **alpha-wrap (BEFORE voxel)** → voxel_repair → face cap → late alpha-wrap. Alpha-wrap runs first so round bores survive watertighting; voxel_repair (res 96) quantizes bores to blocky circles that fail cylinder detection. Alpha-wrap is Linux-only — runs in Docker, no-ops natively.
2. **Surface Segmentation** — region-growing with normal clustering, curvature-based splitting, surface type classification → `FaceRegion` list
3. **Primitive Fitting** — RANSAC + least-squares for plane/cylinder/cone/sphere → `FittedSurface` list
4. **Topology & Edge Classification** — adjacency graph, dihedral angles, sharp/fillet/chamfer/smooth detection → `SurfaceGraph` + `SurfaceEdge` list
5. **Feature Recognition** — base extrusion, through/blind holes, pockets, bosses, slots, fillets, chamfers → `FeatureTree` with dependency DAG
6. **Export**
   - **6a Onshape** — FeatureScript generation, simulated API timeline, volume validation (dry-run or live)
   - **6b CadQuery** — sketch → extrude → cut holes from scratch, STEP export, construction slideshow (GIF + PNG)

## Web UI (viz_server)
- FastAPI backend at `mesh2param/viz_server/app.py`
- Job queue at `mesh2param/viz_server/job_queue.py` — concurrent users, per-job state
- Three.js frontend at `mesh2param/viz_server/static/`
- Endpoints: `POST /api/preview`, `POST /api/run-pipeline`, `GET /api/job/{id}/status`, `GET /api/job/{id}/step/{i}`, `POST /api/chat`, `POST /api/chat/analyze-mesh`
- Left sidebar: Feature tree with typed icons — also an EDITOR: edit dimensions, and pick individual edges to add fillets/chamfers (hover-highlight + drag-arrow in the viewer, measured-radius snap); re-exports an edited STEP. Edits run off a cached base solid for speed.
- Center: 3D orbit viewer with XYZ grid, MeshPhysicalMaterial, studio lighting
- Right sidebar: AI chatbot — describe parts, suggest edits, analyze dimensions
- Bottom: Timeline with 3D-rendered thumbnails per step
- Folder browser: open a local folder (File System Access API — no "upload files" prompt) and get client-side thumbnails. Thumbnails render fully in-browser (STL/OBJ/PLY/3MF loaders + a direct fflate-based 3MF parser for AI exporters that split the mesh into a sub-`.model`); cached in-memory + localStorage by path|size|mtime.
- Preview shows the FULL-RES client mesh (not the decimated server preview); a quality toggle A/Bs full vs decimated so decimation damage is visible.
- Dimension review modal: AI suggests clean dimensions after upload, before pipeline runs
- Complexity meter: scores 0-100, rejects meshes >10K faces
- AI thinking process: streams stage-by-stage reasoning to chat during pipeline
- Runs CadQuery in subprocess to avoid VTK conflicts with Three.js
- Cloud-optional: with `USE_MODAL=1` the heavy pipeline dispatches to an isolated Modal container per job (see Cloud section); default off → local in-process worker.

## Cloud / parallel (Modal) — `modal_app.py`, full details in `docs/MODAL_DEPLOY.md`
One Modal function, `build_mesh`, is the unit of parallelism for BOTH dev and prod.
Each call runs the whole pipeline + exact IoU in its OWN isolated container — which
is the only thing that beats the pymeshlab-can't-run-concurrently wall (local
process pools stall on alpha-wrap parts).
- **Dev sweep (the 15 AI parts in parallel, ~3 min):** `modal run modal_app.py`
  (or `--parts AI9,AI12`). Proven mean IoU 0.813, wall ~175s. Windows: `PYTHONUTF8=1`
  + `python -m modal`; the async-shutdown noise after "✓ App completed" is harmless.
- **Prod (public launch):** `modal deploy modal_app.py` once, then set `USE_MODAL=1`
  + `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` on the server. The worker then SPAWNS each
  customer job into its own container; a poller finalizes them. Auth/billing/UI
  unchanged — only the heavy compute moves out. **FULL parity with local jobs.**
  GEOMETRY: cloud `build_mesh` calls the SAME `pipeline/convert.py:analyze_part`
  (ingest → decompose → per-body `recognize_body` → cross-body
  `stitch_fillet_clusters`) the server runs, then `export_cadquery(body_results,
  stitched_clusters)` — one source of truth, so cloud == local on multi-body parts
  and stitched fillets (full AI-set mean IoU 0.813, up from 0.780 flat). FEATURES:
  live step-streaming (container → shared `modal.Queue` partitioned by job id →
  poller drains into the dual-pane viewer); the feature-tree EDITOR + edge-picking
  + STEP download + mesh Compare (the container returns the rebuild bundle
  `(tree, cleaned, body_results, decomposition)` + STEP bytes; `modal_runner.
  _finalize` lands them on disk and sets `job.features/cleaned/cad_*_path`); and DB
  persistence + token accounting (poller mirrors the local completion path).
  **Keep synced:** the stitched-fillets stage is shared in `convert.py` — never
  re-inline it in `job_queue.py`, or cloud and local will drift.

## Pages
- `/` — Landing page (public) with video hero, scroll animation, pricing, FAQ, waitlist
- `/app` — CAD viewer tool (requires login)
- `/clean` — Free mesh cleaner + resize (requires login)
- `/home` — Signed-in home page with tool picker + usage bar
- `/learn` — 20 SEO guide pages (server-rendered from JSON)
- `/learn/{slug}` — Individual guide with FAQ schema markup
- `/changelog` — What's New + Roadmap + Git History tabs
- `/contact` — Support form → DB + email notification
- `/login` — Sign in / Sign up with cookie auth
- `/privacy`, `/terms` — Legal pages
- `/sitemap.xml` — Dynamic sitemap (28 pages)
- `/robots.txt` — Crawler instructions

## Auth (from operatorDashboard)
- `db/database.py` — User, MeshJob, JobStep, WaitlistEntry, SupportTicket models (SQLite)
- `api/auth.py` — signup, login, verify (PBKDF2-SHA256 salted hashes; legacy SHA-256 hashes upgraded on login; signed expiring tokens; per-IP rate limit; developer accounts via DEVELOPER_USERNAMES env var)
- `api/settings.py` — change password, delete account (cascades to jobs)
- Per-IP token tracking prevents free trial abuse (100 tokens + 10 jobs per IP)
- Database auto-creates `data/mesh2param.db` on first startup

## AI Integration (ai.py + api/chat.py)
- Claude CLI wrapper using OAuth (no API key)
- Three features: chatbot, dimension planner, build order optimizer
- AI support bot widget on landing + cleaner pages
- Security: system prefix, prompt sanitization, 8K prompt cap, 4K response cap, 20 calls/session, 5s rate limit
- Falls back gracefully when CLI unavailable

## Token Pricing
- Free: 100 tokens, 10MB uploads, mesh cleaner unlimited
- Starter $9/mo: 1,000 tokens, 50MB
- Pro $29/mo: 5,000 tokens, 200MB, Onshape + AI chat
- Unlimited $79/mo: unlimited tokens, 500MB
- Cost: ~20 tokens/simple bracket, ~55 tokens/complex mount

## Key Libraries
trimesh, cadquery, pyvista, matplotlib, numpy, scipy, scikit-learn, networkx, fastapi, sqlalchemy, three.js, pydantic

## Key Conventions
- Python 3.11+
- Linting: ruff
- Each stage is an independent module with well-defined input/output dataclasses
- All inter-stage data is JSON-serializable for caching and debugging
- CadQuery imported lazily (subprocess) to avoid VTK conflicts with PyVista
- Model pairs in `models/` auto-discovered for STEP comparison tests
- Font: OpenAI Sans throughout the web UI
- Deployment: Docker + Caddy on Hetzner VPS (port 8081)
