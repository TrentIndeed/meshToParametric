# mesh2param

## Project Overview
Python pipeline that converts triangulated mesh files (STL/OBJ/PLY/3MF) of prismatic mechanical parts into fully parametric CAD models with a reconstructed feature tree. Two export paths: Onshape REST API + FeatureScript, or CadQuery STEP files (offline). Web UI with Three.js viewer and AI chatbot.

Target parts: brackets, mounts, plates — prismatic components with planar faces, cylindrical holes, pockets, bosses, fillets, chamfers. Not freeform/organic geometry.

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

# Web UI (with AI chatbot + dimension planner)
python -m mesh2param.viz_server.app
# Open http://127.0.0.1:8080
```

## Testing
```bash
pytest                                    # all tests
python run_visuals.py                     # visual reports
python run_visuals.py --stage 4 5 6       # specific stages
pytest tests/test_step_comparison.py -v   # STEP comparison
```

## Package Structure
```
mesh2param/
  pipeline/      — stage modules: ingest.py, segment.py, fit.py, topology.py, recognize.py, export_onshape.py, export_cadquery.py
  models/        — dataclasses: mesh.py, region.py, surface.py, feature.py, tree.py
  viz/           — visualization per stage (PyVista 3D + matplotlib 2D)
  viz_server/    — FastAPI + Three.js web UI (app.py, static/index.html, static/app.js)
  db/            — SQLite database models (database.py: User, MeshJob, JobStep)
  api/           — REST endpoints (auth.py, chat.py, settings.py)
  onshape/       — API wrapper: client.py, auth.py, feature_builder.py
  ai.py          — Claude CLI wrapper with injection protection + budget caps
  config.py      — Pydantic settings (IngestConfig, SegmentConfig, FitConfig, TopologyConfig, RecognizeConfig, ExportConfig)
tests/           — pytest suite with fixtures per stage + STEP comparison
models/          — mesh + reference STEP pairs (models/Mesh1/, models/Mesh2/, ...)
```

## Pipeline Stages
1. **Mesh Ingestion & Cleanup** — load, repair, remesh → `CleanedMesh`
2. **Surface Segmentation** — region-growing with normal clustering, curvature-based splitting, surface type classification → `FaceRegion` list
3. **Primitive Fitting** — RANSAC + least-squares for plane/cylinder/cone/sphere → `FittedSurface` list
4. **Topology & Edge Classification** — adjacency graph, dihedral angles, sharp/fillet/chamfer/smooth detection → `SurfaceGraph` + `SurfaceEdge` list
5. **Feature Recognition** — base extrusion, through/blind holes, pockets, bosses, slots, fillets, chamfers → `FeatureTree` with dependency DAG
6. **Export**
   - **6a Onshape** — FeatureScript generation, simulated API timeline, volume validation (dry-run or live)
   - **6b CadQuery** — sketch → extrude → cut holes from scratch, STEP export, construction slideshow (GIF + PNG)

## Web UI (viz_server)
- FastAPI backend at `mesh2param/viz_server/app.py`
- Three.js frontend at `mesh2param/viz_server/static/`
- Endpoints: `POST /api/upload`, `GET /api/status`, `GET /api/step/{index}`, `POST /api/chat`, `POST /api/chat/analyze-mesh`
- Left sidebar: Feature tree with typed icons
- Center: 3D orbit viewer with XYZ grid, MeshPhysicalMaterial, studio lighting
- Right sidebar: AI chatbot — describe parts, suggest edits, analyze dimensions
- Bottom: Timeline with 3D-rendered thumbnails per step
- Dimension review modal: AI suggests clean dimensions after upload, before pipeline runs
- Runs CadQuery in subprocess to avoid VTK conflicts with Three.js

## Auth (from operatorDashboard)
- `db/database.py` — User, MeshJob, JobStep models (SQLite)
- `api/auth.py` — signup, login, verify (SHA-256 + AUTH_SECRET)
- `api/settings.py` — change password, delete account (cascades to jobs)
- Database auto-creates `data/mesh2param.db` on first startup

## AI Integration (ai.py + api/chat.py)
- Claude CLI wrapper using OAuth (no API key)
- Three features: chatbot, dimension planner, build order optimizer
- Security: system prefix, prompt sanitization, 8K prompt cap, 4K response cap, 20 calls/session, 5s rate limit
- Falls back gracefully when CLI unavailable

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
