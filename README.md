# Paramesh AI

Reverse-engineering pipeline that converts triangulated mesh files (STL/OBJ/PLY/3MF) of prismatic mechanical parts into fully parametric CAD models. AI-powered feature recognition, construction timeline, and interactive web viewer.

**Live at [parameshai.com](https://parameshai.com)**

**Target parts:** brackets, mounts, plates — prismatic components with planar faces, cylindrical holes, pockets, bosses, fillets, and chamfers.

## Pipeline

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `ingest.py` | Load mesh, repair topology, fix normals, remove degenerate faces |
| 2 | `segment.py` | Region-growing segmentation with normal clustering, curvature splitting |
| 3 | `fit.py` | RANSAC + least-squares fitting for plane/cylinder/cone/sphere |
| 4 | `topology.py` | Surface adjacency graph, dihedral angles, edge classification |
| 5 | `recognize.py` | Feature recognition: extrusions, holes, pockets, chamfers, fillets |
| 6a | `export_onshape.py` | FeatureScript generation + Onshape REST API (dry-run or live) |
| 6b | `export_cadquery.py` | CadQuery solid: sketch → extrude → cut holes → STEP export |

## Web App

Three-panel CAD viewer with AI assistant at [parameshai.com/app](https://parameshai.com/app).

- **Left**: Feature tree with typed icons (sketch, extrude, hole, chamfer)
- **Center**: Interactive Three.js 3D viewer with orbit controls, XYZ grid
- **Right**: AI chatbot — describe parts, suggest edits, analyze dimensions
- **Bottom**: Timeline with 3D-rendered thumbnails per construction step
- **Upload flow**: preview mesh → complexity meter → AI thinking process → CAD model

### Free Tools

- **Mesh Cleaner** ([parameshai.com/clean](https://parameshai.com/clean)) — fix normals, remove degenerate faces, merge vertices, fill holes, resize. No signup required.
- **Learn** ([parameshai.com/learn](https://parameshai.com/learn)) — 20 SEO guides on mesh-to-CAD conversion

### Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with video hero, pricing, FAQ, waitlist |
| `/app` | CAD viewer tool (requires login) |
| `/clean` | Free mesh cleaner (requires login) |
| `/home` | Signed-in home — tool picker with usage bar |
| `/learn` | 20 SEO guide pages |
| `/changelog` | What's New + Roadmap + Git History |
| `/contact` | Support form (saves to DB + emails) |
| `/login` | Sign in / Sign up |
| `/privacy` | Privacy policy |
| `/terms` | Terms of service |

## Setup

```bash
pip install -e ".[dev]"
pip install cadquery shapely
```

For AI features:
```bash
npm install -g @anthropic-ai/claude-code
claude "hello"  # one-time auth
```

## Running

```bash
# Web app
python -m mesh2param.viz_server.app
# Open http://127.0.0.1:8080

# CLI
mesh2param convert input.stl --dry-run --output feature_tree.json
```

## Testing

```bash
pytest                                    # all tests
python run_visuals.py                     # visual reports
python run_visuals.py --stage 4 5 6       # specific stages
pytest tests/test_step_comparison.py -v   # STEP comparison
pytest tests/test_all_meshes.py -v        # all 5 meshes
```

### Model Pairs

Drop a folder into `models/` with a mesh + reference STEP to auto-test:

```
models/Mesh1/Body1Mesh.3mf + Body1Step.step
models/Mesh2/BottomMount.3mf
models/Mesh3/Front Plate.3mf
models/Mesh4/Bottom.3mf
models/Mesh5/Component3.3mf
```

## Architecture

```
mesh2param/
  pipeline/      — 6 stage modules (ingest → segment → fit → topology → recognize → export)
  models/        — dataclasses (CleanedMesh, FaceRegion, FittedSurface, Feature, FeatureTree)
  viz/           — PyVista 3D + matplotlib 2D visualizations per stage
  viz_server/    — FastAPI web app + Three.js viewer + job queue
  db/            — SQLite (User, MeshJob, JobStep, WaitlistEntry, SupportTicket)
  api/           — REST endpoints (auth, chat, settings)
  ai.py          — Claude CLI wrapper with injection protection + budget caps
  config.py      — Pydantic settings for all stages
tests/           — pytest suite (visual, correctness, STEP comparison, all-mesh)
models/          — mesh + reference STEP pairs
```

## Pricing (Token-Based)

| Plan | Price | Tokens/mo | Upload Limit | Features |
|------|-------|-----------|-------------|----------|
| Free | $0 | 100 | 10 MB | Cleaner unlimited, 3-5 CAD conversions |
| Starter | $9/mo | 1,000 | 50 MB | STEP export, AI dimension planner |
| Pro | $29/mo | 5,000 | 200 MB | Onshape + AI chat + priority support |
| Unlimited | $79/mo | Unlimited | 500 MB | Everything |

Token cost: ~20 for a simple bracket, ~55 for a complex mount with chamfers.

## Security

- Per-IP token tracking prevents free trial abuse (100 tokens + 10 jobs per IP)
- Complexity meter rejects meshes >10,000 faces or non-watertight >5,000 faces
- AI calls: system prefix, prompt sanitization, 8K cap, 20 calls/session, 5s rate limit
- Auth: SHA-256 + AUTH_SECRET, token cookies, session management
- Database: SQLite on Docker volume, survives rebuilds

## Deployment

```bash
# Deploy to Hetzner VPS (same as operatorDashboard)
python deploy.py --ip YOUR_VPS_IP --domain parameshai.com
```

Docker + Caddy on port 8081, alongside operatorDashboard.

## Key Libraries

trimesh, cadquery, pyvista, matplotlib, numpy, scipy, scikit-learn, networkx, fastapi, sqlalchemy, three.js
