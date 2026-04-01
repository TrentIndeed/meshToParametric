# mesh2param

Reverse-engineering pipeline that converts triangulated mesh files (STL/OBJ/PLY/3MF) of prismatic mechanical parts into fully parametric CAD models with a reconstructed feature tree. Exports via Onshape REST API, CadQuery STEP files, or the built-in web viewer with AI assistant.

**Target parts:** brackets, mounts, plates — prismatic components with planar faces, cylindrical holes, pockets, bosses, fillets, and chamfers. Not designed for freeform/organic geometry.

## Pipeline

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `ingest.py` | Load mesh, repair topology, fix normals, remove degenerate faces, optional remesh |
| 2 | `segment.py` | Region-growing segmentation with normal clustering, curvature-based splitting |
| 3 | `fit.py` | RANSAC + least-squares fitting for plane/cylinder/cone/sphere primitives |
| 4 | `topology.py` | Surface adjacency graph, dihedral angles, sharp/fillet/chamfer/smooth edge classification |
| 5 | `recognize.py` | Feature recognition: base extrusion, through/blind holes, pockets, bosses, slots |
| 6a | `export_onshape.py` | FeatureScript generation + Onshape REST API v6 (dry-run or live) |
| 6b | `export_cadquery.py` | CadQuery solid reconstruction: sketch → extrude → cut holes → STEP export |

## Web Viewer

Built-in Three.js CAD viewer with feature tree, AI chatbot, construction timeline, and interactive orbit controls.

```bash
# Start the web UI
python -m mesh2param.viz_server.app

# Open http://127.0.0.1:8080 and upload a mesh file
```

Features:
- Dark theme with Fusion 360-style 3D viewport (MeshPhysicalMaterial, studio lighting)
- **Left sidebar**: Feature tree with typed icons (sketch, extrude, hole)
- **Right sidebar**: AI chatbot — describe parts, suggest edits, analyze dimensions
- **Bottom timeline**: 3D-rendered thumbnails per construction step, click to switch view
- **Dimension planner**: AI analyzes mesh after upload, suggests clean engineering dimensions before building
- Upload File / Send to Onshape action buttons
- User accounts with signup/login (SQLite + SHA-256 auth)

## Setup

```bash
pip install -e ".[dev]"
```

For CadQuery export (offline STEP):
```bash
pip install cadquery
```

For Onshape live export:
```bash
export ONSHAPE_ACCESS_KEY=your_access_key
export ONSHAPE_SECRET_KEY=your_secret_key
```

For AI features (chatbot, dimension planner, build ordering):
```bash
npm install -g @anthropic-ai/claude-code
claude "hello"  # one-time browser auth
```

## Usage

```bash
# Full pipeline via CLI
mesh2param convert input.stl --onshape-doc "My Document" --onshape-workspace main

# Dry run: output feature tree as JSON
mesh2param convert input.stl --dry-run --output feature_tree.json

# Inspect a specific pipeline stage
mesh2param inspect input.stl --stage segment --visualize

# Web viewer with AI assistant
python -m mesh2param.viz_server.app
```

## Testing

```bash
# Run all tests
pytest

# Run all tests and generate visual reports
python run_visuals.py

# Run specific stages
python run_visuals.py --stage 4 5 6

# Visual tests only (generates PNG/GIF reports)
python run_visuals.py --visual-only

# STEP comparison tests (auto-discovers model pairs in models/)
pytest tests/test_step_comparison.py -v
```

### Model Pairs

Drop a folder into `models/` with a mesh file + reference STEP file to auto-test reconstruction accuracy:

```
models/
  Mesh1/
    Body1Mesh.3mf    # input mesh
    Body1Step.step    # reference CAD model
  Mesh2/
    bracket.stl
    bracket.step
```

Visual reports are saved to `tests/visual_reports/` organized by stage.

## Project Structure

```
mesh2param/
  pipeline/      Stage modules (ingest, segment, fit, topology, recognize, export_onshape, export_cadquery)
  models/        Dataclasses (CleanedMesh, FaceRegion, FittedSurface, Feature, FeatureTree)
  viz/           Visualization classes per stage (PyVista 3D + matplotlib 2D)
  viz_server/    FastAPI web UI with Three.js viewer + AI chat
  db/            SQLite database (User, MeshJob, JobStep models)
  api/           REST endpoints (auth, chat, settings)
  onshape/       API wrapper (client, auth, feature_builder)
  ai.py          Claude CLI wrapper with prompt injection protection
  config.py      Pydantic settings for all 6 stages
tests/           Pytest suite with visual + correctness + STEP comparison tests
models/          Mesh + reference STEP pairs for validation
```

## Key Libraries

- **trimesh** — mesh loading, repair, cross-sections
- **cadquery** — parametric solid construction and STEP export
- **PyVista** — 3D visualization and offscreen rendering
- **Three.js** — web-based interactive 3D viewer with orbit controls
- **FastAPI** — web UI backend + REST API
- **SQLAlchemy** — database ORM (SQLite)
- **numpy / scipy** — numerical computation (SVD, least-squares, RANSAC)
- **scikit-learn** — k-means clustering for segmentation
- **networkx** — graph algorithms for topology and feature dependencies
- **pydantic** — configuration and data validation

## AI Integration

Claude CLI (OAuth, no API key needed) powers three features:

1. **AI Chatbot** — right-side panel in the web UI. Ask "describe this part", "make hole 1 larger", etc. Sends feature tree context with each message.
2. **Dimension Planner** — after upload, AI analyzes mesh dimensions, identifies part type, suggests clean engineering values (standard drill sizes, round numbers). User confirms before pipeline builds.
3. **Build Order Optimizer** — sends feature list to Claude for optimal construction sequence. Returns JSON build plan, deterministic code generation unchanged.

Security:
- System prefix constrains model to JSON-only CAD output
- Prompt sanitization strips injection patterns
- 8K prompt cap, 4K response cap
- 20 calls/session max, 5s rate limit
- `--max-turns 1` prevents tool use and looping

## Auth

User accounts ported from operatorDashboard:
- `POST /auth/signup` — create account (username + password)
- `POST /auth/login` — returns SHA-256 token
- `GET /auth/verify?token=xxx` — verify token → username
- `POST /settings/change-password` — change password
- `POST /settings/delete-account` — delete account + all data

Set `AUTH_SECRET` in `.env` for production (changing it breaks all existing passwords).

## Requirements

- Python 3.11+
- Linting: ruff
