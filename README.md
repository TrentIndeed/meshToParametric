# mesh2param

Reverse-engineering pipeline that converts triangulated mesh files (STL/OBJ/PLY/3MF) of prismatic mechanical parts into fully parametric CAD models with a reconstructed feature tree. Exports via Onshape REST API, CadQuery STEP files, or the built-in web viewer.

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

Built-in Three.js CAD viewer with feature tree, construction timeline, and interactive orbit controls.

```bash
# Start the web UI
python -m mesh2param.viz_server.app

# Open http://127.0.0.1:8080 and upload a mesh file
```

Features:
- Dark theme with Fusion 360-style 3D viewport
- Feature tree sidebar with typed icons (sketch, extrude, hole)
- Bottom timeline with 3D-rendered thumbnails per construction step
- Click any step to view the model at that construction stage
- Upload File / Send to Onshape action buttons

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

For LLM-guided build ordering (optional):
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

# Web viewer
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
  viz_server/    FastAPI web UI with Three.js viewer
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
- **Three.js** — web-based interactive 3D viewer
- **FastAPI** — web UI backend
- **numpy / scipy** — numerical computation (SVD, least-squares, RANSAC)
- **scikit-learn** — k-means clustering for segmentation
- **networkx** — graph algorithms for topology and feature dependencies
- **pydantic** — configuration and data validation

## AI Integration

Optional LLM-guided build ordering via Claude CLI (OAuth, no API key):
- Sends feature list to Claude for optimal construction sequence
- Returns JSON build plan, deterministic code generation unchanged
- Protected against prompt injection (system prefix, sanitization, budget caps)
- 20 calls/session max, 5s rate limit, 8K prompt cap, 4K response cap

## Requirements

- Python 3.11+
- Linting: ruff
