# mesh2param

Reverse-engineering pipeline that converts triangulated mesh files (STL/OBJ/PLY/3MF) of prismatic mechanical parts into fully parametric CAD models with a reconstructed feature tree, exported to Onshape via REST API + FeatureScript.

**Target parts:** brackets, mounts, plates — prismatic components with planar faces, cylindrical holes, pockets, bosses, fillets, and chamfers. Not designed for freeform/organic geometry.

## Pipeline

| Stage | Module | Status | Description |
|-------|--------|--------|-------------|
| 1 | `ingest.py` | Implemented | Load mesh, repair topology, fix normals, remove degenerate faces, optional remesh |
| 2 | `segment.py` | Implemented | Region-growing segmentation with normal clustering, curvature-based splitting, surface type classification |
| 3 | `fit.py` | Implemented | RANSAC + least-squares fitting for plane/cylinder/cone/sphere primitives |
| 4 | `topology.py` | Planned | Adjacency graph construction, fillet/chamfer/sharp edge detection |
| 5 | `recognize.py` | Planned | Rule-based feature recognition (extrusions, holes, pockets, fillets) |
| 6 | `export_onshape.py` | Planned | Sequential feature creation via Onshape REST API v6 |

## Setup

```bash
pip install -e ".[dev]"
```

For Onshape export, set credentials via environment variables:
```bash
export ONSHAPE_ACCESS_KEY=your_access_key
export ONSHAPE_SECRET_KEY=your_secret_key
# or
export ONSHAPE_OAUTH_TOKEN=your_oauth_token
```

## Usage

```bash
# Full pipeline: mesh -> Onshape parametric model
mesh2param convert input.stl --onshape-doc "My Document" --onshape-workspace main

# Dry run: output feature tree as JSON without Onshape export
mesh2param convert input.stl --dry-run --output feature_tree.json

# Inspect a specific pipeline stage with visualization
mesh2param inspect input.stl --stage segment --visualize
```

## Testing

```bash
# Run all tests
pytest

# Run all tests and generate visual reports
python run_visuals.py

# Run specific stages
python run_visuals.py --stage 1 2

# Visual tests only (generates PNG reports)
python run_visuals.py --visual-only

# Correctness tests only (no image output)
python run_visuals.py --correct-only

# Skip regenerating images that already exist
python run_visuals.py --no-overwrite
```

Visual reports are saved to `tests/visual_reports/` organized by stage. Each report image includes the test name, pass criteria, and analysis guide.

## Project Structure

```
mesh2param/
  pipeline/    Stage modules (ingest, segment, fit, topology, recognize, export)
  models/      Dataclasses (CleanedMesh, FaceRegion, FittedSurface, Feature, FeatureTree)
  viz/         Visualization classes per stage (PyVista 3D + matplotlib 2D)
  onshape/     API wrapper (client, auth, feature_builder, query_builder)
  utils/       Geometry helpers, RANSAC, parameter snapping, logging
  cli.py       Click CLI entry point
  config.py    Pydantic settings
tests/         Pytest suite with visual + correctness tests per stage
models/        Sample mesh files (Body1Mesh.3mf)
```

## Key Libraries

- **trimesh** — mesh loading, repair, boolean operations
- **PyVista** — 3D visualization and offscreen rendering
- **matplotlib** — 2D charts and tables
- **numpy / scipy** — numerical computation
- **scikit-learn** — clustering for segmentation
- **networkx** — graph algorithms for topology and feature dependencies
- **pydantic** — configuration and data validation

## Requirements

- Python 3.11+
- Linting: ruff
