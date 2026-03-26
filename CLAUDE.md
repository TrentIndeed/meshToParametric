# mesh2param

## Project Overview
Python pipeline that converts triangulated mesh files (STL/OBJ/PLY) of prismatic mechanical parts into fully parametric CAD models with a reconstructed feature tree, delivered to Onshape via REST API + FeatureScript.

Target parts: brackets, mounts, plates — prismatic components with planar faces, cylindrical holes, pockets, bosses, fillets, chamfers. Not freeform/organic geometry.

## Setup
```bash
pip install -e ".[dev]"
```

Onshape credentials via env vars: `ONSHAPE_ACCESS_KEY`, `ONSHAPE_SECRET_KEY`, or `ONSHAPE_OAUTH_TOKEN`.

## Running
```bash
mesh2param convert input.stl --onshape-doc "My Document" --onshape-workspace main
mesh2param convert input.stl --dry-run --output feature_tree.json
mesh2param inspect input.stl --stage segment --visualize
```

## Testing
```bash
pytest
```

## Package Structure
```
mesh2param/
  pipeline/    — stage modules: ingest.py, segment.py, fit.py, topology.py, recognize.py, export_onshape.py
  models/      — dataclasses: mesh.py, region.py, surface.py, feature.py, tree.py
  onshape/     — API wrapper: client.py, auth.py, feature_builder.py, query_builder.py
  utils/       — geometry helpers, RANSAC, snapping, logging
  cli.py       — Click-based CLI entry point
  config.py    — Pydantic settings
tests/         — pytest suite with fixtures per stage
examples/      — sample STL files and expected outputs
```

## Pipeline Stages
1. **Mesh Ingestion & Cleanup** (IMPLEMENTED) — load, repair, remesh → `CleanedMesh`
2. **Surface Segmentation** (IMPLEMENTED) — region-growing with normal clustering, curvature-based splitting, surface type classification → `FaceRegion` list
3. **Primitive Fitting** (IMPLEMENTED) — RANSAC + least-squares for plane/cylinder/cone/sphere → `FittedSurface` list
4. **Topology & Edge Classification** — adjacency graph, fillet/chamfer/sharp edge detection → `SurfaceGraph`
5. **Feature Recognition** — rule-based pattern matching → ordered `Feature` list
6. **Onshape Export** — sequential feature creation via Onshape REST API v6

## Key Libraries
trimesh, numpy, scipy, scikit-learn, networkx, pymeshfix, requests, click, pydantic

## Key Conventions
- Python 3.11+
- Linting: ruff
- Each stage is an independent module with well-defined input/output dataclasses
- All inter-stage data is JSON-serializable for caching and debugging
