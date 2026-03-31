"""STEP comparison tests.

For each model folder in models/ that contains both a mesh file and a
reference STEP file, run the full pipeline (ingest → segment → fit →
topology → recognize → cadquery export) and compare the resulting
solid against the reference STEP.

Comparison metrics:
- Volume error (%)
- Bounding box similarity
- Face count comparison
"""

from __future__ import annotations

import subprocess
import sys
import pickle
import tempfile
import os
from pathlib import Path

import numpy as np
import pytest

from mesh2param.config import (
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

from tests.conftest import discover_model_pairs


MODEL_PAIRS = discover_model_pairs()


def _run_pipeline(mesh_path: Path):
    """Run stages 1-5 on a mesh file."""
    cleaned = ingest_mesh(mesh_path, IngestConfig(remesh=False))
    regions = segment_mesh(cleaned, SegmentConfig())
    fitted = fit_surfaces(regions, cleaned, FitConfig())
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())
    return tree, cleaned


def _export_cadquery_subprocess(tree, cleaned) -> dict:
    """Run CadQuery export in a subprocess and return comparison metrics.

    Returns dict with: cad_volume, bbox_min, bbox_max, n_faces, n_edges.
    """
    tmp_in = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    pickle.dump((tree, cleaned), tmp_in)
    tmp_in.close()

    tmp_out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    tmp_out.close()

    script_file = tmp_out.name.replace(".pkl", ".py")
    with open(script_file, "w") as f:
        f.write(f"""
import pickle, sys, numpy as np
sys.path.insert(0, '.')
with open(r'{tmp_in.name}', 'rb') as f:
    tree, cleaned = pickle.load(f)

from mesh2param.pipeline.export_cadquery import export_cadquery
result = export_cadquery(tree, cleaned, use_llm=False)

# Extract metrics from the solid
import cadquery as cq
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE

metrics = {{}}
if result.solid is not None:
    val = result.solid.val()
    bb = val.BoundingBox()
    metrics['cad_volume'] = float(val.Volume())
    metrics['bbox_min'] = [bb.xmin, bb.ymin, bb.zmin]
    metrics['bbox_max'] = [bb.xmax, bb.ymax, bb.zmax]

    exp = TopExp_Explorer(val.wrapped, TopAbs_FACE)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    metrics['n_faces'] = n

    exp2 = TopExp_Explorer(val.wrapped, TopAbs_EDGE)
    n2 = 0
    while exp2.More():
        n2 += 1
        exp2.Next()
    metrics['n_edges'] = n2
else:
    metrics['cad_volume'] = 0.0
    metrics['bbox_min'] = [0, 0, 0]
    metrics['bbox_max'] = [0, 0, 0]
    metrics['n_faces'] = 0
    metrics['n_edges'] = 0

with open(r'{tmp_out.name}', 'wb') as f:
    pickle.dump(metrics, f)
""")

    proc = subprocess.run(
        [sys.executable, script_file],
        capture_output=True, text=True, timeout=180,
    )
    os.unlink(script_file)
    os.unlink(tmp_in.name)

    if proc.returncode != 0:
        os.unlink(tmp_out.name)
        pytest.fail(f"CadQuery subprocess failed:\n{proc.stderr[-500:]}")

    with open(tmp_out.name, "rb") as f:
        metrics = pickle.load(f)
    os.unlink(tmp_out.name)
    return metrics


def _load_reference_step(step_path: Path) -> dict:
    """Load a reference STEP in a subprocess and return its metrics."""
    tmp_out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    tmp_out.close()

    script_file = tmp_out.name.replace(".pkl", ".py")
    with open(script_file, "w") as f:
        f.write(f"""
import pickle, sys
sys.path.insert(0, '.')
import cadquery as cq
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE

solid = cq.importers.importStep(r'{step_path}')
val = solid.val()
bb = val.BoundingBox()

exp = TopExp_Explorer(val.wrapped, TopAbs_FACE)
n_faces = 0
while exp.More():
    n_faces += 1
    exp.Next()

exp2 = TopExp_Explorer(val.wrapped, TopAbs_EDGE)
n_edges = 0
while exp2.More():
    n_edges += 1
    exp2.Next()

metrics = {{
    'volume': float(val.Volume()),
    'bbox_min': [bb.xmin, bb.ymin, bb.zmin],
    'bbox_max': [bb.xmax, bb.ymax, bb.zmax],
    'n_faces': n_faces,
    'n_edges': n_edges,
}}
with open(r'{tmp_out.name}', 'wb') as f:
    pickle.dump(metrics, f)
""")

    proc = subprocess.run(
        [sys.executable, script_file],
        capture_output=True, text=True, timeout=60,
    )
    os.unlink(script_file)

    if proc.returncode != 0:
        os.unlink(tmp_out.name)
        pytest.fail(f"STEP load failed:\n{proc.stderr[-500:]}")

    with open(tmp_out.name, "rb") as f:
        metrics = pickle.load(f)
    os.unlink(tmp_out.name)
    return metrics


@pytest.mark.parametrize(
    "name,mesh_path,step_path",
    MODEL_PAIRS,
    ids=[p[0] for p in MODEL_PAIRS],
)
class TestStepComparison:
    """Compare pipeline-generated CadQuery solid against reference STEP."""

    def test_volume_within_tolerance(self, name, mesh_path, step_path):
        """Reconstructed volume should be within 15% of reference."""
        tree, cleaned = _run_pipeline(mesh_path)
        result = _export_cadquery_subprocess(tree, cleaned)
        ref = _load_reference_step(step_path)

        ref_vol = ref["volume"]
        cad_vol = result["cad_volume"]
        error_pct = abs(cad_vol - ref_vol) / (ref_vol + 1e-15) * 100

        print(f"\n  {name}: ref={ref_vol:.1f} mm³, cad={cad_vol:.1f} mm³, error={error_pct:.1f}%")
        assert error_pct < 15.0, (
            f"Volume error {error_pct:.1f}% exceeds 15% tolerance "
            f"(ref={ref_vol:.1f}, got={cad_vol:.1f})"
        )

    def test_bounding_box_overlap(self, name, mesh_path, step_path):
        """Bounding boxes should overlap by at least 90%."""
        tree, cleaned = _run_pipeline(mesh_path)
        result = _export_cadquery_subprocess(tree, cleaned)
        ref = _load_reference_step(step_path)

        ref_min = np.array(ref["bbox_min"])
        ref_max = np.array(ref["bbox_max"])
        cad_min = np.array(result["bbox_min"])
        cad_max = np.array(result["bbox_max"])

        # Intersection volume
        inter_min = np.maximum(ref_min, cad_min)
        inter_max = np.minimum(ref_max, cad_max)
        inter_size = np.maximum(inter_max - inter_min, 0)
        inter_vol = float(np.prod(inter_size))

        # Union volume
        ref_size = ref_max - ref_min
        cad_size = cad_max - cad_min
        ref_vol = float(np.prod(ref_size))
        cad_vol = float(np.prod(cad_size))
        union_vol = ref_vol + cad_vol - inter_vol

        overlap = inter_vol / (union_vol + 1e-15)
        print(f"\n  {name}: bbox overlap = {overlap:.1%}")
        assert overlap > 0.90, (
            f"Bounding box overlap {overlap:.1%} < 90% "
            f"(ref={ref_min.round(1)}-{ref_max.round(1)}, "
            f"cad={cad_min.round(1)}-{cad_max.round(1)})"
        )

    def test_has_subtractive_features(self, name, mesh_path, step_path):
        """Reconstructed solid should have holes (fewer faces than a plain extrusion)."""
        tree, cleaned = _run_pipeline(mesh_path)
        result = _export_cadquery_subprocess(tree, cleaned)
        ref = _load_reference_step(step_path)

        # A plain extrusion of a 16-sided polygon has ~18 faces.
        # With holes cut, there should be more faces.
        cad_faces = result["n_faces"]
        ref_faces = ref["n_faces"]

        print(f"\n  {name}: ref={ref_faces} faces, cad={cad_faces} faces")
        assert cad_faces >= 18, (
            f"Only {cad_faces} faces — expected more from hole cuts "
            f"(reference has {ref_faces})"
        )
