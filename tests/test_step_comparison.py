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
from mesh2param.viz.stage6_export import Stage6CadQueryViz

from tests.conftest import discover_model_pairs, VISUAL_REPORTS_DIR


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

    # Tessellate for visual comparison
    tess = val.tessellate(tolerance=1.0, angularTolerance=0.3)
    import numpy as np
    metrics['tess_verts'] = np.array([(v.x, v.y, v.z) for v in tess[0]], dtype=np.float64)
    metrics['tess_faces'] = np.array(tess[1], dtype=np.int64)
else:
    metrics['cad_volume'] = 0.0
    metrics['bbox_min'] = [0, 0, 0]
    metrics['bbox_max'] = [0, 0, 0]
    metrics['n_faces'] = 0
    metrics['n_edges'] = 0
    metrics['tess_verts'] = None
    metrics['tess_faces'] = None

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
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder

solid = cq.importers.importStep(r'{step_path}')
val = solid.val()
bb = val.BoundingBox()

# Every cylindrical face: bores, counterbores, fillet barrels.
cylinders = []
exp_c = TopExp_Explorer(val.wrapped, TopAbs_FACE)
while exp_c.More():
    face = TopoDS.Face_s(exp_c.Current())
    ad = BRepAdaptor_Surface(face)
    if ad.GetType() == GeomAbs_Cylinder:
        cyl = ad.Cylinder()
        loc = cyl.Axis().Location()
        d = cyl.Axis().Direction()
        cylinders.append({{
            'radius': float(cyl.Radius()),
            'axis_point': [loc.X(), loc.Y(), loc.Z()],
            'axis_dir': [d.X(), d.Y(), d.Z()],
        }})
    exp_c.Next()

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

import numpy as np
tess = val.tessellate(tolerance=1.0, angularTolerance=0.3)
metrics = {{
    'volume': float(val.Volume()),
    'bbox_min': [bb.xmin, bb.ymin, bb.zmin],
    'bbox_max': [bb.xmax, bb.ymax, bb.zmax],
    'n_faces': n_faces,
    'n_edges': n_edges,
    'cylinders': cylinders,
    'tess_verts': np.array([(v.x, v.y, v.z) for v in tess[0]], dtype=np.float64),
    'tess_faces': np.array(tess[1], dtype=np.int64),
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

    def test_hole_accuracy(self, name, mesh_path, step_path):
        """Recognized holes must land on reference bores with the right
        radius and axis.

        Aggregate volume error hides these failures: a bore fitted 2mm
        oversize barely moves volume but makes the part unusable. The
        0.35mm bound is tight enough to catch the old max-of-group
        radius rule (0.44mm error on Mesh1) while current accuracy is
        ~0.17mm.
        """
        from mesh2param.pipeline.metrics import (
            match_holes_to_reference, recognized_holes,
        )

        tree, cleaned = _run_pipeline(mesh_path)
        holes = recognized_holes(tree)
        ref = _load_reference_step(step_path)
        report = match_holes_to_reference(holes, ref["cylinders"])

        print(f"\n  {name}: {report['n_holes']} holes, "
              f"{report['n_matched']} matched, "
              f"{report['n_unmatched']} unmatched, "
              f"{report['n_axis_wrong']} wrong axis, "
              f"max radius error {report['max_radius_error']:.3f} mm")
        for m in report["matched"]:
            print(f"    {m['type']}: r={m['radius']:.3f} vs ref "
                  f"{m['ref_radius']:.3f} (err {m['radius_error']:.3f}, "
                  f"axis_dot {m['axis_dot']:.2f})")

        # Recall ratchet: minimum recognized-hole records per model.
        # Mesh1 has 6 physical bores (4 plate corner + 2 Z-side holes);
        # the full-length-cylinder rejection used to silently drop the
        # 4 corner holes.
        expected_min_holes = {"Mesh1": 6}
        assert report["n_holes"] >= expected_min_holes.get(name, 1), (
            f"Only {report['n_holes']} holes recognized "
            f"(expected >= {expected_min_holes.get(name, 1)})"
        )
        assert report["n_unmatched"] == 0, (
            f"{report['n_unmatched']} recognized holes have no colocated "
            f"reference cylinder: {report['unmatched']}"
        )
        assert report["n_axis_wrong"] == 0, (
            "Holes matched a reference bore but with the wrong axis "
            "direction (thin-plate override regression?)"
        )
        assert report["max_radius_error"] < 0.35, (
            f"Max hole radius error {report['max_radius_error']:.3f} mm "
            f">= 0.35 mm"
        )

    @pytest.mark.visual
    def test_visual_comparison(self, name, mesh_path, step_path):
        """Side-by-side render: reference STEP vs reconstructed CadQuery solid."""
        tree, cleaned = _run_pipeline(mesh_path)
        result = _export_cadquery_subprocess(tree, cleaned)
        ref = _load_reference_step(step_path)

        out_dir = os.path.join(VISUAL_REPORTS_DIR, "step_comparison")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}_comparison.png")

        _render_comparison(ref, result, name, out_path)
        assert os.path.exists(out_path)


def _render_comparison(ref: dict, cad: dict, name: str, save_path: str):
    """Render reference STEP vs reconstructed solid side-by-side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz = Stage6CadQueryViz()

    # Compute fixed bounds from both
    all_verts = []
    if ref.get("tess_verts") is not None:
        all_verts.append(ref["tess_verts"])
    if cad.get("tess_verts") is not None:
        all_verts.append(cad["tess_verts"])
    if not all_verts:
        return
    combined = np.concatenate(all_verts)
    fixed_bounds = (combined.min(axis=0) - 5, combined.max(axis=0) + 5)

    # Render reference STEP
    ref_img = viz._render_single_panel(
        ref["tess_verts"], ref["tess_faces"],
        "Reference STEP",
        f'{ref["volume"]:,.0f} mm³  |  {ref["n_faces"]} faces',
        fixed_bounds=fixed_bounds,
    )

    # Render reconstructed solid
    cad_img = viz._render_single_panel(
        cad["tess_verts"], cad["tess_faces"],
        "Reconstructed (CadQuery)",
        f'{cad["cad_volume"]:,.0f} mm³  |  {cad["n_faces"]} faces',
        fixed_bounds=fixed_bounds,
    )

    # Composite side-by-side with stats
    vol_err = abs(cad["cad_volume"] - ref["volume"]) / (ref["volume"] + 1e-15) * 100
    status_color = "#27AE60" if vol_err < 5 else "#F39C12" if vol_err < 15 else "#E74C3C"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#F0F2F4")

    axes[0].imshow(ref_img)
    axes[0].axis("off")
    axes[0].set_title("Reference STEP", fontsize=13, color="#2D3436", pad=8)

    axes[1].imshow(cad_img)
    axes[1].axis("off")
    axes[1].set_title("Reconstructed (CadQuery)", fontsize=13, color="#2D3436", pad=8)

    fig.suptitle(
        f"{name}  —  Volume error: {vol_err:.1f}%",
        fontsize=15, fontweight="bold", color=status_color, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#F0F2F4")
    plt.close(fig)
