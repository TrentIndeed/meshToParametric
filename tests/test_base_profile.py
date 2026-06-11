"""Base-profile cutout policy: only LARGE cross-section cutouts (a tube
bore, a frame window) are baked into the base sketch as holes (one
annular/windowed extrude). Small bolt holes must stay separate drilled
features — baking them put a spurious hole in Body1Mesh's base sketch and
left the extrude tessellating as edges only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cadquery")

from mesh2param.config import (
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.export_cadquery import export_cadquery
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

MODELS = Path(__file__).parent.parent / "models"


def _build(mesh_path):
    cleaned = ingest_mesh(str(mesh_path), IngestConfig(remesh=False))
    regions = segment_mesh(cleaned, SegmentConfig())
    fitted = fit_surfaces(regions, cleaned, FitConfig())
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())
    return export_cadquery(tree, cleaned, use_llm=False)


def test_bracket_bolt_holes_not_baked_into_base():
    """Body1Mesh's small bolt holes are drilled as their own Hole step,
    not baked into the base sketch (which made the base extrude degenerate
    and rendered as edges only)."""
    res = _build(MODELS / "Mesh1" / "Body1Mesh.3mf")
    steps = [s.name for s in res.construction_steps]
    # The base extrude must be a real solid (non-trivial tessellation),
    # not the edge-only shell a holed/degenerate profile produced.
    base = next(s for s in res.construction_steps if s.name == "Extrude Base")
    assert base.tess_faces is not None and len(base.tess_faces) >= 12, (
        f"base extrude tessellation looks degenerate: "
        f"{0 if base.tess_faces is None else len(base.tess_faces)} faces"
    )
    assert base.volume > 0
    # Bolt holes appear as a separate drilled step.
    assert any("Hole" in s for s in steps), f"no separate hole step: {steps}"
    assert res.status == "PASS"


def test_tapered_plate_uses_trapezoid_side_profile():
    """BottomMount is a drafted plate (one face wider than the other). The
    base must extrude along the constant-length axis with the SILHOUETTE
    side profile — a TRAPEZOID (two parallel edges of different width) —
    not a thin-axis slice (a flat rectangle that misses the taper)."""
    import numpy as np

    from mesh2param.config import RecognizeConfig
    from mesh2param.pipeline.export_cadquery import _extract_profile_and_holes
    cleaned = ingest_mesh(str(MODELS / "Mesh2" / "BottomMount.3mf"), IngestConfig(remesh=False))
    regions = segment_mesh(cleaned, SegmentConfig())
    fitted = fit_surfaces(regions, cleaned, FitConfig())
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())
    base = tree.features[0]
    direction = np.asarray(base.direction, dtype=float)

    # Extrudes along Y (the constant-length axis), not the thin Z axis.
    assert int(np.argmax(np.abs(direction))) == 1, (
        f"tapered plate should extrude along axis 1, got dir {direction.tolist()}"
    )
    profiles, *_ = _extract_profile_and_holes(tree, cleaned, direction)
    prof = np.asarray(profiles[0], dtype=float)
    # Trapezoid: at the two ends of the thickness (v axis) the width (u
    # span) differs. A flat rectangle would have equal widths.
    v = prof[:, 1]
    lo_w = np.ptp(prof[v < v.mean()][:, 0])
    hi_w = np.ptp(prof[v >= v.mean()][:, 0])
    assert min(lo_w, hi_w) / max(lo_w, hi_w) < 0.95, (
        f"side profile is not a trapezoid: widths {lo_w:.1f} vs {hi_w:.1f}"
    )


def test_tube_bore_is_baked_annular():
    """The tube's large concentric bore IS baked into the base — annular
    sketch, one extrude, no separate bore cut."""
    res = _build(MODELS / "Mesh4" / "tube.3mf")
    steps = [s.name for s in res.construction_steps]
    assert steps == ["Sketch Profile", "Extrude Base"], (
        f"tube should be a single annular extrude, got {steps}"
    )
    assert res.status == "PASS"
    assert res.volume_error_pct < 5.0
