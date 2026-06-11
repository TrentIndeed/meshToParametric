"""Iterative extrusion engine: prismatic-segment decomposition.

Locks the engine's segment decisions on the reference parts and the
cross-section fidelity metric that gates extrusion modeling choices
(volume error can be gamed by an averaged cross-section; section IoU
cannot).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mesh2param.config import IngestConfig
from mesh2param.pipeline.extrusion import plan_extrusions
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.metrics import cross_section_fidelity

MODELS = Path(__file__).parent.parent / "models"


def _plan(rel):
    cleaned = ingest_mesh(str(MODELS / rel), IngestConfig(remesh=False))
    return cleaned, plan_extrusions(cleaned)


def test_tapered_plate_single_side_extrude():
    """BottomMount (drafted plate) is prismatic ONLY along its constant
    side axis: one segment, axis Y, ~full coverage. The taper must break
    thin-axis runs (no averaging through it) — this is the engine
    deriving the trapezoid side profile from first principles."""
    _, plan = _plan("Mesh2/BottomMount.3mf")
    assert len(plan.segments) == 1, (
        f"expected 1 segment, got {len(plan.segments)}"
    )
    assert plan.segments[0].axis_idx == 1
    assert plan.coverage_frac > 0.97


def test_flat_plate_primary_through_thickness():
    """Body1Mesh: primary segment through the plate thickness (Y)."""
    _, plan = _plan("Mesh1/Body1Mesh.3mf")
    assert plan.segments, "no segments found"
    assert plan.segments[0].axis_idx == 1
    assert plan.segments[0].gain_frac > 0.5


def test_stepped_plate_finds_coaxial_segments():
    """Front Plate is a stepped part (raised borders): the engine finds
    multiple coaxial Y segments covering nearly everything — the basis
    for future multi-extrude modeling."""
    _, plan = _plan("Mesh3/Front Plate.3mf")
    assert len(plan.segments) >= 2
    assert all(s.axis_idx == 1 for s in plan.segments)
    assert plan.coverage_frac > 0.95


def test_rackmount_base_plus_joins():
    """RackMount (one connected solid, not an assembly): largest segment
    is the base plate along Z; additional join segments cover the side
    plates. This is the part the user flagged as '3 extrusions that are
    connected', not bodies."""
    _, plan = _plan("Mesh5/RackMount.3mf")
    assert len(plan.segments) >= 3
    assert plan.segments[0].axis_idx == 2, (
        f"base should sweep along Z, got axis {plan.segments[0].axis_idx}"
    )
    assert plan.coverage_frac > 0.8


def test_fidelity_metric_identity():
    """A solid compared with itself scores IoU 1.0 on every section."""
    cleaned, _ = _plan("Mesh4/tube.3mf")
    fid = cross_section_fidelity(
        cleaned.vertices, cleaned.faces, cleaned.vertices, cleaned.faces,
    )
    assert fid["mean_iou"] > 0.999
    assert fid["worst_iou"] > 0.999


def test_fidelity_metric_penalizes_wrong_profile():
    """A bounding-box solid of the same volume as the part must score
    clearly below the part itself — the metric can't be gamed by
    volume-matching (the original sin of the slice ranker)."""
    import trimesh

    cleaned, _ = _plan("Mesh1/Body1Mesh.3mf")
    box = trimesh.creation.box(
        bounds=np.array([cleaned.bbox[0], cleaned.bbox[1]]),
    )
    fid = cross_section_fidelity(
        box.vertices, box.faces, cleaned.vertices, cleaned.faces,
    )
    assert fid["mean_iou"] < 0.75, (
        f"bbox stand-in scored {fid['mean_iou']:.3f} — metric too lenient"
    )
