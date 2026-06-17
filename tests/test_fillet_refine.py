"""Opt-in fillet-by-IoU sweep (export_cadquery._fillet_refine_solid) — the
volume/IoU-based alternative to the blend-graph + stitching fillet path.

A sharp-box solid whose TARGET has rounded edges should pick up a fillet (IoU
improves); a sharp-box solid whose target is also sharp should stay sharp.
"""
from __future__ import annotations

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")
trimesh = pytest.importorskip("trimesh")

from mesh2param.models.mesh import CleanedMesh  # noqa: E402
from mesh2param.pipeline.export_cadquery import (  # noqa: E402
    _fillet_refine_solid, _tessellate_solid,
)


def _cleaned_from_solid(solid):
    tv, tf = _tessellate_solid(solid)
    v = np.asarray(tv, float).reshape(-1, 3)
    f = np.asarray(tf, np.int64).reshape(-1, 3)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return CleanedMesh(vertices=v, faces=f,
                       normals=np.asarray(m.face_normals, float),
                       is_watertight=bool(m.is_watertight),
                       bbox=np.array([v.min(0), v.max(0)]))


def _max_min_edge_radius(solid):
    """Crude roundedness probe: how far the solid's surface is from sharp."""
    tv, _ = _tessellate_solid(solid)
    return np.asarray(tv).reshape(-1, 3)


def test_filleted_target_pulls_in_a_fillet():
    # Target = box with 3mm filleted edges; start solid = sharp box of same size.
    target_solid = cq.Workplane().box(40, 30, 20).edges().fillet(3.0)
    target = _cleaned_from_solid(target_solid)
    sharp = cq.Workplane().box(40, 30, 20)

    refined = _fillet_refine_solid(sharp, target, max_radius_frac=0.08, n_radii=6)
    # The refined solid should have LESS volume than the sharp box (fillets
    # remove corner material) — i.e. a fillet was actually applied.
    sharp_vol = abs(sharp.val().Volume())
    refined_vol = abs(refined.val().Volume())
    assert refined_vol < sharp_vol - 1.0, (sharp_vol, refined_vol)


def test_sharp_target_keeps_solid_sharp():
    # Target is a sharp box; start solid is the same sharp box -> no fillet.
    target_solid = cq.Workplane().box(40, 30, 20)
    target = _cleaned_from_solid(target_solid)
    sharp = cq.Workplane().box(40, 30, 20)

    refined = _fillet_refine_solid(sharp, target, max_radius_frac=0.08, n_radii=6)
    assert abs(abs(refined.val().Volume()) - abs(sharp.val().Volume())) < 1.0


def test_failure_is_noop():
    sharp = cq.Workplane().box(10, 10, 10)
    bad = CleanedMesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), np.int64),
                      normals=np.zeros((0, 3)), is_watertight=False,
                      bbox=np.zeros((2, 3)))
    refined = _fillet_refine_solid(sharp, bad)
    assert refined is sharp
