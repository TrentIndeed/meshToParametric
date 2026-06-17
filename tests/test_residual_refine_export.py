"""Residual-refinement post-pass wired into export_cadquery
(_residual_refine_solid). Builds a CadQuery solid that deliberately under- or
over-shoots a target mesh and checks the post-pass patches it toward the target
(volume error drops) — and never makes a correct solid worse.
"""
from __future__ import annotations

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")
trimesh = pytest.importorskip("trimesh")

from mesh2param.models.mesh import CleanedMesh  # noqa: E402
from mesh2param.pipeline.export_cadquery import (  # noqa: E402
    _residual_refine_solid, _solid_volume,
)


def _cleaned(extents):
    """A CleanedMesh box centered at the origin (same frame as cq boxes)."""
    m = trimesh.creation.box(extents=extents)
    return CleanedMesh(
        vertices=np.asarray(m.vertices, float),
        faces=np.asarray(m.faces, np.int64),
        normals=np.asarray(m.face_normals, float),
        is_watertight=True,
        bbox=np.array([m.bounds[0], m.bounds[1]]),
    )


def _vol_err(vol, mesh_vol):
    return abs(vol - mesh_vol) / mesh_vol


def test_under_volume_solid_gets_patched_up():
    # Target is a 20x20x12 box; solid is only 20x20x6 (half the height).
    target = _cleaned([20, 20, 12])
    mesh_vol = 20 * 20 * 12.0
    solid = cq.Workplane().box(20, 20, 6)
    err_before = _vol_err(_solid_volume(solid), mesh_vol)

    refined, new_vol = _residual_refine_solid(solid, target, mesh_vol)
    err_after = _vol_err(new_vol, mesh_vol)
    assert err_after < err_before - 0.05, (err_before, err_after)


def test_over_volume_solid_gets_trimmed():
    # Target is a 20x20x6 box; solid is 20x20x12 (twice as tall).
    target = _cleaned([20, 20, 6])
    mesh_vol = 20 * 20 * 6.0
    solid = cq.Workplane().box(20, 20, 12)
    err_before = _vol_err(_solid_volume(solid), mesh_vol)

    refined, new_vol = _residual_refine_solid(solid, target, mesh_vol)
    err_after = _vol_err(new_vol, mesh_vol)
    assert err_after < err_before - 0.05, (err_before, err_after)


def test_correct_solid_is_left_unchanged():
    # Solid already matches the target -> no patches, volume essentially equal.
    target = _cleaned([15, 10, 8])
    mesh_vol = 15 * 10 * 8.0
    solid = cq.Workplane().box(15, 10, 8)
    vol_before = _solid_volume(solid)
    refined, new_vol = _residual_refine_solid(solid, target, mesh_vol)
    assert abs(new_vol - vol_before) / vol_before < 0.02


def test_failure_is_a_noop():
    # mesh_volume <= 0 -> returns the solid unchanged, no exception.
    target = _cleaned([10, 10, 10])
    solid = cq.Workplane().box(10, 10, 10)
    refined, vol = _residual_refine_solid(solid, target, 0.0)
    assert refined is solid
