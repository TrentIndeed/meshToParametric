"""Segmentation-free reconstruction (mesh2param.pipeline.contour_recon).

A plain box and an L-prism are reconstructed from the extrusion plan + residual
loop WITHOUT any segmentation; the rebuilt volume should match the input. This
is the Path-B builder used by the Phase 5 segmentation A/B.
"""
from __future__ import annotations

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")
trimesh = pytest.importorskip("trimesh")

from mesh2param.models.mesh import CleanedMesh  # noqa: E402
from mesh2param.pipeline.contour_recon import reconstruct_segmentation_free  # noqa: E402


def _cleaned(mesh: trimesh.Trimesh) -> CleanedMesh:
    return CleanedMesh(
        vertices=np.asarray(mesh.vertices, float),
        faces=np.asarray(mesh.faces, np.int64),
        normals=np.asarray(mesh.face_normals, float),
        is_watertight=bool(mesh.is_watertight),
        bbox=np.array([mesh.bounds[0], mesh.bounds[1]]),
    )


def test_box_reconstructs_without_segmentation():
    box = trimesh.creation.box(extents=[40, 24, 12])
    res = reconstruct_segmentation_free(_cleaned(box))
    assert res.solid is not None, res.info
    assert res.n_built >= 1
    # Volume recovered within the PASS tolerance, no segmentation involved.
    assert res.volume_error_pct < 10.0, res.volume_error_pct


def test_l_prism_reconstructs():
    # L-shaped prism extruded along Z: union of two boxes.
    a = trimesh.creation.box(extents=[40, 12, 10]); a.apply_translation([0, -10, 0])
    b = trimesh.creation.box(extents=[12, 40, 10]); b.apply_translation([-10, 0, 0])
    L = trimesh.util.concatenate([a, b])
    # Make it a clean watertight solid via convex-decomposition-free union proxy:
    L.merge_vertices()
    res = reconstruct_segmentation_free(_cleaned(L))
    assert res.solid is not None, res.info
    # Residual loop should bring the volume error into a reasonable range even
    # if the raw extrusion plan is imperfect on the concave shape.
    assert res.volume_error_pct < 25.0, res.volume_error_pct


def test_empty_on_non_watertight_returns_gracefully():
    # An open shell: plan_extrusions declines -> graceful empty result.
    plane = trimesh.creation.box(extents=[10, 10, 10])
    plane.faces = plane.faces[:6]  # break watertightness
    res = reconstruct_segmentation_free(_cleaned(plane))
    # Either it declined (no solid) or still produced something — must not raise.
    assert res is not None
