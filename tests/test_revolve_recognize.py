"""Body-of-revolution recognition + CadQuery revolve build (Phase 4).

A cylinder, a cone, and a stepped shaft are detected as revolves and rebuild to
the right volume; a box and a thin plate are NOT detected (they fall through to
the prismatic path).
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from mesh2param.models.mesh import CleanedMesh  # noqa: E402
from mesh2param.models.feature import FeatureType  # noqa: E402
from mesh2param.pipeline.revolve_recognize import (  # noqa: E402
    detect_body_of_revolution, try_revolve_tree,
)


def _cleaned(mesh: trimesh.Trimesh) -> CleanedMesh:
    return CleanedMesh(
        vertices=np.asarray(mesh.vertices, float),
        faces=np.asarray(mesh.faces, np.int64),
        normals=np.asarray(mesh.face_normals, float),
        is_watertight=bool(mesh.is_watertight),
        bbox=np.array([mesh.bounds[0], mesh.bounds[1]]),
    )


def test_cylinder_is_revolve():
    cyl = trimesh.creation.cylinder(radius=10.0, height=40.0, sections=64)
    res = detect_body_of_revolution(_cleaned(cyl))
    assert res is not None
    origin, axis, profile = res
    assert abs(abs(axis[2]) - 1.0) < 0.05  # cylinder axis is Z
    assert profile.shape[1] == 2
    # constant radius ~10
    assert abs(profile[:, 1].mean() - 10.0) < 1.0


def test_cone_is_revolve():
    cone = trimesh.creation.cone(radius=12.0, height=30.0, sections=64)
    res = detect_body_of_revolution(_cleaned(cone))
    assert res is not None
    _, _, profile = res
    # radius varies (linear) along the axis -> spread in radii
    assert profile[:, 1].max() - profile[:, 1].min() > 3.0


def test_box_is_not_revolve():
    box = trimesh.creation.box(extents=[20, 20, 40])
    assert detect_body_of_revolution(_cleaned(box)) is None


def test_rounded_hexagon_is_not_revolve():
    # A hexagonal prism (rounded polygon) has a coherent 6-fold harmonic and
    # must NOT be called a revolve — the failure mode the harmonic test fixes.
    hexp = trimesh.creation.cylinder(radius=15.0, height=40.0, sections=6)
    assert detect_body_of_revolution(_cleaned(hexp)) is None


def test_noisy_cylinder_still_detected():
    # Add radial noise (like an AI/Meshy mesh) — the harmonic test must still
    # see it as round (this is why we don't use a strict circle fit).
    cyl = trimesh.creation.cylinder(radius=12.0, height=40.0, sections=80)
    rng = np.random.default_rng(0)
    v = np.asarray(cyl.vertices, float)
    r = np.hypot(v[:, 0], v[:, 1])
    scale = 1.0 + rng.normal(0, 0.04, size=len(v))   # ~4% radial noise
    nz = r > 1e-6
    v[nz, 0] *= scale[nz]; v[nz, 1] *= scale[nz]
    noisy = trimesh.Trimesh(vertices=v, faces=cyl.faces, process=False)
    assert detect_body_of_revolution(_cleaned(noisy)) is not None


def test_try_revolve_tree_emits_revolve_feature():
    cyl = trimesh.creation.cylinder(radius=8.0, height=30.0, sections=64)
    tree = try_revolve_tree(_cleaned(cyl))
    assert tree is not None
    assert tree.features[0].type == FeatureType.REVOLVE
    sg = tree.features[0].sketch_geometry
    assert sg["type"] == "revolve" and len(sg["profile"]) >= 4


def test_revolve_builds_to_correct_volume():
    cq = pytest.importorskip("cadquery")
    from mesh2param.pipeline.export_cadquery import _build_revolve_result
    cyl = trimesh.creation.cylinder(radius=10.0, height=40.0, sections=96)
    c = _cleaned(cyl)
    tree = try_revolve_tree(c)
    assert tree is not None
    mesh_vol = float(abs(cyl.volume))
    result = _build_revolve_result(tree.features[0], c, mesh_vol)
    assert result is not None
    # pi r^2 h = pi*100*40 ~= 125664
    assert result.volume_error_pct < 10.0, result.volume_error_pct
    assert result.status == "PASS"


def test_cone_revolve_builds():
    cq = pytest.importorskip("cadquery")
    from mesh2param.pipeline.export_cadquery import _build_revolve_result
    cone = trimesh.creation.cone(radius=12.0, height=30.0, sections=96)
    c = _cleaned(cone)
    tree = try_revolve_tree(c)
    assert tree is not None
    result = _build_revolve_result(tree.features[0], c, float(abs(cone.volume)))
    assert result is not None
    assert result.volume_error_pct < 15.0, result.volume_error_pct
