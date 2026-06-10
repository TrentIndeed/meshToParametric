"""Tests for Stage 1: Mesh Ingestion & Cleanup."""

from __future__ import annotations

import os

import numpy as np
import pyvista as pv
import pytest

from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.viz.stage1_cleanup import Stage1Viz


def _to_pv(mesh) -> pv.PolyData:
    """Convert trimesh or CleanedMesh to PyVista PolyData."""
    verts = mesh.vertices if hasattr(mesh, "vertices") else mesh.vertices
    fcs = mesh.faces
    faces = np.column_stack([np.full(len(fcs), 3), fcs])
    return pv.PolyData(verts.copy(), faces.ravel())


@pytest.fixture(scope="session")
def cleaned_box(simple_box_mesh, tmp_path_factory):
    path = tmp_path_factory.mktemp("s1") / "box.stl"
    simple_box_mesh.export(str(path))
    return ingest_mesh(path, IngestConfig(remesh=False))


# --- Visual tests (use Body1Mesh.3mf by default) ---


class TestStage1Visual:

    @pytest.mark.visual
    def test_before_after_overlay(self, body1mesh_cleaned, save_path):
        viz = Stage1Viz()
        raw_pv = _to_pv(body1mesh_cleaned)
        out = save_path("before_after.png")
        viz.before_after_overlay(raw_pv, raw_pv, save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_normal_consistency_map(self, body1mesh_cleaned, save_path):
        viz = Stage1Viz()
        out = save_path("normal_map.png")
        viz.normal_consistency_map(_to_pv(body1mesh_cleaned), save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_mesh_quality_histogram(self, body1mesh_cleaned, save_path):
        viz = Stage1Viz()
        out = save_path("quality_hist.png")
        viz.mesh_quality_histogram(_to_pv(body1mesh_cleaned), save_path=out)
        assert os.path.exists(out)


# --- Correctness tests ---


class TestStage1Correctness:

    def test_cleaned_mesh_is_watertight(self, body1mesh_cleaned):
        assert body1mesh_cleaned.is_watertight is True

    def test_no_degenerate_faces(self, body1mesh_cleaned):
        for face in body1mesh_cleaned.faces:
            v0, v1, v2 = body1mesh_cleaned.vertices[face]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            assert area > 1e-10

    def test_normals_are_unit(self, body1mesh_cleaned):
        norms = np.linalg.norm(body1mesh_cleaned.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_l_bracket_loads(self, tmp_path, l_bracket_mesh):
        path = tmp_path / "l_bracket.stl"
        l_bracket_mesh.export(str(path))
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.num_faces > 0

    def test_motor_mount_loads(self, tmp_path, motor_mount_mesh):
        path = tmp_path / "motor_mount.stl"
        motor_mount_mesh.export(str(path))
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.num_faces > 0
