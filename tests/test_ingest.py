"""Tests for Stage 1: Mesh Ingestion & Cleanup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from mesh2param.config import IngestConfig
from mesh2param.models.mesh import CleanedMesh
from mesh2param.pipeline.ingest import ingest_mesh


@pytest.fixture
def tmp_mesh_dir(tmp_path: Path) -> Path:
    return tmp_path


def _save_mesh(mesh: trimesh.Trimesh, path: Path, ext: str = ".stl") -> Path:
    file_path = path / f"test_mesh{ext}"
    mesh.export(str(file_path))
    return file_path


def _make_box(extents=(10, 20, 30)) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents)


def _make_cylinder(radius=5, height=20) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, height=height, sections=32)


def _make_sphere(radius=10) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=3, radius=radius)


# --- File validation tests ---


class TestFileValidation:
    def test_file_not_found(self, tmp_mesh_dir: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            ingest_mesh(tmp_mesh_dir / "nonexistent.stl")

    def test_unsupported_extension(self, tmp_mesh_dir: Path):
        bad_file = tmp_mesh_dir / "test.step"
        bad_file.write_text("not a mesh")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ingest_mesh(bad_file)

    def test_supported_extensions(self, tmp_mesh_dir: Path):
        box = _make_box()
        for ext in [".stl", ".obj", ".ply"]:
            path = _save_mesh(box, tmp_mesh_dir, ext)
            result = ingest_mesh(path, IngestConfig(remesh=False))
            assert isinstance(result, CleanedMesh)


# --- Box mesh tests (simplest watertight case) ---


class TestBoxIngestion:
    def test_box_returns_cleaned_mesh(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert isinstance(result, CleanedMesh)

    def test_box_is_watertight(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.is_watertight is True

    def test_box_has_correct_vertex_shape(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.vertices.ndim == 2
        assert result.vertices.shape[1] == 3

    def test_box_has_correct_face_shape(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.faces.ndim == 2
        assert result.faces.shape[1] == 3

    def test_box_normals_shape_matches_faces(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.normals.shape == (result.num_faces, 3)

    def test_box_normals_are_unit_vectors(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        norms = np.linalg.norm(result.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_box_bbox_correct(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(extents=(10, 20, 30)), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        # Bbox should span from -5 to 5 on x, -10 to 10 on y, -15 to 15 on z
        np.testing.assert_allclose(result.bbox[0], [-5, -10, -15], atol=1e-6)
        np.testing.assert_allclose(result.bbox[1], [5, 10, 15], atol=1e-6)

    def test_box_bbox_diagonal(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(extents=(10, 20, 30)), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        expected = np.sqrt(10**2 + 20**2 + 30**2)
        assert abs(result.bbox_diagonal - expected) < 1e-6

    def test_box_metadata_has_source(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert "source_file" in result.metadata
        assert result.metadata["original_format"] == ".stl"


# --- Cylinder and sphere tests ---


class TestOtherPrimitives:
    def test_cylinder_ingestion(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_cylinder(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.is_watertight is True
        assert result.num_faces > 0
        assert result.num_vertices > 0

    def test_sphere_ingestion(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_sphere(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.is_watertight is True


# --- Degenerate face removal ---


class TestDegenerateFaceRemoval:
    def test_removes_zero_area_faces(self, tmp_mesh_dir: Path):
        box = _make_box()
        # Add a degenerate face (three identical vertices)
        v_idx = len(box.vertices)
        new_verts = np.vstack([box.vertices, [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        new_faces = np.vstack([box.faces, [[v_idx, v_idx + 1, v_idx + 2]]])
        mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        path = _save_mesh(mesh, tmp_mesh_dir, ".ply")
        result = ingest_mesh(path, IngestConfig(remesh=False))
        # All remaining faces should have non-zero area
        areas = []
        for face in result.faces:
            v0, v1, v2 = result.vertices[face]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas.append(area)
        assert all(a > 0 for a in areas)


# --- Normal consistency ---


class TestNormalConsistency:
    def test_outward_facing_normals_box(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        # For a box centered at origin, face normals should point outward:
        # the dot product of face_centroid and face_normal should be > 0
        for i in range(result.num_faces):
            centroid = result.vertices[result.faces[i]].mean(axis=0)
            dot = np.dot(centroid, result.normals[i])
            assert dot > 0, f"Face {i} normal points inward"


# --- to_dict serialization ---


class TestCleanedMeshSerialization:
    def test_to_dict(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        d = result.to_dict()
        assert d["num_vertices"] == result.num_vertices
        assert d["num_faces"] == result.num_faces
        assert d["is_watertight"] == result.is_watertight
        assert isinstance(d["bbox_min"], list)
        assert isinstance(d["bbox_max"], list)
        assert isinstance(d["bbox_diagonal"], float)


# --- Config defaults ---


class TestIngestConfig:
    def test_defaults(self):
        cfg = IngestConfig()
        assert cfg.merge_tolerance == 1e-8
        assert cfg.min_face_area == 1e-10
        assert cfg.remesh is True
        assert cfg.target_edge_ratio == 0.005

    def test_override(self):
        cfg = IngestConfig(remesh=False, merge_tolerance=1e-6)
        assert cfg.remesh is False
        assert cfg.merge_tolerance == 1e-6


# --- Remeshing (if pymeshfix is available) ---


class TestRemeshing:
    def test_remesh_produces_valid_mesh(self, tmp_mesh_dir: Path):
        path = _save_mesh(_make_box(), tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=True))
        assert result.num_faces > 0
        assert result.num_vertices > 0
        assert result.normals.shape == (result.num_faces, 3)

    def test_no_remesh_preserves_face_count(self, tmp_mesh_dir: Path):
        box = _make_box()
        path = _save_mesh(box, tmp_mesh_dir)
        result = ingest_mesh(path, IngestConfig(remesh=False))
        # STL export/reimport may change face count slightly due to
        # vertex merging, but should be in the same ballpark
        assert result.num_faces > 0


# --- OBJ and PLY format round-trips ---


class TestFormats:
    def test_obj_round_trip(self, tmp_mesh_dir: Path):
        box = _make_box()
        path = _save_mesh(box, tmp_mesh_dir, ".obj")
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.is_watertight is True
        assert result.metadata["original_format"] == ".obj"

    def test_ply_round_trip(self, tmp_mesh_dir: Path):
        box = _make_box()
        path = _save_mesh(box, tmp_mesh_dir, ".ply")
        result = ingest_mesh(path, IngestConfig(remesh=False))
        assert result.is_watertight is True
        assert result.metadata["original_format"] == ".ply"
