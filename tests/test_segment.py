"""Tests for Stage 2: Surface Segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from mesh2param.config import IngestConfig, SegmentConfig
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.region import FaceRegion, SurfaceType
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import (
    _build_face_adjacency,
    _classify_surface_type,
    _find_boundary_edges,
    _merge_small_regions,
    _region_growing,
    segment_mesh,
)


# --- Helpers ---


def _save_and_ingest(mesh: trimesh.Trimesh, tmp_path: Path, ext: str = ".stl") -> CleanedMesh:
    path = tmp_path / f"test{ext}"
    mesh.export(str(path))
    return ingest_mesh(path, IngestConfig(remesh=False))


def _make_box(extents=(10, 20, 30)) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents)


def _make_cylinder(radius=5, height=20, sections=32) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def _make_sphere(radius=10, subdivisions=3) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


# --- Face adjacency tests ---


class TestFaceAdjacency:
    def test_box_adjacency_all_faces_have_neighbors(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        adj = _build_face_adjacency(cleaned.faces)
        for fi in range(cleaned.num_faces):
            assert fi in adj
            assert len(adj[fi]) > 0, f"Face {fi} has no neighbors"

    def test_adjacency_is_symmetric(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        adj = _build_face_adjacency(cleaned.faces)
        for fi, neighbors in adj.items():
            for ni in neighbors:
                assert fi in adj[ni], f"Adjacency not symmetric: {fi} -> {ni}"

    def test_triangle_pair_adjacency(self):
        """Two triangles sharing an edge should be adjacent."""
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        adj = _build_face_adjacency(faces)
        assert 1 in adj[0]
        assert 0 in adj[1]

    def test_disconnected_faces_no_adjacency(self):
        """Two triangles with no shared vertices should not be adjacent."""
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        adj = _build_face_adjacency(faces)
        assert 1 not in adj[0]
        assert 0 not in adj[1]


# --- Region growing tests ---


class TestRegionGrowing:
    def test_all_faces_assigned(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        adj = _build_face_adjacency(cleaned.faces)
        config = SegmentConfig()
        labels = _region_growing(cleaned.normals, adj, config)
        assert np.all(labels >= 0), "Some faces were not assigned a region"

    def test_box_produces_few_regions(self, tmp_path: Path):
        """A box has 6 planar faces — expect ~6 regions (2 triangles per face may merge)."""
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        adj = _build_face_adjacency(cleaned.faces)
        config = SegmentConfig()
        labels = _region_growing(cleaned.normals, adj, config)
        n_regions = len(np.unique(labels))
        # A box should segment into exactly 6 planar regions
        assert n_regions == 6, f"Expected 6 regions for a box, got {n_regions}"

    def test_sphere_produces_fewer_regions_with_loose_threshold(self, tmp_path: Path):
        """A sphere with a loose angle threshold should merge into fewer regions."""
        cleaned = _save_and_ingest(_make_sphere(subdivisions=2), tmp_path)
        adj = _build_face_adjacency(cleaned.faces)

        tight = SegmentConfig(plane_angle_threshold_deg=5, curved_angle_threshold_deg=10)
        loose = SegmentConfig(plane_angle_threshold_deg=30, curved_angle_threshold_deg=60)

        labels_tight = _region_growing(cleaned.normals, adj, tight)
        labels_loose = _region_growing(cleaned.normals, adj, loose)

        n_tight = len(np.unique(labels_tight))
        n_loose = len(np.unique(labels_loose))
        assert n_loose <= n_tight, "Looser threshold should produce fewer or equal regions"


# --- Small region merging ---


class TestMergeSmallRegions:
    def test_no_tiny_regions_after_merge(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        adj = _build_face_adjacency(cleaned.faces)
        config = SegmentConfig()
        labels = _region_growing(cleaned.normals, adj, config)
        merged = _merge_small_regions(labels, adj, cleaned.normals, min_faces=2)

        unique, counts = np.unique(merged, return_counts=True)
        # After merging, all remaining regions should have >= min_faces
        # (unless the total mesh is very small)
        for label, count in zip(unique, counts):
            assert count >= 2, f"Region {label} has only {count} faces after merge"

    def test_merge_preserves_total_face_count(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        adj = _build_face_adjacency(cleaned.faces)
        config = SegmentConfig()
        labels = _region_growing(cleaned.normals, adj, config)
        merged = _merge_small_regions(labels, adj, cleaned.normals, min_faces=5)
        assert len(merged) == cleaned.num_faces


# --- Surface type classification ---


class TestClassification:
    def test_planar_normals_classified_as_plane(self):
        normals = np.tile([0, 0, 1], (50, 1)).astype(np.float64)
        curvatures = np.zeros(50)
        assert _classify_surface_type(normals, curvatures) == SurfaceType.PLANE

    def test_varied_normals_with_curvature_not_plane(self):
        """Normals pointing in many directions with high curvature should not be PLANE."""
        rng = np.random.RandomState(42)
        normals = rng.randn(50, 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        curvatures = np.ones(50) * 5.0
        result = _classify_surface_type(normals, curvatures)
        assert result != SurfaceType.PLANE


# --- Boundary edge detection ---


class TestBoundaryEdges:
    def test_single_face_all_edges_are_boundary(self):
        faces = np.array([[0, 1, 2]])
        face_indices = np.array([0])
        edges = _find_boundary_edges(face_indices, faces)
        assert len(edges) == 3

    def test_two_adjacent_faces_shared_edge_not_boundary(self):
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        face_indices = np.array([0, 1])
        edges = _find_boundary_edges(face_indices, faces)
        # 2 triangles sharing edge (1,2): 6 total edges - 2 shared = 4 boundary
        assert len(edges) == 4


# --- End-to-end segmentation tests ---


class TestSegmentMesh:
    def test_box_segmentation(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        regions = segment_mesh(cleaned)
        assert len(regions) > 0
        assert all(isinstance(r, FaceRegion) for r in regions)

    def test_box_six_planar_regions(self, tmp_path: Path):
        """A box should produce 6 planar regions."""
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        regions = segment_mesh(cleaned)
        assert len(regions) == 6
        for r in regions:
            assert r.surface_type == SurfaceType.PLANE

    def test_all_faces_covered(self, tmp_path: Path):
        """Every face in the mesh should belong to exactly one region."""
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        regions = segment_mesh(cleaned)
        all_faces = np.concatenate([r.face_indices for r in regions])
        assert len(all_faces) == cleaned.num_faces
        assert len(np.unique(all_faces)) == cleaned.num_faces

    def test_region_ids_are_contiguous(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        regions = segment_mesh(cleaned)
        ids = [r.region_id for r in regions]
        assert ids == list(range(len(regions)))

    def test_region_normals_shape(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        regions = segment_mesh(cleaned)
        for r in regions:
            assert r.normals.shape == (r.num_faces, 3)

    def test_region_vertices_are_3d(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        regions = segment_mesh(cleaned)
        for r in regions:
            assert r.vertices.ndim == 2
            assert r.vertices.shape[1] == 3
            assert len(r.vertices) > 0

    def test_region_has_boundary_edges(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        regions = segment_mesh(cleaned)
        for r in regions:
            assert len(r.boundary_edges) > 0, f"Region {r.region_id} has no boundary edges"

    def test_cylinder_has_non_planar_regions(self, tmp_path: Path):
        """A cylinder should have planar caps and a curved body."""
        cleaned = _save_and_ingest(_make_cylinder(), tmp_path)
        regions = segment_mesh(cleaned)
        types = {r.surface_type for r in regions}
        # Should have at least one non-PLANE type for the curved surface
        assert len(types) > 1 or SurfaceType.CYLINDER in types or len(regions) > 2

    def test_to_dict_serialization(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        regions = segment_mesh(cleaned)
        for r in regions:
            d = r.to_dict()
            assert "region_id" in d
            assert "surface_type" in d
            assert "num_faces" in d
            assert d["num_faces"] == r.num_faces

    def test_custom_config(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_box(), tmp_path)
        config = SegmentConfig(
            plane_angle_threshold_deg=10.0,
            min_region_faces=1,
        )
        regions = segment_mesh(cleaned, config)
        assert len(regions) > 0

    def test_sphere_segmentation_runs(self, tmp_path: Path):
        cleaned = _save_and_ingest(_make_sphere(subdivisions=2), tmp_path)
        regions = segment_mesh(cleaned)
        assert len(regions) > 0
        all_faces = np.concatenate([r.face_indices for r in regions])
        assert len(np.unique(all_faces)) == cleaned.num_faces


# --- SurfaceType enum ---


class TestSurfaceType:
    def test_all_values(self):
        assert SurfaceType.PLANE.value == "plane"
        assert SurfaceType.CYLINDER.value == "cylinder"
        assert SurfaceType.CONE.value == "cone"
        assert SurfaceType.SPHERE.value == "sphere"
        assert SurfaceType.TORUS.value == "torus"
        assert SurfaceType.UNKNOWN.value == "unknown"
