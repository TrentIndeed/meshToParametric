"""Tests for Stage 2: Surface Segmentation."""

from __future__ import annotations

import os

import numpy as np
import pyvista as pv
import pytest
import trimesh

from mesh2param.config import IngestConfig, SegmentConfig
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.viz.stage2_segment import Stage2Viz


def _to_pv(cleaned) -> pv.PolyData:
    faces = np.column_stack([np.full(len(cleaned.faces), 3), cleaned.faces])
    return pv.PolyData(cleaned.vertices.copy(), faces.ravel())


@pytest.fixture(scope="session")
def box_cleaned(simple_box_mesh, tmp_path_factory):
    path = tmp_path_factory.mktemp("s2") / "box.stl"
    simple_box_mesh.export(str(path))
    return ingest_mesh(path, IngestConfig(remesh=False))


@pytest.fixture(scope="session")
def box_regions(box_cleaned):
    return segment_mesh(box_cleaned, SegmentConfig())


@pytest.fixture(scope="session")
def body1mesh_regions(body1mesh_cleaned):
    return segment_mesh(body1mesh_cleaned, SegmentConfig())


# --- Visual tests (use Body1Mesh.3mf by default) ---


class TestStage2Visual:

    @pytest.mark.visual
    def test_region_color_map(self, body1mesh_cleaned, body1mesh_regions, save_path):
        viz = Stage2Viz()
        out = save_path("region_map.png")
        viz.region_color_map(_to_pv(body1mesh_cleaned), body1mesh_regions, save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_boundary_edges(self, body1mesh_cleaned, body1mesh_regions, save_path):
        viz = Stage2Viz()
        out = save_path("boundary_edges.png")
        viz.boundary_edges(_to_pv(body1mesh_cleaned), body1mesh_regions, body1mesh_cleaned.vertices, save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_region_stats_table(self, body1mesh_regions, save_path):
        viz = Stage2Viz()
        out = save_path("region_stats.png")
        viz.region_stats_table(body1mesh_regions, save_path=out)
        assert os.path.exists(out)


# --- Correctness tests ---


class TestStage2Correctness:

    def test_box_has_6_planar_regions(self, box_regions):
        assert len(box_regions) == 6

    def test_all_faces_assigned(self, box_cleaned, box_regions):
        all_indices = set()
        for r in box_regions:
            all_indices.update(r.face_indices.tolist())
        assert len(all_indices) == box_cleaned.num_faces

    def test_no_overlapping_regions(self, box_regions):
        seen = set()
        for r in box_regions:
            for fi in r.face_indices:
                assert fi not in seen, f"Face {fi} in multiple regions"
                seen.add(fi)

    def test_all_regions_classified_as_plane(self, box_regions):
        for r in box_regions:
            assert r.surface_type.value == "plane"

    def test_regions_have_boundary_edges(self, box_regions):
        for r in box_regions:
            assert len(r.boundary_edges) > 0

    def test_cylinder_segmentation(self, tmp_path):
        cyl = trimesh.creation.cylinder(radius=5, height=20, sections=32)
        path = tmp_path / "cyl.stl"
        cyl.export(str(path))
        cleaned = ingest_mesh(path, IngestConfig(remesh=False))
        regions = segment_mesh(cleaned, SegmentConfig())
        types = {r.surface_type.value for r in regions}
        assert len(regions) >= 2
        assert "plane" in types

    def test_sphere_segmentation(self, tmp_path):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=10)
        path = tmp_path / "sphere.stl"
        sphere.export(str(path))
        cleaned = ingest_mesh(path, IngestConfig(remesh=False))
        regions = segment_mesh(cleaned, SegmentConfig())
        assert len(regions) >= 1
        types = {r.surface_type.value for r in regions}
        assert types != {"plane"}, "Sphere should not be all-planar regions"
