"""Shared fixtures: benchmark meshes, pipeline runner, viz output dir."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

from mesh2param.config import ExportConfig, FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig
from mesh2param.pipeline.export_onshape import export_to_onshape
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

VISUAL_REPORTS_DIR = os.path.join(os.path.dirname(__file__), "visual_reports")
BODY1MESH_PATH = Path(__file__).parent.parent / "models" / "Body1Mesh.3mf"

# Map test module names to stage folder names
_STAGE_FOLDERS = {
    "test_stage1_cleanup": "stage1_cleanup",
    "test_stage2_segment": "stage2_segment",
    "test_stage3_fit": "stage3_fit",
    "test_stage4_topology": "stage4_topology",
    "test_stage5_features": "stage5_features",
    "test_stage6_export": "stage6_export",
}


@pytest.fixture(scope="session")
def visual_report_dir():
    os.makedirs(VISUAL_REPORTS_DIR, exist_ok=True)
    return VISUAL_REPORTS_DIR


@pytest.fixture(scope="session")
def l_bracket_mesh():
    """Generate a synthetic L-bracket for testing.

    Base plate: 100x60x10mm + vertical plate: 10x60x80mm.
    """
    base = trimesh.creation.box(extents=[100, 60, 10])
    base.apply_translation([50, 30, 5])

    vertical = trimesh.creation.box(extents=[10, 60, 80])
    vertical.apply_translation([5, 30, 50])

    try:
        bracket = trimesh.boolean.union([base, vertical])
        if not bracket.is_volume:
            raise ValueError("Union result is not a volume")
    except Exception:
        bracket = trimesh.util.concatenate([base, vertical])
        bracket.merge_vertices()
        bracket.fill_holes()
        bracket.fix_normals()

    return bracket


@pytest.fixture(scope="session")
def motor_mount_mesh():
    """Rectangular plate 120x80x8mm."""
    plate = trimesh.creation.box(extents=[120, 80, 8])
    plate.apply_translation([60, 40, 4])
    return plate


@pytest.fixture(scope="session")
def simple_box_mesh():
    """A simple box mesh for quick tests."""
    return trimesh.creation.box(extents=[10, 20, 30])


@pytest.fixture(scope="session")
def simple_cylinder_mesh():
    """A simple cylinder mesh for quick tests."""
    return trimesh.creation.cylinder(radius=5, height=20, sections=32)


@pytest.fixture(scope="session")
def simple_sphere_mesh():
    """A simple sphere mesh for quick tests."""
    return trimesh.creation.icosphere(subdivisions=3, radius=10)


@pytest.fixture
def save_path(request, visual_report_dir):
    """Generate a save path for CI screenshots.

    Files go directly into the stage folder (e.g. visual_reports/stage1_cleanup/normal_map.png).
    Overwrites existing files by default.

    Usage:
        def test_something(save_path):
            visualize(data, save_path=save_path("my_viz.png"))
    """

    def _make_path(filename):
        module_name = request.node.module.__name__.split(".")[-1]
        stage_folder = _STAGE_FOLDERS.get(module_name, module_name)
        stage_dir = os.path.join(visual_report_dir, stage_folder)
        os.makedirs(stage_dir, exist_ok=True)
        return os.path.join(stage_dir, filename)

    return _make_path


@pytest.fixture(scope="session")
def body1mesh_cleaned():
    """Load and clean the default Body1Mesh.3mf test model."""
    return ingest_mesh(BODY1MESH_PATH, IngestConfig(remesh=False))


@pytest.fixture(scope="session")
def body1mesh_regions(body1mesh_cleaned):
    """Segment the Body1Mesh into surface regions."""
    return segment_mesh(body1mesh_cleaned, SegmentConfig())


@pytest.fixture(scope="session")
def body1mesh_fitted(body1mesh_regions, body1mesh_cleaned):
    """Fit analytic primitives to Body1Mesh regions."""
    return fit_surfaces(body1mesh_regions, body1mesh_cleaned, FitConfig())


@pytest.fixture(scope="session")
def body1mesh_topology(body1mesh_regions, body1mesh_fitted, body1mesh_cleaned):
    """Build topology graph and classify edges for Body1Mesh."""
    return build_topology(body1mesh_regions, body1mesh_fitted, body1mesh_cleaned, TopologyConfig())


@pytest.fixture(scope="session")
def body1mesh_features(body1mesh_regions, body1mesh_fitted, body1mesh_topology, body1mesh_cleaned):
    """Recognize features for Body1Mesh."""
    graph, edges = body1mesh_topology
    return recognize_features(
        body1mesh_regions, body1mesh_fitted, graph, edges,
        body1mesh_cleaned, RecognizeConfig(),
    )


@pytest.fixture(scope="session")
def body1mesh_export(body1mesh_features, body1mesh_cleaned):
    """Dry-run export of Body1Mesh features."""
    return export_to_onshape(body1mesh_features, body1mesh_cleaned, ExportConfig(dry_run=True))


@pytest.fixture
def tmp_stl(tmp_path, simple_box_mesh):
    """Save a simple box mesh to a temp STL file."""
    path = tmp_path / "box.stl"
    simple_box_mesh.export(str(path))
    return path
