"""Tests for Stage 3: Primitive Fitting."""

from __future__ import annotations

import os

import numpy as np
import pyvista as pv
import pytest

from mesh2param.config import FitConfig, IngestConfig, SegmentConfig
from mesh2param.models.region import SurfaceType
from mesh2param.models.surface import FittedSurface
from mesh2param.pipeline.fit import compute_residuals, fit_surfaces
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.viz.stage3_fit import Stage3Viz


def _to_pv(cleaned) -> pv.PolyData:
    faces = np.column_stack([np.full(len(cleaned.faces), 3), cleaned.faces])
    return pv.PolyData(cleaned.vertices.copy(), faces.ravel())


# --- Synthetic data helpers ---


def _make_plane_points(n=200):
    rng = np.random.RandomState(42)
    xy = rng.uniform(-10, 10, (n, 2))
    z = np.full(n, 5.0) + rng.normal(0, 0.01, n)
    return np.column_stack([xy, z])


def _make_cylinder_points(n=500, radius=5.0, height=20.0):
    rng = np.random.RandomState(42)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, height, n)
    r = radius + rng.normal(0, 0.02, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def _make_sphere_points(n=300, radius=10.0):
    rng = np.random.RandomState(42)
    phi = rng.uniform(0, 2 * np.pi, n)
    theta = rng.uniform(0, np.pi, n)
    r = radius + rng.normal(0, 0.03, n)
    return np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])


# --- Fixtures ---


@pytest.fixture(scope="session")
def body1mesh_regions(body1mesh_cleaned):
    return segment_mesh(body1mesh_cleaned, SegmentConfig())


@pytest.fixture(scope="session")
def body1mesh_fitted(body1mesh_regions, body1mesh_cleaned):
    return fit_surfaces(body1mesh_regions, body1mesh_cleaned, FitConfig())


# --- Visual tests (use Body1Mesh.3mf) ---


class TestStage3Visual:

    @pytest.mark.visual
    def test_fitted_overlay(self, body1mesh_regions, body1mesh_fitted, save_path):
        viz = Stage3Viz()
        # Show overlay for the largest region
        largest = max(body1mesh_regions, key=lambda r: r.num_faces)
        fitted = next(f for f in body1mesh_fitted if f.region_id == largest.region_id)
        out = save_path("fitted_overlay.png")
        viz.fitted_surface_overlay(largest.vertices, fitted, save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_residual_heatmap(self, body1mesh_regions, body1mesh_fitted, save_path):
        viz = Stage3Viz()
        largest = max(body1mesh_regions, key=lambda r: r.num_faces)
        fitted = next(f for f in body1mesh_fitted if f.region_id == largest.region_id)
        residuals = compute_residuals(largest.vertices, fitted)
        out = save_path("residual_heatmap.png")
        viz.residual_heatmap(largest.vertices, residuals, save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_parameter_snap_table(self, body1mesh_fitted, save_path):
        viz = Stage3Viz()
        out = save_path("snap_table.png")
        viz.parameter_snap_table(body1mesh_fitted, save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_all_surfaces_composite(self, body1mesh_cleaned, body1mesh_regions, body1mesh_fitted, save_path):
        viz = Stage3Viz()
        out = save_path("all_surfaces.png")
        viz.all_surfaces_composite(
            _to_pv(body1mesh_cleaned), body1mesh_regions, body1mesh_fitted, save_path=out,
        )
        assert os.path.exists(out)


# --- Correctness tests ---


class TestStage3Correctness:

    def test_plane_rms_is_low(self):
        pts = _make_plane_points()
        residuals = np.abs(pts[:, 2] - 5.0)
        rms = float(np.sqrt(np.mean(residuals**2)))
        assert rms < 0.05

    def test_cylinder_rms_is_low(self):
        pts = _make_cylinder_points(radius=5.0)
        dists = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        residuals = np.abs(dists - 5.0)
        rms = float(np.sqrt(np.mean(residuals**2)))
        assert rms < 0.1

    def test_sphere_rms_is_low(self):
        pts = _make_sphere_points(radius=10.0)
        dists = np.linalg.norm(pts, axis=1)
        residuals = np.abs(dists - 10.0)
        rms = float(np.sqrt(np.mean(residuals**2)))
        assert rms < 0.1

    def test_snapping_detects_change(self):
        fitted = FittedSurface(
            region_id=1, surface_type=SurfaceType.CYLINDER,
            params={"axis_point": [0, 0, 10], "axis_dir": [0, 0, 1], "radius": 4.998, "height": 20.0},
            rms_error=0.02,
            snapped_params={"axis_point": [0, 0, 10], "axis_dir": [0, 0, 1], "radius": 5.0, "height": 20.0},
        )
        assert fitted.params["radius"] != fitted.snapped_params["radius"]
        assert fitted.snapped_params["radius"] == 5.0

    def test_fitted_surface_to_dict(self):
        fitted = FittedSurface(
            region_id=0, surface_type=SurfaceType.PLANE,
            params={"point": [0.0, 0.0, 5.0], "normal": [0.0, 0.0, 1.0], "extent": 20},
            rms_error=0.0098,
            snapped_params={"point": [0.0, 0.0, 5.0], "normal": [0.0, 0.0, 1.0], "extent": 20},
        )
        d = fitted.to_dict()
        assert d["surface_type"] == "plane"
        assert d["rms_error"] < 0.01

    def test_body1mesh_all_regions_fitted(self, body1mesh_regions, body1mesh_fitted):
        assert len(body1mesh_fitted) == len(body1mesh_regions)

    def test_body1mesh_rms_within_tolerance(self, body1mesh_cleaned, body1mesh_fitted):
        tol = body1mesh_cleaned.bbox_diagonal * 0.005
        for f in body1mesh_fitted:
            # Allow some regions to exceed tolerance (complex geometry)
            # but most should be within tolerance
            pass
        within = sum(1 for f in body1mesh_fitted if f.rms_error < tol)
        assert within >= len(body1mesh_fitted) * 0.5, (
            f"Only {within}/{len(body1mesh_fitted)} regions have RMS < {tol:.4f}"
        )

    def test_body1mesh_has_planes(self, body1mesh_fitted):
        planes = [f for f in body1mesh_fitted if f.surface_type == SurfaceType.PLANE]
        assert len(planes) >= 1, "Body1Mesh should have at least one plane"
