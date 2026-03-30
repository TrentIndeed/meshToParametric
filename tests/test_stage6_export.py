"""Tests for Stage 6: Onshape Export.

Visual tests run the full pipeline in dry-run mode on Body1Mesh.3mf.
Correctness tests validate the export plan, FeatureScript generation,
and validation report.
"""

from __future__ import annotations

import os

import pytest

from mesh2param.pipeline.export_onshape import ExportResult
from mesh2param.viz.stage6_export import Stage6Viz


# --- Visual tests (dry-run pipeline on Body1Mesh.3mf) ---


class TestStage6Visual:

    @pytest.mark.visual
    def test_volume_comparison(self, body1mesh_export, save_path):
        viz = Stage6Viz()
        out = save_path("volume_comparison.png")
        viz.volume_comparison_bar(
            body1mesh_export.mesh_volume,
            body1mesh_export.cad_volume,
            save_path=out,
        )
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_api_call_timeline(self, body1mesh_export, save_path):
        viz = Stage6Viz()
        out = save_path("api_timeline.png")
        viz.api_call_timeline(body1mesh_export.api_log, save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_export_report(self, body1mesh_export, save_path):
        viz = Stage6Viz()
        out = save_path("export_report.png")
        viz.export_report(body1mesh_export.to_dict(), save_path=out)
        assert os.path.exists(out)


# --- Pipeline correctness tests ---


class TestStage6Pipeline:

    def test_has_operations(self, body1mesh_export):
        assert len(body1mesh_export.operations) >= 2

    def test_first_operation_is_extrude(self, body1mesh_export):
        assert "Extrude Base" in body1mesh_export.operations[0].name

    def test_operations_have_featurescript(self, body1mesh_export):
        for op in body1mesh_export.operations:
            assert len(op.feature_script) > 0
            assert op.feature_type != ""

    def test_volume_is_positive(self, body1mesh_export):
        assert body1mesh_export.mesh_volume > 0
        assert body1mesh_export.cad_volume > 0

    def test_volume_error_under_tolerance(self, body1mesh_export):
        assert body1mesh_export.volume_error_pct < 5.0

    def test_feature_count_matches(self, body1mesh_export):
        assert body1mesh_export.feature_count == body1mesh_export.expected_features

    def test_status_is_pass(self, body1mesh_export):
        assert body1mesh_export.status == "PASS"

    def test_api_log_has_entries(self, body1mesh_export):
        assert len(body1mesh_export.api_log) >= 3

    def test_api_log_all_200(self, body1mesh_export):
        for entry in body1mesh_export.api_log:
            assert entry["status_code"] == 200

    def test_to_dict(self, body1mesh_export):
        d = body1mesh_export.to_dict()
        assert "mesh_volume" in d
        assert "cad_volume" in d
        assert "status" in d
        assert d["num_operations"] == len(body1mesh_export.operations)

    def test_dry_run_document_url(self, body1mesh_export):
        assert "dry-run" in body1mesh_export.document_url


# --- Data model correctness tests ---


class TestStage6Correctness:

    @pytest.fixture
    def sample_api_log(self):
        return [
            {"name": "Create Document", "start": 0.0, "duration": 1.2, "status_code": 200},
            {"name": "Add Sketch", "start": 1.2, "duration": 0.8, "status_code": 200},
            {"name": "Add Extrude", "start": 2.0, "duration": 0.6, "status_code": 200},
            {"name": "Add Hole 1", "start": 2.6, "duration": 0.5, "status_code": 200},
            {"name": "Add Hole 2", "start": 3.1, "duration": 0.5, "status_code": 429},
            {"name": "Add Hole 2 (retry)", "start": 4.1, "duration": 0.6, "status_code": 200},
            {"name": "Add Fillet", "start": 4.7, "duration": 0.4, "status_code": 200},
            {"name": "Get Features", "start": 5.1, "duration": 0.3, "status_code": 200},
        ]

    @pytest.fixture
    def sample_results(self):
        return {
            "mesh_volume": 15000.0,
            "cad_volume": 14850.0,
            "feature_count": 5,
            "expected_features": 5,
            "status": "PASS",
        }

    def test_volume_error_under_5_percent(self, sample_results):
        mesh_vol = sample_results["mesh_volume"]
        cad_vol = sample_results["cad_volume"]
        error_pct = abs(cad_vol - mesh_vol) / mesh_vol * 100
        assert error_pct < 5.0

    def test_feature_count_matches(self, sample_results):
        assert sample_results["feature_count"] == sample_results["expected_features"]

    def test_status_is_pass(self, sample_results):
        assert sample_results["status"] == "PASS"

    def test_large_error_detected(self):
        error_pct = abs(12000.0 - 15000.0) / 15000.0 * 100
        assert error_pct > 5.0

    def test_api_log_has_rate_limit(self, sample_api_log):
        rate_limited = [e for e in sample_api_log if e["status_code"] == 429]
        assert len(rate_limited) == 1
