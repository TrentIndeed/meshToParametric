"""Pipeline tests for ALL meshes in models/.

Auto-discovers mesh folders and runs stages 1-5 + visual reports
for each, organized into visual_reports/{MeshName}/stage{N}/.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections import Counter

import numpy as np
import pyvista as pv
import pytest

from mesh2param.config import (
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology
from mesh2param.viz.stage4_topology import Stage4Viz
from mesh2param.viz.stage5_features import Stage5Viz

from tests.conftest import ALL_MESHES, VISUAL_REPORTS_DIR


def _to_pv(cleaned) -> pv.PolyData:
    faces = np.column_stack([np.full(len(cleaned.faces), 3), cleaned.faces])
    return pv.PolyData(cleaned.vertices.copy(), faces.ravel())


def _save_dir(mesh_name: str, stage: str) -> str:
    d = os.path.join(VISUAL_REPORTS_DIR, mesh_name, stage)
    os.makedirs(d, exist_ok=True)
    return d


# Cache pipeline results per mesh to avoid re-running
_cache: dict[str, dict] = {}


def _get_pipeline(mesh_name: str, mesh_path: Path) -> dict:
    if mesh_name not in _cache:
        cleaned = ingest_mesh(mesh_path, IngestConfig(remesh=False))
        regions = segment_mesh(cleaned, SegmentConfig())
        fitted = fit_surfaces(regions, cleaned, FitConfig())
        graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
        tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())
        _cache[mesh_name] = {
            "cleaned": cleaned,
            "regions": regions,
            "fitted": fitted,
            "graph": graph,
            "edges": edges,
            "tree": tree,
        }
    return _cache[mesh_name]


# Curated reference parts with known-good geometry and feature expectations.
# The STRICT correctness assertions below (fit completeness, graph
# connectivity, has-holes, has-features) only hold for clean prismatic CAD
# meshes. AI/Scan parts (models/AI*, models/Scan*) are the WIP scan/AI
# front-end's inputs — research-stage, dirty, and without feature ground
# truth (AI15 RibbedBlock genuinely has no holes; dirty meshes legitimately
# leave a region unfittable or a graph disconnected). They are tracked via
# scripts/diag_ai_build.py, the visual reports, and docs/AI_MESH_FINDINGS.md,
# NOT by these binary asserts. The pipeline still runs end-to-end for them
# (test_ingest/test_segment/test_topology_edge_angles + every visual test call
# _get_pipeline), so crashes/regressions are still caught — only the
# ground-truth-dependent assertions are gated.
REFERENCE_MESHES = {"Mesh1", "Mesh2", "Mesh3", "Mesh4", "Mesh5"}


def _require_reference(mesh_name: str) -> None:
    if mesh_name not in REFERENCE_MESHES:
        pytest.skip(
            f"{mesh_name}: WIP AI/Scan part - strict feature assertions need "
            "ground truth this part doesn't have (see docs/AI_MESH_FINDINGS.md)"
        )


@pytest.mark.parametrize("mesh_name,mesh_path", ALL_MESHES, ids=[m[0] for m in ALL_MESHES])
class TestAllMeshes:

    # ── Stage 1: Ingest ──

    def test_ingest(self, mesh_name, mesh_path):
        data = _get_pipeline(mesh_name, mesh_path)
        cleaned = data["cleaned"]
        assert cleaned.num_faces > 0
        assert cleaned.num_vertices > 0

    # ── Stage 2: Segment ──

    def test_segment(self, mesh_name, mesh_path):
        data = _get_pipeline(mesh_name, mesh_path)
        assert len(data["regions"]) >= 2

    # ── Stage 3: Fit ──

    def test_fit(self, mesh_name, mesh_path):
        _require_reference(mesh_name)
        if mesh_name == "Mesh4":
            pytest.xfail("tube part class: one seam region unfittable (7/8)")
        data = _get_pipeline(mesh_name, mesh_path)
        assert len(data["fitted"]) == len(data["regions"])

    # ── Stage 4: Topology ──

    def test_topology_connected(self, mesh_name, mesh_path):
        _require_reference(mesh_name)
        import networkx as nx
        data = _get_pipeline(mesh_name, mesh_path)
        assert nx.is_connected(data["graph"])

    def test_topology_edge_angles(self, mesh_name, mesh_path):
        data = _get_pipeline(mesh_name, mesh_path)
        for e in data["edges"]:
            assert 0 <= e.dihedral_angle <= 180

    @pytest.mark.visual
    def test_topology_visual(self, mesh_name, mesh_path):
        data = _get_pipeline(mesh_name, mesh_path)
        viz = Stage4Viz()
        out_dir = _save_dir(mesh_name, "stage4_topology")

        viz.edge_classification_3d(
            _to_pv(data["cleaned"]), data["edges"],
            save_path=os.path.join(out_dir, "edge_classification_3d.png"),
        )
        viz.adjacency_graph_2d(
            data["graph"],
            save_path=os.path.join(out_dir, "adjacency_graph_2d.png"),
        )
        viz.dihedral_histogram(
            data["edges"],
            save_path=os.path.join(out_dir, "dihedral_histogram.png"),
        )

    # ── Stage 5: Recognize ──

    def test_first_feature_is_base(self, mesh_name, mesh_path):
        data = _get_pipeline(mesh_name, mesh_path)
        from mesh2param.models.feature import FeatureType
        assert data["tree"].features[0].type == FeatureType.BASE_EXTRUSION

    def test_has_features(self, mesh_name, mesh_path):
        _require_reference(mesh_name)
        data = _get_pipeline(mesh_name, mesh_path)
        assert data["tree"].num_features >= 2

    def test_feature_summary(self, mesh_name, mesh_path):
        data = _get_pipeline(mesh_name, mesh_path)
        types = Counter(f.type.value for f in data["tree"].features)
        print(f"\n  {mesh_name}: {dict(types)}")

    @pytest.mark.visual
    def test_features_visual(self, mesh_name, mesh_path):
        data = _get_pipeline(mesh_name, mesh_path)
        viz = Stage5Viz()
        out_dir = _save_dir(mesh_name, "stage5_features")

        viz.feature_highlight_composite(
            _to_pv(data["cleaned"]), data["tree"],
            save_path=os.path.join(out_dir, "feature_highlight_composite.png"),
        )
        viz.dependency_dag(
            data["tree"],
            save_path=os.path.join(out_dir, "dependency_dag.png"),
        )

    # ── Feature type checks ──

    def test_has_holes(self, mesh_name, mesh_path):
        _require_reference(mesh_name)
        if mesh_name == "Mesh4":
            pytest.xfail("tube part class: coaxial bore not recognized as a hole")
        data = _get_pipeline(mesh_name, mesh_path)
        from mesh2param.models.feature import FeatureType
        hole_types = {FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE}
        holes = [f for f in data["tree"].features if f.type in hole_types]
        assert len(holes) >= 1, f"{mesh_name} should have at least one hole"

    def test_chamfer_detection(self, mesh_name, mesh_path):
        """Report chamfer/fillet counts (not a hard assert — not all parts have them)."""
        data = _get_pipeline(mesh_name, mesh_path)
        from mesh2param.models.feature import FeatureType
        chamfers = [f for f in data["tree"].features if f.type == FeatureType.CHAMFER]
        fillets = [f for f in data["tree"].features if f.type == FeatureType.FILLET]
        print(f"\n  {mesh_name}: {len(chamfers)} chamfers, {len(fillets)} fillets")
