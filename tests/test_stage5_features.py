"""Tests for Stage 5: Feature Recognition.

Visual tests run the full pipeline on Body1Mesh.3mf.
Correctness tests validate both data model contracts and pipeline outputs.
"""

from __future__ import annotations

import os

import networkx as nx
import numpy as np
import pyvista as pv
import pytest

from mesh2param.models.feature import Feature, FeatureType
from mesh2param.models.tree import FeatureTree
from mesh2param.viz.stage5_features import Stage5Viz


# --- Helpers ---


def _to_pv(cleaned) -> pv.PolyData:
    faces = np.column_stack([np.full(len(cleaned.faces), 3), cleaned.faces])
    return pv.PolyData(cleaned.vertices.copy(), faces.ravel())


def _make_sample_tree() -> FeatureTree:
    features = [
        Feature(type=FeatureType.BASE_EXTRUSION, face_indices=np.arange(0, 4),
                depth=20.0, direction=np.array([0, 0, 1.0]), operation="NEW", dependencies=[]),
        Feature(type=FeatureType.THROUGH_HOLE, face_indices=np.arange(4, 6),
                depth=20.0, direction=np.array([0, 0, -1.0]), operation="REMOVE", dependencies=[0]),
        Feature(type=FeatureType.POCKET, face_indices=np.arange(6, 9),
                depth=5.0, direction=np.array([0, 0, -1.0]), operation="REMOVE", dependencies=[0]),
        Feature(type=FeatureType.FILLET, face_indices=np.arange(9, 11),
                depth=0.0, direction=np.array([0, 0, 1.0]), operation="REMOVE", dependencies=[0, 2]),
    ]
    G = nx.DiGraph()
    for i, f in enumerate(features):
        G.add_node(i, type=f.type.value)
        for dep in f.dependencies:
            G.add_edge(dep, i)
    return FeatureTree(features=features, dependency_graph=G, metadata={"source": "synthetic"})


@pytest.fixture
def sample_tree():
    return _make_sample_tree()


# --- Visual tests (run full pipeline on Body1Mesh.3mf) ---


class TestStage5Visual:

    @pytest.mark.visual
    def test_feature_highlight_composite(self, body1mesh_cleaned, body1mesh_features, save_path):
        viz = Stage5Viz()
        out = save_path("feature_highlight_composite.png")
        viz.feature_highlight_composite(
            _to_pv(body1mesh_cleaned), body1mesh_features, save_path=out,
        )
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_dependency_dag(self, body1mesh_features, save_path):
        viz = Stage5Viz()
        out = save_path("dependency_dag.png")
        viz.dependency_dag(body1mesh_features, save_path=out)
        assert os.path.exists(out)


# --- Pipeline correctness tests ---


class TestStage5Pipeline:

    def test_first_feature_is_base_extrusion(self, body1mesh_features):
        assert body1mesh_features.features[0].type == FeatureType.BASE_EXTRUSION

    def test_base_is_new_operation(self, body1mesh_features):
        assert body1mesh_features.features[0].operation == "NEW"

    def test_has_at_least_two_features(self, body1mesh_features):
        assert body1mesh_features.num_features >= 2

    def test_has_holes(self, body1mesh_features):
        hole_types = {FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE}
        holes = [f for f in body1mesh_features.features if f.type in hole_types]
        assert len(holes) >= 1, "Body1Mesh should have at least one hole"

    def test_holes_are_remove_ops(self, body1mesh_features):
        hole_types = {FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE}
        for f in body1mesh_features.features:
            if f.type in hole_types:
                assert f.operation == "REMOVE"

    def test_dependency_graph_is_dag(self, body1mesh_features):
        assert nx.is_directed_acyclic_graph(body1mesh_features.dependency_graph)

    def test_topological_order_starts_with_base(self, body1mesh_features):
        order = list(nx.topological_sort(body1mesh_features.dependency_graph))
        assert order[0] == 0

    def test_all_features_have_faces(self, body1mesh_features):
        for f in body1mesh_features.features:
            assert len(f.face_indices) >= 1

    def test_features_cover_all_faces(self, body1mesh_cleaned, body1mesh_features):
        all_faces = set()
        for f in body1mesh_features.features:
            all_faces.update(f.face_indices.tolist())
        coverage = len(all_faces) / body1mesh_cleaned.num_faces
        assert coverage > 0.8, f"Features only cover {coverage:.0%} of mesh faces"

    def test_tree_to_dict(self, body1mesh_features):
        d = body1mesh_features.to_dict()
        assert "num_features" in d
        assert "features" in d
        assert len(d["features"]) == body1mesh_features.num_features

    def test_non_base_features_have_dependencies(self, body1mesh_features):
        for i, f in enumerate(body1mesh_features.features):
            if i > 0:
                assert len(f.dependencies) >= 1, (
                    f"Feature {i} ({f.type.value}) has no dependencies"
                )


# --- Data model correctness tests ---


class TestStage5Correctness:

    def test_tree_has_correct_feature_count(self, sample_tree):
        assert sample_tree.num_features == 4

    def test_first_feature_is_base_extrusion(self, sample_tree):
        assert sample_tree.features[0].type == FeatureType.BASE_EXTRUSION

    def test_dependency_graph_is_dag(self, sample_tree):
        assert nx.is_directed_acyclic_graph(sample_tree.dependency_graph)

    def test_topological_order_valid(self, sample_tree):
        order = list(nx.topological_sort(sample_tree.dependency_graph))
        assert order[0] == 0

    def test_feature_to_dict(self, sample_tree):
        d = sample_tree.features[1].to_dict()
        assert d["type"] == "THROUGH_HOLE"
        assert d["operation"] == "REMOVE"

    def test_tree_to_dict(self, sample_tree):
        d = sample_tree.to_dict()
        assert d["num_features"] == 4

    def test_fillet_depends_on_base_and_pocket(self, sample_tree):
        fillet = sample_tree.features[3]
        assert 0 in fillet.dependencies
        assert 2 in fillet.dependencies
