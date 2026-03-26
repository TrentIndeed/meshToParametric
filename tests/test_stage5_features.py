"""Tests for Stage 5: Feature Recognition.

Visual tests are skipped — Stage 5 pipeline (recognize.py) is not yet implemented.
Correctness tests validate data model contracts using synthetic data.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from mesh2param.models.feature import Feature, FeatureType
from mesh2param.models.tree import FeatureTree


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


# --- Visual tests (skipped until pipeline exists) ---


class TestStage5Visual:

    @pytest.mark.visual
    def test_feature_highlight_composite(self, save_path):
        pytest.fail("Stage 5 pipeline (recognize.py) not implemented — no visual output")

    @pytest.mark.visual
    def test_feature_sequence_animation(self, save_path):
        pytest.fail("Stage 5 pipeline (recognize.py) not implemented — no visual output")

    @pytest.mark.visual
    def test_dependency_dag(self, save_path):
        pytest.fail("Stage 5 pipeline (recognize.py) not implemented — no visual output")


# --- Correctness tests (data model contracts) ---


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
