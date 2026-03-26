"""Tests for Stage 4: Topology & Edge Classification.

Visual tests are skipped — Stage 4 pipeline (topology.py) is not yet implemented.
Correctness tests validate data model contracts using synthetic data.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from mesh2param.models.surface import SurfaceEdge, TransitionType


# --- Synthetic data helpers ---


def _make_box_graph() -> nx.Graph:
    G = nx.Graph()
    for i, name in enumerate(["top", "bottom", "front", "back", "left", "right"]):
        G.add_node(i, surface_type="PLANE", area=100.0, label=name)
    for a, b in [(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]:
        G.add_edge(a, b, transition_type="SHARP", dihedral_angle=90.0)
    return G


def _make_box_edges() -> list[SurfaceEdge]:
    edges = []
    for a, b in [(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]:
        v = np.array([[a * 2.0, 0.0, 0.0], [a * 2.0 + 1.0, 0.0, 0.0]])
        edges.append(SurfaceEdge(
            surface_a_id=a, surface_b_id=b,
            transition_type=TransitionType.SHARP, dihedral_angle=90.0, edge_vertices=v,
        ))
    return edges


def _make_mixed_edges() -> list[SurfaceEdge]:
    rng = np.random.RandomState(42)
    edges = []
    for i, (ttype, angle) in enumerate([
        (TransitionType.SHARP, 90.0), (TransitionType.SHARP, 90.0),
        (TransitionType.FILLET, 170.0), (TransitionType.CHAMFER, 135.0),
        (TransitionType.SMOOTH, 179.0), (TransitionType.SHARP, 45.0),
        (TransitionType.FILLET, 165.0),
    ]):
        v = rng.randn(2, 3) * 5
        edges.append(SurfaceEdge(
            surface_a_id=i, surface_b_id=i + 1,
            transition_type=ttype, dihedral_angle=angle, edge_vertices=v,
        ))
    return edges


# --- Visual tests (skipped until pipeline exists) ---


class TestStage4Visual:

    @pytest.mark.visual
    def test_edge_classification_3d(self, save_path):
        pytest.fail("Stage 4 pipeline (topology.py) not implemented — no visual output")

    @pytest.mark.visual
    def test_adjacency_graph_2d(self, save_path):
        pytest.fail("Stage 4 pipeline (topology.py) not implemented — no visual output")

    @pytest.mark.visual
    def test_dihedral_histogram(self, save_path):
        pytest.fail("Stage 4 pipeline (topology.py) not implemented — no visual output")


# --- Correctness tests (data model contracts) ---


class TestStage4Correctness:

    def test_box_graph_connectivity(self):
        G = _make_box_graph()
        assert nx.is_connected(G)
        assert G.number_of_nodes() == 6
        assert G.number_of_edges() == 12

    def test_all_box_edges_are_sharp(self):
        for e in _make_box_edges():
            assert e.transition_type == TransitionType.SHARP
            assert e.dihedral_angle == 90.0

    def test_mixed_edges_have_variety(self):
        types = {e.transition_type for e in _make_mixed_edges()}
        assert TransitionType.SHARP in types
        assert TransitionType.FILLET in types
        assert TransitionType.CHAMFER in types
        assert TransitionType.SMOOTH in types

    def test_surface_edge_to_dict(self):
        e = SurfaceEdge(
            surface_a_id=0, surface_b_id=1,
            transition_type=TransitionType.SHARP, dihedral_angle=90.0,
        )
        d = e.to_dict()
        assert d["transition_type"] == "sharp"
        assert d["dihedral_angle"] == 90.0

    def test_graph_node_attributes(self):
        G = _make_box_graph()
        for n in G.nodes():
            assert G.nodes[n]["surface_type"] == "PLANE"
            assert G.nodes[n]["area"] == 100.0
