"""Tests for Stage 4: Topology & Edge Classification.

Visual tests run the full pipeline on Body1Mesh.3mf.
Correctness tests validate both data model contracts and pipeline outputs.
"""

from __future__ import annotations

import os

import networkx as nx
import numpy as np
import pyvista as pv
import pytest

from mesh2param.config import TopologyConfig
from mesh2param.models.surface import SurfaceEdge, TransitionType
from mesh2param.pipeline.topology import build_topology
from mesh2param.viz.stage4_topology import Stage4Viz


# --- Helpers ---


def _to_pv(cleaned) -> pv.PolyData:
    faces = np.column_stack([np.full(len(cleaned.faces), 3), cleaned.faces])
    return pv.PolyData(cleaned.vertices.copy(), faces.ravel())


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


# --- Visual tests (run full pipeline on Body1Mesh.3mf) ---


class TestStage4Visual:

    @pytest.mark.visual
    def test_edge_classification_3d(self, body1mesh_cleaned, body1mesh_topology, save_path):
        graph, edges = body1mesh_topology
        viz = Stage4Viz()
        out = save_path("edge_classification_3d.png")
        viz.edge_classification_3d(_to_pv(body1mesh_cleaned), edges, save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_adjacency_graph_2d(self, body1mesh_topology, save_path):
        graph, edges = body1mesh_topology
        viz = Stage4Viz()
        out = save_path("adjacency_graph_2d.png")
        viz.adjacency_graph_2d(graph, save_path=out)
        assert os.path.exists(out)

    @pytest.mark.visual
    def test_dihedral_histogram(self, body1mesh_topology, save_path):
        graph, edges = body1mesh_topology
        viz = Stage4Viz()
        out = save_path("dihedral_histogram.png")
        viz.dihedral_histogram(edges, save_path=out)
        assert os.path.exists(out)


# --- Pipeline correctness tests ---


class TestStage4Pipeline:

    def test_graph_has_all_regions(self, body1mesh_regions, body1mesh_topology):
        graph, _ = body1mesh_topology
        assert graph.number_of_nodes() == len(body1mesh_regions)

    def test_graph_is_connected(self, body1mesh_topology):
        graph, _ = body1mesh_topology
        assert nx.is_connected(graph), "Surface graph should be connected for a single part"

    def test_edges_have_valid_angles(self, body1mesh_topology):
        _, edges = body1mesh_topology
        for e in edges:
            assert 0 <= e.dihedral_angle <= 180.0, (
                f"Edge ({e.surface_a_id}, {e.surface_b_id}) has invalid angle {e.dihedral_angle}"
            )

    def test_edges_reference_valid_surfaces(self, body1mesh_regions, body1mesh_topology):
        _, edges = body1mesh_topology
        region_ids = {r.region_id for r in body1mesh_regions}
        for e in edges:
            assert e.surface_a_id in region_ids
            assert e.surface_b_id in region_ids

    def test_edges_have_vertices(self, body1mesh_topology):
        _, edges = body1mesh_topology
        for e in edges:
            assert e.edge_vertices is not None
            assert len(e.edge_vertices) >= 2, (
                f"Edge ({e.surface_a_id}, {e.surface_b_id}) has < 2 vertices"
            )

    def test_has_sharp_edges(self, body1mesh_topology):
        _, edges = body1mesh_topology
        sharp = [e for e in edges if e.transition_type == TransitionType.SHARP]
        assert len(sharp) >= 1, "Body1Mesh should have at least one sharp edge"

    def test_graph_node_attributes(self, body1mesh_topology):
        graph, _ = body1mesh_topology
        for n in graph.nodes():
            assert "surface_type" in graph.nodes[n]
            assert "area" in graph.nodes[n]
            assert "label" in graph.nodes[n]

    def test_graph_edge_attributes(self, body1mesh_topology):
        graph, _ = body1mesh_topology
        for u, v, d in graph.edges(data=True):
            assert "transition_type" in d
            assert "dihedral_angle" in d

    def test_edge_ids_are_ordered(self, body1mesh_topology):
        """surface_a_id < surface_b_id by convention."""
        _, edges = body1mesh_topology
        for e in edges:
            assert e.surface_a_id < e.surface_b_id

    def test_to_dict_roundtrip(self, body1mesh_topology):
        _, edges = body1mesh_topology
        for e in edges:
            d = e.to_dict()
            assert d["surface_a_id"] == e.surface_a_id
            assert d["surface_b_id"] == e.surface_b_id
            assert d["transition_type"] == e.transition_type.value
            assert d["dihedral_angle"] == e.dihedral_angle


# --- Data model correctness tests ---


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
