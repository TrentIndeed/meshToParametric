"""Stage 4 visualizations: Topology & Edge Classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv

from mesh2param.models.surface import SurfaceEdge, TransitionType
from mesh2param.viz.base import BaseVisualizer, add_report_banner_mpl, add_report_banner_pv
from mesh2param.viz.colors import DARK_ACCENT, DARK_BG, DARK_GRID, EDGE_COLORS, SURFACE_COLORS

_LINE_WIDTHS = {
    TransitionType.SMOOTH: 1,
    TransitionType.CHAMFER: 2,
    TransitionType.FILLET: 3,
    TransitionType.SHARP: 4,
}


class Stage4Viz(BaseVisualizer):
    """Visualizations for topology & edge classification results."""

    def edge_classification_3d(
        self,
        mesh: pv.PolyData,
        edges: list[SurfaceEdge],
        save_path: str | Path | None = None,
    ) -> str | None:
        """Render classified edges on a semi-transparent mesh."""
        p = self._make_plotter(save_path)
        p.add_mesh(mesh, color="#94A3B8", opacity=0.3)

        present_types = {e.transition_type for e in edges}
        for etype in TransitionType:
            type_edges = [e for e in edges if e.transition_type == etype]
            color = EDGE_COLORS.get(etype.value.upper(), "#6B7280")
            width = _LINE_WIDTHS.get(etype, 2)

            all_pts = []
            cells = []
            idx = 0
            for e in type_edges:
                if e.edge_vertices is not None and len(e.edge_vertices) >= 2:
                    for i in range(len(e.edge_vertices) - 1):
                        all_pts.append(e.edge_vertices[i])
                        all_pts.append(e.edge_vertices[i + 1])
                        cells.extend([2, idx, idx + 1])
                        idx += 2

            if all_pts:
                line_mesh = pv.PolyData(np.array(all_pts), lines=cells)
                p.add_mesh(line_mesh, color=color, line_width=width)

        # Always show all four types in the legend as a color key;
        # dim the label for types not present in this model.
        legend_entries = []
        for etype in TransitionType:
            color = EDGE_COLORS.get(etype.value.upper(), "#6B7280")
            label = etype.value.upper()
            if etype not in present_types:
                label = f"({label})"
            legend_entries.append([label, color])

        p.add_legend(legend_entries, bcolor=DARK_BG, face=None)
        p.camera_position = "iso"

        add_report_banner_pv(
            p,
            test_name="Edge Classification 3D",
            pass_criteria="Sharp edges (green) appear at 90° corners; fillet edges (blue) at rounded "
            "transitions; chamfer edges (orange) at beveled edges; no misclassified smooth edges on sharp corners",
            analysis_guide="Green = sharp, blue = fillet, orange = chamfer, gray = smooth. Line thickness "
            "indicates confidence. Check that hole rims are classified as sharp, fillet arcs as fillet, "
            "and flat-face junctions as sharp. Misclassified edges will cause wrong features in Stage 5.",
        )

        return self._finalize(p, save_path)

    def adjacency_graph_2d(
        self,
        graph: nx.Graph,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """2D node-link diagram of the surface adjacency graph.

        Nodes are colored by surface type, edges by transition type.
        """
        if save_path is not None:
            matplotlib.use("Agg")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.axis("off")

        pos = nx.spring_layout(graph, seed=42)

        # Draw edges by transition type
        for ttype_str, color in EDGE_COLORS.items():
            edge_list = [
                (u, v)
                for u, v, d in graph.edges(data=True)
                if d.get("transition_type", "").upper() == ttype_str
            ]
            if edge_list:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=edge_list, edge_color=color, width=2, ax=ax
                )

        # Draw nodes by surface type
        for stype_str, color in SURFACE_COLORS.items():
            nodes = [
                n
                for n, d in graph.nodes(data=True)
                if d.get("surface_type", "").upper() == stype_str
            ]
            if nodes:
                sizes = [graph.nodes[n].get("area", 100) * 50 for n in nodes]
                nx.draw_networkx_nodes(
                    graph, pos, nodelist=nodes, node_color=color, node_size=sizes, ax=ax
                )

        # Fallback for nodes without matching type
        uncolored = [
            n
            for n in graph.nodes()
            if graph.nodes[n].get("surface_type", "").upper() not in SURFACE_COLORS
        ]
        if uncolored:
            nx.draw_networkx_nodes(
                graph, pos, nodelist=uncolored, node_color="#6B7280", node_size=200, ax=ax
            )

        nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white", ax=ax)
        ax.set_title("Surface Adjacency Graph", color="white", fontsize=16)

        add_report_banner_mpl(
            fig,
            test_name="Surface Adjacency Graph",
            pass_criteria="Graph is connected (no isolated nodes); node count matches region count; "
            "edge colors correctly reflect transition types between adjacent surfaces",
            analysis_guide="Nodes = surfaces (sized by area, colored by type: purple=plane, pink=cylinder, "
            "yellow=cone, green=sphere). Edges = adjacencies colored by transition type. Isolated nodes "
            "mean a surface has no neighbors (likely a segmentation bug). Check that cylinders (holes) "
            "connect to their surrounding planar faces.",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.78])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)

        return fig

    def dihedral_histogram(
        self,
        edges: list[SurfaceEdge],
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Histogram of dihedral angles across all edges."""
        if save_path is not None:
            matplotlib.use("Agg")

        angles = [e.dihedral_angle for e in edges]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=DARK_ACCENT)
        ax.spines["bottom"].set_color(DARK_GRID)
        ax.spines["left"].set_color(DARK_GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.hist(angles, bins=36, range=(0, 180), color="#6366F1", edgecolor=DARK_BG)

        # Annotate peaks
        for angle, label, color in [
            (90, "90° corners", "#22C55E"),
            (135, "~135° chamfers", "#F97316"),
            (180, "180° smooth", "#6B7280"),
        ]:
            ax.axvline(angle, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
            ax.annotate(
                label,
                xy=(angle, 0),
                xytext=(angle + 3, ax.get_ylim()[1] * 0.8),
                color=color,
                fontsize=9,
            )

        ax.set_xlabel("Dihedral Angle (°)", color=DARK_ACCENT)
        ax.set_ylabel("Count", color=DARK_ACCENT)
        ax.set_title("Dihedral Angle Distribution", color="#F1F5F9", fontsize=14)

        add_report_banner_mpl(
            fig,
            test_name="Dihedral Angle Distribution",
            pass_criteria="Clear peaks at 90° (box corners) and/or 180° (coplanar); no unexpected peaks "
            "at random angles; prismatic parts should cluster at 90° and 180°",
            analysis_guide="For a prismatic part (boxes, plates), expect a strong peak at 90° (right-angle "
            "corners) and 180° (tangent/smooth joins). A peak near 135° suggests chamfers. Peaks at odd "
            "angles may indicate mesh artifacts or misclassified surfaces. The histogram shape gives a "
            "quick fingerprint of the part's geometric complexity.",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.78])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)

        return fig
