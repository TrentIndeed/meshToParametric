"""Stage 5 visualizations: Feature Recognition."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv

from mesh2param.models.tree import FeatureTree
from mesh2param.viz.base import BaseVisualizer, add_report_banner_mpl, add_report_banner_pv
from mesh2param.viz.colors import DARK_BG, FEATURE_COLORS


class Stage5Viz(BaseVisualizer):
    """Visualizations for feature recognition results."""

    def feature_highlight_composite(
        self,
        mesh: pv.PolyData,
        feature_tree: FeatureTree,
        save_path: str | Path | None = None,
    ) -> str | None:
        """Subplot grid with one panel per feature, highlighted on the mesh."""
        n = feature_tree.num_features
        if n == 0:
            return None

        cols = min(4, n)
        rows = math.ceil(n / cols)

        p = self._make_plotter(
            save_path,
            shape=(rows, cols),
            window_size=(400 * cols, 400 * rows),
        )

        for i, feature in enumerate(feature_tree.features):
            r, c = divmod(i, cols)
            p.subplot(r, c)
            p.set_background(self.BACKGROUND_COLOR)

            p.add_mesh(mesh, color="#E5E7EB", opacity=0.3)

            # Extract and highlight feature faces
            valid_indices = feature.face_indices[feature.face_indices < mesh.n_cells]
            if len(valid_indices) > 0:
                feat_mesh = mesh.extract_cells(valid_indices)
                color = FEATURE_COLORS.get(feature.type.value, "#8B5CF6")
                p.add_mesh(feat_mesh, color=color, opacity=0.9)

            p.add_text(
                f"F{i + 1}: {feature.type.value}",
                font_size=10,
                position="upper_left",
                color="white",
            )
            p.camera_position = "iso"

        # Add banner to the first subplot
        p.subplot(0, 0)
        add_report_banner_pv(
            p,
            test_name="Feature Highlight Composite",
            pass_criteria="Feature 1 is BASE_EXTRUSION covering the main body; holes/pockets are REMOVE ops; "
            "all faces are covered by at least one feature",
            analysis_guide="Each subplot highlights one feature's faces in color on a gray mesh. "
            "BASE_EXTRUSION (purple) should cover the largest surfaces. THROUGH_HOLE (amber) should "
            "highlight cylindrical faces. Uncovered gray areas indicate missed features. Check that "
            "feature types match the geometry (e.g. a round hole is not labeled as a POCKET).",
        )

        return self._finalize(p, save_path)

    def feature_sequence_animation(
        self,
        mesh: pv.PolyData,
        feature_tree: FeatureTree,
        output_dir: str | Path,
    ) -> list[str]:
        """Render one PNG per feature step, then combine into an animated GIF."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        cumulative_indices = np.array([], dtype=int)

        for i, feature in enumerate(feature_tree.features):
            p = self._make_plotter(save_path="dummy")  # force offscreen

            p.add_mesh(mesh, color="#E5E7EB", opacity=0.3)

            valid = feature.face_indices[feature.face_indices < mesh.n_cells]
            cumulative_indices = np.unique(np.concatenate([cumulative_indices, valid]))

            if len(cumulative_indices) > 0:
                highlighted = mesh.extract_cells(cumulative_indices)
                color = FEATURE_COLORS.get(feature.type.value, "#8B5CF6")
                p.add_mesh(highlighted, color=color, opacity=0.8)

            p.add_text(
                f"Step {i + 1}/{feature_tree.num_features}: {feature.type.value}",
                font_size=12,
                position="upper_left",
                color="white",
            )
            p.camera_position = "iso"

            add_report_banner_pv(
                p,
                test_name=f"Feature Sequence Step {i + 1}/{feature_tree.num_features}",
                pass_criteria="Each step adds geometry without removing previous features; final frame "
                "covers the entire part",
                analysis_guide="Watch the build-up: the part should be 'constructed' step by step. "
                "Base extrusion first, then additive features (bosses), then subtractive (holes, pockets), "
                "then finish features (fillets, chamfers). If the order looks wrong, check the dependency DAG.",
            )

            frame_path = str(output_dir / f"feature_step_{i + 1:03d}.png")
            p.screenshot(frame_path, transparent_background=False)
            p.close()
            frames.append(frame_path)

        # Generate animated GIF
        if frames:
            try:
                from PIL import Image

                images = [Image.open(f) for f in frames]
                gif_path = str(output_dir / "feature_sequence.gif")
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=800,
                    loop=0,
                )
            except ImportError:
                pass

        return frames

    def dependency_dag(
        self,
        feature_tree: FeatureTree,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Directed acyclic graph of feature dependencies."""
        if save_path is not None:
            matplotlib.use("Agg")

        G = feature_tree.dependency_graph
        if G.number_of_nodes() == 0:
            # Build a simple graph from the feature list
            G = nx.DiGraph()
            for i, f in enumerate(feature_tree.features):
                G.add_node(i, type=f.type.value)
                for dep in f.dependencies:
                    G.add_edge(dep, i)

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.axis("off")

        # Layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        # Node colors
        node_colors = []
        labels = {}
        for n in G.nodes():
            ftype = G.nodes[n].get("type", "UNKNOWN")
            node_colors.append(FEATURE_COLORS.get(ftype, "#6B7280"))
            labels[n] = f"F{n + 1}\n{ftype}"

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
        nx.draw_networkx_edges(
            G, pos, edge_color="#94A3B8", arrows=True, arrowsize=15,
            width=2, ax=ax,
        )
        nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color="white", ax=ax)

        ax.set_title("Feature Dependency DAG", color="white", fontsize=14)

        add_report_banner_mpl(
            fig,
            test_name="Feature Dependency DAG",
            pass_criteria="DAG is acyclic; BASE_EXTRUSION has no incoming edges; fillets/chamfers depend "
            "on the features whose edges they reference; topological order matches the feature list order",
            analysis_guide="Arrows show dependencies (A → B means B depends on A). BASE_EXTRUSION should "
            "be a root node (no parents). Fillets/chamfers should point back to the features that created "
            "their edges. Cycles indicate a bug in dependency resolution. Check that the left-to-right "
            "flow matches the manufacturing sequence.",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.78])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)

        return fig
