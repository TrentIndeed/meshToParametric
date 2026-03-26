"""Stage 1 visualizations: Mesh Ingestion & Cleanup."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from mesh2param.viz.base import BaseVisualizer, add_report_banner_mpl, add_report_banner_pv
from mesh2param.viz.colors import DARK_ACCENT, DARK_BG, DARK_GRID


class Stage1Viz(BaseVisualizer):
    """Visualizations for mesh cleanup results."""

    def before_after_overlay(
        self,
        original: pv.PolyData,
        cleaned: pv.PolyData,
        save_path: str | Path | None = None,
    ) -> str | None:
        """Overlay original (red wireframe) and cleaned (green solid) meshes.

        Degenerate faces (area < 1e-10) highlighted in yellow.
        """
        p = self._make_plotter(save_path)

        p.add_mesh(
            original,
            style="wireframe",
            color="#EF4444",
            opacity=0.5,
            line_width=3,
            label="Original",
        )
        p.add_mesh(
            cleaned,
            color="#10B981",
            opacity=0.8,
            label="Cleaned",
        )

        # Highlight degenerate faces
        sized = cleaned.compute_cell_sizes()
        areas = sized["Area"]
        degen_mask = areas < 1e-10
        if degen_mask.any():
            degen_ids = np.where(degen_mask)[0]
            degen = cleaned.extract_cells(degen_ids)
            p.add_mesh(degen, color="#FBBF24", label="Degenerate")

        p.add_legend(bcolor=DARK_BG, face=None)
        p.camera_position = "iso"

        add_report_banner_pv(
            p,
            test_name="Before/After Mesh Overlay",
            pass_criteria="Green solid mesh fully covers red wireframe; no yellow degenerate faces visible",
            analysis_guide="Red wireframe = original, green solid = cleaned. Yellow patches = degenerate "
            "faces that should have been removed. Check that green mesh is closed and smooth.",
        )

        return self._finalize(p, save_path)

    def normal_consistency_map(
        self,
        mesh: pv.PolyData,
        save_path: str | Path | None = None,
    ) -> str | None:
        """Color each face by normal direction (nx,ny,nz -> RGB)."""
        p = self._make_plotter(save_path)

        normals = mesh.cell_normals
        colors = ((normals + 1) / 2 * 255).astype(np.uint8)
        mesh.cell_data["NormalRGB"] = colors

        p.add_mesh(mesh, scalars="NormalRGB", rgb=True)
        p.camera_position = "iso"

        add_report_banner_pv(
            p,
            test_name="Normal Consistency Map",
            pass_criteria="Each flat face is a single uniform color; no sudden color flips between adjacent coplanar faces",
            analysis_guide="Colors map normal direction to RGB. Coplanar faces should be identical color. "
            "A face that is a drastically different color from its neighbors has a flipped normal. "
            "Smooth curved surfaces should show a gradual color gradient.",
        )

        return self._finalize(p, save_path)

    def mesh_quality_histogram(
        self,
        mesh: pv.PolyData,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Three subplots: aspect ratio, edge length, face area distributions."""
        if save_path is not None:
            matplotlib.use("Agg")

        try:
            qual = mesh.cell_quality(quality_measure="aspect_ratio")
            aspect_ratios = qual["aspect_ratio"]
        except AttributeError:
            qual = mesh.compute_cell_quality(quality_measure="aspect_ratio")
            aspect_ratios = qual["CellQuality"]

        # Extract edge lengths
        edge_lengths = []
        points = mesh.points
        if mesh.is_all_triangles:
            faces_arr = mesh.faces.reshape(-1, 4)[:, 1:]
            for f in faces_arr:
                for i in range(3):
                    v0, v1 = points[f[i]], points[f[(i + 1) % 3]]
                    edge_lengths.append(float(np.linalg.norm(v1 - v0)))
        edge_lengths = np.array(edge_lengths) if edge_lengths else np.array([0])

        # Face areas
        sized = mesh.compute_cell_sizes()
        face_areas = sized["Area"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        fig.patch.set_facecolor(DARK_BG)

        for ax in axes:
            ax.set_facecolor(DARK_BG)
            ax.tick_params(colors=DARK_ACCENT)
            ax.spines["bottom"].set_color(DARK_GRID)
            ax.spines["left"].set_color(DARK_GRID)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.label.set_color(DARK_ACCENT)
            ax.yaxis.label.set_color(DARK_ACCENT)
            ax.title.set_color("#F1F5F9")

        axes[0].hist(aspect_ratios, bins=30, color="#6366F1", edgecolor=DARK_BG)
        axes[0].set_title("Aspect Ratio")
        axes[0].set_xlabel("Ratio")

        axes[1].hist(edge_lengths, bins=30, color="#10B981", edgecolor=DARK_BG)
        axes[1].set_title("Edge Length")
        axes[1].set_xlabel("Length")

        axes[2].hist(face_areas, bins=30, color="#F59E0B", edgecolor=DARK_BG)
        axes[2].set_title("Face Area")
        axes[2].set_xlabel("Area")
        # Annotate degenerate count instead of invisible axvline at 1e-10
        n_degen = int((face_areas < 1e-10).sum())
        axes[2].annotate(
            f"Degenerate (area < 1e-10): {n_degen}",
            xy=(0.98, 0.95), xycoords="axes fraction",
            ha="right", va="top",
            fontsize=9, color="#EF4444" if n_degen > 0 else "#22C55E",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=DARK_GRID),
        )

        add_report_banner_mpl(
            fig,
            test_name="Mesh Quality Histogram",
            pass_criteria="No faces below the red degenerate threshold; aspect ratios clustered near 1.0; "
            "edge lengths unimodal without extreme outliers",
            analysis_guide="Left: aspect ratio near 1.0 = equilateral triangles (ideal). "
            "Center: edge lengths should be tightly grouped — wide spread means inconsistent tessellation. "
            "Right: face areas — any bars left of the red dashed line are degenerate faces that need removal.",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.65])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)

        return fig
