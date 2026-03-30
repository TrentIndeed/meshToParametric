"""Stage 6 visualizations: Onshape Export validation & CadQuery solid."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from mesh2param.viz.base import BaseVisualizer, add_report_banner_mpl, add_report_banner_pv
from mesh2param.viz.colors import DARK_ACCENT, DARK_BG, DARK_GRID


class Stage6Viz:
    """Visualizations for Onshape export validation. No Onshape dependency."""

    def volume_comparison_bar(
        self,
        mesh_volume: float,
        cad_volume: float,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Bar chart comparing mesh volume vs reconstructed CAD volume."""
        if save_path is not None:
            matplotlib.use("Agg")

        error_pct = abs(cad_volume - mesh_volume) / (mesh_volume + 1e-15) * 100

        if error_pct < 2:
            bar_color = "#22C55E"
            status = "PASS"
        elif error_pct < 5:
            bar_color = "#F59E0B"
            status = "WARN"
        else:
            bar_color = "#EF4444"
            status = "FAIL"

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)

        ax.bar(
            ["Mesh", "CAD"],
            [mesh_volume, cad_volume],
            color=[DARK_ACCENT, bar_color],
            edgecolor=DARK_GRID,
            width=0.5,
        )

        ax.set_ylabel("Volume (mm³)", color=DARK_ACCENT)
        ax.tick_params(colors=DARK_ACCENT)
        ax.spines["bottom"].set_color(DARK_GRID)
        ax.spines["left"].set_color(DARK_GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.annotate(
            f"Error: {error_pct:.2f}%  [{status}]",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=bar_color,
        )

        ax.set_title("Volume Comparison", color="#F1F5F9", fontsize=14)

        add_report_banner_mpl(
            fig,
            test_name="Volume Comparison",
            pass_criteria="Volume error < 2% = PASS (green); < 5% = WARN (yellow); > 5% = FAIL (red)",
            analysis_guide="Left bar = original mesh volume. Right bar = reconstructed CAD volume. "
            "The bars should be nearly identical height. Green = excellent reconstruction. Yellow = "
            "acceptable but some features may be approximated. Red = significant geometry was lost or "
            "added during reconstruction — check which features are missing.",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.78])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)

        return fig

    def api_call_timeline(
        self,
        api_log: list[dict],
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Timeline of API calls with status codes and response times.

        Args:
            api_log: List of dicts with keys: name, start, duration, status_code.
        """
        if save_path is not None:
            matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(14, max(4, 0.4 * len(api_log) + 2)))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=DARK_ACCENT)
        ax.spines["bottom"].set_color(DARK_GRID)
        ax.spines["left"].set_color(DARK_GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for i, entry in enumerate(api_log):
            status = entry.get("status_code", 200)
            if status == 200:
                color = "#22C55E"
            elif status == 429:
                color = "#F59E0B"
            else:
                color = "#EF4444"

            start = entry.get("start", i)
            duration = entry.get("duration", 0.5)
            ax.barh(i, duration, left=start, color=color, height=0.6, edgecolor=DARK_GRID)
            ax.text(
                start + duration + 0.1,
                i,
                f"{entry.get('name', '')} ({status}) {duration:.1f}s",
                va="center",
                color=DARK_ACCENT,
                fontsize=8,
            )

        ax.set_xlabel("Time (s)", color=DARK_ACCENT)
        ax.set_ylabel("API Call", color=DARK_ACCENT)
        ax.set_title("API Call Timeline", color="#F1F5F9", fontsize=14)
        ax.set_yticks(range(len(api_log)))
        ax.set_yticklabels([e.get("name", f"call_{i}") for i, e in enumerate(api_log)])
        ax.invert_yaxis()

        add_report_banner_mpl(
            fig,
            test_name="API Call Timeline",
            pass_criteria="All calls green (200 OK); no red error bars; total elapsed time reasonable "
            "for feature count; no 429 rate-limit retries",
            analysis_guide="Green = success (200), yellow = rate-limited (429), red = error. "
            "Each bar's width = response time. Long bars indicate slow API calls. Clusters of yellow "
            "bars mean rate limiting is a problem — increase backoff delay. Red bars need investigation: "
            "check the Onshape error response for invalid geometry or auth issues.",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.78])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)

        return fig

    def export_report(
        self,
        results: dict,
        save_path: str | Path,
    ) -> plt.Figure:
        """Composite summary: volume bar + feature count + pass/fail."""
        matplotlib.use("Agg")

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.patch.set_facecolor(DARK_BG)

        for ax in axes:
            ax.set_facecolor(DARK_BG)
            ax.tick_params(colors=DARK_ACCENT)

        # Panel 1: Volume comparison
        mesh_vol = results.get("mesh_volume", 100)
        cad_vol = results.get("cad_volume", 98)
        error_pct = abs(cad_vol - mesh_vol) / (mesh_vol + 1e-15) * 100
        bar_color = "#22C55E" if error_pct < 2 else "#F59E0B" if error_pct < 5 else "#EF4444"
        axes[0].bar(["Mesh", "CAD"], [mesh_vol, cad_vol], color=[DARK_ACCENT, bar_color], width=0.5)
        axes[0].set_title(f"Volume Error: {error_pct:.2f}%", color="white", fontsize=11)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["bottom"].set_color(DARK_GRID)
        axes[0].spines["left"].set_color(DARK_GRID)

        # Panel 2: Feature count
        feat_count = results.get("feature_count", 0)
        expected = results.get("expected_features", 0)
        match_color = "#22C55E" if feat_count == expected else "#EF4444"
        axes[1].bar(["Expected", "Actual"], [expected, feat_count], color=[DARK_ACCENT, match_color], width=0.5)
        axes[1].set_title("Feature Count", color="white", fontsize=11)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["bottom"].set_color(DARK_GRID)
        axes[1].spines["left"].set_color(DARK_GRID)

        # Panel 3: Overall status
        axes[2].axis("off")
        status = results.get("status", "PASS")
        status_color = "#22C55E" if status == "PASS" else "#EF4444"
        axes[2].text(
            0.5, 0.5, status, transform=axes[2].transAxes,
            ha="center", va="center", fontsize=48, fontweight="bold", color=status_color,
        )
        axes[2].set_title("Overall", color="white", fontsize=11)

        fig.suptitle("Export Report", color="#F1F5F9", fontsize=16, y=0.80)

        add_report_banner_mpl(
            fig,
            test_name="Export Report Summary",
            pass_criteria="Volume error < 5%; feature count matches expected; overall status = PASS",
            analysis_guide="Three-panel summary. Left: volume bars should be equal height (green = < 2% error). "
            "Center: feature count bars should match (green = exact match). Right: overall status. "
            "If FAIL, check which panel is red to identify the problem — volume mismatch means geometry "
            "loss, count mismatch means missed or extra features.",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.78])
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)

        return fig


class Stage6CadQueryViz(BaseVisualizer):
    """Visualization for CadQuery solid reconstruction."""

    def cadquery_solid_3d(
        self,
        cq_result,
        original_mesh: pv.PolyData | None = None,
        save_path: str | Path | None = None,
    ) -> str | None:
        """Render the CadQuery solid alongside the original mesh.

        Args:
            cq_result: CadQueryResult with .solid attribute.
            original_mesh: Optional original mesh for side-by-side comparison.
            save_path: Path to save the screenshot.
        """
        if cq_result.tess_vertices is None or cq_result.tess_faces is None:
            return None

        # Build PyVista mesh from pre-tessellated data
        verts = cq_result.tess_vertices
        faces = cq_result.tess_faces
        pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
        cq_mesh = pv.PolyData(verts, pv_faces)

        if original_mesh is not None:
            # Side-by-side: original mesh (left) vs CadQuery solid (right)
            p = self._make_plotter(save_path, shape=(1, 2), window_size=(1600, 800))

            p.subplot(0, 0)
            p.set_background(self.BACKGROUND_COLOR)
            p.add_mesh(original_mesh, color="#94A3B8", opacity=0.7)
            p.add_text("Original Mesh", position="upper_left", font_size=14, color="white")
            p.camera_position = "iso"

            p.subplot(0, 1)
            p.set_background(self.BACKGROUND_COLOR)
            p.add_mesh(cq_mesh, color="#6366F1", opacity=0.9, show_edges=True, edge_color="#1E293B")
            vol_err = cq_result.volume_error_pct
            status_color = "#22C55E" if vol_err < 5 else "#F59E0B" if vol_err < 10 else "#EF4444"
            p.add_text(
                f"CadQuery Solid (vol err: {vol_err:.1f}%)",
                position="upper_left", font_size=14, color=status_color,
            )
            p.camera_position = "iso"

            p.subplot(0, 0)
            add_report_banner_pv(
                p,
                test_name="CadQuery Solid Reconstruction",
                pass_criteria="CadQuery solid (right) matches original mesh (left) in overall shape; "
                "holes are present at correct positions; volume error < 5%",
                analysis_guide="Left = original mesh. Right = reconstructed CadQuery solid with edges. "
                "The solid should reproduce the base profile shape and hole positions. Missing holes "
                "indicate failed cuts. Shape mismatch means the profile extraction missed features.",
            )
        else:
            p = self._make_plotter(save_path, window_size=(1024, 768))
            p.add_mesh(cq_mesh, color="#6366F1", opacity=0.9, show_edges=True, edge_color="#1E293B")
            p.camera_position = "iso"
            add_report_banner_pv(
                p,
                test_name="CadQuery Solid",
                pass_criteria="Solid has correct base profile and holes",
                analysis_guide="Purple solid = CadQuery reconstruction from feature tree.",
            )

        return self._finalize(p, save_path)


