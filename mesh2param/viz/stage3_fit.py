"""Stage 3 visualizations: Primitive Fitting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from mesh2param.models.region import SurfaceType
from mesh2param.models.surface import FittedSurface
from mesh2param.viz.base import BaseVisualizer, add_report_banner_mpl, add_report_banner_pv
from mesh2param.models.region import FaceRegion
from mesh2param.viz.colors import DARK_ACCENT, DARK_BG, DARK_GRID, SURFACE_COLORS


class Stage3Viz(BaseVisualizer):
    """Visualizations for primitive fitting results."""

    def fitted_surface_overlay(
        self,
        points: np.ndarray,
        fitted: FittedSurface,
        save_path: str | Path | None = None,
    ) -> str | None:
        """Show analytic surface overlaid on mesh region point cloud.

        Args:
            points: (N, 3) point cloud of the mesh region vertices.
            fitted: The fitted analytic surface.
            save_path: Path for offscreen render, or None for interactive.
        """
        p = self._make_plotter(save_path)

        cloud = pv.PolyData(points)
        p.add_mesh(cloud, color="#9CA3AF", point_size=3, label="Mesh Points")

        analytic = _make_analytic_mesh(fitted)
        if analytic is not None:
            p.add_mesh(analytic, color="#3B82F6", opacity=0.4, label="Fitted")

        p.add_text(
            f"{fitted.surface_type.value.upper()}  RMS: {fitted.rms_error:.4f}",
            position="upper_right",
            font_size=12,
            color=DARK_ACCENT,
        )
        p.add_legend(bcolor=DARK_BG, face=None)
        p.camera_position = "iso"

        add_report_banner_pv(
            p,
            test_name="Fitted Surface Overlay",
            pass_criteria="Blue analytic surface closely envelops the gray point cloud; RMS error < 0.1% of "
            "bounding box diagonal",
            analysis_guide="Gray dots = original mesh vertices. Blue translucent surface = fitted primitive. "
            "The blue surface should pass through the center of the point cloud. Large gaps between points "
            "and the blue surface indicate a bad fit. Check the RMS value in the upper-right corner.",
        )

        return self._finalize(p, save_path)

    def residual_heatmap(
        self,
        points: np.ndarray,
        residuals: np.ndarray,
        save_path: str | Path | None = None,
    ) -> str | None:
        """Color vertices by signed distance to the fitted surface.

        Args:
            points: (N, 3) point cloud.
            residuals: (N,) signed distances.
            save_path: Path for offscreen render, or None for interactive.
        """
        p = self._make_plotter(save_path)

        cloud = pv.PolyData(points)
        cloud["distance"] = residuals

        p.add_mesh(
            cloud,
            scalars="distance",
            cmap="RdYlGn_r",
            point_size=5,
            scalar_bar_args={"title": "Residual (mm)", "color": DARK_ACCENT},
        )
        p.camera_position = "iso"

        add_report_banner_pv(
            p,
            test_name="Residual Heatmap",
            pass_criteria="Predominantly green (near zero residual); no large red or blue clusters "
            "indicating systematic misfit",
            analysis_guide="Green = on surface (zero residual). Red = outside the fitted surface. "
            "Dark green/blue = inside. A uniform green cloud means excellent fit. Clusters of red/blue "
            "indicate the fitted primitive type may be wrong, or the region includes faces from two "
            "different surfaces. Check the scalar bar for the magnitude range.",
        )

        return self._finalize(p, save_path)

    def parameter_snap_table(
        self,
        fitted_surfaces: list[FittedSurface],
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Table showing raw vs snapped parameters for each surface."""
        if save_path is not None:
            matplotlib.use("Agg")

        headers = ["Region", "Type", "Parameter", "Raw", "Snapped", "Changed"]
        rows = []

        for fs in fitted_surfaces:
            for key in fs.params:
                raw_val = fs.params[key]
                snapped_val = fs.snapped_params.get(key, raw_val)
                changed = "Yes" if raw_val != snapped_val else ""
                rows.append([
                    str(fs.region_id),
                    fs.surface_type.value,
                    key,
                    _fmt_val(raw_val),
                    _fmt_val(snapped_val),
                    changed,
                ])

        fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(rows) + 5)))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.axis("off")

        if rows:
            table = ax.table(
                cellText=rows,
                colLabels=headers,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)

            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor(DARK_GRID)
                if row == 0:
                    cell.set_facecolor(DARK_GRID)
                    cell.set_text_props(color=DARK_ACCENT, fontweight="bold")
                else:
                    # Highlight changed rows
                    if col == 5 and rows[row - 1][5] == "Yes":
                        cell.set_facecolor("#F59E0B22")
                    else:
                        cell.set_facecolor(DARK_BG)
                    cell.set_text_props(color="#E2E8F0")

        add_report_banner_mpl(
            fig,
            test_name="Parameter Snap Table",
            pass_criteria="Snapped values are round numbers (e.g. 5.000 not 4.998); axis directions "
            "snapped to principal axes where appropriate; no excessive snapping that changes geometry",
            analysis_guide="Rows highlighted in yellow = parameter was snapped. Check that radii snap to "
            "standard sizes (e.g. 2.5, 3.0, 4.0, 5.0 mm). Axis directions should snap to (1,0,0), (0,1,0), "
            "or (0,0,1) if within 2°. Large snap deltas (raw vs snapped differ significantly) may indicate "
            "the snap tolerance is too aggressive.",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.65])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)

        return fig


    def all_surfaces_composite(
        self,
        mesh: pv.PolyData,
        regions: list[FaceRegion],
        fitted_surfaces: list[FittedSurface],
        save_path: str | Path | None = None,
    ) -> str | None:
        """Render 4-view composite (iso, front, side, top) colored by fitted surface type.

        Each region is colored by its fitted type (plane=purple, cylinder=pink,
        sphere=green, etc.) and the fitted analytic primitives are shown as
        translucent overlays.
        """
        # Build a per-cell color array from fitted surface types
        face_colors = np.full((mesh.n_cells, 3), 100, dtype=np.uint8)
        fitted_by_id = {f.region_id: f for f in fitted_surfaces}

        legend_entries = set()
        for region in regions:
            fitted = fitted_by_id.get(region.region_id)
            if fitted is None:
                continue
            stype_key = fitted.surface_type.value.upper()
            hex_c = SURFACE_COLORS.get(stype_key, SURFACE_COLORS["UNKNOWN"])
            r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            for fi in region.face_indices:
                if fi < mesh.n_cells:
                    face_colors[fi] = [r, g, b]
            legend_entries.add((stype_key, hex_c))

        mesh.cell_data["FitTypeRGB"] = face_colors

        # Summary text
        type_counts = {}
        for f in fitted_surfaces:
            key = f.surface_type.value.upper()
            type_counts[key] = type_counts.get(key, 0) + 1
        summary = "  ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))

        views = [
            ("Isometric", "iso"),
            ("Front (-Y)", [(0, -300, 0), (0, 0, 0), (0, 0, 1)]),
            ("Right (+X)", [(300, 0, 0), (0, 0, 0), (0, 1, 0)]),
            ("Top (-Z)", [(0, 0, 300), (0, 0, 0), (0, 1, 0)]),
        ]

        off_screen = save_path is not None or self.headless
        p = pv.Plotter(off_screen=off_screen, shape=(2, 2))

        for idx, (label, cam) in enumerate(views):
            p.subplot(idx // 2, idx % 2)
            p.set_background(self.BACKGROUND_COLOR)

            p.add_mesh(mesh, scalars="FitTypeRGB", rgb=True, show_edges=True,
                       edge_color="#1E293B", line_width=0.5)

            # Highlight non-plane regions with wireframe
            for region in regions:
                fitted = fitted_by_id.get(region.region_id)
                if fitted is None or fitted.surface_type == SurfaceType.PLANE:
                    continue
                valid = region.face_indices[region.face_indices < mesh.n_cells]
                if len(valid) == 0:
                    continue
                region_mesh = mesh.extract_cells(valid)
                stype_key = fitted.surface_type.value.upper()
                hex_c = SURFACE_COLORS.get(stype_key, "#6B7280")
                p.add_mesh(region_mesh, color=hex_c, style="wireframe", line_width=2)

            p.add_text(label, position="upper_left", font_size=14, color=DARK_ACCENT)

            if cam == "iso":
                p.camera_position = cam
                # Add legend and summary to the iso view
                if legend_entries:
                    p.add_legend(
                        [[name, color] for name, color in sorted(legend_entries)],
                        bcolor=DARK_BG,
                        face=None,
                    )
                p.add_text(summary, position="upper_right", font_size=12, color=DARK_ACCENT)
            else:
                p.camera_position = cam

        # Add banner to subplot 0
        p.subplot(0, 0)
        add_report_banner_pv(
            p,
            test_name="All Fitted Surfaces",
            pass_criteria="Every region colored by its fitted type; no large UNKNOWN (gray) "
            "regions on clearly analytic geometry; holes are pink, flat faces are purple",
            analysis_guide="Purple = plane, pink = cylinder, green = sphere, yellow = cone, gray = unknown. "
            "Non-plane regions have a bright wireframe outline to stand out. "
            "Check that cylindrical holes are pink, flat faces are purple, and fillet bands are correctly typed.",
        )

        if save_path:
            p.screenshot(save_path, transparent_background=False)
            p.close()
            return str(save_path)
        else:
            p.show()
            return None


def _make_analytic_mesh(fitted: FittedSurface) -> pv.PolyData | None:
    """Generate a PyVista mesh for the fitted analytic surface."""
    params = fitted.params
    if fitted.surface_type == SurfaceType.PLANE:
        return pv.Plane(
            center=params.get("point", [0, 0, 0]),
            direction=params.get("normal", [0, 0, 1]),
            i_size=params.get("extent", 20),
            j_size=params.get("extent", 20),
        )
    elif fitted.surface_type == SurfaceType.CYLINDER:
        return pv.Cylinder(
            center=params.get("axis_point", [0, 0, 0]),
            direction=params.get("axis_dir", [0, 0, 1]),
            radius=params.get("radius", 5),
            height=params.get("height", 20),
        )
    elif fitted.surface_type == SurfaceType.SPHERE:
        return pv.Sphere(
            center=params.get("center", [0, 0, 0]),
            radius=params.get("radius", 5),
        )
    return None


def _fmt_val(val) -> str:
    """Format a value for display."""
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val)
        if arr.ndim == 1 and len(arr) <= 4:
            return "(" + ", ".join(f"{v:.3f}" for v in arr) + ")"
    return str(val)
