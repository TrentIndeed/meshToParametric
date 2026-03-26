"""Stage 2 visualizations: Surface Segmentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from mesh2param.models.region import FaceRegion
from mesh2param.viz.base import BaseVisualizer, add_report_banner_mpl, add_report_banner_pv
from mesh2param.viz.colors import DARK_ACCENT, DARK_BG, DARK_GRID, get_n_colors, surface_type_color


class Stage2Viz(BaseVisualizer):
    """Visualizations for surface segmentation results."""

    def region_color_map(
        self,
        mesh: pv.PolyData,
        regions: list[FaceRegion],
        save_path: str | Path | None = None,
    ) -> str | None:
        """Color each segmented region with a unique color.

        Region ID labels are placed at each region's centroid.
        """
        p = self._make_plotter(save_path)
        n_regions = len(regions)
        colors = get_n_colors(n_regions)

        face_colors = np.full((mesh.n_cells, 3), 128, dtype=np.uint8)
        centroids = []
        labels = []

        for i, region in enumerate(regions):
            hex_c = colors[i]
            r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            for fi in region.face_indices:
                if fi < mesh.n_cells:
                    face_colors[fi] = [r, g, b]
            centroid = region.vertices.mean(axis=0) if len(region.vertices) > 0 else np.zeros(3)
            centroids.append(centroid)
            labels.append(f"R{region.region_id}: {region.surface_type.value}")

        mesh.cell_data["RegionRGB"] = face_colors
        p.add_mesh(mesh, scalars="RegionRGB", rgb=True)

        if centroids:
            pts = np.array(centroids)
            p.add_point_labels(
                pts,
                labels,
                font_size=10,
                text_color="white",
                shape_color=DARK_BG,
                shape_opacity=0.7,
                point_size=1,
            )

        p.camera_position = "iso"

        add_report_banner_pv(
            p,
            test_name="Region Color Map",
            pass_criteria="Every face assigned to a region (no gray patches); each flat surface is one "
            "color; cylindrical holes are distinct from planar faces",
            analysis_guide="Each color = one segmented region. Gray faces = unassigned (bad). "
            "Adjacent regions should have distinct colors. Labels show region ID and classified type. "
            "Check that planar walls aren't split across multiple regions, and holes aren't merged with flat faces.",
        )

        return self._finalize(p, save_path)

    def boundary_edges(
        self,
        mesh: pv.PolyData,
        regions: list[FaceRegion],
        vertices: np.ndarray,
        save_path: str | Path | None = None,
    ) -> str | None:
        """Draw region boundary edges as black lines on a semi-transparent mesh."""
        p = self._make_plotter(save_path)
        p.add_mesh(mesh, color="#94A3B8", opacity=0.4)

        for region in regions:
            if not region.boundary_edges:
                continue
            lines = []
            for v0, v1 in region.boundary_edges:
                if v0 < len(vertices) and v1 < len(vertices):
                    lines.append(vertices[v0])
                    lines.append(vertices[v1])
            if lines:
                pts = np.array(lines)
                n_segs = len(pts) // 2
                cells = []
                for i in range(n_segs):
                    cells.extend([2, i * 2, i * 2 + 1])
                line_mesh = pv.PolyData(pts, lines=cells)
                p.add_mesh(line_mesh, color="black", line_width=2)

        p.camera_position = "iso"

        add_report_banner_pv(
            p,
            test_name="Region Boundary Edges",
            pass_criteria="Black boundary lines trace all sharp edges and surface transitions; "
            "no boundaries cutting through the middle of a flat face",
            analysis_guide="Black lines = boundaries between segmented regions. They should follow "
            "geometric edges (corners, holes, fillets). If a boundary cuts through a clearly flat "
            "surface, the region was incorrectly split. Missing boundaries between different surfaces "
            "means regions were incorrectly merged.",
        )

        return self._finalize(p, save_path)

    def region_stats_table(
        self,
        regions: list[FaceRegion],
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Table of region statistics: ID, face count, surface type, avg normal, curvature."""
        if save_path is not None:
            matplotlib.use("Agg")

        headers = ["Region", "Faces", "Type", "Avg Normal", "Curv Mean", "Curv Var"]
        rows = []
        row_colors = []

        for region in sorted(regions, key=lambda r: r.num_faces, reverse=True):
            avg_n = region.normals.mean(axis=0) if len(region.normals) > 0 else np.zeros(3)
            norm = np.linalg.norm(avg_n)
            if norm > 1e-15:
                avg_n /= norm
            avg_n_str = f"({avg_n[0]:.2f}, {avg_n[1]:.2f}, {avg_n[2]:.2f})"

            # Curvature stats from normal spread (proxy)
            if len(region.normals) > 1:
                dots = region.normals @ avg_n
                curv_mean = f"{1.0 - float(np.mean(dots)):.4f}"
                curv_var = f"{float(np.var(dots)):.4f}"
            else:
                curv_mean = "0.0"
                curv_var = "0.0"

            rows.append([
                f"R{region.region_id}",
                str(region.num_faces),
                region.surface_type.value,
                avg_n_str,
                curv_mean,
                curv_var,
            ])
            hex_c = surface_type_color(region.surface_type.value)
            r, g, b = int(hex_c[1:3], 16) / 255, int(hex_c[3:5], 16) / 255, int(hex_c[5:7], 16) / 255
            row_colors.append((r, g, b, 0.2))

        fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(rows) + 5)))
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
            table.scale(1, 1.5)

            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor(DARK_GRID)
                if row == 0:
                    cell.set_facecolor(DARK_GRID)
                    cell.set_text_props(color=DARK_ACCENT, fontweight="bold")
                else:
                    cell.set_facecolor(row_colors[row - 1] if row - 1 < len(row_colors) else DARK_BG)
                    cell.set_text_props(color="#E2E8F0")

        add_report_banner_mpl(
            fig,
            test_name="Region Statistics Table",
            pass_criteria="All faces accounted for across regions; plane regions have curv_mean ≈ 0; "
            "cylinder/sphere regions have non-zero curvature; no UNKNOWN regions on clearly analytic surfaces",
            analysis_guide="Sorted by face count descending. Plane regions should show near-zero curvature "
            "mean and variance. Cylinder regions have moderate curvature in one direction. "
            "UNKNOWN type means the classifier couldn't decide — check if those regions are fillet transitions "
            "or genuinely ambiguous geometry.",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.65])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)

        return fig
