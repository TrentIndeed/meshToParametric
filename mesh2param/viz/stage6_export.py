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
    """Visualization for CadQuery solid reconstruction.

    Styled to match Fusion 360's viewport appearance:
    - Light gray gradient background
    - Silver/aluminum PBR metallic body material
    - Dark edge outlines
    - Clean sans-serif labels
    """

    # Fusion 360-style background: light gray gradient
    _BG_TOP = "#D4D8DC"
    _BG_BOTTOM = "#F0F2F4"

    # Fusion 360-style body color: aluminum silver
    _BODY_COLOR = "#B0B8C0"
    _BODY_METALLIC = 0.45
    _BODY_ROUGHNESS = 0.35

    # Highlight for newly-changed geometry in each step
    _HIGHLIGHT_COLOR = "#FF6B35"
    _HIGHLIGHT_METALLIC = 0.2
    _HIGHLIGHT_ROUGHNESS = 0.5

    # Edge styling
    _EDGE_COLOR = "#2D3436"
    _EDGE_WIDTH = 1

    # Text
    _LABEL_COLOR = "#2D3436"
    _SUBLABEL_COLOR = "#636E72"

    def _setup_fusion_panel(self, plotter):
        """Apply Fusion 360 viewport styling to a plotter/subplot."""
        plotter.set_background(self._BG_BOTTOM, top=self._BG_TOP)
        try:
            plotter.enable_anti_aliasing("ssaa")
        except Exception:
            pass

    def _add_body_mesh(self, plotter, mesh, highlight=False):
        """Add a mesh with Fusion 360-style material."""
        color = self._HIGHLIGHT_COLOR if highlight else self._BODY_COLOR
        try:
            plotter.add_mesh(
                mesh,
                color=color,
                pbr=True,
                metallic=self._BODY_METALLIC,
                roughness=self._BODY_ROUGHNESS,
                show_edges=True,
                edge_color=self._EDGE_COLOR,
                line_width=self._EDGE_WIDTH,
            )
        except Exception:
            # PBR fallback for subplots that don't support it
            plotter.add_mesh(
                mesh,
                color=color,
                show_edges=True,
                edge_color=self._EDGE_COLOR,
                line_width=self._EDGE_WIDTH,
                specular=0.6,
                specular_power=30,
            )

    def cadquery_solid_3d(
        self,
        cq_result,
        original_mesh: pv.PolyData | None = None,
        save_path: str | Path | None = None,
    ) -> str | None:
        """Render the construction process as a Fusion 360-style slideshow.

        Renders each step as a separate frame, then composites them into
        a single grid image using matplotlib.
        """
        import matplotlib
        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from io import BytesIO

        steps = getattr(cq_result, "construction_steps", [])

        # Build list of (label, sub_label, tess_verts, tess_faces, is_sketch)
        panels = []
        for i, step in enumerate(steps):
            if step.tess_faces is None or len(step.tess_faces) == 0:
                continue
            is_sketch = "sketch" in step.name.lower()
            vol_text = "2D Profile" if is_sketch else f"{step.volume:,.0f} mm\u00B3"
            panels.append((
                f"Step {i + 1}: {step.name}",
                vol_text,
                step.tess_vertices,
                step.tess_faces,
                is_sketch,
            ))

        # Add original mesh as final panel
        if original_mesh is not None:
            verts_orig = np.array(original_mesh.points)
            faces_orig = original_mesh.faces.reshape(-1, 4)[:, 1:]
            vol_err = cq_result.volume_error_pct
            panels.append((
                "Original Mesh",
                f"Volume error: {vol_err:.1f}%",
                verts_orig,
                faces_orig,
                False,
            ))

        if not panels:
            return None

        # Compute fixed bounding box from all panels for consistent camera
        all_verts = np.concatenate([v for _, _, v, _, _ in panels])
        fixed_bounds = (
            all_verts.min(axis=0) - 5,
            all_verts.max(axis=0) + 5,
        )

        # Render each panel to an image buffer
        panel_images = []
        for label, sub_label, verts, faces, is_sketch in panels:
            img = self._render_single_panel(verts, faces, label, sub_label,
                                            is_sketch=is_sketch,
                                            fixed_bounds=fixed_bounds)
            panel_images.append(img)

        # Composite into a grid using matplotlib
        cols = min(4, len(panel_images))
        rows = max(1, (len(panel_images) + cols - 1) // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        fig.patch.set_facecolor(self._BG_BOTTOM)

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for i, img in enumerate(panel_images):
            r, c = divmod(i, cols)
            axes[r, c].imshow(img)
            axes[r, c].axis("off")

        # Hide empty panels
        for i in range(len(panel_images), rows * cols):
            r, c = divmod(i, cols)
            axes[r, c].axis("off")

        fig.suptitle(
            "CadQuery Construction Sequence",
            fontsize=16, fontweight="bold", color=self._LABEL_COLOR, y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight",
                        facecolor=self._BG_BOTTOM)
            plt.close(fig)
            return str(save_path)
        else:
            plt.show()
            return None

    def _render_single_panel(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        label: str,
        sub_label: str,
        is_sketch: bool = False,
        fixed_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        camera_angle: int = 0,
    ) -> np.ndarray:
        """Render a single construction step to an image array.

        All panels use the same camera bounds so sizing stays consistent.
        XYZ origin axes are shown on every panel.
        """
        p = pv.Plotter(off_screen=True, window_size=(600, 600))
        p.set_background(self._BG_BOTTOM, top=self._BG_TOP)

        # XYZ origin axes on every panel
        ref_verts = verts if fixed_bounds is None else np.vstack([
            fixed_bounds[0], fixed_bounds[1],
        ])
        self._add_origin_axes(p, ref_verts)

        if is_sketch:
            # Draw the profile as a closed polyline
            n_pts = len(verts)
            line_pts = []
            line_cells = []
            idx = 0
            for i in range(n_pts):
                j = (i + 1) % n_pts
                line_pts.append(verts[i])
                line_pts.append(verts[j])
                line_cells.extend([2, idx, idx + 1])
                idx += 2
            line_mesh = pv.PolyData(np.array(line_pts), lines=line_cells)
            p.add_mesh(line_mesh, color="#2C3E50", line_width=3)
        else:
            # Solid body rendering — only show feature edges (not internal
            # triangle diagonals on flat faces)
            pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
            mesh = pv.PolyData(verts.copy(), pv_faces)
            mesh.compute_normals(cell_normals=True, point_normals=True, inplace=True)

            # Add the solid surface (no triangle edges)
            try:
                p.add_mesh(
                    mesh,
                    color=self._BODY_COLOR,
                    pbr=True,
                    metallic=self._BODY_METALLIC,
                    roughness=self._BODY_ROUGHNESS,
                )
            except Exception:
                p.add_mesh(
                    mesh,
                    color=self._BODY_COLOR,
                    specular=0.6,
                    specular_power=30,
                )

            # Add only the sharp feature edges (where adjacent faces meet
            # at > 30°), hiding the internal triangle splits on flat faces
            try:
                feat_edges = mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=True,
                    manifold_edges=False,
                    non_manifold_edges=False,
                    feature_angle=30,
                )
                p.add_mesh(feat_edges, color=self._EDGE_COLOR,
                           line_width=self._EDGE_WIDTH + 1)
            except Exception:
                pass

        p.add_text(label, position="upper_left", font_size=14,
                   color=self._LABEL_COLOR, font="arial")
        p.add_text(sub_label, position="lower_left", font_size=11,
                   color=self._SUBLABEL_COLOR, font="arial")

        try:
            p.enable_anti_aliasing("ssaa")
        except Exception:
            pass

        # Fixed camera: use bounds to set consistent view across all panels
        if fixed_bounds is not None:
            self._set_camera(p, fixed_bounds, camera_angle)
        else:
            p.camera_position = "iso"

        img = p.screenshot(return_img=True)
        p.close()
        return img

    # Three camera angles for the GIF rotation
    _CAMERA_ANGLES = [
        # angle 0: front-right isometric (default)
        (0.5, 0.7, 0.5),
        # angle 1: bottom-right (see underside + blind holes)
        (0.5, -0.6, 0.5),
        # angle 2: bottom-up angled
        (0.4, -0.8, 0.4),
    ]

    def _set_camera(self, plotter, fixed_bounds, angle_index=0):
        """Set camera position from fixed bounds and angle index."""
        bmin, bmax = fixed_bounds
        center = (bmin + bmax) / 2
        size = float(np.linalg.norm(bmax - bmin))
        cam_dist = size * 1.8
        dx, dy, dz = self._CAMERA_ANGLES[angle_index % len(self._CAMERA_ANGLES)]
        plotter.camera.focal_point = center.tolist()
        plotter.camera.position = (
            center[0] + cam_dist * dx,
            center[1] + cam_dist * dy,
            center[2] + cam_dist * dz,
        )
        plotter.camera.up = (0, 1, 0)
        plotter.reset_camera_clipping_range()

    def _add_origin_axes(self, plotter, verts: np.ndarray):
        """Add Fusion 360-style XYZ origin axes and a grid plane."""
        center = verts.mean(axis=0)
        extent = max(verts.max(axis=0) - verts.min(axis=0)) * 0.7

        # Grid lines on the sketch plane (XZ for this bracket)
        grid_color = "#CBD5E1"
        n_lines = 8
        for i in range(-n_lines, n_lines + 1):
            offset = i * extent / n_lines
            # Horizontal grid lines (along X)
            p1 = [center[0] - extent, center[1], center[2] + offset]
            p2 = [center[0] + extent, center[1], center[2] + offset]
            line = pv.Line(p1, p2)
            plotter.add_mesh(line, color=grid_color, line_width=0.5, opacity=0.3)
            # Vertical grid lines (along Z)
            p3 = [center[0] + offset, center[1], center[2] - extent]
            p4 = [center[0] + offset, center[1], center[2] + extent]
            line2 = pv.Line(p3, p4)
            plotter.add_mesh(line2, color=grid_color, line_width=0.5, opacity=0.3)

        # X axis (red)
        ax_len = extent * 1.2
        x_line = pv.Line(
            [center[0] - ax_len, center[1], center[2]],
            [center[0] + ax_len, center[1], center[2]],
        )
        plotter.add_mesh(x_line, color="#E74C3C", line_width=2)

        # Y axis (green)
        y_line = pv.Line(
            [center[0], center[1] - ax_len * 0.3, center[2]],
            [center[0], center[1] + ax_len * 0.3, center[2]],
        )
        plotter.add_mesh(y_line, color="#2ECC71", line_width=2)

        # Z axis (blue)
        z_line = pv.Line(
            [center[0], center[1], center[2] - ax_len],
            [center[0], center[1], center[2] + ax_len],
        )
        plotter.add_mesh(z_line, color="#3498DB", line_width=2)

    def construction_gif(
        self,
        cq_result,
        original_mesh: pv.PolyData | None = None,
        save_path: str | Path | None = None,
        frame_duration_ms: int = 1200,
        final_hold_ms: int = 3000,
    ) -> str | None:
        """Render an animated GIF of the construction process.

        Each frame shows the solid after one feature operation, with
        Fusion 360-style rendering. The final frame (original mesh
        comparison) holds longer.
        """
        from PIL import Image

        steps = getattr(cq_result, "construction_steps", [])
        if not steps:
            return None

        frames: list[Image.Image] = []
        durations: list[int] = []

        # Compute fixed bounds from all steps + original mesh
        all_step_verts = [s.tess_vertices for s in steps
                         if s.tess_vertices is not None and len(s.tess_vertices) > 0]
        if original_mesh is not None:
            all_step_verts.append(np.array(original_mesh.points))
        if not all_step_verts:
            return None
        all_verts = np.concatenate(all_step_verts)
        fixed_bounds = (all_verts.min(axis=0) - 5, all_verts.max(axis=0) + 5)

        angle_labels = ["Front-Right", "Bottom-Right", "Bottom-Up"]

        # Loop through all steps 3 times, each from a different camera angle
        for angle_idx in range(3):
            for i, step in enumerate(steps):
                if step.tess_faces is None or len(step.tess_faces) == 0:
                    continue
                is_sketch = "sketch" in step.name.lower()
                vol_text = "2D Profile" if is_sketch else f"{step.volume:,.0f} mm\u00B3"
                img = self._render_single_panel(
                    step.tess_vertices, step.tess_faces,
                    f"[{angle_labels[angle_idx]}]  Step {i + 1}/{len(steps)}: {step.name}",
                    vol_text,
                    is_sketch=is_sketch,
                    fixed_bounds=fixed_bounds,
                    camera_angle=angle_idx,
                )
                frames.append(Image.fromarray(img))
                durations.append(frame_duration_ms)

            # Original mesh comparison at the end of each pass
            if original_mesh is not None:
                verts = np.array(original_mesh.points)
                faces = original_mesh.faces.reshape(-1, 4)[:, 1:]
                vol_err = cq_result.volume_error_pct
                img = self._render_single_panel(
                    verts, faces,
                    f"[{angle_labels[angle_idx]}]  Original Mesh",
                    f"Volume error: {vol_err:.1f}%",
                    fixed_bounds=fixed_bounds,
                    camera_angle=angle_idx,
                )
                frames.append(Image.fromarray(img))
                durations.append(final_hold_ms)

        if not frames:
            return None

        if save_path:
            save_path = str(save_path)
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
            )
            return save_path
        return None


