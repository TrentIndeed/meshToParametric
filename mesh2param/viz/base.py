"""Base visualizer with dev/CI mode switching."""

from __future__ import annotations

import os
import textwrap

import matplotlib.pyplot as plt
import pyvista as pv

from mesh2param.viz.colors import DARK_ACCENT, DARK_BG, DARK_GRID


def add_report_banner_pv(
    plotter: pv.Plotter,
    test_name: str,
    pass_criteria: str,
    analysis_guide: str,
) -> None:
    """Add a 3-line annotation banner to a PyVista plotter.

    Appears in the lower-left corner of the rendered image:
        TEST: <test_name>
        PASS: <pass_criteria>
        LOOK: <analysis_guide>
    """
    plotter.add_text(
        f"TEST: {test_name}",
        position="upper_left",
        font_size=32,
        color="#94A3B8",
    )
    detail = (
        f"PASS:\n{pass_criteria}\n\n"
        f"LOOK:\n{analysis_guide}"
    )
    plotter.add_text(
        detail,
        position="lower_left",
        font_size=22,
        color="#94A3B8",
    )


def add_report_banner_mpl(
    fig: plt.Figure,
    test_name: str,
    pass_criteria: str,
    analysis_guide: str,
) -> None:
    """Add a 3-line annotation banner to a matplotlib figure.

    Appears at the bottom of the figure.
    """
    fig.text(
        0.5, 0.99, f"TEST: {test_name}",
        ha="center", va="top",
        fontsize=28, color="#94A3B8",
        fontweight="bold",
    )
    pass_wrapped = textwrap.fill(f"PASS: {pass_criteria}", width=110)
    look_wrapped = textwrap.fill(f"LOOK: {analysis_guide}", width=110)
    detail = f"{pass_wrapped}\n\n{look_wrapped}"
    fig.text(
        0.5, 0.96, detail,
        ha="center", va="top",
        fontsize=20, color="#94A3B8",
        fontstyle="italic",
    )


class BaseVisualizer:
    """All viz classes inherit from this.

    Handles automatic switching between interactive dev mode
    and offscreen CI rendering.

    Usage:
        viz = Stage1Viz()
        viz.show(mesh)            # interactive dev window
        viz.save(mesh, "out.png") # offscreen CI render
    """

    BACKGROUND_COLOR = "#0F172A"

    def __init__(self):
        self.headless = (
            os.environ.get("CI") == "true" or not _display_available()
        )
        if self.headless and os.name == "posix":
            try:
                pv.start_xvfb()
            except Exception:
                pass

    def _make_plotter(self, save_path=None, **kwargs):
        off_screen = save_path is not None or self.headless
        p = pv.Plotter(off_screen=off_screen, **kwargs)
        p.set_background(self.BACKGROUND_COLOR)
        return p

    def _finalize(self, plotter, save_path=None):
        if save_path:
            plotter.screenshot(save_path, transparent_background=False)
            plotter.close()
            return str(save_path)
        else:
            plotter.show()
            return None


def _display_available() -> bool:
    """Check if a display is available for interactive rendering."""
    if os.name == "nt":
        return True  # Windows always has a display
    return os.environ.get("DISPLAY") is not None
