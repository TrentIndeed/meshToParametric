"""Shared color constants and utilities for all visualizations."""

from __future__ import annotations

from distinctipy import distinctipy

# Feature type colors
FEATURE_COLORS = {
    "BASE_EXTRUSION": "#6366F1",
    "POCKET": "#EF4444",
    "THROUGH_HOLE": "#F59E0B",
    "BLIND_HOLE": "#F97316",
    "BOSS": "#10B981",
    "SLOT": "#EC4899",
    "FILLET": "#06B6D4",
    "CHAMFER": "#8B5CF6",
}

# Edge type colors
EDGE_COLORS = {
    "SHARP": "#22C55E",
    "FILLET": "#3B82F6",
    "CHAMFER": "#F97316",
    "SMOOTH": "#6B7280",
}

# Surface type colors
SURFACE_COLORS = {
    "PLANE": "#818CF8",
    "CYLINDER": "#F472B6",
    "CONE": "#FBBF24",
    "SPHERE": "#34D399",
    "TORUS": "#FB923C",
    "UNKNOWN": "#6B7280",
}

# Dark theme constants
DARK_BG = "#0F172A"
DARK_TEXT = "#E2E8F0"
DARK_GRID = "#1E293B"
DARK_ACCENT = "#94A3B8"


def get_n_colors(n: int) -> list[str]:
    """Generate n maximally distinct hex colors."""
    if n == 0:
        return []
    rgbs = distinctipy.get_colors(n)
    return [
        "#" + "".join(f"{int(c * 255):02x}" for c in rgb)
        for rgb in rgbs
    ]


def surface_type_color(surface_type: str) -> str:
    """Get the hex color for a surface type string."""
    return SURFACE_COLORS.get(surface_type.upper(), SURFACE_COLORS["UNKNOWN"])
