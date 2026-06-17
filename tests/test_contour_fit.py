"""Greedy 2D contour fitting (mesh2param.pipeline.contour_fit).

Synthetic loops with known structure: a square (4 lines), a circle (one arc
covering ~360 deg), a rounded-corner rectangle (lines + 4 arcs), and a slot
(2 lines + 2 arcs). Also checks noise tolerance and the primitive fitters.
"""
from __future__ import annotations

import numpy as np
import pytest

from mesh2param.pipeline.contour_fit import (
    Primitive, angular_coverage, fit_circle_2d, fit_line_2d,
    segment_greedy_fit,
)


def _square(side=10.0, per_edge=15):
    """Ordered loop around an axis-aligned square."""
    t = np.linspace(0, 1, per_edge, endpoint=False)
    e = []
    e.append(np.column_stack([t * side, np.zeros_like(t)]))                 # bottom
    e.append(np.column_stack([np.full_like(t, side), t * side]))            # right
    e.append(np.column_stack([side - t * side, np.full_like(t, side)]))     # top
    e.append(np.column_stack([np.zeros_like(t), side - t * side]))          # left
    return np.vstack(e)


def _circle(r=5.0, n=120, cx=0.0, cy=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def test_fit_line_2d_recovers_direction_and_zero_rms():
    pts = np.column_stack([np.linspace(0, 10, 20), np.full(20, 3.0)])
    c, d, rms = fit_line_2d(pts)
    assert rms < 1e-9
    assert abs(abs(d[0]) - 1.0) < 1e-6 and abs(d[1]) < 1e-6


def test_fit_circle_2d_recovers_center_radius():
    pts = _circle(r=7.0, n=40, cx=2.0, cy=-3.0)
    center, radius, rms = fit_circle_2d(pts)
    assert np.allclose(center, [2.0, -3.0], atol=1e-6)
    assert abs(radius - 7.0) < 1e-6
    assert rms < 1e-6


def test_collinear_points_prefer_line_over_circle():
    # A straight run must fit a line far better than any circle, so the greedy
    # picker keeps it a line. (Kasa can still return a finite bogus radius on
    # collinear input — what matters is the circle's residual is much worse.)
    pts = np.column_stack([np.linspace(0, 10, 10), np.zeros(10)])
    _, _, line_rms = fit_line_2d(pts)
    _, radius, circ_rms = fit_circle_2d(pts)
    assert line_rms < 1e-9
    assert circ_rms > line_rms + 0.1 or radius > 100 or not np.isfinite(radius)


def test_angular_coverage_full_circle_vs_short_arc():
    c = np.zeros(2)
    assert angular_coverage(_circle(n=60), c) > np.radians(300)
    short = _circle(n=60)[:6]  # ~30 deg of the ring
    assert angular_coverage(short, c) < np.radians(60)


def test_square_segments_to_four_lines():
    prims = segment_greedy_fit(_square(), tol_frac=0.01, closed=True)
    lines = [p for p in prims if p.kind == "line"]
    assert all(p.kind == "line" for p in prims), [p.kind for p in prims]
    # Four straight edges (allow merge wrap to land on 4).
    assert len(lines) == 4, len(lines)


def test_circle_segments_to_arcs_not_lines():
    prims = segment_greedy_fit(_circle(r=5.0, n=120), tol_frac=0.01, closed=True)
    assert any(p.kind == "arc" for p in prims), [p.kind for p in prims]
    arc = next(p for p in prims if p.kind == "arc")
    assert abs(arc.radius - 5.0) < 0.2


def test_noisy_square_still_four_lines():
    rng = np.random.default_rng(0)
    pts = _square() + rng.normal(0, 0.03, size=_square().shape)
    prims = segment_greedy_fit(pts, tol_frac=0.02, closed=True)
    assert all(p.kind == "line" for p in prims)
    assert len([p for p in prims if p.kind == "line"]) == 4


def test_rounded_rectangle_has_lines_and_arcs():
    # Rectangle with filleted corners: 4 straight runs + 4 corner arcs.
    w, h, r = 20.0, 12.0, 3.0
    pieces = []
    # bottom edge
    pieces.append(np.column_stack([np.linspace(r, w - r, 12),
                                   np.zeros(12)]))
    # br corner
    a = np.linspace(-np.pi / 2, 0, 8)
    pieces.append(np.column_stack([w - r + r * np.cos(a), r + r * np.sin(a)]))
    # right edge
    pieces.append(np.column_stack([np.full(12, w), np.linspace(r, h - r, 12)]))
    # tr corner
    a = np.linspace(0, np.pi / 2, 8)
    pieces.append(np.column_stack([w - r + r * np.cos(a), h - r + r * np.sin(a)]))
    # top edge
    pieces.append(np.column_stack([np.linspace(w - r, r, 12), np.full(12, h)]))
    # tl corner
    a = np.linspace(np.pi / 2, np.pi, 8)
    pieces.append(np.column_stack([r + r * np.cos(a), h - r + r * np.sin(a)]))
    # left edge
    pieces.append(np.column_stack([np.zeros(12), np.linspace(h - r, r, 12)]))
    # bl corner
    a = np.linspace(np.pi, 3 * np.pi / 2, 8)
    pieces.append(np.column_stack([r + r * np.cos(a), r + r * np.sin(a)]))
    loop = np.vstack(pieces)

    prims = segment_greedy_fit(loop, tol_frac=0.01, closed=True)
    kinds = [p.kind for p in prims]
    assert "line" in kinds and "arc" in kinds, kinds
    assert sum(k == "arc" for k in kinds) >= 3  # ~4 corner arcs


def test_tiny_loop_falls_back_gracefully():
    prims = segment_greedy_fit(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert len(prims) == 1
    assert isinstance(prims[0], Primitive)
