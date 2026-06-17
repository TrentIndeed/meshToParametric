"""Revolve-axis detection (mesh2param.pipeline.revolve_detect).

Profiles with known symmetry: a rectangle (mirror-symmetric about its two
mid-axes -> revolve candidate), a circle (symmetric about every axis), an
asymmetric L-shape (no symmetry axis -> None), and a tilted rectangle (axis
recovered at the tilt angle). Plus the 3D lift.
"""
from __future__ import annotations

import numpy as np

from mesh2param.pipeline.revolve_detect import (
    detect_revolve_axis, revolve_axis_3d,
)


def _rect_loop(w=20.0, h=8.0, per_edge=20, cx=0.0, cy=0.0):
    t = np.linspace(0, 1, per_edge, endpoint=False)
    e = [
        np.column_stack([cx - w / 2 + t * w, np.full_like(t, cy - h / 2)]),
        np.column_stack([np.full_like(t, cx + w / 2), cy - h / 2 + t * h]),
        np.column_stack([cx + w / 2 - t * w, np.full_like(t, cy + h / 2)]),
        np.column_stack([np.full_like(t, cx - w / 2), cy + h / 2 - t * h]),
    ]
    return np.vstack(e)


def _circle(r=5.0, n=120):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _l_shape():
    return np.array([
        [0, 0], [10, 0], [10, 3], [3, 3], [3, 10], [0, 10],
        [0, 5], [1, 5], [1, 2], [0, 2],
    ], dtype=float)


def test_rectangle_is_revolve_candidate():
    cand = detect_revolve_axis(_rect_loop(w=20, h=8))
    assert cand is not None
    # Mirror axis is one of the two midlines: direction ~ horizontal or vertical.
    d = np.abs(cand.axis_dir)
    assert (d[0] > 0.98 or d[1] > 0.98), cand.axis_dir
    assert cand.mismatch < 0.02
    assert len(cand.half_profile) > 0


def test_circle_is_symmetric():
    cand = detect_revolve_axis(_circle())
    assert cand is not None
    assert cand.mismatch < 0.02


def test_l_shape_has_no_symmetry_axis():
    cand = detect_revolve_axis(_l_shape(), max_mismatch_frac=0.02)
    assert cand is None


def test_tilted_rectangle_axis_follows_tilt():
    rect = _rect_loop(w=20, h=8)
    theta = np.radians(30.0)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    tilted = rect @ R.T
    cand = detect_revolve_axis(tilted)
    assert cand is not None
    # Axis should align with the long edge (30 deg) or its perpendicular.
    ang = np.degrees(np.arctan2(cand.axis_dir[1], cand.axis_dir[0])) % 180
    assert min(abs(ang - 30), abs(ang - 120)) < 6, ang


def test_revolve_axis_3d_lift_on_xy_plane():
    res = revolve_axis_3d(
        _rect_loop(w=20, h=8),
        plane_origin=np.array([0.0, 0.0, 5.0]),
        plane_xdir=np.array([1.0, 0.0, 0.0]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )
    assert res is not None
    start, end, mismatch = res
    # Axis lies in the z=5 plane.
    assert abs(start[2] - 5.0) < 1e-6 and abs(end[2] - 5.0) < 1e-6
    assert mismatch < 0.02


def test_too_few_points_returns_none():
    assert detect_revolve_axis(np.array([[0.0, 0.0], [1.0, 1.0]])) is None
