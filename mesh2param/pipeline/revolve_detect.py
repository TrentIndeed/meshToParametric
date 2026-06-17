"""Revolve-axis detection from a planar profile.

This is the capability CADFit (github.com/ghadinehme/CADFit,
`analyze_revolve_axis.py`) has and this repo lacks: recognizing a
solid-of-revolution. The mesh→parametric pipeline here is otherwise
extrusion-only, so turned/round parts (shafts, bosses with conical tops,
domed caps) can only be approximated. This module supplies the geometric
test; wiring it into recognize.py/export is a follow-up.

Idea: a solid of revolution cut by a plane CONTAINING its axis yields a
profile that is mirror-symmetric across that axis. So we look for the
reflective-symmetry axis of the 2D profile. Candidate axes pass through the
profile centroid at a set of angles (the four 0/45/90/135 deg axes CADFit
uses, plus the two PCA principal directions); for each we reflect the points
and measure a symmetric Chamfer distance back to the original. The lowest
mismatch, if below tolerance, is the revolve axis. The half-profile on one
side of that axis is what you would revolve 360 deg.

Pure numpy + scipy KD-tree. Deterministic, no kernel build, no ML.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RevolveCandidate:
    """A detected revolve axis in the sketch plane's 2D coordinates.

    axis_point / axis_dir: a point on, and the unit direction of, the
        symmetry axis (2D, in the profile's own coordinates).
    mismatch: symmetric Chamfer distance after reflection, as a FRACTION of
        the profile bbox diagonal (0 = perfect symmetry). Lower is better.
    half_profile: the loop points lying on one side of the axis (the part you
        would revolve). May be empty if the split is degenerate.
    """

    axis_point: np.ndarray
    axis_dir: np.ndarray
    mismatch: float
    half_profile: np.ndarray


def _reflect(points: np.ndarray, p0: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Reflect `points` across the line through p0 with unit direction d."""
    rel = points - p0
    proj = np.outer(rel @ d, d)        # component along the axis
    perp = rel - proj                  # component across it
    return p0 + proj - perp            # flip the perpendicular part


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    try:
        from scipy.spatial import cKDTree
        ta, tb = cKDTree(a), cKDTree(b)
        da, _ = tb.query(a)
        db, _ = ta.query(b)
        return float(0.5 * (da.mean() + db.mean()))
    except Exception:
        # O(n^2) fallback (small loops only).
        da = np.min(np.linalg.norm(a[:, None] - b[None], axis=2), axis=1)
        db = np.min(np.linalg.norm(b[:, None] - a[None], axis=2), axis=1)
        return float(0.5 * (da.mean() + db.mean()))


def _candidate_directions(points: np.ndarray) -> list[np.ndarray]:
    """The fixed 0/45/90/135 deg axes plus the two PCA principal directions."""
    dirs = [np.array([np.cos(t), np.sin(t)])
            for t in np.radians([0.0, 45.0, 90.0, 135.0])]
    d = points - points.mean(0)
    try:
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        dirs.append(vt[0])
        dirs.append(vt[1])
    except np.linalg.LinAlgError:
        pass
    return dirs


def detect_revolve_axis(profile_2d: np.ndarray,
                        max_mismatch_frac: float = 0.02,
                        refine: bool = True) -> RevolveCandidate | None:
    """Return the best reflective-symmetry axis of a 2D profile, or None.

    Args:
        profile_2d: (N,2) ordered or unordered loop points in the sketch plane.
        max_mismatch_frac: accept the axis only if the post-reflection Chamfer
            distance is below this fraction of the profile bbox diagonal.
        refine: after the coarse candidate search, locally refine the axis
            angle (+/- a few degrees) to minimize mismatch.

    A circle is symmetric about every axis, so the FIRST acceptable candidate
    is returned (callers that care can treat tiny mismatch as "any axis").
    """
    pts = np.asarray(profile_2d, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        return None
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    centroid = pts.mean(0)

    def mismatch_for(d: np.ndarray) -> float:
        d = d / (np.linalg.norm(d) + 1e-12)
        return _chamfer(pts, _reflect(pts, centroid, d)) / diag

    best_dir, best_m = None, float("inf")
    for d in _candidate_directions(pts):
        m = mismatch_for(d)
        if m < best_m:
            best_m, best_dir = m, d / (np.linalg.norm(d) + 1e-12)

    if best_dir is None:
        return None

    if refine:
        base = np.arctan2(best_dir[1], best_dir[0])
        for dt in np.radians(np.linspace(-5, 5, 21)):
            d = np.array([np.cos(base + dt), np.sin(base + dt)])
            m = mismatch_for(d)
            if m < best_m:
                best_m, best_dir = m, d

    if best_m > max_mismatch_frac:
        return None

    # Half-profile: points on the positive perpendicular side of the axis.
    normal = np.array([-best_dir[1], best_dir[0]])
    side = (pts - centroid) @ normal
    half = pts[side >= -1e-9]
    return RevolveCandidate(axis_point=centroid, axis_dir=best_dir,
                            mismatch=best_m, half_profile=half)


def revolve_axis_3d(profile_2d: np.ndarray,
                    plane_origin: np.ndarray,
                    plane_xdir: np.ndarray,
                    plane_normal: np.ndarray,
                    **kwargs) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Lift a detected 2D revolve axis into 3D world coordinates.

    Returns (axis_start_3d, axis_end_3d, mismatch) or None. The sketch plane is
    defined by origin + xdir + normal; ydir is normal x xdir.
    """
    cand = detect_revolve_axis(profile_2d, **kwargs)
    if cand is None:
        return None
    xdir = np.asarray(plane_xdir, float)
    xdir = xdir / (np.linalg.norm(xdir) + 1e-12)
    ydir = np.cross(np.asarray(plane_normal, float), xdir)
    ydir = ydir / (np.linalg.norm(ydir) + 1e-12)
    origin = np.asarray(plane_origin, float)

    def to3d(p2):
        return origin + p2[0] * xdir + p2[1] * ydir

    ext = float(np.linalg.norm(profile_2d.max(0) - profile_2d.min(0)))
    p_lo = cand.axis_point - cand.axis_dir * ext
    p_hi = cand.axis_point + cand.axis_dir * ext
    return to3d(p_lo), to3d(p_hi), cand.mismatch
