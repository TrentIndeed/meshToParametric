"""Greedy 2D contour fitting: turn a noisy ordered loop of 2D points into a
clean sequence of line + arc primitives.

This is the technique CADFit (github.com/ghadinehme/CADFit) uses in
`fit_primitives.py`, ported here to harden base-profile extraction on noisy
AI/scan silhouettes — where a slice outline arrives as hundreds of jittery
points that should collapse to a few lines and arcs.

Algorithm (`segment_greedy_fit`): walk the loop; at each start index extend the
segment as far as a line OR a circular arc still fits within tolerance; commit
whichever primitive has the lower RMS; advance. Lines come from a PCA fit
(direction = dominant eigenvector, error = perpendicular spread); circles/arcs
from Kasa's algebraic fit (center/radius from a linear system, error = radial
residual). Tolerances scale with the loop's bbox diagonal so the same code
handles mm parts and unit-box-normalized profiles. Pure numpy — no kernel, no
optimization loop, no ML.

Deliberately dependency-light and deterministic, matching the rest of the
pipeline (no RANSAC randomness): the greedy sweep is reproducible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Primitive:
    """One fitted contour primitive.

    kind: "line" | "arc" | "circle".
    points: the (start, end) endpoints (2,2) for line/arc; for a full circle
        the single representative point on the ring.
    center, radius, start_angle, end_angle: set for arc/circle only.
    rms: fit residual (same units as the input points).
    n_points: how many raw loop points this primitive absorbed.
    """

    kind: str
    points: np.ndarray
    rms: float
    n_points: int
    center: np.ndarray | None = None
    radius: float = 0.0
    start_angle: float = 0.0
    end_angle: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "rms": float(self.rms),
             "n_points": int(self.n_points),
             "points": np.asarray(self.points, dtype=float).tolist()}
        if self.center is not None:
            d.update(center=np.asarray(self.center, float).tolist(),
                     radius=float(self.radius),
                     start_angle=float(self.start_angle),
                     end_angle=float(self.end_angle))
        return d


def fit_line_2d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA line fit. Returns (point_on_line, unit_direction, perpendicular RMS)."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    d = pts - c
    # Dominant eigenvector of the 2x2 scatter = line direction.
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    perp = d @ normal
    rms = float(np.sqrt(np.mean(perp ** 2))) if len(pts) else 0.0
    return c, direction, rms


def fit_circle_2d(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Kasa's algebraic circle fit. Returns (center(2,), radius, radial RMS).

    Solves the linear system for the circle  x^2 + y^2 + a x + b y + c = 0;
    far cheaper and more stable than iterative geometric fitting and good
    enough as a segmentation test. radius=inf / huge rms when the points are
    near-collinear (the system is ill-conditioned), which correctly makes the
    line fit win for straight runs.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return np.zeros(2), float("inf"), float("inf")
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(pts))])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(2), float("inf"), float("inf")
    cx, cy, c = sol
    r2 = c + cx ** 2 + cy ** 2
    if r2 <= 0:
        return np.array([cx, cy]), float("inf"), float("inf")
    r = float(np.sqrt(r2))
    center = np.array([cx, cy])
    radial = np.linalg.norm(pts - center, axis=1) - r
    rms = float(np.sqrt(np.mean(radial ** 2)))
    return center, r, rms


def angular_coverage(pts: np.ndarray, center: np.ndarray) -> float:
    """Angular span (radians, 0..2pi) the points cover about `center`, robust
    to wraparound. A near-straight run covers a tiny span (so it is NOT an
    arc); a real arc covers a meaningful sweep."""
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ang = np.sort(ang)
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = (ang[0] + 2 * np.pi) - ang[-1]
    largest_gap = max(float(gaps.max()), wrap)
    return float(2 * np.pi - largest_gap)


def _dedup(points: np.ndarray, tol: float) -> np.ndarray:
    """Drop consecutive near-duplicate points (slice outlines repeat vertices)."""
    if len(points) == 0:
        return points
    keep = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    return np.asarray(keep, dtype=float)


def segment_greedy_fit(points: np.ndarray,
                       tol_frac: float = 0.01,
                       min_arc_span_deg: float = 25.0,
                       closed: bool = True) -> list[Primitive]:
    """Segment an ordered 2D loop into line + arc primitives (greedy).

    Args:
        points: (N,2) ordered loop vertices.
        tol_frac: fit tolerance as a fraction of the loop's bbox diagonal.
        min_arc_span_deg: a run only becomes an ARC if it sweeps at least this
            many degrees about the fitted center; otherwise it stays a line
            (prevents calling a slightly-bent straight edge an arc).
        closed: treat the loop as closed (wrap the last segment to the start).

    Returns the list of primitives in order. Falls back to a single "spline"
    primitive carrying the raw points if the loop is too small to segment.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return [Primitive("spline", pts.reshape(-1, 2), 0.0, len(pts))]

    diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    tol = tol_frac * diag
    pts = _dedup(pts, tol * 0.25)
    n = len(pts)
    if n < 3:
        return [Primitive("line", pts[[0, -1]], 0.0, n)]

    # A closed loop is walked over [0, n]; index n maps back to point 0.
    seq = np.vstack([pts, pts[0]]) if closed else pts
    limit = len(seq)

    out: list[Primitive] = []
    i = 0
    while i < limit - 1:
        best_j = i + 1          # at least a 2-point line
        best_prim: Primitive | None = None
        # Extend the run as far as a line OR arc still fits.
        j = i + 2
        while j <= limit - 1 if not closed else j <= limit:
            seg = seq[i:j + 1] if j <= limit - 1 else np.vstack([seq[i:], seq[:1]])
            if len(seg) < 2:
                j += 1
                continue
            _, _, line_rms = fit_line_2d(seg)
            center, radius, circ_rms = (fit_circle_2d(seg)
                                        if len(seg) >= 3
                                        else (None, float("inf"), float("inf")))
            span = (np.degrees(angular_coverage(seg, center))
                    if center is not None and np.isfinite(radius) else 0.0)
            line_ok = line_rms <= tol
            arc_ok = (circ_rms <= tol and span >= min_arc_span_deg
                      and radius < 50 * diag)
            if not (line_ok or arc_ok):
                break
            # Prefer the lower-RMS primitive among those that fit.
            if arc_ok and (not line_ok or circ_rms < line_rms):
                best_prim = Primitive("arc", seg[[0, -1]].copy(), circ_rms,
                                      len(seg), center=center, radius=radius,
                                      start_angle=float(np.arctan2(
                                          seg[0, 1] - center[1],
                                          seg[0, 0] - center[0])),
                                      end_angle=float(np.arctan2(
                                          seg[-1, 1] - center[1],
                                          seg[-1, 0] - center[0])))
            else:
                best_prim = Primitive("line", seg[[0, -1]].copy(), line_rms,
                                      len(seg))
            best_j = j
            j += 1
        if best_prim is None:
            best_prim = Primitive("line", seq[[i, i + 1]].copy(), 0.0, 2)
            best_j = i + 1
        out.append(best_prim)
        i = best_j

    return _merge_adjacent_lines(out, tol)


def _merge_adjacent_lines(prims: list[Primitive], tol: float) -> list[Primitive]:
    """Collapse consecutive near-collinear line primitives into one."""
    if len(prims) < 2:
        return prims
    merged: list[Primitive] = [prims[0]]
    for p in prims[1:]:
        last = merged[-1]
        if p.kind == "line" and last.kind == "line":
            seg = np.vstack([last.points[0], last.points[1], p.points[1]])
            _, _, rms = fit_line_2d(seg)
            if rms <= tol:
                merged[-1] = Primitive("line",
                                       np.vstack([last.points[0], p.points[1]]),
                                       rms, last.n_points + p.n_points)
                continue
        merged.append(p)
    return merged
