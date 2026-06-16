"""FeatureTree -> Onshape BTM feature JSON.

Builds the ordered list of native Onshape features (construction
planes, sketches, extrudes) that reconstruct a recognized FeatureTree
as a real, editable parametric model. Every serialization detail here
was validated live against cad.onshape.com (scripts/onshape_probe*.py):

  - Default plane FACE deterministic IDs: Right=JEC, Front=JCC, Top=JDC.
    Sketch (u,v) maps to the remaining world axes in ASCENDING order —
    identical to the pipeline's own convention (axis 0 -> u=Y,v=Z;
    axis 1 -> u=X,v=Z; axis 2 -> u=X,v=Y) — so profile points transfer
    with no swizzle, only mm -> meters.
  - Plane normals (= default extrude direction): Right +X, Front -Y,
    Top +Z. A cPlane's offset must be a POSITIVE expression (negative
    violates the feature spec); going against the parent normal takes
    oppositeDirection=true. Extrudes from a cPlane keep the PARENT
    plane's normal as their default direction regardless of the
    cPlane's own flip.
  - Sketch plane references: default planes by deterministicIds;
    cPlanes by raw queryString 'query=qCreatedBy(makeId("<fid>"),
    EntityType.FACE);' (BTMIndividualCreatedByQuery-137 is rejected).
  - filterInnerLoops=true on the sketch-region query keeps inner rings
    (windows, baked bores) as holes when extruding a profile.
  - Hole cuts: circles sketched on a cPlane OUTSIDE the part, extrude
    REMOVE with endBound THROUGH_ALL pointed into the part.

Units: sketch entity geometry is in METERS; quantity parameters are
"<value> mm" expressions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from mesh2param.models.feature import Feature, FeatureType
from mesh2param.models.tree import FeatureTree

logger = logging.getLogger(__name__)

# (default-plane deterministic FACE id, plane normal sign on that axis)
PLANE_FOR_AXIS = {0: ("JEC", +1.0), 1: ("JCC", -1.0), 2: ("JDC", +1.0)}
UV_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


@dataclass
class PlannedFeature:
    """One Onshape feature plus bookkeeping for the export loop."""

    name: str
    btm: dict
    # Sketch features that must reference the featureId Onshape assigns
    # to an EARLIER planned feature (a cPlane) carry that dependency
    # here; the export loop patches the queryString after the cPlane is
    # created. Same for extrudes referencing their sketch.
    plane_dep: int | None = None   # index into the plan list
    sketch_dep: int | None = None  # index into the plan list
    feature_id: str = ""           # filled in as features are created
    # Fallback definitions tried in order when the primary errors (the
    # errored feature is deleted first). Used for blends: an explicit
    # edge set Parasolid refuses often applies as a single seed edge
    # with tangent propagation (RackMount's outer ring chains — which
    # even OCC could only partially roll — apply COMPLETELY this way).
    retry_btms: list = field(default_factory=list)


@dataclass
class OnshapePlan:
    features: list[PlannedFeature] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def add(self, pf: PlannedFeature) -> int:
        self.features.append(pf)
        return len(self.features) - 1


# ---------------------------------------------------------------------------
# BTM building blocks
# ---------------------------------------------------------------------------


def _line(eid: str, px: float, py: float, dx: float, dy: float,
          p0: float, p1: float) -> dict:
    return {
        "btType": "BTMSketchCurveSegment-155",
        "entityId": eid,
        "startPointId": f"{eid}.start",
        "endPointId": f"{eid}.end",
        "startParam": p0,
        "endParam": p1,
        "geometry": {
            "btType": "BTCurveGeometryLine-117",
            "pntX": px, "pntY": py, "dirX": dx, "dirY": dy,
        },
    }


def _circle(eid: str, cx: float, cy: float, r: float) -> dict:
    return {
        "btType": "BTMSketchCurve-4",
        "entityId": eid,
        "geometry": {
            "btType": "BTCurveGeometryCircle-115",
            "radius": r, "xCenter": cx, "yCenter": cy,
            "xDir": 1.0, "yDir": 0.0, "clockwise": False,
        },
    }


def _clean_polygon(pts_mm: list) -> list:
    """CadQuery's profile cleaning (repair + simplify) — REQUIRED for
    parity: raw engine rings carry duplicate/noisy points that defeat
    circle recognition, and CadQuery draws the cleaned points."""
    try:
        from mesh2param.pipeline.export_cadquery import (
            _clean_engine_polygon,
        )
        cleaned = _clean_engine_polygon(pts_mm)
        if len(cleaned) >= 3:
            return cleaned
    except Exception:
        logger.debug("Polygon cleaning failed; using raw", exc_info=True)
    return pts_mm


def _ring_entities(prefix: str, pts_mm: list) -> list[dict]:
    """Entities for one closed ring: a true CIRCLE when the points fit
    one, else a polyline.

    Mesh-derived inner rings (bores, tube walls) arrive as ~60-gon
    point lists; drawing them literally makes the Onshape bore a
    62-sided prism whose "rim" is 62 sliver edges — rim fillets then
    fail and the bore isn't a real cylinder. This is EXACTLY CadQuery's
    circle recognition (clean -> algebraic least-squares -> max-residual
    gate, n >= 12): the mean-center fit on raw points rejected rings
    CadQuery accepted (resid 0.008 after cleaning vs std 1.2 raw), and
    the n >= 12 floor matters because any 4 rectangle corners lie
    exactly on their circumcircle (resid 0.000).
    """
    pts = _clean_polygon(pts_mm)
    arr = np.asarray(pts, dtype=float)
    if len(arr) >= 12:
        A = np.c_[2 * arr[:, 0], 2 * arr[:, 1], np.ones(len(arr))]
        sol, *_ = np.linalg.lstsq(A, (arr**2).sum(axis=1), rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        cr = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
        resid = np.abs(np.hypot(arr[:, 0] - cx, arr[:, 1] - cy) - cr)
        if cr > 0.5 and float(resid.max()) < max(0.15, 0.02 * cr):
            return [_circle(
                f"{prefix}_c", cx / 1000.0, cy / 1000.0, cr / 1000.0,
            )]
    return _polygon_entities(prefix, pts)


def _arc_entity(eid: str, cx: float, cy: float, r: float,
                a0: float, a1: float) -> dict:
    """Arc segment: CCW from angle a0 to a1 (radians from +x at the
    center). clockwise=True is rejected by the extrude (probe6); a CW
    arc is expressed by swapping the angles."""
    return {
        "btType": "BTMSketchCurveSegment-155",
        "entityId": eid,
        "startPointId": f"{eid}.start",
        "endPointId": f"{eid}.end",
        "startParam": a0,
        "endParam": a1,
        "geometry": {
            "btType": "BTCurveGeometryCircle-115",
            "radius": r, "xCenter": cx, "yCenter": cy,
            "xDir": 1.0, "yDir": 0.0, "clockwise": False,
        },
    }


def _fit_circle_2d(arr: np.ndarray) -> tuple[float, float, float]:
    """Algebraic least-squares circle fit. Returns (cx, cy, r) in input
    units."""
    A = np.c_[2 * arr[:, 0], 2 * arr[:, 1], np.ones(len(arr))]
    sol, *_ = np.linalg.lstsq(A, (arr ** 2).sum(axis=1), rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    return cx, cy, r


def _nearest_axis(d, tol_deg: float = 15.0):
    """Return the sketch axis (±x / ±y unit vector) nearest direction d
    when within tol_deg, else None."""
    d = np.asarray(d, dtype=float)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return None
    d = d / nrm
    best, best_ang = None, tol_deg
    for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                 np.array([-1.0, 0.0]), np.array([0.0, -1.0])):
        ang = float(np.degrees(np.arccos(np.clip(d @ axis, -1, 1))))
        if ang < best_ang:
            best, best_ang = axis, ang
    return best


def _line_intersect(p1, d1, p2, d2):
    denom = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(denom) < 1e-9:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom
    return np.asarray(p1, dtype=float) + t * np.asarray(d1, dtype=float)


def _regularize_profile(pts_mm: list):
    """Reduce a sampled outline to SHARP corners + per-corner radii.

    Driven by the STRAIGHT sides (prismatic parts are axis-aligned), so
    it handles mixed sharp and rounded corners uniformly:

      1. Mark each segment that is axis-aligned and long enough as
         straight; group consecutive same-axis straights into runs and
         fit each run's line (snapped exactly to its axis).
      2. Each consecutive pair of straight runs meets at a corner =
         their line intersection. If sampled bend vertices sit between
         the runs, circle-fit them for the rounding radius; otherwise
         the corner is sharp (no fillet).

    Feeding the sharp corners + radii to the analytic tangent-arc builder
    yields exactly-90-deg corners with arcs exactly tangent to the sides,
    so the outline reads clean AND the outer edge is one tangent loop a
    single fillet propagates around. Returns (sharp_pts_mm, {idx: r}) or
    (None, None) when the outline isn't a prismatic axis-aligned shape.
    """
    pts = _clean_polygon(pts_mm)
    P = np.asarray(pts, dtype=float)
    n = len(P)
    if n < 6:
        return None, None
    seg = np.roll(P, -1, axis=0) - P
    L = np.linalg.norm(seg, axis=1)
    axis_of = [_nearest_axis(seg[i]) for i in range(n)]
    straight = [axis_of[i] is not None and L[i] >= 2.0 for i in range(n)]
    if sum(straight) < 3:
        return None, None
    # start at the first straight segment whose predecessor is not the
    # same straight axis (a run boundary)
    def _same(a, b):
        return a is not None and b is not None and bool(np.allclose(a, b))

    start = next(
        (i for i in range(n)
         if straight[i] and not (straight[(i - 1) % n]
                                 and _same(axis_of[(i - 1) % n], axis_of[i]))),
        None,
    )
    if start is None:
        return None, None
    runs = []  # (axis, [segment indices])
    i = start
    guard = 0
    while guard < 4 * n:
        guard += 1
        if straight[i]:
            ax = axis_of[i]
            members = [i]
            j = (i + 1) % n
            while (guard < 4 * n and straight[j]
                   and _same(axis_of[j], ax)):
                members.append(j)
                j = (j + 1) % n
                guard += 1
            runs.append((ax, members))
            i = j
        else:
            i = (i + 1) % n
        if i == start:
            break
    if len(runs) < 3:
        return None, None
    # line per run: axis direction, centroid of its spanned vertices
    run_lines = []
    for ax, members in runs:
        vidx = members + [(members[-1] + 1) % n]
        ctr = P[vidx].mean(axis=0)
        run_lines.append((ctr, np.asarray(ax, dtype=float), members))
    m = len(runs)
    sharp_pts = []
    fillets = {}
    for k in range(m):
        ctr1, ax1, mem1 = run_lines[k]
        ctr2, ax2, mem2 = run_lines[(k + 1) % m]
        X = _line_intersect(ctr1, ax1, ctr2, ax2)
        if X is None:
            return None, None
        # corner (bend) vertices strictly between the two runs
        bend = []
        j = (mem1[-1] + 1) % n
        g = 0
        while j != mem2[0] and g < n:
            bend.append(j)
            j = (j + 1) % n
            g += 1
        r = 0.0
        if len(bend) >= 2:
            try:
                _, _, rr = _fit_circle_2d(P[bend])
                if 0.3 < rr < 60.0:
                    r = rr
            except np.linalg.LinAlgError:
                r = 0.0
        if r > 0:
            fillets[len(sharp_pts)] = r
        sharp_pts.append((float(X[0]), float(X[1])))
    if len(sharp_pts) < 3 or not fillets:
        return None, None
    return sharp_pts, fillets


def _analytic_tangent_arcs(prefix: str, sharp, fillets):
    """Sharp polygon + per-corner radii -> line/arc entities with arcs
    EXACTLY tangent to the flanking legs (tangent length r/tan(phi/2),
    center on the bisector at r/sin(phi/2)). Returns None if nothing
    rounds."""
    P = np.asarray(sharp, dtype=float) / 1000.0  # meters
    n = len(P)
    node_pts: list[tuple] = []  # (entry_pt, exit_pt, arc or None)
    any_arc = False
    for i in range(n):
        V = P[i]
        r_mm = fillets.get(i)
        if not r_mm:
            node_pts.append((V, V, None))
            continue
        r = float(r_mm) / 1000.0
        a = P[(i - 1) % n] - V
        b = P[(i + 1) % n] - V
        la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if la < 1e-9 or lb < 1e-9:
            node_pts.append((V, V, None))
            continue
        a, b = a / la, b / lb
        phi = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
        if phi < np.radians(10) or phi > np.radians(170):
            node_pts.append((V, V, None))
            continue
        t = r / np.tan(phi / 2.0)
        if t > 0.6 * min(la, lb):
            node_pts.append((V, V, None))
            continue
        T1 = V + a * t
        T2 = V + b * t
        C = V + (a + b) / np.linalg.norm(a + b) * (r / np.sin(phi / 2.0))
        ang1 = float(np.arctan2(T1[1] - C[1], T1[0] - C[0]))
        ang2 = float(np.arctan2(T2[1] - C[1], T2[0] - C[0]))
        sweep = (ang2 - ang1) % (2.0 * np.pi)
        if sweep > np.pi:
            ang1, ang2 = ang2, ang1
        if ang2 < ang1:
            ang2 += 2.0 * np.pi
        node_pts.append((T1, T2, (C, r, ang1, ang2)))
        any_arc = True
    if not any_arc:
        return None
    ents: list[dict] = []
    for i in range(n):
        _, exit_pt, _ = node_pts[i]
        entry_next, _, arc_next = node_pts[(i + 1) % n]
        d = entry_next - exit_pt
        seg_len = float(np.linalg.norm(d))
        if seg_len > 1e-9:
            ents.append(_line(
                f"{prefix}_l{i}", float(exit_pt[0]), float(exit_pt[1]),
                float(d[0] / seg_len), float(d[1] / seg_len), 0.0, seg_len,
            ))
        if arc_next is not None:
            C, r, a0, a1 = arc_next
            ents.append(_arc_entity(
                f"{prefix}_a{(i + 1) % n}", float(C[0]), float(C[1]),
                float(r), a0, a1,
            ))
    return ents


def _fit_arc_entity(eid: str, run_pts_mm: np.ndarray, E_m, X_m):
    """One arc through a sampled corner run: circle-fit the run, sweep
    the run's winding direction from endpoint E to endpoint X. Smooth
    but only approximately tangent (the secondary path when exact
    regularization fails)."""
    arr = np.asarray(run_pts_mm, dtype=float) / 1000.0
    if len(arr) < 3:
        return None
    try:
        cx, cy, cr = _fit_circle_2d(arr)
    except np.linalg.LinAlgError:
        return None
    if cr < 0.0003 or cr > 0.060:
        return None
    E = np.asarray(E_m, dtype=float)
    X = np.asarray(X_m, dtype=float)
    if (abs(np.hypot(E[0] - cx, E[1] - cy) - cr) > 0.5 * cr
            or abs(np.hypot(X[0] - cx, X[1] - cy) - cr) > 0.5 * cr):
        return None
    mid = arr[len(arr) // 2]
    turn_sign = np.sign(
        (mid[0] - E[0]) * (X[1] - mid[1]) - (mid[1] - E[1]) * (X[0] - mid[0])
    )
    Ea = float(np.arctan2(E[1] - cy, E[0] - cx))
    Xa = float(np.arctan2(X[1] - cy, X[0] - cx))
    if turn_sign >= 0:  # CCW
        a0, a1 = Ea, Ea + (Xa - Ea) % (2 * np.pi)
    else:               # CW expressed as CCW from X to E
        a0, a1 = Xa, Xa + (Ea - Xa) % (2 * np.pi)
    return _arc_entity(eid, cx, cy, cr, a0, a1)


def _line_seg(eid: str, p0_m, p1_m):
    d = np.asarray(p1_m, dtype=float) - np.asarray(p0_m, dtype=float)
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        return None
    return _line(eid, float(p0_m[0]), float(p0_m[1]),
                 float(d[0] / L), float(d[1] / L), 0.0, L)


def _walk_outline_with_arcs(prefix: str, pts_mm: list):
    """Replace each detected corner arc-run with one fitted arc, joining
    straight spans with lines. Smooth corners without exact tangency."""
    try:
        from mesh2param.pipeline.export_cadquery import (
            _detect_profile_arc_runs,
        )
    except Exception:
        return None
    runs = _detect_profile_arc_runs(pts_mm)
    if not runs:
        return None
    P = np.asarray(pts_mm, dtype=float)
    n = len(P)
    in_run = {}
    member = set()
    for (a, b) in runs:
        idxs = []
        j = a
        while True:
            idxs.append(j)
            if j == b:
                break
            j = (j + 1) % n
        in_run[a] = idxs
        member.update(idxs[1:])
    start = next((k for k in range(n)
                  if k not in member and k not in in_run), None)
    if start is None:
        return None
    ents = []
    cur = P[start] / 1000.0
    i = (start + 1) % n
    guard = 0
    eid = 0
    while i != start and guard < 2 * n:
        guard += 1
        if i in in_run:
            idxs = in_run[i]
            E = P[idxs[0]] / 1000.0
            X = P[idxs[-1]] / 1000.0
            seg = _line_seg(f"{prefix}_l{eid}", cur, E)
            if seg:
                ents.append(seg)
                eid += 1
            arc = _fit_arc_entity(f"{prefix}_a{eid}", P[idxs], E, X)
            ents.append(arc if arc else _line_seg(f"{prefix}_l{eid}", E, X))
            cur = X
            i = (idxs[-1] + 1) % n
        else:
            nxt = P[i] / 1000.0
            seg = _line_seg(f"{prefix}_l{eid}", cur, nxt)
            if seg:
                ents.append(seg)
            cur = nxt
            i = (i + 1) % n
        eid += 1
    seg = _line_seg(f"{prefix}_l{eid}", cur, P[start] / 1000.0)
    if seg:
        ents.append(seg)
    ents = [e for e in ents if e is not None]
    return ents if len(ents) >= 3 else None


def _entity_endpoints(e: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(start, end) of a line/arc sketch segment in METERS, or None for a
    full circle / unrecognized geometry."""
    g = e.get("geometry") or {}
    bt = g.get("btType", "")
    if "Line" in bt:
        px, py = g["pntX"], g["pntY"]
        dx, dy = g["dirX"], g["dirY"]
        p0, p1 = e.get("startParam", 0.0), e.get("endParam", 0.0)
        return (np.array([px + dx * p0, py + dy * p0]),
                np.array([px + dx * p1, py + dy * p1]))
    if "Circle" in bt:
        if "startParam" not in e or "endParam" not in e:
            return None  # full circle, not an arc segment
        cx, cy, r = g["xCenter"], g["yCenter"], g["radius"]
        a0, a1 = e["startParam"], e["endParam"]
        return (np.array([cx + r * np.cos(a0), cy + r * np.sin(a0)]),
                np.array([cx + r * np.cos(a1), cy + r * np.sin(a1)]))
    return None


def _chain_closes(ents: list[dict], tol_mm: float = 0.05) -> bool:
    """True when the ordered entities form a closed loop — each segment's
    end meets the next segment's start (and last meets first) within
    tol_mm. Onshape only forms a sketch REGION from a closed loop, so an
    arc whose endpoint misses the adjacent line by more than the merge
    tolerance yields an OK sketch with no region and the extrude ERRORs
    (Mesh3's corner arcs missed by ~4.7 mm). Sub-tolerance: reject and
    fall back to the polyline, which closes by construction."""
    eps = tol_mm / 1000.0
    n = len(ents)
    if n < 3:
        return False
    pts = [_entity_endpoints(e) for e in ents]
    if any(p is None for p in pts):
        return False
    for i in range(n):
        end = pts[i][1]
        start_next = pts[(i + 1) % n][0]
        if float(np.linalg.norm(end - start_next)) > eps:
            return False
    return True


def _outline_entities(prefix: str, pts_mm: list) -> list[dict]:
    """Outer-profile entities. Three paths, best first:

      A. Regularize to sharp axis-aligned corners + radii, build EXACT
         tangent arcs -> clean 90 deg corners + a tangent-continuous
         outer edge (a single fillet propagates the whole ring).
      B. Fit one arc per sampled corner run -> smooth but only
         approximately tangent (complex outlines the regularizer rejects).
      C. Raw polyline (last resort).

    Every arc-based path is verified to form a CLOSED loop before use; a
    chain that doesn't close (Onshape would build the sketch but fail the
    extrude with no region) is discarded in favor of the next path. The
    polyline always closes, so the base extrude never silently fails.
    """
    pts = _clean_polygon(pts_mm)
    sharp, fillets = _regularize_profile(pts_mm)
    if not (sharp and fillets):
        try:
            from mesh2param.pipeline.export_cadquery import (
                _sharp_polygon_with_fillets,
            )
            sharp, fillets = _sharp_polygon_with_fillets(pts)
        except Exception:
            sharp = fillets = None
    if sharp and fillets:
        ents = _analytic_tangent_arcs(prefix, sharp, fillets)
        if ents and _chain_closes(ents):
            return ents
        if ents:
            logger.info("Outline %s: analytic-arc chain open (gap > tol); "
                        "trying next path", prefix)
    ents = _walk_outline_with_arcs(prefix, pts)
    if ents and _chain_closes(ents):
        return ents
    if ents:
        logger.info("Outline %s: fitted-arc chain open; using polyline",
                    prefix)
    return _polygon_entities(prefix, pts)


def _polygon_entities(prefix: str, pts_mm: list) -> list[dict]:
    """Closed polyline entities for one ring (mm -> meters)."""
    ents = []
    n = len(pts_mm)
    for i in range(n):
        x0, y0 = float(pts_mm[i][0]) / 1000.0, float(pts_mm[i][1]) / 1000.0
        x1, y1 = (float(pts_mm[(i + 1) % n][0]) / 1000.0,
                  float(pts_mm[(i + 1) % n][1]) / 1000.0)
        dx, dy = x1 - x0, y1 - y0
        seg_len = float(np.hypot(dx, dy))
        if seg_len < 1e-9:
            continue
        ents.append(_line(
            f"{prefix}_{i}", x0, y0, dx / seg_len, dy / seg_len,
            0.0, seg_len,
        ))
    return ents


def _plane_param_default(det_id: str) -> dict:
    return {
        "btType": "BTMParameterQueryList-148",
        "parameterId": "sketchPlane",
        "queries": [{
            "btType": "BTMIndividualQuery-138",
            "deterministicIds": [det_id],
        }],
    }


def _plane_param_created_by(feature_id: str) -> dict:
    qs = f'query=qCreatedBy(makeId("{feature_id}"), EntityType.FACE);'
    return {
        "btType": "BTMParameterQueryList-148",
        "parameterId": "sketchPlane",
        "queries": [{
            "btType": "BTMIndividualQuery-138",
            "queryString": qs,
        }],
    }


def _sketch(name: str, plane_param: dict, entities: list[dict]) -> dict:
    return {
        "btType": "BTMSketch-151",
        "featureType": "newSketch",
        "name": name,
        "parameters": [plane_param],
        "entities": entities,
        "constraints": [],
    }


def _cplane(name: str, axis_idx: int, position_mm: float,
            size_mm: float = 15.0) -> dict:
    """Construction plane at world position `position_mm` on the axis.

    An explicit small `size` keeps the rendered plane local instead of
    the giant auto-fit rectangles that visually swamped the model (the
    plane is mathematically infinite for sketching either way).
    """
    det_id, normal_sign = PLANE_FOR_AXIS[axis_idx]
    offset = position_mm * normal_sign
    opposite = offset < 0
    return {
        "btType": "BTMFeature-134",
        "featureType": "cPlane",
        "name": name,
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{
                 "btType": "BTMIndividualQuery-138",
                 "deterministicIds": [det_id],
             }]},
            {"btType": "BTMParameterEnum-145", "parameterId": "cplaneType",
             "value": "OFFSET", "enumName": "CPlaneType"},
            {"btType": "BTMParameterQuantity-147", "parameterId": "offset",
             "expression": f"{abs(offset):.6f} mm"},
            {"btType": "BTMParameterBoolean-144",
             "parameterId": "oppositeDirection", "value": bool(opposite)},
            {"btType": "BTMParameterQuantity-147", "parameterId": "size",
             "expression": f"{size_mm:.1f} mm"},
        ],
    }


def _blend_feature(name: str, kind: str, size_mm: float,
                   edge_midpoints_mm: list,
                   tangent_propagation: bool = False) -> dict:
    """Fillet/chamfer with edges selected by closest-point queries.

    The Onshape rebuild is geometrically identical to the CadQuery
    rebuild (same profiles, same hole cylinders, same blend order), so
    the midpoints of the edges OCC actually rolled transfer directly:
    each qClosestTo picks the nearest edge to that midpoint on the
    current regenerated body. Validated live (probe5: 4-edge box fillet
    volume exact to 0.1 mm^3).
    """
    qparts = ", ".join(
        f"qClosestTo(qEverything(EntityType.EDGE), "
        f"vector({m[0]:.4f}, {m[1]:.4f}, {m[2]:.4f}) * millimeter)"
        for m in edge_midpoints_mm
    )
    params = [
        {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
         "queries": [{
             "btType": "BTMIndividualQuery-138",
             "queryString": f"query=qUnion([{qparts}]);",
         }]},
    ]
    if kind == "chamfer":
        params += [
            {"btType": "BTMParameterEnum-145", "parameterId": "chamferType",
             "value": "EQUAL_OFFSETS", "enumName": "ChamferType"},
            {"btType": "BTMParameterQuantity-147", "parameterId": "width",
             "expression": f"{size_mm:.4f} mm"},
        ]
    else:
        params.append(
            {"btType": "BTMParameterQuantity-147", "parameterId": "radius",
             "expression": f"{size_mm:.4f} mm"},
        )
    if tangent_propagation:
        params.append({
            "btType": "BTMParameterBoolean-144",
            "parameterId": "tangentPropagation",
            "value": True,
        })
    return {
        "btType": "BTMFeature-134",
        "featureType": "chamfer" if kind == "chamfer" else "fillet",
        "name": name,
        "parameters": params,
    }


def _extrude(name: str, sketch_fid: str, op: str, end_bound: str,
             depth_mm: float | None, opposite: bool,
             filter_inner: bool) -> dict:
    params = [
        {"btType": "BTMParameterEnum-145", "parameterId": "bodyType",
         "value": "SOLID", "enumName": "ExtendedToolBodyType"},
        {"btType": "BTMParameterEnum-145", "parameterId": "operationType",
         "value": op, "enumName": "NewBodyOperationType"},
        {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
         "queries": [{
             "btType": "BTMIndividualSketchRegionQuery-140",
             "featureId": sketch_fid,
             "filterInnerLoops": bool(filter_inner),
         }]},
        {"btType": "BTMParameterEnum-145", "parameterId": "endBound",
         "value": end_bound, "enumName": "BoundingType"},
        {"btType": "BTMParameterBoolean-144",
         "parameterId": "oppositeDirection", "value": bool(opposite)},
    ]
    if depth_mm is not None:
        params.append({
            "btType": "BTMParameterQuantity-147", "parameterId": "depth",
            "expression": f"{abs(depth_mm):.6f} mm",
        })
    return {
        "btType": "BTMFeature-134",
        "featureType": "extrude",
        "name": name,
        "parameters": params,
    }


def _extrude_opposite_for(axis_idx: int, want_positive: bool) -> bool:
    """oppositeDirection value to extrude toward +axis (or -axis)."""
    _, normal_sign = PLANE_FOR_AXIS[axis_idx]
    goes_positive_by_default = normal_sign > 0
    return goes_positive_by_default != want_positive


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _axis_of(direction) -> tuple[int, float]:
    d = np.asarray(direction, dtype=float)
    ax = int(np.argmax(np.abs(d)))
    return ax, float(np.sign(d[ax]) or 1.0)


def _profile_extrusion(plan: OnshapePlan, feat: Feature, label: str,
                       op: str, offset_mm: np.ndarray) -> bool:
    """Sketch+extrude for a base/join profile carried by the engine.

    sketch_geometry: {points: [[u,v]...] mm, span: [lo,hi] on the axis,
    inner: [rings], axis_idx (joins) — base derives axis from
    feature.direction}. offset_mm translates the model so it builds at
    the origin instead of wherever the scan happened to sit.
    """
    sg = feat.sketch_geometry or {}
    pts = sg.get("points")
    span = sg.get("span")
    if not pts or not span or len(pts) < 3:
        plan.skipped.append(f"{label}: no engine profile/span")
        return False
    if "axis_idx" in sg:
        ax = int(sg["axis_idx"])
    else:
        ax, _ = _axis_of(feat.direction)
    u_ax, v_ax = UV_AXES[ax]
    off_uv = np.array([offset_mm[u_ax], offset_mm[v_ax]])
    pts = (np.asarray(pts, dtype=float) + off_uv).tolist()
    lo = float(span[0]) + float(offset_mm[ax])
    hi = float(span[1]) + float(offset_mm[ax])

    # Sketch ON a default origin plane when the profile sits at the
    # origin (the base plate after anchoring) — no construction plane,
    # the part is built directly on Onshape's own Top/Front/Right plane
    # (user request). Other profiles (joins) sit away from the origin
    # and keep their own construction planes.
    if abs(lo) < 1e-6:
        det_id, _ = PLANE_FOR_AXIS[ax]
        plane_param = _plane_param_default(det_id)
        plane_dep = None
    else:
        cp_idx = plan.add(PlannedFeature(
            name=f"{label} plane",
            btm=_cplane(f"{label} plane", ax, lo),
        ))
        plane_param = _plane_param_created_by("PENDING")
        plane_dep = cp_idx
    # Prefer the PARAMETRIC arc representation (sharp corners + per-corner
    # radii): emit true line + tangent-arc sketch entities — the way Onshape
    # models a rounded outline, and identical to what the CadQuery build draws
    # from the same data. Fall back to point-based arc detection if absent.
    entities = None
    _corners = sg.get("corners")
    _fillets = sg.get("fillets")
    if _corners and len(_corners) >= 3 and _fillets:
        _fdict = {i: float(r) for i, r in enumerate(_fillets) if r and r > 0.05}
        if _fdict:
            _corners_off = (np.asarray(_corners, dtype=float) + off_uv).tolist()
            entities = _analytic_tangent_arcs("outer", _corners_off, _fdict)
    if entities is None:
        entities = _outline_entities("outer", pts)
    for ri, ring in enumerate(sg.get("inner") or []):
        if len(ring) >= 3:
            ring_s = (np.asarray(ring, dtype=float) + off_uv).tolist()
            entities += _ring_entities(f"inner{ri}", ring_s)
    sk_idx = plan.add(PlannedFeature(
        name=f"{label} sketch",
        btm=_sketch(f"{label} sketch", plane_param, entities),
        plane_dep=plane_dep,
    ))
    plan.add(PlannedFeature(
        name=label,
        btm=_extrude(
            label, "PENDING", op, "BLIND", hi - lo,
            _extrude_opposite_for(ax, want_positive=True),
            filter_inner=True,
        ),
        sketch_dep=sk_idx,
    ))
    return True


def _hole_cut(plan: OnshapePlan, feat: Feature, label: str,
              bbox_mm: tuple, offset_mm: np.ndarray) -> bool:
    sg = feat.sketch_geometry or {}
    radius = float(sg.get("radius", 0.0) or 0.0)
    centers = sg.get("centers") or (
        [sg.get("center")] if sg.get("center") is not None else []
    )
    centers = [c for c in centers if c is not None]
    if radius <= 0 or not centers:
        plan.skipped.append(f"{label}: no center/radius")
        return False
    centers = (np.asarray(centers, dtype=float) + offset_mm).tolist()
    return _hole_cut_raw(
        plan, centers, radius, feat.direction,
        feat.type == FeatureType.THROUGH_HOLE, float(feat.depth),
        label, bbox_mm,
    )


def _hole_cut_raw(plan: OnshapePlan, centers: list, radius: float,
                  direction, through: bool, depth: float, label: str,
                  bbox_mm: tuple) -> bool:
    ax, dir_sign = _axis_of(direction)
    u_ax, v_ax = UV_AXES[ax]

    if through:
        # Sketch just outside the LOCAL material at the holes, then cut
        # THROUGH_ALL into the part. Using the global bbox extreme put
        # the tab-hole sketch ~80 mm 'in front of' the model (the bbox
        # min on the hole axis is the far side-plate edge, nowhere near
        # the base-plate tabs — user report). Anchor to the holes
        # instead: their entry face is ~ center - depth/2.
        c = np.asarray(centers, dtype=float)
        half = float(depth) / 2.0 if depth and depth > 0 else 5.0
        plane_pos = float(c[:, ax].min()) - half - 2.0
        want_positive = True
        end_bound, depth = "THROUGH_ALL", None
    else:
        # Blind: the recognized center sits at the bore's axial middle;
        # the opening is half a depth back along the cut direction.
        c0 = np.asarray(centers[0], dtype=float)
        depth = float(depth)
        plane_pos = float(c0[ax]) - dir_sign * depth / 2.0
        want_positive = dir_sign > 0
        end_bound = "BLIND"

    cp_idx = plan.add(PlannedFeature(
        name=f"{label} plane",
        btm=_cplane(f"{label} plane", ax, plane_pos),
    ))
    entities = [
        _circle(f"h{i}", float(c[u_ax]) / 1000.0, float(c[v_ax]) / 1000.0,
                radius / 1000.0)
        for i, c in enumerate(centers)
    ]
    sk_idx = plan.add(PlannedFeature(
        name=f"{label} sketch",
        btm=_sketch(f"{label} sketch", _plane_param_created_by("PENDING"),
                    entities),
        plane_dep=cp_idx,
    ))
    plan.add(PlannedFeature(
        name=label,
        btm=_extrude(
            label, "PENDING", "REMOVE", end_bound, depth,
            _extrude_opposite_for(ax, want_positive=want_positive),
            filter_inner=False,
        ),
        sketch_dep=sk_idx,
    ))
    return True


def build_onshape_plan(feature_tree: FeatureTree,
                       bbox_mm: tuple,
                       as_built: dict | None = None) -> OnshapePlan:
    """Convert the recognized FeatureTree into an ordered Onshape plan.

    Mirrors the CadQuery timeline: base extrusion -> join extrusions ->
    hole cuts -> blends. When the CadQuery export's AS-BUILT record is
    provided, its hole list replaces the tree's hole features (the tree
    misses every hole the CadQuery recovery passes found — 3 of 4
    RackMount tab holes came only from 2D ring recall) and its blend
    list becomes fillet/chamfer features with closest-point edge
    queries. Without as_built, blends are skipped and tree holes are
    used as-is.

    The model is TRANSLATED so the BASE plate's own sketch plane lands
    on the origin (user request: "build ON origin, not offset from it").
    The base plate is CENTERED on the origin: its sketch-plane axis is
    anchored at its own span-lo (so the sketch sits on a default plane,
    not floated ~76 mm up by the side plates that form the bbox minimum)
    AND its in-plane center is moved to (0,0) (so the origin sits at the
    middle of the base plate, not off to one side by the full-part
    bbox-min — which is dominated by the wider side plates). Every other
    feature rides the same translation, so the assembly stays intact.
    """
    plan = OnshapePlan()
    features = list(feature_tree.features)
    bmin = np.asarray(bbox_mm[0], dtype=float)
    off = -bmin.copy()  # fallback: bbox-min -> origin

    base = next(
        (f for f in features if f.type == FeatureType.BASE_EXTRUSION), None,
    )
    # Prefer the AS-BUILT base profile: CadQuery's slice path captures the
    # interior cutouts (frame windows/pockets) the recognizer's engine
    # profile misses (outer outline only). Replaying it keeps the Onshape
    # base identical to the CadQuery solid — same outer, same holes — so a
    # frame no longer extrudes as a solid plate (Mesh3 ~2x volume).
    ab_base = (as_built or {}).get("base") if as_built else None
    if base is not None and ab_base and ab_base.get("outer"):
        import dataclasses
        try:
            base = dataclasses.replace(base, sketch_geometry={
                "points": ab_base["outer"],
                "inner": ab_base.get("inner") or [],
                "span": ab_base.get("span") or (base.sketch_geometry or {}).get("span"),
                "axis_idx": ab_base.get("axis_idx", _axis_of(base.direction)[0]),
            })
        except Exception:
            logger.debug("Could not apply as-built base profile", exc_info=True)
    if base is not None:
        bsg = base.sketch_geometry or {}
        ax_b, _ = _axis_of(base.direction)
        bspan = bsg.get("span")
        bpts = bsg.get("points")
        if bspan is not None:
            # base sketch-plane axis -> origin plane
            off[ax_b] = -float(bspan[0])
        if bpts:
            # base in-plane center -> origin
            u_ax, v_ax = UV_AXES[ax_b]
            BP = np.asarray(bpts, dtype=float)
            off[u_ax] = -0.5 * (float(BP[:, 0].min()) + float(BP[:, 0].max()))
            off[v_ax] = -0.5 * (float(BP[:, 1].min()) + float(BP[:, 1].max()))
    bbox_shifted = (bmin + off, np.asarray(bbox_mm[1], dtype=float) + off)

    if base is not None:
        _profile_extrusion(plan, base, "Extrude Base", "NEW", off)
    else:
        plan.skipped.append("no BASE_EXTRUSION in tree")

    n_join = 0
    for f in features:
        if f.type == FeatureType.BOSS and f.operation == "JOIN":
            n_join += 1
            _profile_extrusion(plan, f, f"Extrude Join {n_join}", "ADD", off)

    # Additional decomposed bodies (RackMount's side plates). Each is its own
    # prismatic chunk in the SAME coordinate frame as the base, so it rides
    # the same origin offset. Built with ADD (union) — like the joins — so the
    # whole part becomes one solid before the holes/fillets, letting blends
    # roll across body junctions and the perimeter of every plate.
    n_body = 0
    for eb in (as_built or {}).get("extra_bodies") or []:
        if not eb.get("outer"):
            continue
        n_body += 1
        ax_e = int(eb.get("axis_idx", 0))
        d_e = np.zeros(3, dtype=float)
        d_e[ax_e] = 1.0
        eb_feat = Feature(
            type=FeatureType.BASE_EXTRUSION,
            face_indices=np.asarray([], dtype=int),
            sketch_geometry={
                "points": eb["outer"], "inner": eb.get("inner") or [],
                "span": eb["span"], "axis_idx": ax_e,
            },
            direction=d_e,
            depth=abs(float(eb["span"][1]) - float(eb["span"][0])),
            operation="ADD",
        )
        _profile_extrusion(plan, eb_feat, f"Extrude Side Body {n_body}",
                           "ADD", off)

    n_hole = 0
    if as_built and as_built.get("holes"):
        for h in as_built["holes"]:
            n_hole += 1
            n_c = len(h["centers"])
            kind = "Through" if h.get("through") else "Blind"
            label = (f"{kind} Hole {n_hole}" if n_c == 1
                     else f"{kind} Hole Pattern {n_hole} ({n_c}x)")
            centers = (np.asarray(h["centers"], dtype=float) + off).tolist()
            _hole_cut_raw(
                plan, centers, float(h["radius"]), h["axis_dir"],
                bool(h.get("through")), float(h.get("depth", 0.0)),
                label, bbox_shifted,
            )
    else:
        for f in features:
            if f.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
                n_hole += 1
                sg = f.sketch_geometry or {}
                n_c = len(sg.get("centers") or []) or 1
                kind = ("Through" if f.type == FeatureType.THROUGH_HOLE
                        else "Blind")
                label = (f"{kind} Hole {n_hole}" if n_c == 1
                         else f"{kind} Hole Pattern {n_hole} ({n_c}x)")
                _hole_cut(plan, f, label, bbox_shifted, off)

    if as_built and as_built.get("blends"):
        # Emit in the exact order CadQuery applied them — each blend
        # trims neighboring edges, so the closest-point selections only
        # transfer if the regen sequence matches.
        n_f = n_c2 = 0
        for b in as_built["blends"]:
            kind = b.get("kind", "fillet")
            r = float(b["radius"])
            if kind == "chamfer":
                n_c2 += 1
                label = f"Chamfer {n_c2}"
            else:
                n_f += 1
                label = f"Fillet {n_f} (r={r:.0f}mm)"
            mids = (np.asarray(b.get("edge_midpoints") or [], dtype=float)
                    + off).tolist()
            if len(mids) > 1:
                # Outline ring: seed EVERY recorded edge AND propagate
                # tangentially. The outer edge of a plate is a loop of
                # straight sides and tangent corner arcs broken only at
                # the (few) sharp corners; seeding every side + tangent
                # propagation rolls the whole loop, where a single seed
                # stopped at the first sharp corner (user: "only the
                # long edge filleted"). Fallbacks: explicit set only,
                # then single-seed propagation.
                primary = _blend_feature(label, kind, r, mids,
                                         tangent_propagation=True)
                retries = [
                    _blend_feature(label, kind, r, mids),
                    _blend_feature(label, kind, r, mids[:1],
                                   tangent_propagation=True),
                ]
            else:
                primary = _blend_feature(label, kind, r, mids)
                retries = []
            plan.add(PlannedFeature(
                name=label, btm=primary, retry_btms=retries,
            ))
    else:
        for f in features:
            if f.type in (FeatureType.FILLET, FeatureType.CHAMFER):
                sg = f.sketch_geometry or {}
                r = sg.get("radius") or sg.get("width") or 0
                plan.skipped.append(
                    f"{f.type.value} r={r} (needs the CadQuery as-built "
                    f"record for edge queries)"
                )

    logger.info(
        "Onshape plan: %d features (%d skipped: %s)",
        len(plan.features), len(plan.skipped),
        "; ".join(plan.skipped) or "none",
    )
    return plan


def resolve_pending(plan: OnshapePlan, pf: PlannedFeature) -> dict:
    """Patch a planned feature's PENDING references using the feature
    ids Onshape assigned to its dependencies, returning the final BTM."""
    btm = pf.btm
    if pf.plane_dep is not None:
        dep_fid = plan.features[pf.plane_dep].feature_id
        for p in btm["parameters"]:
            if p.get("parameterId") == "sketchPlane":
                p["queries"][0]["queryString"] = (
                    f'query=qCreatedBy(makeId("{dep_fid}"), '
                    f'EntityType.FACE);'
                )
    if pf.sketch_dep is not None:
        dep_fid = plan.features[pf.sketch_dep].feature_id
        for p in btm["parameters"]:
            if p.get("parameterId") == "entities":
                for q in p["queries"]:
                    if q.get("btType") == "BTMIndividualSketchRegionQuery-140":
                        q["featureId"] = dep_fid
    return btm
