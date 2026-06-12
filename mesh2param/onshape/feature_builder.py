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


def _outline_entities(prefix: str, pts_mm: list) -> list[dict]:
    """Outer-profile entities with EXACT tangent corner arcs.

    Mesh-derived profiles arrive with corner radii sampled as chains of
    short kinked segments. CadQuery collapses them to sharp corners +
    radii (_sharp_polygon_with_fillets) and rebuilds exact tangent arcs
    with MakeFillet2d; drawing the sampled polygon here instead left
    the Onshape body with different edges near every corner, and the
    side plates' outline ring fillets failed on the mismatched chains.
    This computes the same tangent arcs analytically (for straight-line
    corners the construction is identical to MakeFillet2d): tangent
    length r/tan(phi/2) along each leg, center on the bisector at
    r/sin(phi/2).
    """
    pts = _clean_polygon(pts_mm)
    sharp = fillets = None
    try:
        from mesh2param.pipeline.export_cadquery import (
            _sharp_polygon_with_fillets,
        )
        sharp, fillets = _sharp_polygon_with_fillets(pts)
    except Exception:
        logger.debug("Sharp-corner reconstruction failed", exc_info=True)
    if not sharp or not fillets:
        return _polygon_entities(prefix, pts)

    P = np.asarray(sharp, dtype=float) / 1000.0  # meters
    n = len(P)
    # Per-vertex tangent trim distances (0 for sharp vertices).
    node_pts: list[tuple] = []  # (entry_pt, exit_pt, arc or None)
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
        cos_phi = float(np.clip(a @ b, -1.0, 1.0))
        phi = float(np.arccos(cos_phi))
        if phi < np.radians(10) or phi > np.radians(170):
            node_pts.append((V, V, None))
            continue
        t = r / np.tan(phi / 2.0)
        if t > 0.6 * min(la, lb):
            # radius too big for the legs — keep the corner sharp
            # rather than emit a self-intersecting outline
            node_pts.append((V, V, None))
            continue
        T1 = V + a * t
        T2 = V + b * t
        C = V + (a + b) / np.linalg.norm(a + b) * (r / np.sin(phi / 2.0))
        ang1 = float(np.arctan2(T1[1] - C[1], T1[0] - C[0]))
        ang2 = float(np.arctan2(T2[1] - C[1], T2[0] - C[0]))
        # minor arc from T1 to T2 (corner sweep is pi - phi < 170 deg):
        # express CCW, swapping ends if the CCW sweep is the long way.
        sweep = (ang2 - ang1) % (2.0 * np.pi)
        if sweep > np.pi:
            ang1, ang2 = ang2, ang1
        if ang2 < ang1:
            ang2 += 2.0 * np.pi
        node_pts.append((T1, T2, (C, r, ang1, ang2)))

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

    cp_idx = plan.add(PlannedFeature(
        name=f"{label} plane",
        btm=_cplane(f"{label} plane", ax, lo),
    ))
    entities = _outline_entities("outer", pts)
    for ri, ring in enumerate(sg.get("inner") or []):
        if len(ring) >= 3:
            ring_s = (np.asarray(ring, dtype=float) + off_uv).tolist()
            entities += _ring_entities(f"inner{ri}", ring_s)
    sk_idx = plan.add(PlannedFeature(
        name=f"{label} sketch",
        btm=_sketch(f"{label} sketch", _plane_param_created_by("PENDING"),
                    entities),
        plane_dep=cp_idx,
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
        # Sketch just outside the part on the LOW side, cut THROUGH_ALL
        # toward +axis: plane position can't affect the result as long
        # as it's outside the material.
        plane_pos = float(bbox_mm[0][ax]) - 2.0
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

    The whole model is TRANSLATED so its bbox-min corner lands on the
    origin — mesh coordinates are wherever the scanner/exporter put
    them, and a part floating 100 mm from the origin reads as broken
    in CAD (user request).
    """
    plan = OnshapePlan()
    features = list(feature_tree.features)
    off = -np.asarray(bbox_mm[0], dtype=float)
    size = np.asarray(bbox_mm[1], dtype=float) + off
    bbox_shifted = (np.zeros(3), size)

    base = next(
        (f for f in features if f.type == FeatureType.BASE_EXTRUSION), None,
    )
    if base is not None:
        _profile_extrusion(plan, base, "Extrude Base", "NEW", off)
    else:
        plan.skipped.append("no BASE_EXTRUSION in tree")

    n_join = 0
    for f in features:
        if f.type == FeatureType.BOSS and f.operation == "JOIN":
            n_join += 1
            _profile_extrusion(plan, f, f"Extrude Join {n_join}", "ADD", off)

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
            if kind == "chamfer":
                n_c2 += 1
                label = f"Chamfer {n_c2}"
            else:
                n_f += 1
                label = f"Fillet {n_f} (r={b['radius']:.0f}mm)"
            mids = (np.asarray(b.get("edge_midpoints") or [], dtype=float)
                    + off).tolist()
            retries = []
            if len(mids) > 1:
                retries.append(_blend_feature(
                    label, kind, float(b["radius"]), mids[:1],
                    tangent_propagation=True,
                ))
            plan.add(PlannedFeature(
                name=label,
                btm=_blend_feature(
                    label, kind, float(b["radius"]), mids,
                ),
                retry_btms=retries,
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
