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


def _cplane(name: str, axis_idx: int, position_mm: float) -> dict:
    """Construction plane at world position `position_mm` on the axis."""
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
        ],
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
                       op: str) -> bool:
    """Sketch+extrude for a base/join profile carried by the engine.

    sketch_geometry: {points: [[u,v]...] mm, span: [lo,hi] on the axis,
    inner: [rings], axis_idx (joins) — base derives axis from
    feature.direction}.
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
    lo, hi = float(span[0]), float(span[1])

    cp_idx = plan.add(PlannedFeature(
        name=f"{label} plane",
        btm=_cplane(f"{label} plane", ax, lo),
    ))
    entities = _polygon_entities("outer", pts)
    for ri, ring in enumerate(sg.get("inner") or []):
        if len(ring) >= 3:
            entities += _polygon_entities(f"inner{ri}", ring)
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
              bbox_mm: tuple) -> bool:
    sg = feat.sketch_geometry or {}
    radius = float(sg.get("radius", 0.0) or 0.0)
    centers = sg.get("centers") or (
        [sg.get("center")] if sg.get("center") is not None else []
    )
    centers = [c for c in centers if c is not None]
    if radius <= 0 or not centers:
        plan.skipped.append(f"{label}: no center/radius")
        return False
    ax, dir_sign = _axis_of(feat.direction)
    u_ax, v_ax = UV_AXES[ax]
    through = feat.type == FeatureType.THROUGH_HOLE

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
        depth = float(feat.depth)
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
                       bbox_mm: tuple) -> OnshapePlan:
    """Convert the recognized FeatureTree into an ordered Onshape plan.

    Mirrors the CadQuery timeline: base extrusion -> join extrusions ->
    hole cuts. Fillets/chamfers are not exported yet (they need edge
    queries against Onshape's own topology — phase 2); they're listed
    in plan.skipped so the caller can report them.
    """
    plan = OnshapePlan()
    features = list(feature_tree.features)

    base = next(
        (f for f in features if f.type == FeatureType.BASE_EXTRUSION), None,
    )
    if base is not None:
        _profile_extrusion(plan, base, "Extrude Base", "NEW")
    else:
        plan.skipped.append("no BASE_EXTRUSION in tree")

    n_join = 0
    for f in features:
        if f.type == FeatureType.BOSS and f.operation == "JOIN":
            n_join += 1
            _profile_extrusion(plan, f, f"Extrude Join {n_join}", "ADD")

    n_hole = 0
    for f in features:
        if f.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
            n_hole += 1
            sg = f.sketch_geometry or {}
            n_c = len(sg.get("centers") or []) or 1
            kind = ("Through" if f.type == FeatureType.THROUGH_HOLE
                    else "Blind")
            label = (f"{kind} Hole {n_hole}" if n_c == 1
                     else f"{kind} Hole Pattern {n_hole} ({n_c}x)")
            _hole_cut(plan, f, label, bbox_mm)

    for f in features:
        if f.type in (FeatureType.FILLET, FeatureType.CHAMFER):
            sg = f.sketch_geometry or {}
            r = sg.get("radius") or sg.get("width") or 0
            plan.skipped.append(
                f"{f.type.value} r={r} (edge-query export is phase 2)"
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
