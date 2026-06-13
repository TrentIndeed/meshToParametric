"""Onshape export: offline plan-builder tests + opt-in live smoke.

The BTM serialization itself was validated live (scripts/onshape_probe*)
— these tests lock the PLAN construction: feature ordering, operation
types, plane/sign conventions, unit conversion, and the deferred-
reference patching. They run with no network.

The live smoke test (sessioninfo round-trip) only runs when
ONSHAPE_LIVE_TESTS=1 and credentials are configured, so CI and
credential-less checkouts skip it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from mesh2param.models.feature import Feature, FeatureType
from mesh2param.models.tree import FeatureTree
from mesh2param.onshape.feature_builder import (
    PLANE_FOR_AXIS, build_onshape_plan, resolve_pending,
)

BBOX = (np.array([-10.0, -20.0, 0.0]), np.array([50.0, 80.0, 12.0]))


def _tree() -> FeatureTree:
    base = Feature(
        type=FeatureType.BASE_EXTRUSION,
        face_indices=np.array([], dtype=np.int64),
        sketch_geometry={
            "points": [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]],
            "span": [0.0, 12.0],
            "inner": [[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0],
                       [10.0, 20.0]]],
        },
        depth=12.0,
        direction=np.array([0.0, 0.0, 1.0]),
        operation="NEW",
    )
    join = Feature(
        type=FeatureType.BOSS,
        face_indices=np.array([], dtype=np.int64),
        sketch_geometry={
            "type": "polygon",
            "points": [[0.0, 0.0], [30.0, 0.0], [30.0, 60.0], [0.0, 60.0]],
            "span": [40.0, 47.0],
            "axis_idx": 1,
            "inner": [],
        },
        depth=7.0,
        direction=np.array([0.0, 1.0, 0.0]),
        operation="JOIN",
    )
    through = Feature(
        type=FeatureType.THROUGH_HOLE,
        face_indices=np.array([], dtype=np.int64),
        sketch_geometry={
            "type": "circle",
            "centers": [[10.0, 10.0, 6.0], [30.0, 10.0, 6.0]],
            "center": [10.0, 10.0, 6.0],
            "radius": 3.0,
        },
        depth=12.0,
        direction=np.array([0.0, 0.0, -1.0]),
        operation="REMOVE",
    )
    blind = Feature(
        type=FeatureType.BLIND_HOLE,
        face_indices=np.array([], dtype=np.int64),
        sketch_geometry={
            "type": "circle",
            "center": [20.0, 25.0, 9.0],
            "radius": 2.0,
        },
        depth=6.0,
        direction=np.array([0.0, 0.0, -1.0]),
        operation="REMOVE",
    )
    fillet = Feature(
        type=FeatureType.FILLET,
        face_indices=np.array([], dtype=np.int64),
        sketch_geometry={"type": "arc", "radius": 2.0},
        depth=0.0,
        operation="REMOVE",
    )
    return FeatureTree(features=[base, join, through, blind, fillet])


def test_plan_structure_and_order():
    plan = build_onshape_plan(_tree(), BBOX)
    names = [pf.name for pf in plan.features]
    # base on a DEFAULT origin plane (no cPlane) => sketch+extrude only;
    # join triplet; two hole triplets = 11 features
    assert len(plan.features) == 11, names
    assert names[0:2] == ["Extrude Base sketch", "Extrude Base"]
    assert names[2] == "Extrude Join 1 plane"
    assert any("Through Hole Pattern" in n and "(2x)" in n for n in names)
    assert any(n.startswith("Blind Hole") for n in names)
    # fillet deferred, reported
    assert any("FILLET" in s for s in plan.skipped)


def test_base_on_default_origin_plane():
    plan = build_onshape_plan(_tree(), BBOX)
    sk, ex = plan.features[0:2]
    # base axis Z, span lo 0 -> sketched directly on the Top plane (JDC),
    # NOT a construction plane; no plane dependency to patch
    assert sk.name == "Extrude Base sketch"
    assert sk.plane_dep is None
    plane_q = next(p for p in sk.btm["parameters"]
                   if p["parameterId"] == "sketchPlane")["queries"][0]
    assert plane_q["deterministicIds"] == [PLANE_FOR_AXIS[2][0]]
    # sketch is mm -> meters: first outer segment param length 0.040 m
    line0 = sk.btm["entities"][0]
    assert line0["geometry"]["btType"] == "BTCurveGeometryLine-117"
    assert abs(line0["endParam"] - 0.040) < 1e-9
    # inner ring entities present (window baked into the profile)
    assert any(e["entityId"].startswith("inner0") for e in sk.btm["entities"])
    p = {x["parameterId"]: x for x in ex.btm["parameters"]}
    assert p["operationType"]["value"] == "NEW"
    assert p["endBound"]["value"] == "BLIND"
    assert p["depth"]["expression"].startswith("12.")
    assert p["oppositeDirection"]["value"] is False
    # inner loops must stay holes when extruding
    assert p["entities"]["queries"][0]["filterInnerLoops"] is True


def test_join_y_axis_sign_conventions():
    plan = build_onshape_plan(_tree(), BBOX)
    cp = next(pf for pf in plan.features if pf.name == "Extrude Join 1 plane")
    ex = next(pf for pf in plan.features if pf.name == "Extrude Join 1")
    pcp = {x["parameterId"]: x for x in cp.btm["parameters"]}
    # base is centered on origin: base (u,v) center is (20,15) so the Y
    # shift is -15. Join span lo 40 -> world y=25 on the Front plane
    # (normal -Y): positive offset with oppositeDirection=True (negative
    # offset expressions are rejected by the feature spec)
    assert pcp["offset"]["expression"].startswith("25.")
    assert pcp["oppositeDirection"]["value"] is True
    pex = {x["parameterId"]: x for x in ex.btm["parameters"]}
    # extruding toward +Y from a Front-derived plane needs the flip
    assert pex["oppositeDirection"]["value"] is True
    assert pex["operationType"]["value"] == "ADD"


def test_through_hole_outside_part_and_blind_opening():
    plan = build_onshape_plan(_tree(), BBOX)
    th_cp = next(pf for pf in plan.features
                 if "Through Hole" in pf.name and pf.name.endswith("plane"))
    p = {x["parameterId"]: x for x in th_cp.btm["parameters"]}
    # plane sits below bbox z lo (0.0) minus margin
    assert p["offset"]["expression"].startswith("2.")
    assert p["oppositeDirection"]["value"] is True  # z = -2 on Top plane
    bl_cp = next(pf for pf in plan.features
                 if pf.name.startswith("Blind Hole")
                 and pf.name.endswith("plane"))
    p = {x["parameterId"]: x for x in bl_cp.btm["parameters"]}
    # center z=9, depth 6, cut direction -Z -> opening at z=12
    assert p["offset"]["expression"].startswith("12.")
    bl_ex = next(pf for pf in plan.features
                 if pf.name.startswith("Blind Hole")
                 and not pf.name.endswith(("plane", "sketch")))
    pex = {x["parameterId"]: x for x in bl_ex.btm["parameters"]}
    assert pex["endBound"]["value"] == "BLIND"
    assert pex["depth"]["expression"].startswith("6.")
    # cut toward -Z from a Top-derived plane needs the flip
    assert pex["oppositeDirection"]["value"] is True


def test_as_built_overrides_tree_holes_and_adds_blends():
    as_built = {
        "holes": [{
            "centers": [[5.0, 5.0, 6.0], [15.0, 5.0, 6.0],
                        [25.0, 5.0, 6.0], [35.0, 5.0, 6.0]],
            "radius": 4.0,
            "axis_dir": [0.0, 0.0, -1.0],
            "through": True,
            "depth": 12.0,
        }],
        "blends": [
            {"kind": "fillet", "radius": 2.0,
             "edge_midpoints": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
            {"kind": "chamfer", "radius": 1.0,
             "edge_midpoints": [[7.0, 8.0, 9.0]]},
        ],
    }
    plan = build_onshape_plan(_tree(), BBOX, as_built=as_built)
    names = [pf.name for pf in plan.features]
    # tree's 2 hole features replaced by the single as-built 4x group
    assert sum(1 for n in names if "Hole" in n and not n.endswith(
        ("plane", "sketch"))) == 1
    assert any("(4x)" in n for n in names)
    # blends emitted in order, as real features (not skipped)
    fillet_pf = next(pf for pf in plan.features if "Fillet" in pf.name)
    qs = next(
        p for p in fillet_pf.btm["parameters"]
        if p["parameterId"] == "entities"
    )["queries"][0]["queryString"]
    assert qs.count("qClosestTo") == 2
    # midpoints carry the base-centering shift: base (u,v) center (20,15)
    # -> off (-20,-15,0); midpoint (1,2,3) -> (-19,-13,3)
    assert "vector(-19.0000, -13.0000, 3.0000)" in qs
    cham_pf = next(pf for pf in plan.features if "Chamfer" in pf.name)
    assert cham_pf.btm["featureType"] == "chamfer"
    assert not any("FILLET" in s for s in plan.skipped)


def test_ring_circle_recognition():
    from mesh2param.onshape.feature_builder import _ring_entities
    # 60-gon circle ring -> single CIRCLE entity
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ring = [[20 + 14.5 * np.cos(a), 30 + 14.5 * np.sin(a)] for a in ang]
    ents = _ring_entities("r", ring)
    assert len(ents) == 1
    g = ents[0]["geometry"]
    assert g["btType"] == "BTCurveGeometryCircle-115"
    assert abs(g["radius"] - 0.0145) < 1e-4
    # rectangle (4 corners lie exactly on their circumcircle!) must NOT
    # become a circle
    rect = [[0, 0], [40, 0], [40, 30], [0, 30]]
    ents = _ring_entities("r2", rect)
    assert all(
        e["geometry"]["btType"] == "BTCurveGeometryLine-117" for e in ents
    )


def test_resolve_pending_patches_references():
    plan = build_onshape_plan(_tree(), BBOX)
    # the JOIN keeps a construction plane (it's away from the origin) —
    # use its plane->sketch->extrude chain to exercise the patching
    cp = next(pf for pf in plan.features if pf.name == "Extrude Join 1 plane")
    sk = next(pf for pf in plan.features if pf.name == "Extrude Join 1 sketch")
    ex = next(pf for pf in plan.features if pf.name == "Extrude Join 1")
    cp.feature_id = "FCPLANE"
    btm = resolve_pending(plan, sk)
    qs = next(p for p in btm["parameters"]
              if p["parameterId"] == "sketchPlane")["queries"][0]["queryString"]
    assert 'makeId("FCPLANE")' in qs
    sk.feature_id = "FSKETCH"
    btm = resolve_pending(plan, ex)
    q = next(p for p in btm["parameters"]
             if p["parameterId"] == "entities")["queries"][0]
    assert q["featureId"] == "FSKETCH"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("ONSHAPE_LIVE_TESTS") != "1",
    reason="live Onshape test (set ONSHAPE_LIVE_TESTS=1)",
)
def test_live_sessioninfo():
    from mesh2param.onshape.client import OnshapeClient
    client = OnshapeClient.from_env()
    assert client is not None, "ONSHAPE_ACCESS_KEY/SECRET_KEY not configured"
    info = client.sessioninfo()
    assert info.get("id")
