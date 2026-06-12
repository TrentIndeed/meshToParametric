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
    # base triplet, join triplet, two hole triplets = 12 features
    assert len(plan.features) == 12, names
    assert names[0:3] == ["Extrude Base plane", "Extrude Base sketch",
                          "Extrude Base"]
    assert names[3] == "Extrude Join 1 plane"
    assert any("Through Hole Pattern" in n and "(2x)" in n for n in names)
    assert any(n.startswith("Blind Hole") for n in names)
    # fillet deferred, reported
    assert any("FILLET" in s for s in plan.skipped)


def test_base_btm_conventions():
    plan = build_onshape_plan(_tree(), BBOX)
    cp, sk, ex = plan.features[0:3]
    # base axis is Z -> Top plane (JDC), extrude +Z needs no flip
    det = cp.btm["parameters"][0]["queries"][0]["deterministicIds"]
    assert det == [PLANE_FOR_AXIS[2][0]]
    # sketch is mm -> meters: outer ring first point at origin, second
    # segment param length 0.040 m
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
    # world y=+40 on the Front plane (normal -Y): positive offset with
    # oppositeDirection=True (negative offset expressions are rejected
    # by the feature spec)
    assert pcp["offset"]["expression"].startswith("40.")
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


def test_resolve_pending_patches_references():
    plan = build_onshape_plan(_tree(), BBOX)
    cp, sk, ex = plan.features[0:3]
    cp.feature_id = "FCPLANE"
    btm = resolve_pending(plan, sk)
    qs = btm["parameters"][0]["queries"][0]["queryString"]
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
