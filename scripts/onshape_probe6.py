"""Probe round 6: sketch ARC entity serialization.

Draw a 20x20 mm square with ONE r=5 rounded corner (lines + one arc),
extrude 5 mm, check the volume: 394.63 * 5 = 1973.2 mm^3 if the arc
parameterization (startParam/endParam angles vs clockwise flag) is
right; 2000 or ERROR otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onshape_probe5 import add, volume  # noqa: E402
from onshape_scratch import BASE, _session  # noqa: E402
from mesh2param.onshape.feature_builder import (  # noqa: E402
    _extrude, _line, _plane_param_default, _sketch,
)


def arc_entity(eid, cx, cy, r, a0, a1, clockwise):
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
            "xDir": 1.0, "yDir": 0.0, "clockwise": clockwise,
        },
    }


def main() -> int:
    s = _session()
    r = s.post(f"{BASE}/documents", json={
        "name": "mesh2param-scratch3", "isPublic": True,
    })
    doc = r.json()
    did, wid = doc["id"], doc["defaultWorkspace"]["id"]
    eid = next(e["id"] for e in s.get(
        f"{BASE}/documents/d/{did}/w/{wid}/elements").json()
        if e["elementType"] == "PARTSTUDIO")
    print(f"scratch3: https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}")

    W = 0.020
    R = 0.005
    # square corners: (0,0) (W,0) (W,W) (0,W); round the (W,W) corner.
    # tangent points: (W, W-R) and (W-R, W); arc center (W-R, W-R).
    # ccw outline: angle of T1 from center = 0 rad; T2 = pi/2 rad.
    variants = [
        ("ccw params 0->pi/2, clockwise=False", 0.0, np.pi / 2, False),
        ("cw flag, params 0->pi/2", 0.0, np.pi / 2, True),
    ]
    for label, a0, a1, cwise in variants:
        ents = [
            _line("l1", 0, 0, 1, 0, 0.0, W),
            _line("l2", W, 0, 0, 1, 0.0, W - R),
            arc_entity("a1", W - R, W - R, R, a0, a1, cwise),
            _line("l3", W - R, W, -1, 0, 0.0, W - R),
            _line("l4", 0, W, 0, -1, 0.0, W),
        ]
        sk, st = add(s, did, wid, eid, _sketch(
            f"arc sketch {label}", _plane_param_default("JDC"), ents,
        ))
        ex, st2 = add(s, did, wid, eid, _extrude(
            f"arc ex {label}", sk, "NEW", "BLIND", 5.0, False, True,
        ))
        vol = volume(s, did, wid, eid)
        print(f"{label}: sketch {st}, extrude {st2}, vol={vol} "
              f"(expect 1973.2)")
        # remove for next variant
        for fid in (ex, sk):
            if fid:
                s.delete(f"{BASE}/partstudios/d/{did}/w/{wid}/e/{eid}"
                         f"/features/featureid/{fid}")
    print(f"DID={did} (delete when done)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
