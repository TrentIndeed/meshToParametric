"""Probe round 2: sign conventions, offset cPlane, circles.

1. Rect on each default plane, extrude, print bbox LO/HI (not just dims)
   -> recover u/v world mapping signs and extrude direction sign.
2. cPlane offset 20mm from Top, sketch on it via CreatedBy query,
   extrude -> does sketch-on-cPlane work, and where does the pad land?
3. Circle entity (hole-style) inside a rect region, extrude -> does the
   even-odd region give a plate with a hole (volume check)?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onshape_iterate import (  # noqa: E402
    BASE, DID, EID, WID, add, delete_feature, line,
    plane_q_deterministic, sketch,
)
from onshape_planes import extrude_feature  # noqa: E402
from onshape_scratch import _session  # noqa: E402


def bbox(s):
    r = s.get(f"{BASE}/partstudios/d/{DID}/w/{WID}/e/{EID}/boundingboxes")
    if r.status_code != 200:
        return None
    b = r.json()
    return ([round(b[k] * 1000, 1) for k in ("lowX", "lowY", "lowZ")],
            [round(b[k] * 1000, 1) for k in ("highX", "highY", "highZ")])


def rect(w, h):
    return [
        line("l1", 0, 0, 1, 0, 0.0, w),
        line("l2", w, 0, 0, 1, 0.0, h),
        line("l3", w, h, -1, 0, 0.0, w),
        line("l4", 0, h, 0, -1, 0.0, h),
    ]


def circle(eid_, cx, cy, r):
    return {
        "btType": "BTMSketchCurve-4",
        "entityId": eid_,
        "geometry": {
            "btType": "BTCurveGeometryCircle-115",
            "radius": r, "xCenter": cx, "yCenter": cy,
            "xDir": 1.0, "yDir": 0.0, "clockwise": False,
        },
    }


def cplane(name, base_det_id, offset_mm, opposite=False):
    return {
        "btType": "BTMFeature-134",
        "featureType": "cPlane",
        "name": name,
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{
                 "btType": "BTMIndividualQuery-138",
                 "deterministicIds": [base_det_id],
             }]},
            {"btType": "BTMParameterEnum-145", "parameterId": "cplaneType",
             "value": "OFFSET", "enumName": "CPlaneType"},
            {"btType": "BTMParameterQuantity-147", "parameterId": "offset",
             "expression": f"{offset_mm} mm"},
            {"btType": "BTMParameterBoolean-144",
             "parameterId": "oppositeDirection", "value": opposite},
        ],
    }


def sketch_on_created_by(name, feature_id, entities):
    return {
        "btType": "BTMSketch-151",
        "featureType": "newSketch",
        "name": name,
        "parameters": [{
            "btType": "BTMParameterQueryList-148",
            "parameterId": "sketchPlane",
            "queries": [{
                "btType": "BTMIndividualCreatedByQuery-137",
                "featureId": feature_id,
            }],
        }],
        "entities": entities,
        "constraints": [],
    }


def main() -> int:
    s = _session()
    created = []

    # --- 1. sign probes on each default plane ---
    for det, label in (("JDC", "Top"), ("JCC", "Front"), ("JEC", "Right")):
        sk, st = add(s, sketch(f"sign {label}", plane_q_deterministic(det),
                               rect(0.040, 0.030)))
        ex, st2 = add(s, extrude_feature(f"sign ex {label}", sk, 5.0))
        print(f"{label} ({det}): {st}/{st2} bbox={bbox(s)}")
        for fid in (ex, sk):
            if fid:
                delete_feature(s, fid)

    # --- 2. cPlane offset 20mm from Top, sketch via CreatedBy ---
    cp, st = add(s, cplane("offset plane", "JDC", 20.0))
    print(f"cPlane: {st} (id={cp})")
    if st == "OK":
        sk, st = add(s, sketch_on_created_by("on cplane", cp,
                                             rect(0.040, 0.030)))
        print(f"sketch on cPlane via CreatedBy: {st}")
        if st == "OK":
            ex, st2 = add(s, extrude_feature("cplane ex", sk, 5.0))
            print(f"extrude from cPlane sketch: {st2} bbox={bbox(s)}")
            if ex:
                created.append(ex)
        if sk:
            created.append(sk)
        created.append(cp)

    # --- 3. circle-in-rect (plate with hole) on Top ---
    ents = rect(0.040, 0.030) + [circle("c1", 0.020, 0.015, 0.005)]
    sk, st = add(s, sketch("plate with hole", plane_q_deterministic("JDC"),
                           ents))
    print(f"plate-with-hole sketch: {st}")
    if st == "OK":
        ex, st2 = add(s, extrude_feature("plate ex", sk, 5.0))
        r = s.get(
            f"{BASE}/partstudios/d/{DID}/w/{WID}/e/{EID}/massproperties",
            params={"massAsComputed": "true"},
        )
        vol = None
        if r.status_code == 200:
            try:
                vol = r.json()["bodies"]["-all-"]["volume"][0] * 1e9
            except Exception:
                vol = f"parse fail: {str(r.json())[:200]}"
        # expected: 40*30*5 - pi*25*5 = 6000 - 392.7 = 5607.3 mm^3
        print(f"plate-with-hole extrude: {st2} volume={vol} mm^3 "
              f"(expect ~5607 if hole cut, 6000 if filled)")
        for fid in (ex, sk):
            if fid:
                created.append(fid)

    for fid in reversed(created):
        delete_feature(s, fid)
    print("cleaned up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
