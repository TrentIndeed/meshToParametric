"""Probe round 3:
a. sketch plane referenced by raw queryString qCreatedBy(makeId(fid))
   (the form human sketches serialize) -> unlock sketch-on-cPlane
b. filterInnerLoops=true on the sketch region query -> plate WITH hole
c. circle sketch + extrude REMOVE THROUGH_ALL -> hole cut path
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onshape_iterate import (  # noqa: E402
    BASE, DID, EID, WID, add, delete_feature, line,
    plane_q_deterministic, sketch,
)
from onshape_probe2 import bbox, circle, cplane, rect  # noqa: E402
from onshape_scratch import _session  # noqa: E402


def sketch_on_query_string(name, feature_id, entities):
    qs = f'query=qCreatedBy(makeId("{feature_id}"), EntityType.FACE);'
    return {
        "btType": "BTMSketch-151",
        "featureType": "newSketch",
        "name": name,
        "parameters": [{
            "btType": "BTMParameterQueryList-148",
            "parameterId": "sketchPlane",
            "queries": [{
                "btType": "BTMIndividualQuery-138",
                "queryString": qs,
            }],
        }],
        "entities": entities,
        "constraints": [],
    }


def extrude2(name, sketch_fid, op="NEW", end_bound="BLIND",
             depth_mm=None, filter_inner=False):
    params = [
        {"btType": "BTMParameterEnum-145", "parameterId": "bodyType",
         "value": "SOLID", "enumName": "ExtendedToolBodyType"},
        {"btType": "BTMParameterEnum-145", "parameterId": "operationType",
         "value": op, "enumName": "NewBodyOperationType"},
        {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
         "queries": [{
             "btType": "BTMIndividualSketchRegionQuery-140",
             "featureId": sketch_fid,
             "filterInnerLoops": filter_inner,
         }]},
        {"btType": "BTMParameterEnum-145", "parameterId": "endBound",
         "value": end_bound, "enumName": "BoundingType"},
    ]
    if depth_mm is not None:
        params.append({
            "btType": "BTMParameterQuantity-147", "parameterId": "depth",
            "expression": f"{depth_mm} mm",
        })
    return {
        "btType": "BTMFeature-134",
        "featureType": "extrude",
        "name": name,
        "parameters": params,
    }


def volume(s):
    r = s.get(
        f"{BASE}/partstudios/d/{DID}/w/{WID}/e/{EID}/massproperties",
        params={"massAsComputed": "true"},
    )
    if r.status_code != 200:
        return None
    try:
        return round(r.json()["bodies"]["-all-"]["volume"][0] * 1e9, 1)
    except Exception:
        return None


def main() -> int:
    s = _session()
    created = []

    # --- a. cPlane + queryString sketch ---
    cp, st = add(s, cplane("qp plane", "JDC", 20.0))
    created.append(cp)
    sk, st2 = add(s, sketch_on_query_string("qs sketch", cp,
                                            rect(0.040, 0.030)))
    print(f"cPlane {st}; sketch via queryString {st2}")
    if st2 == "OK":
        ex, st3 = add(s, extrude2("qs ex", sk, depth_mm=5.0))
        print(f"  extrude: {st3} bbox={bbox(s)} (expect z 20..25)")
        created.append(ex)
    created.append(sk)
    for fid in reversed([c for c in created if c]):
        delete_feature(s, fid)
    created.clear()

    # --- b. plate with hole via filterInnerLoops ---
    ents = rect(0.040, 0.030) + [circle("c1", 0.020, 0.015, 0.005)]
    sk, st = add(s, sketch("fil sketch", plane_q_deterministic("JDC"), ents))
    ex, st2 = add(s, extrude2("fil ex", sk, depth_mm=5.0, filter_inner=True))
    print(f"filterInnerLoops plate: {st}/{st2} volume={volume(s)} "
          f"(expect ~5607)")
    created += [ex, sk]
    for fid in reversed([c for c in created if c]):
        delete_feature(s, fid)
    created.clear()

    # --- c. base plate + separate circle sketch + REMOVE THROUGH_ALL ---
    sk, _ = add(s, sketch("base", plane_q_deterministic("JDC"),
                          rect(0.040, 0.030)))
    ex, _ = add(s, extrude2("base ex", sk, depth_mm=5.0))
    sk2, st = add(s, sketch("hole sketch", plane_q_deterministic("JDC"),
                            [circle("h1", 0.020, 0.015, 0.005)]))
    ex2, st2 = add(s, extrude2("hole cut", sk2, op="REMOVE",
                               end_bound="THROUGH_ALL"))
    print(f"hole cut: sketch {st}, remove THROUGH_ALL {st2} "
          f"volume={volume(s)} (expect ~5607)")
    for fid in reversed([f for f in (sk, ex, sk2, ex2) if f]):
        delete_feature(s, fid)
    print("cleaned up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
