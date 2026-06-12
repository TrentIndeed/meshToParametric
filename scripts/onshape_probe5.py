"""Probe round 5 (fresh scratch doc):
a. fillet feature with edges selected by qClosestTo midpoint queries
b. extrude startOffset parameters (would remove cPlane clutter)
c. cPlane "size" parameter (smaller construction planes)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onshape_scratch import BASE, _session  # noqa: E402


def add(s, did, wid, eid, feature):
    r = s.post(
        f"{BASE}/partstudios/d/{did}/w/{wid}/e/{eid}/features",
        json={"feature": feature},
    )
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:800]}")
        return None, None
    d = r.json()
    f = d.get("feature") or {}
    fid = f.get("featureId") or (f.get("message") or {}).get("featureId")
    return fid, (d.get("featureState") or {}).get("featureStatus")


def volume(s, did, wid, eid):
    r = s.get(
        f"{BASE}/partstudios/d/{did}/w/{wid}/e/{eid}/massproperties",
        params={"massAsComputed": "true"},
    )
    try:
        return round(r.json()["bodies"]["-all-"]["volume"][0] * 1e9, 1)
    except Exception:
        return None


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mesh2param.onshape.feature_builder import (
        _cplane, _extrude, _plane_param_default, _polygon_entities, _sketch,
    )

    s = _session()
    r = s.post(f"{BASE}/documents", json={
        "name": "mesh2param-scratch2", "isPublic": True,
    })
    doc = r.json()
    did, wid = doc["id"], doc["defaultWorkspace"]["id"]
    eid = next(e["id"] for e in s.get(
        f"{BASE}/documents/d/{did}/w/{wid}/elements").json()
        if e["elementType"] == "PARTSTUDIO")
    print(f"scratch2: https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}")

    # base box 40x30x10 on Top
    pts = [[0, 0], [40, 0], [40, 30], [0, 30]]
    sk, st = add(s, did, wid, eid, _sketch(
        "box sketch", _plane_param_default("JDC"),
        _polygon_entities("o", pts),
    ))
    ex = _extrude("box", sk, "NEW", "BLIND", 10.0, False, True)
    exf, st2 = add(s, did, wid, eid, ex)
    print(f"box: {st}/{st2} vol={volume(s, did, wid, eid)} (expect 12000)")

    # --- a. fillet the 4 vertical edges via qClosestTo midpoints ---
    mids = [(0, 0, 5), (40, 0, 5), (40, 30, 5), (0, 30, 5)]
    qparts = ", ".join(
        f"qClosestTo(qEverything(EntityType.EDGE), "
        f"vector({x}, {y}, {z}) * millimeter)"
        for x, y, z in mids
    )
    fillet = {
        "btType": "BTMFeature-134",
        "featureType": "fillet",
        "name": "probe fillet",
        "parameters": [
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{
                 "btType": "BTMIndividualQuery-138",
                 "queryString": f"query=qUnion([{qparts}]);",
             }]},
            {"btType": "BTMParameterQuantity-147", "parameterId": "radius",
             "expression": "3 mm"},
        ],
    }
    ff, st = add(s, did, wid, eid, fillet)
    # 4 vertical r=3 fillets remove 4*(9-9pi/4)*10 = 77.3 mm^3
    print(f"fillet: {st} vol={volume(s, did, wid, eid)} (expect ~11922.7)")

    # --- b. extrude with startOffset ---
    sk2, _ = add(s, did, wid, eid, _sketch(
        "off sketch", _plane_param_default("JDC"),
        _polygon_entities("o2", [[60, 0], [80, 0], [80, 20], [60, 20]]),
    ))
    ex2 = _extrude("off ex", sk2, "NEW", "BLIND", 5.0, False, True)
    ex2["parameters"] += [
        {"btType": "BTMParameterBoolean-144", "parameterId": "startOffset",
         "value": True},
        {"btType": "BTMParameterQuantity-147",
         "parameterId": "startOffsetDistance", "expression": "15 mm"},
    ]
    f2, st = add(s, did, wid, eid, ex2)
    r2 = s.get(f"{BASE}/partstudios/d/{did}/w/{wid}/e/{eid}/boundingboxes")
    b = r2.json()
    print(f"startOffset extrude: {st} bbox z hi={round(b['highZ']*1000,1)} "
          f"(expect 20 if offset works, 10 if param ignored)")

    # --- c. cPlane with size param ---
    cp = _cplane("sized plane", 2, 40.0)
    cp["parameters"].append({
        "btType": "BTMParameterQuantity-147", "parameterId": "size",
        "expression": "20 mm",
    })
    f3, st = add(s, did, wid, eid, cp)
    print(f"cPlane with size param: {st}")

    print("(scratch2 left for inspection; delete after)")
    print(f"DID={did}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
