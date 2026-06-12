"""Identify the deterministic IDs of the three default planes by
extruding a thin pad from each candidate id and reading the part
bounding box (the thin axis = plane normal). Front is known: JCC."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onshape_iterate import (  # noqa: E402
    BASE, DID, EID, WID, add, delete_feature, line,
    plane_q_deterministic, sketch,
)
from onshape_scratch import _session  # noqa: E402


def extrude_feature(name, sketch_fid, depth_mm, op="NEW"):
    return {
        "btType": "BTMFeature-134",
        "featureType": "extrude",
        "name": name,
        "parameters": [
            {"btType": "BTMParameterEnum-145", "parameterId": "bodyType",
             "value": "SOLID", "enumName": "ExtendedToolBodyType"},
            {"btType": "BTMParameterEnum-145", "parameterId": "operationType",
             "value": op, "enumName": "NewBodyOperationType"},
            {"btType": "BTMParameterQueryList-148", "parameterId": "entities",
             "queries": [{
                 "btType": "BTMIndividualSketchRegionQuery-140",
                 "featureId": sketch_fid,
             }]},
            {"btType": "BTMParameterEnum-145", "parameterId": "endBound",
             "value": "BLIND", "enumName": "BoundingType"},
            {"btType": "BTMParameterQuantity-147", "parameterId": "depth",
             "expression": f"{depth_mm} mm"},
        ],
    }


def main() -> int:
    s = _session()
    rect = [
        line("l1", 0, 0, 1, 0, 0.0, 0.040),
        line("l2", 0.040, 0, 0, 1, 0.0, 0.030),
        line("l3", 0.040, 0.030, -1, 0, 0.0, 0.040),
        line("l4", 0, 0.030, 0, -1, 0.0, 0.030),
    ]
    for cand in ("JBC", "JCC", "JDC", "JEC", "JFC"):
        sk_fid, sk_state = add(s, sketch(
            f"plane probe {cand}", plane_q_deterministic(cand), rect,
        ))
        if sk_state != "OK":
            print(f"{cand}: sketch {sk_state}; skipping")
            if sk_fid:
                delete_feature(s, sk_fid)
            continue
        ex_fid, ex_state = add(s, extrude_feature(
            f"probe extrude {cand}", sk_fid, 5.0,
        ))
        bbox = None
        if ex_state == "OK":
            r = s.get(
                f"{BASE}/partstudios/d/{DID}/w/{WID}/e/{EID}/boundingboxes",
            )
            if r.status_code == 200:
                b = r.json()
                lo = [b["lowX"], b["lowY"], b["lowZ"]]
                hi = [b["highX"], b["highY"], b["highZ"]]
                dims = [round((h - l) * 1000, 1) for l, h in zip(lo, hi)]
                bbox = dims
        print(f"{cand}: sketch OK, extrude {ex_state}, bbox dims mm = {bbox}")
        # clean up so the next probe's bbox is unambiguous
        if ex_fid:
            delete_feature(s, ex_fid)
        if sk_fid:
            delete_feature(s, sk_fid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
