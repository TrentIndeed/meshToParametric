"""Iterate BTM feature variants against the scratch document and report
each feature's resulting state. Bisects which part of the serialization
Onshape rejects (plane query vs sketch entities vs extrude params)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onshape_scratch import BASE, _session  # noqa: E402

DID = "4efab4096d64aa8eefee8011"
WID = "332334179c3252ce8c8bbaf5"
EID = "71a5087f959d546ba54f04ea"


def add(s, feature):
    r = s.post(
        f"{BASE}/partstudios/d/{DID}/w/{WID}/e/{EID}/features",
        json={"feature": feature},
    )
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:1500]}")
        return None, None
    d = r.json()
    f = d.get("feature") or {}
    fid = f.get("featureId") or (f.get("message") or {}).get("featureId")
    state = (d.get("featureState") or {}).get("featureStatus")
    return fid, state


def delete_feature(s, fid):
    r = s.delete(
        f"{BASE}/partstudios/d/{DID}/w/{WID}/e/{EID}/features/featureid/{fid}",
    )
    return r.status_code


def line(eid_, px, py, dx, dy, p0, p1):
    return {
        "btType": "BTMSketchCurveSegment-155",
        "entityId": eid_,
        "startPointId": f"{eid_}.start",
        "endPointId": f"{eid_}.end",
        "startParam": p0,
        "endParam": p1,
        "geometry": {
            "btType": "BTCurveGeometryLine-117",
            "pntX": px, "pntY": py, "dirX": dx, "dirY": dy,
        },
    }


def plane_q_deterministic(det_id):
    return {
        "btType": "BTMParameterQueryList-148",
        "parameterId": "sketchPlane",
        "queries": [{
            "btType": "BTMIndividualQuery-138",
            "deterministicIds": [det_id],
        }],
    }


def plane_q_created_by(feature_id):
    return {
        "btType": "BTMParameterQueryList-148",
        "parameterId": "sketchPlane",
        "queries": [{
            "btType": "BTMIndividualCreatedByQuery-137",
            "featureId": feature_id,
        }],
    }


def sketch(name, plane_param, entities):
    return {
        "btType": "BTMSketch-151",
        "featureType": "newSketch",
        "name": name,
        "parameters": [plane_param],
        "entities": entities,
        "constraints": [],
    }


def main() -> int:
    s = _session()
    W, H = 0.040, 0.030
    rect = [
        line("l1", 0, 0, 1, 0, 0.0, W),
        line("l2", W, 0, 0, 1, 0.0, H),
        line("l3", W, H, -1, 0, 0.0, W),
        line("l4", 0, H, 0, -1, 0.0, H),
    ]
    variants = [
        ("empty sketch, deterministic JCC (Front)",
         sketch("v1", plane_q_deterministic("JCC"), [])),
        ("empty sketch, createdBy Top",
         sketch("v2", plane_q_created_by("Top"), [])),
        ("rect sketch, deterministic JCC",
         sketch("v3", plane_q_deterministic("JCC"), rect)),
    ]
    for label, feat in variants:
        fid, state = add(s, feat)
        print(f"[{state}] {label} (id={fid})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
