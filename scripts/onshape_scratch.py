"""Scratch: prototype Onshape BTM feature JSON against the live API.

Creates a private scratch document, inspects the default features,
adds a sketch + extrude, evaluates the part volume, and prints the
document URL. Iterating here is how we learn the exact JSON the
Features API accepts before committing to a builder design.

Usage:
    python scripts/onshape_scratch.py [step]
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://cad.onshape.com/api/v6"


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _session() -> requests.Session:
    _load_env()
    ak = os.environ["ONSHAPE_ACCESS_KEY"]
    sk = os.environ["ONSHAPE_SECRET_KEY"]
    s = requests.Session()
    tok = base64.b64encode(f"{ak}:{sk}".encode()).decode()
    s.headers.update({
        "Authorization": f"Basic {tok}",
        "Accept": "application/json;charset=UTF-8; qs=0.09",
        "Content-Type": "application/json",
    })
    return s


def main() -> int:
    s = _session()

    # 1. create scratch document. Free-tier accounts can only create
    # PUBLIC documents (private creation returns 409) — try private
    # first, fall back to public.
    r = s.post(f"{BASE}/documents", json={
        "name": "mesh2param-scratch", "isPublic": False,
    })
    if r.status_code == 409:
        print("private creation refused (free tier?) -> creating public")
        r = s.post(f"{BASE}/documents", json={
            "name": "mesh2param-scratch", "isPublic": True,
        })
    r.raise_for_status()
    doc = r.json()
    did = doc["id"]
    wid = doc["defaultWorkspace"]["id"]
    print(f"doc {did} workspace {wid}")
    print(f"URL: https://cad.onshape.com/documents/{did}/w/{wid}")

    # 2. find the Part Studio element
    r = s.get(f"{BASE}/documents/d/{did}/w/{wid}/elements")
    r.raise_for_status()
    eid = next(e["id"] for e in r.json() if e["elementType"] == "PARTSTUDIO")
    print(f"partstudio {eid}")

    # 3. inspect default features
    r = s.get(f"{BASE}/partstudios/d/{did}/w/{wid}/e/{eid}/features")
    r.raise_for_status()
    feats = r.json()
    print("top-level keys:", sorted(feats.keys()))
    for df in feats.get("defaultFeatures", []):
        print("default feature:", df.get("message", {}).get("name"),
              "| featureId:", df.get("message", {}).get("featureId"),
              "| btType:", df.get("btType") or df.get("type"))

    # 4. sketch a 40x30 mm rectangle on the Top plane
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

    W, H = 0.040, 0.030  # meters
    sketch = {
        "btType": "BTMSketch-151",
        "featureType": "newSketch",
        "name": "m2p sketch test",
        "parameters": [
            {
                "btType": "BTMParameterQueryList-148",
                "parameterId": "sketchPlane",
                "queries": [{
                    "btType": "BTMIndividualCreatedByQuery-137",
                    "featureId": "Top",
                }],
            },
        ],
        "entities": [
            line("l1", 0, 0, 1, 0, 0.0, W),
            line("l2", W, 0, 0, 1, 0.0, H),
            line("l3", W, H, -1, 0, 0.0, W),
            line("l4", 0, H, 0, -1, 0.0, H),
        ],
        "constraints": [],
    }
    r = s.post(
        f"{BASE}/partstudios/d/{did}/w/{wid}/e/{eid}/features",
        json={"feature": sketch},
    )
    print("sketch add:", r.status_code)
    if r.status_code != 200:
        print(r.text[:3000])
        return 1
    sk_resp = r.json()
    sk_fid = sk_resp["feature"]["message"]["featureId"] \
        if "message" in sk_resp.get("feature", {}) else sk_resp["feature"]["featureId"]
    print("sketch featureId:", sk_fid)
    fes = sk_resp.get("featureState") or {}
    print("featureState:", json.dumps(fes)[:300])

    # 5. extrude the sketch region 10 mm
    extrude = {
        "btType": "BTMFeature-134",
        "featureType": "extrude",
        "name": "m2p extrude test",
        "parameters": [
            {
                "btType": "BTMParameterEnum-145",
                "parameterId": "bodyType",
                "value": "SOLID",
                "enumName": "ExtendedToolBodyType",
            },
            {
                "btType": "BTMParameterEnum-145",
                "parameterId": "operationType",
                "value": "NEW",
                "enumName": "NewBodyOperationType",
            },
            {
                "btType": "BTMParameterQueryList-148",
                "parameterId": "entities",
                "queries": [{
                    "btType": "BTMIndividualSketchRegionQuery-140",
                    "featureId": sk_fid,
                }],
            },
            {
                "btType": "BTMParameterEnum-145",
                "parameterId": "endBound",
                "value": "BLIND",
                "enumName": "BoundingType",
            },
            {
                "btType": "BTMParameterQuantity-147",
                "parameterId": "depth",
                "expression": "10 mm",
            },
        ],
    }
    r = s.post(
        f"{BASE}/partstudios/d/{did}/w/{wid}/e/{eid}/features",
        json={"feature": extrude},
    )
    print("extrude add:", r.status_code)
    if r.status_code != 200:
        print(r.text[:3000])
        return 1
    ex_resp = r.json()
    print("extrude featureState:", json.dumps(ex_resp.get("featureState") or {})[:300])

    # 6. evaluate volume via FeatureScript
    script = (
        "function(context is Context, queries) { "
        "return evVolume(context, {entities: qEverything(EntityType.BODY)}); }"
    )
    r = s.post(
        f"{BASE}/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript",
        json={"script": script, "queries": []},
    )
    print("fs eval:", r.status_code)
    print(json.dumps(r.json())[:600])
    print(f"\nINSPECT: https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
