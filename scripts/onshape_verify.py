"""Verify a live doc: bbox/origin placement, base sketch plane, and the
through-hole sketch plane position (should be near the holes)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesh2param.onshape.client import OnshapeClient  # noqa: E402


def main() -> int:
    did = sys.argv[1]
    c = OnshapeClient.from_env()
    doc = c.get(f"/documents/{did}")
    wid = doc["defaultWorkspace"]["id"]
    eid = c.partstudio_id(did, wid)
    lo, hi = c.bounding_box_mm(did, wid, eid)
    print("bbox lo:", [round(x, 2) for x in lo])
    print("bbox hi:", [round(x, 2) for x in hi])
    # base sketch: which plane? find through-hole plane offset
    d = c.features(did, wid, eid)
    for f in d.get("features", []):
        nm = f.get("name", "")
        if nm == "Extrude Base sketch":
            sp = next((p for p in f["parameters"]
                       if p.get("parameterId") == "sketchPlane"), None)
            q = sp["queries"][0] if sp else {}
            print("base sketch plane:",
                  q.get("deterministicIds") or q.get("queryString", "?"))
        if nm.endswith("plane") and "Through Hole" in nm:
            off = next((p for p in f["parameters"]
                        if p.get("parameterId") == "offset"), {})
            opp = next((p for p in f["parameters"]
                        if p.get("parameterId") == "oppositeDirection"), {})
            print(f"through-hole plane: offset={off.get('expression')} "
                  f"opposite={opp.get('value')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
