"""Evaluate what edge a qClosestTo midpoint query actually selects in
an Onshape part studio: bounding box + curve length of the result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mesh2param.onshape.client import OnshapeClient  # noqa: E402


def main() -> int:
    did = sys.argv[1]
    pts = [
        (45.6304, 111.0453, -133.4204),
        (-43.6304, 139.2071, -128.5655),
        (39.6797, 111.0453, -133.4204),
        (-37.6797, 139.2071, -128.5655),
    ]
    c = OnshapeClient.from_env()
    doc = c.get(f"/documents/{did}")
    wid = doc["defaultWorkspace"]["id"]
    eid = c.partstudio_id(did, wid)
    for x, y, z in pts:
        script = (
            "function(context is Context, queries is map) {"
            f"var q = qClosestTo(qEverything(EntityType.EDGE), "
            f"vector({x}, {y}, {z}) * millimeter);"
            "var bb = evBox3d(context, {topology: q});"
            "var ln = evLength(context, {entities: q});"
            'return {"theBox": bb, "theLen": ln};'
            "}"
        )
        r = c.post(
            f"/partstudios/d/{did}/w/{wid}/e/{eid}/featurescript",
            json={"script": script, "queries": {}},
        )
        out = json.dumps(r.get("result") or r)
        # crude extraction of the numbers
        print(f"point ({x:.1f},{y:.1f},{z:.1f}):")
        print(" ", out[:1200])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
