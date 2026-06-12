"""Dump the BTM JSON of features in an existing Onshape document —
ground truth for the exact serialization the Features API expects."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onshape_scratch import BASE, _session  # noqa: E402


def main() -> int:
    s = _session()
    did = sys.argv[1] if len(sys.argv) > 1 else "f51a8cde4c8e603d116ef2de"
    want = sys.argv[2] if len(sys.argv) > 2 else None
    r = s.get(f"{BASE}/documents/{did}")
    doc = r.json()
    wid = doc["defaultWorkspace"]["id"]
    r = s.get(f"{BASE}/documents/d/{did}/w/{wid}/elements")
    for e in r.json():
        print(e["elementType"], e["id"], e["name"])
        if e["elementType"] != "PARTSTUDIO":
            continue
        rf = s.get(
            f"{BASE}/partstudios/d/{did}/w/{wid}/e/{e['id']}/features",
        )
        fd = rf.json()
        for f in fd.get("features", []):
            print("  feature:", f.get("featureType"), "|", f.get("name"),
                  "|", f.get("featureId"))
            if want and f.get("featureType") == want:
                print(json.dumps(f, indent=1)[:8000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
