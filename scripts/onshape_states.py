"""Print per-feature states for a document (debug helper)."""
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
    d = c.features(did, wid, eid)
    states = d.get("featureStates") or {}
    for f in d.get("features", []):
        fid = f.get("featureId")
        st = (states.get(fid) or {}).get("featureStatus")
        print(f"{st:>7}  {f.get('featureType'):<10} {f.get('name')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
