"""Debug: re-add the first failing rim fillet to the v3 doc and dump
the FULL feature-add response (looking for error notices)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mesh2param.config import IngestConfig
from mesh2param.pipeline.export_cadquery import export_cadquery
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.onshape.client import OnshapeClient
from mesh2param.onshape.feature_builder import _blend_feature
import match_fillets_to_edges as mfe


def main() -> int:
    did = sys.argv[1]
    cleaned = ingest_mesh(
        str(ROOT / "models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False),
    )
    raw, body_results, _ = mfe.build_clusters(cleaned, single_body=True)
    clusters = [{
        "group_id": cl["gid"], "vertices": cl["verts"],
        "radius": cl["radius"], "category": cl["category"],
        "kind": cl.get("kind", "fillet"), "n_records": 0,
    } for cl in raw]
    _, _, tree = body_results[0]
    cq_res = export_cadquery(
        tree, cleaned, use_llm=False, stitched_clusters=clusters,
    )
    blends = (cq_res.as_built or {}).get("blends") or []
    print(f"{len(blends)} blends; first 4:")
    for b in blends[:4]:
        print(" ", b["kind"], b["radius"], b["edge_midpoints"])

    c = OnshapeClient.from_env()
    doc = c.get(f"/documents/{did}")
    wid = doc["defaultWorkspace"]["id"]
    eid = c.partstudio_id(did, wid)
    b = blends[0]
    feat = _blend_feature("debug fillet", b["kind"], b["radius"],
                          b["edge_midpoints"])
    r = c.post(
        f"/partstudios/d/{did}/w/{wid}/e/{eid}/features",
        json={"feature": feat},
    )
    print(json.dumps(r, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
