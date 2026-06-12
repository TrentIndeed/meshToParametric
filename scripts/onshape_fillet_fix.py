"""Probe: can the two failing ring fillets apply on the v5 doc with
tangent propagation (roll the whole tangent chain from the longest
edge), or with the full edge set + propagation?"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

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
    off = -np.asarray(cleaned.bbox[0], dtype=float)
    for i, b in enumerate(blends):
        print(f"blend {i}: {b['kind']} r={b['radius']} "
              f"n_mids={len(b['edge_midpoints'])}")

    c = OnshapeClient.from_env()
    doc = c.get(f"/documents/{did}")
    wid = doc["defaultWorkspace"]["id"]
    eid = c.partstudio_id(did, wid)

    # failing blends are index 5 and 7 (Fillet 6 / Fillet 8)
    for bi in (5, 7):
        b = blends[bi]
        mids = (np.asarray(b["edge_midpoints"], dtype=float) + off).tolist()
        for label, use_mids, prop in (
            (f"b{bi} all mids + tangentPropagation", mids, True),
            (f"b{bi} single mid + tangentPropagation", mids[:1], True),
        ):
            feat = _blend_feature(label, b["kind"], b["radius"], use_mids)
            feat["parameters"].append({
                "btType": "BTMParameterBoolean-144",
                "parameterId": "tangentPropagation",
                "value": prop,
            })
            r = c.post(
                f"/partstudios/d/{did}/w/{wid}/e/{eid}/features",
                json={"feature": feat},
            )
            f = r.get("feature") or {}
            fid = f.get("featureId")
            st = (r.get("featureState") or {}).get("featureStatus")
            print(f"{label}: {st}")
            if st != "OK" and fid:
                c.delete_feature(did, wid, eid, fid)
            elif st == "OK":
                print(f"  KEPT as {fid}")
                break
    vol = c.volume_mm3(did, wid, eid)
    print(f"volume now: {vol:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
