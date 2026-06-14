"""Diagnose Onshape feature errors for Mesh2 (BottomMount).

Builds Mesh2 exactly like the server's multi-body path, live-exports it
to a REAL Onshape document via env API keys, then fetches the FINAL
feature states so we can see which features end up red (ERROR) after the
whole tree rebuilds — not just which were OK at add-time.

Usage:
    python scripts/onshape_diag_mesh2.py [document name]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
for noisy in ("trimesh", "matplotlib", "PIL", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from mesh2param.config import ExportConfig, IngestConfig
from mesh2param.onshape.client import OnshapeClient
from mesh2param.pipeline.export_cadquery import export_cadquery
from mesh2param.pipeline.export_onshape import export_to_onshape
from mesh2param.pipeline.ingest import ingest_mesh
import match_fillets_to_edges as mfe


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "mesh2param BottomMount DIAG"
    cleaned = ingest_mesh(
        str(ROOT / "models/Mesh2/BottomMount.3mf"), IngestConfig(remesh=False),
    )
    raw, body_results, decomp = mfe.build_clusters(cleaned)
    clusters = [{
        "group_id": cl["gid"], "vertices": cl["verts"],
        "radius": cl["radius"], "category": cl["category"],
        "kind": cl.get("kind", "fillet"), "n_records": 0,
    } for cl in raw]
    tree = body_results[0][2]
    print(f"multi-body={decomp.is_multi_body}  bodies={len(body_results)}  "
          f"tree(body0)={tree.num_features} features")

    cq_res = export_cadquery(
        tree, cleaned, use_llm=False, num_variants=1,
        body_results=body_results if decomp.is_multi_body else None,
        stitched_clusters=clusters,
    )
    ab = cq_res.as_built or {}
    print(f"as-built: {len(ab.get('holes', []))} hole group(s), "
          f"{len(ab.get('blends', []))} blend(s); "
          f"CadQuery vol err {cq_res.volume_error_pct:.1f}%")

    res = export_to_onshape(tree, cleaned, ExportConfig(
        dry_run=False, document_name=name,
    ), as_built=ab)
    print()
    print(f"status:    {res.status}")
    print(f"features:  {res.feature_count}/{res.expected_features} OK at add-time")
    print(f"volume:    {res.cad_volume:.0f} vs mesh {res.mesh_volume:.0f} "
          f"({res.volume_error_pct:.1f}%)")
    print(f"document:  {res.document_url}")

    # --- FINAL state: which features are red after the whole tree rebuilt ---
    client = OnshapeClient.from_env()
    data = client.features(res.document_id, res.workspace_id, res.element_id)
    states = data.get("featureStates", {}) or {}
    feats = data.get("features", []) or []
    print("\n── final feature states ──")
    bad = []
    for f in feats:
        fid = f.get("featureId", "")
        st = (states.get(fid) or {}).get("featureStatus", "?")
        nm = f.get("name", "?")
        flag = "" if st == "OK" else f"   <<< {st}"
        print(f"  [{st:<7}] {nm}{flag}")
        if st not in ("OK",):
            bad.append((nm, st, fid))
    print(f"\n{len(bad)} feature(s) NOT ok out of {len(feats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
