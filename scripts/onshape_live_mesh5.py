"""End-to-end live Onshape export of Mesh5 (RackMount).

Builds the recognized feature tree exactly like the server's
single-body path, then exports it to a REAL Onshape document and
prints the URL + validation numbers.

Usage:
    python scripts/onshape_live_mesh5.py [document name]
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
from mesh2param.pipeline.export_cadquery import export_cadquery
from mesh2param.pipeline.export_onshape import export_to_onshape
from mesh2param.pipeline.ingest import ingest_mesh
import match_fillets_to_edges as mfe


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "mesh2param RackMount"
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
    print(f"tree: {tree.num_features} features")
    # CadQuery rebuild FIRST: its as-built record (post-recovery holes +
    # actually-applied blends with edge midpoints) is what the Onshape
    # export replays — the raw tree is missing recovered holes.
    cq_res = export_cadquery(
        tree, cleaned, use_llm=False, stitched_clusters=clusters,
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
    print(f"features:  {res.feature_count}/{res.expected_features}")
    print(f"volume:    {res.cad_volume:.0f} vs mesh {res.mesh_volume:.0f} "
          f"({res.volume_error_pct:.1f}%)")
    print(f"document:  {res.document_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
