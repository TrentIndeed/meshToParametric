"""End-to-end check: 12 stitched clusters -> production matcher/apply.

Prints the per-cluster match + apply outcome with logging enabled, to
verify both outline rings of EACH side plate get their own edge chain
(intent coverage) and how many edges each applies.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
for noisy in ("trimesh", "matplotlib", "PIL"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from mesh2param.config import IngestConfig
from mesh2param.pipeline.export_cadquery import export_cadquery
from mesh2param.pipeline.ingest import ingest_mesh
import match_fillets_to_edges as mfe


def main() -> int:
    cleaned = ingest_mesh(
        str(ROOT / "models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False),
    )
    raw, body_results, _ = mfe.build_clusters(cleaned, single_body=True)
    clusters = [{
        "group_id": cl["gid"], "vertices": cl["verts"],
        "radius": cl["radius"], "category": cl["category"],
        "kind": cl.get("kind", "fillet"), "n_records": 0,
    } for cl in raw]
    print(f"{len(clusters)} clusters in")
    _, _, tree = body_results[0]
    res = export_cadquery(
        tree, cleaned, use_llm=False, stitched_clusters=clusters,
    )
    print(f"\nstatus={res.status} vol_err={res.volume_error_pct:.1f}%")
    n_fillet_steps = sum(
        1 for s in res.construction_steps if s.name.startswith(("Fillet", "Chamfer"))
    )
    print(f"fillet/chamfer steps applied: {n_fillet_steps}")
    for d in (getattr(res, "fillet_match_diag", None) or []):
        print(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
