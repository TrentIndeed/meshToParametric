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
from mesh2param.pipeline.export_onshape import export_to_onshape
from mesh2param.pipeline.ingest import ingest_mesh
import match_fillets_to_edges as mfe


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "mesh2param RackMount"
    cleaned = ingest_mesh(
        str(ROOT / "models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False),
    )
    _, body_results, _ = mfe.build_clusters(cleaned, single_body=True)
    _, _, tree = body_results[0]
    print(f"tree: {tree.num_features} features")
    res = export_to_onshape(tree, cleaned, ExportConfig(
        dry_run=False, document_name=name,
    ))
    print()
    print(f"status:    {res.status}")
    print(f"features:  {res.feature_count}/{res.expected_features}")
    print(f"volume:    {res.cad_volume:.0f} vs mesh {res.mesh_volume:.0f} "
          f"({res.volume_error_pct:.1f}%)")
    print(f"document:  {res.document_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
