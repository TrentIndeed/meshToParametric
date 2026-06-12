"""Check CQ's exact circle-fit recipe on the engine inner rings."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from mesh2param.config import IngestConfig
from mesh2param.pipeline.export_cadquery import _clean_engine_polygon
from mesh2param.pipeline.ingest import ingest_mesh
import match_fillets_to_edges as mfe


def main() -> int:
    cleaned = ingest_mesh(
        str(ROOT / "models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False),
    )
    _, body_results, _ = mfe.build_clusters(cleaned, single_body=True)
    _, _, tree = body_results[0]
    for f in tree.features:
        sg = f.sketch_geometry or {}
        for ri, ring in enumerate(sg.get("inner") or []):
            pts = _clean_engine_polygon(ring)
            arr = np.asarray(pts, dtype=float)
            A = np.c_[2 * arr[:, 0], 2 * arr[:, 1], np.ones(len(arr))]
            sol, *_ = np.linalg.lstsq(A, (arr**2).sum(axis=1), rcond=None)
            cx, cy = float(sol[0]), float(sol[1])
            cr = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
            resid = np.abs(np.hypot(arr[:, 0] - cx, arr[:, 1] - cy) - cr)
            tol = max(0.15, 0.02 * cr)
            verdict = ("CIRCLE" if cr > 0.5 and float(resid.max()) < tol
                       else "polygon")
            print(f"{f.type.value} ring{ri}: n_raw={len(ring)} "
                  f"n_clean={len(pts)} r={cr:.2f} "
                  f"resid_max={float(resid.max()):.3f} tol={tol:.3f} "
                  f"-> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
