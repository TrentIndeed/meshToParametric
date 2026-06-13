"""Offline check of the profile regularizer on RackMount profiles:
corner count, radii, corner angles (want ~90 deg), and reconstructed
area/bbox vs the original (want close)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mesh2param.config import IngestConfig
from mesh2param.models.feature import FeatureType
from mesh2param.onshape import feature_builder as fb
from mesh2param.pipeline.ingest import ingest_mesh
import match_fillets_to_edges as mfe


def poly_area(pts):
    P = np.asarray(pts, dtype=float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def check(name, pts):
    print(f"\n{name}: {len(pts)} raw pts, area={poly_area(pts):.0f} "
          f"bbox={np.ptp(np.asarray(pts), axis=0).round(1).tolist()}")
    sharp, fillets = fb._regularize_profile(pts)
    if not sharp:
        print("  regularize -> None (would fall back)")
        ents = fb._outline_entities("t", pts)
        kinds = {}
        for e in ents:
            g = e.get("geometry", {}).get("btType", "line")
            kinds[g] = kinds.get(g, 0) + 1
        print(f"  outline path -> {len(ents)} entities {kinds}")
        return
    P = np.asarray(sharp, dtype=float)
    n = len(P)
    angs = []
    for i in range(n):
        a = P[(i - 1) % n] - P[i]
        b = P[(i + 1) % n] - P[i]
        a /= np.linalg.norm(a) + 1e-12
        b /= np.linalg.norm(b) + 1e-12
        angs.append(float(np.degrees(np.arccos(np.clip(a @ b, -1, 1)))))
    print(f"  regularize -> {n} sharp corners, "
          f"radii={sorted(set(round(r,1) for r in fillets.values()))}")
    print(f"  corner angles: {[round(a) for a in angs]}")
    print(f"  sharp-polygon area={poly_area(sharp):.0f} "
          f"bbox={np.ptp(P, axis=0).round(1).tolist()}")
    ents = fb._outline_entities("t", pts)
    kinds = {}
    for e in ents:
        g = e.get("geometry", {}).get("btType", "line")
        kinds[g] = kinds.get(g, 0) + 1
    print(f"  outline entities: {len(ents)} {kinds}")


def main() -> int:
    cleaned = ingest_mesh(
        str(ROOT / "models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False),
    )
    _, body_results, _ = mfe.build_clusters(cleaned, single_body=True)
    _, _, tree = body_results[0]
    for f in tree.features:
        sg = f.sketch_geometry or {}
        if f.type == FeatureType.BASE_EXTRUSION and sg.get("points"):
            check("BASE", sg["points"])
        elif (f.type == FeatureType.BOSS and f.operation == "JOIN"
              and sg.get("points")):
            check("JOIN", sg["points"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
