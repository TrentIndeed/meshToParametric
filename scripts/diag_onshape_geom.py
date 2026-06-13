"""Dump the geometry the Onshape export sees: base/join axis+span,
as-built hole axes+positions, and corner-angle analysis of each engine
profile (to locate the 'slight angle near corner' the user reports)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mesh2param.config import IngestConfig
from mesh2param.models.feature import FeatureType
from mesh2param.pipeline.export_cadquery import (
    _clean_engine_polygon, _sharp_polygon_with_fillets,
)
from mesh2param.pipeline.ingest import ingest_mesh
import match_fillets_to_edges as mfe

AXN = "XYZ"


def corner_report(name, pts):
    P = np.asarray(_clean_engine_polygon(pts), dtype=float)
    n = len(P)
    seg = np.roll(P, -1, axis=0) - P
    L = np.linalg.norm(seg, axis=1)
    D = seg / (L[:, None] + 1e-12)
    Dp = np.roll(D, 1, axis=0)
    cosv = np.clip(np.einsum("ij,ij->i", Dp, D), -1.0, 1.0)
    turn = np.degrees(np.arccos(cosv))
    # axis-alignment of each segment (deg off nearest axis)
    ang = np.degrees(np.arctan2(np.abs(D[:, 1]), np.abs(D[:, 0])))
    off_axis = np.minimum(ang, 90 - ang)
    print(f"  {name}: n_clean={n}")
    # report segments that are nearly axis-aligned but a few deg off,
    # and short segments near corners
    for i in range(n):
        flag = ""
        if 0.5 < off_axis[i] < 20 and L[i] > 1.0:
            flag = " <-- OFF-AXIS"
        if L[i] < 0.15 * L.max():
            flag += " (short)"
        if flag:
            print(f"    seg{i}: len={L[i]:6.2f} off_axis={off_axis[i]:5.2f}deg "
                  f"turn_in={turn[i]:6.1f}deg{flag}")
    sharp, fillets = _sharp_polygon_with_fillets(pts)
    print(f"    sharp_polygon: "
          f"{'%d pts, %d fillets' % (len(sharp), len(fillets)) if sharp else 'None'}")


def main() -> int:
    cleaned = ingest_mesh(
        str(ROOT / "models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False),
    )
    print("mesh bbox lo:", [round(float(x), 2) for x in cleaned.bbox[0]])
    print("mesh bbox hi:", [round(float(x), 2) for x in cleaned.bbox[1]])
    raw, body_results, _ = mfe.build_clusters(cleaned, single_body=True)
    clusters = [{
        "group_id": cl["gid"], "vertices": cl["verts"],
        "radius": cl["radius"], "category": cl["category"],
        "kind": cl.get("kind", "fillet"), "n_records": 0,
    } for cl in raw]
    _, _, tree = body_results[0]

    print("\n=== profile features ===")
    for f in tree.features:
        sg = f.sketch_geometry or {}
        if f.type == FeatureType.BASE_EXTRUSION:
            ax = int(np.argmax(np.abs(f.direction)))
            print(f"\nBASE axis={AXN[ax]} span={sg.get('span')} "
                  f"dir={np.round(f.direction,2).tolist()}")
            if sg.get("points"):
                corner_report("base outer", sg["points"])
        elif f.type == FeatureType.BOSS and f.operation == "JOIN":
            ax = int(sg.get("axis_idx", 0))
            print(f"\nJOIN axis={AXN[ax]} span={sg.get('span')}")
            if sg.get("points"):
                corner_report("join outer", sg["points"])

    from mesh2param.pipeline.export_cadquery import export_cadquery
    res = export_cadquery(
        tree, cleaned, use_llm=False, stitched_clusters=clusters,
    )
    ab = res.as_built or {}
    print("\n=== as-built holes ===")
    for h in ab.get("holes", []):
        ad = np.asarray(h["axis_dir"], dtype=float)
        ax = int(np.argmax(np.abs(ad)))
        cs = np.asarray(h["centers"], dtype=float)
        print(f"  {len(cs)}x r={h['radius']:.1f} axis={AXN[ax]} "
              f"through={h['through']} depth={h['depth']:.1f}")
        print(f"    centers {AXN[ax]}: "
              f"{[round(float(c[ax]),1) for c in cs]}")
        print(f"    center {AXN[(ax+1)%3]}/{AXN[(ax+2)%3]} range: "
              f"{round(float(cs[:,(ax+1)%3].min()),1)}.."
              f"{round(float(cs[:,(ax+1)%3].max()),1)} / "
              f"{round(float(cs[:,(ax+2)%3].min()),1)}.."
              f"{round(float(cs[:,(ax+2)%3].max()),1)}")
    print("\n=== as-built blends (outer-ring chains) ===")
    for i, b in enumerate(ab.get("blends", [])):
        mids = np.asarray(b["edge_midpoints"], dtype=float)
        print(f"  blend{i}: {b['kind']} r={b['radius']} "
              f"n_edges={len(mids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
