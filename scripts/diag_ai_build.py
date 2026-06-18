"""End-to-end build of AI parts through the NEW front-end, to see if they
actually convert to a valid CAD model (volume error, feature count).

Mirrors the server's single-body pipeline INCLUDING the scan/AI front-end:
ingest(scan_auto) -> segment(noise_robust) -> fit -> regularize -> re-seg/fit
-> topology -> recognize -> export_cadquery. Prints CAD volume error + feature
count per part. A low volume error + sane feature count = it converts.

Run unbuffered:  python -u scripts/diag_ai_build.py [AIn ...]
"""
from __future__ import annotations

import glob
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for n in ("trimesh", "matplotlib", "PIL", "urllib3", "pymeshlab", "cadquery"):
    logging.getLogger(n).setLevel(logging.ERROR)

import numpy as np  # noqa: E402

from mesh2param.config import (  # noqa: E402
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.export_cadquery import export_cadquery  # noqa: E402
from mesh2param.pipeline.fit import fit_surfaces  # noqa: E402
from mesh2param.pipeline.ingest import ingest_mesh  # noqa: E402
from mesh2param.pipeline.recognize import recognize_features  # noqa: E402
from mesh2param.pipeline.regularize import regularize_mesh  # noqa: E402
from mesh2param.pipeline.segment import segment_mesh  # noqa: E402
from mesh2param.pipeline.topology import build_topology  # noqa: E402


def build(part_dir: Path) -> None:
    f = glob.glob(str(part_dir / "*.3mf"))
    if not f:
        return
    nm = part_dir.name
    t = time.time()
    try:
        c = ingest_mesh(f[0], IngestConfig(remesh=False, scan_auto=True))
        dirty = bool(c.metadata.get("scan_mode"))
        seg = SegmentConfig(noise_robust=dirty)
        regions = segment_mesh(c, seg)
        fitted = fit_surfaces(regions, c, FitConfig())
        if dirty:
            regularize_mesh(c, regions, fitted, snap_angle_deg=12.0)
            regions = segment_mesh(c, seg)
            fitted = fit_surfaces(regions, c, FitConfig())
        graph, edges = build_topology(regions, fitted, c, TopologyConfig())
        tree = recognize_features(regions, fitted, graph, edges, c,
                                  RecognizeConfig())
        res = export_cadquery(tree, c, use_llm=False, num_variants=1)
        # Shape IoU (exact-boolean) vs the cleaned target — the primary quality
        # metric now (cad_vol_err is volume-only and shape-blind).
        iou = float("nan")
        try:
            import numpy as _np
            import trimesh as _tm
            from mesh2param.pipeline.iou_refine import mesh_iou_exact
            if res.tess_vertices is not None and res.tess_faces is not None:
                tgt = _tm.Trimesh(vertices=c.vertices, faces=c.faces,
                                  process=False)
                rec = _tm.Trimesh(
                    vertices=_np.asarray(res.tess_vertices, float).reshape(-1, 3),
                    faces=_np.asarray(res.tess_faces, _np.int64).reshape(-1, 3),
                    process=False)
                iou = mesh_iou_exact(tgt, rec)
        except Exception:
            pass
        print(f"{nm:<16} regions={len(regions):>4} feats={tree.num_features:>3} "
              f"IoU={iou:>5.3f}  cad_vol_err={res.volume_error_pct:>6.1f}% "
              f"({time.time() - t:.0f}s)", flush=True)
    except Exception as e:
        import traceback
        print(f"{nm:<16} ERR {str(e)[:80]}", flush=True)
        traceback.print_exc()


def main() -> int:
    which = sys.argv[1:] or ["AI3", "AI14", "AI1", "AI15", "AI11"]
    print(f"{'part':<16}{'result':>40}", flush=True)
    for w in which:
        d = ROOT / "models" / w
        if d.exists():
            build(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
