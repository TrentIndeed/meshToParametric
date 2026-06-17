"""Phase 5 A/B: does segmentation earn its keep on BUILD QUALITY?

For each part, reconstruct two ways from the SAME ingested mesh and compare
end-to-end CAD volume error:

  Path A (recognition): segment -> fit -> (regularize) -> topology -> recognize
                        -> export_cadquery        [segmentation IS load-bearing]
  Path B (contour):     plan_extrusions -> build extrudes -> residual refine
                        (reconstruct_segmentation_free)   [NO segmentation]

Both paths get the residual loop, so the delta isolates segmentation's
contribution. Decision gate (docs/CADFIT_PORTING_PLAN.md Phase 5):
  - B >= A on AI/Scan, A wins clean-CAD editability  -> route by input
  - B >= A everywhere                                 -> replace segmentation
  - A wins broadly                                    -> keep segmentation

Run:  python -u scripts/diag_seg_ab.py [Mesh1 AI1 ...]
"""
from __future__ import annotations

import glob
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.CRITICAL)

from mesh2param.config import (  # noqa: E402
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.contour_recon import reconstruct_segmentation_free  # noqa: E402
from mesh2param.pipeline.export_cadquery import export_cadquery  # noqa: E402
from mesh2param.pipeline.fit import fit_surfaces  # noqa: E402
from mesh2param.pipeline.ingest import ingest_mesh  # noqa: E402
from mesh2param.pipeline.recognize import recognize_features  # noqa: E402
from mesh2param.pipeline.regularize import regularize_mesh  # noqa: E402
from mesh2param.pipeline.segment import segment_mesh  # noqa: E402
from mesh2param.pipeline.topology import build_topology  # noqa: E402


def _find(part: str):
    for ext in ("*.3mf", "*.stl", "*.obj", "*.ply"):
        f = glob.glob(str(ROOT / "models" / part / ext))
        if f:
            return f[0]
    return None


def _path_a(cleaned):
    """Full recognition path -> cad_vol_err."""
    dirty = bool(cleaned.metadata.get("scan_mode"))
    seg = SegmentConfig(noise_robust=dirty)
    regions = segment_mesh(cleaned, seg)
    fitted = fit_surfaces(regions, cleaned, FitConfig())
    if dirty:
        try:
            regularize_mesh(cleaned, regions, fitted, snap_angle_deg=12.0)
            regions = segment_mesh(cleaned, seg)
            fitted = fit_surfaces(regions, cleaned, FitConfig())
        except Exception:
            pass
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    tree = recognize_features(regions, fitted, graph, edges, cleaned,
                              RecognizeConfig())
    r = export_cadquery(tree, cleaned, use_llm=False, num_variants=1)
    return len(regions), r.volume_error_pct, r.status


def main() -> int:
    parts = sys.argv[1:] or [
        "Mesh1", "Mesh2", "Mesh3", "AI1", "AI6", "AI11", "AI13", "AI14",
    ]
    print(f"{'part':<8}{'A_err':>8}{'A_st':>6}{'B_err':>8}{'B_st':>6}"
          f"{'regs':>6}{'segB':>6}{'winner':>9}", flush=True)
    for part in parts:
        f = _find(part)
        if not f:
            print(f"{part:<8}  no file", flush=True)
            continue
        try:
            cleaned = ingest_mesh(f, IngestConfig(remesh=False, scan_auto=True))
        except Exception as e:
            print(f"{part:<8}  ingest ERR {str(e)[:40]}", flush=True)
            continue
        try:
            regs, a_err, a_st = _path_a(cleaned)
        except Exception as e:
            regs, a_err, a_st = -1, float("nan"), f"ERR:{str(e)[:20]}"
        try:
            b = reconstruct_segmentation_free(cleaned)
            b_err, b_st, segb = b.volume_error_pct, b.status, b.n_built
        except Exception as e:
            b_err, b_st, segb = float("nan"), f"ERR:{str(e)[:20]}", 0
        winner = ("A" if (a_err == a_err and (b_err != b_err or a_err < b_err))
                  else "B" if b_err == b_err else "?")
        print(f"{part:<8}{a_err:>8.1f}{a_st:>6}{b_err:>8.1f}{b_st:>6}"
              f"{regs:>6}{segb:>6}{winner:>9}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
