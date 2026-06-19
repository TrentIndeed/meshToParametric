#!/usr/bin/env python3
"""Parallel AI-set build — same per-part pipeline as diag_ai_build, but runs the
parts CONCURRENTLY across a process pool, so a full 15-part validation finishes
in roughly the slowest single part's time instead of the sum (~15-20 min -> a
few minutes). The parts are fully independent, so this is embarrassingly
parallel; the only limits are CPU cores and RAM (each worker loads its own
OCC + pymeshlab and the mesh, so heavy parts like AI13 use real memory).

Run in the mesh2param container (16 cores there):
  python scripts/diag_ai_parallel.py                       # all AI1..AI15
  python scripts/diag_ai_parallel.py AI9 AI13 --workers 8
  python scripts/diag_ai_parallel.py --workers 12          # dial to fit RAM

If you OOM, lower --workers (or raise Docker Desktop's memory in
Settings -> Resources). Prints each part as it finishes + a mean-IoU summary.
"""
import os

# Cap (but don't kill) each worker's internal threads. Full multithreading per
# worker × many workers oversubscribes the cores; 1 thread/worker is worse (each
# part's numpy/OCC genuinely needs a few threads). A few threads/worker × a
# moderate worker count fills the cores without thrash. Tunable via THREADS_PER.
_tpw = os.environ.get("THREADS_PER", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _tpw)

import argparse
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# NOTE: do NOT preload pymeshlab (or anything importing it, like
# export_cadquery/iou_refine) in the parent. pymeshlab is Qt-based and NOT
# fork-safe — forking after it loads deadlocks the workers' alpha-wrap call, so
# the parts that need alpha-wrap hang forever on "ingest". Each worker imports it
# fresh after fork instead. (OCC/cadquery would be fork-safe to preload, but the
# memory saving isn't worth the footgun, so we just let workers import.)


def _build_one(part: str):
    """Full pipeline + exact IoU for one part. Imports live INSIDE the worker so
    each process owns its OCC/pymeshlab state. Returns a result tuple."""
    import logging
    for n in ("trimesh", "matplotlib", "PIL", "urllib3", "pymeshlab",
              "cadquery", "mesh2param"):
        logging.getLogger(n).setLevel(logging.ERROR)
    t = time.time()
    try:
        import numpy as np
        import trimesh
        from mesh2param.config import (
            FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig)
        from mesh2param.pipeline.ingest import ingest_mesh
        from mesh2param.pipeline.segment import segment_mesh
        from mesh2param.pipeline.fit import fit_surfaces
        from mesh2param.pipeline.topology import build_topology
        from mesh2param.pipeline.recognize import recognize_features
        from mesh2param.pipeline.export_cadquery import export_cadquery
        from mesh2param.pipeline.iou_refine import mesh_iou_exact

        # Per-stage progress so `docker logs` shows who's on which stage live
        # (workers interleave; each line is tagged with the part). Set
        # NO_STAGE_LOG=1 to silence.
        _quiet = os.environ.get("NO_STAGE_LOG") == "1"

        def stage(s):
            if not _quiet:
                print(f"  [{part:<6}] {s:<22} (+{time.time() - t:.0f}s)", flush=True)

        d = ROOT / "models" / part
        files = (glob.glob(str(d / "*.3mf")) + glob.glob(str(d / "*.stl"))
                 + glob.glob(str(d / "*.obj")) + glob.glob(str(d / "*.ply")))
        if not files:
            return (part, None, None, float("nan"), None, time.time() - t, "no mesh")

        stage("ingest...")
        c = ingest_mesh(files[0], IngestConfig(remesh=False, scan_auto=True))
        dirty = bool((c.metadata or {}).get("scan_mode"))
        stage(f"ingest done ({len(c.faces)} faces)")
        seg = SegmentConfig(noise_robust=dirty)
        regions = segment_mesh(c, seg)
        fitted = fit_surfaces(regions, c, FitConfig())
        stage(f"segment+fit ({len(regions)} regions)")
        if dirty:
            from mesh2param.pipeline.regularize import regularize_mesh
            regularize_mesh(c, regions, fitted, snap_angle_deg=12.0)
            regions = segment_mesh(c, seg)
            fitted = fit_surfaces(regions, c, FitConfig())
            stage(f"regularize+reseg ({len(regions)} regions)")
        graph, edges = build_topology(regions, fitted, c, TopologyConfig())
        stage(f"topology ({len(edges)} edges)")
        tree = recognize_features(regions, fitted, graph, edges, c, RecognizeConfig())
        stage(f"recognize ({tree.num_features} feats)")
        res = export_cadquery(tree, c, use_llm=False, num_variants=1)
        stage("export done; IoU...")

        iou = float("nan")
        try:
            tgt = trimesh.Trimesh(vertices=c.vertices, faces=c.faces, process=False)
            rec = trimesh.Trimesh(
                vertices=np.asarray(res.tess_vertices, float).reshape(-1, 3),
                faces=np.asarray(res.tess_faces, np.int64).reshape(-1, 3),
                process=False)
            iou = mesh_iou_exact(tgt, rec)
        except Exception:
            pass
        return (part, len(regions), tree.num_features, iou,
                res.volume_error_pct, time.time() - t, None)
    except Exception as exc:
        return (part, None, None, float("nan"), None, time.time() - t, str(exc)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes (default: cpu_count-1, capped to #parts)")
    a = ap.parse_args()
    parts = a.parts or [f"AI{i}" for i in range(1, 16)]
    cpu = os.cpu_count() or 4
    workers = a.workers or min(len(parts), max(2, cpu - 1))

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    # SPAWN, not fork: pymeshlab (Qt) deadlocks in a forked child, so the parts
    # that call alpha-wrap hang forever. Spawn starts each worker as a fresh
    # interpreter that imports cleanly — costs ~15s import per worker (parallel),
    # but it's the only safe option with pymeshlab in the mix.
    ctx = mp.get_context("spawn")
    print(f"building {len(parts)} parts on {workers}/{cpu} workers (spawn)...",
          flush=True)
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_build_one, p): p for p in parts}
        for fu in as_completed(futs):
            part, nreg, nf, iou, ve, el, err = fu.result()
            results[part] = (iou, ve)
            if err:
                print(f"{part:<10} ERR {err} ({el:.0f}s)", flush=True)
            else:
                print(f"{part:<10} regions={nreg:>4} feats={nf:>3} "
                      f"IoU={iou:>5.3f} vol_err={ve:>6.1f}% ({el:.0f}s)", flush=True)

    ious = [results[p][0] for p in parts
            if p in results and results[p][0] == results[p][0]]
    if ious:
        print(f"\nmean IoU = {sum(ious) / len(ious):.3f} over {len(ious)} parts "
              f"| wall {time.time() - t0:.0f}s "
              f"(sequential would be ~{sum(1 for _ in parts) * 70}s)", flush=True)


if __name__ == "__main__":
    main()
