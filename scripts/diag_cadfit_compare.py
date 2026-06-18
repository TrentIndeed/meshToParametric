"""Head-to-head: this repo vs CADFit (github.com/ghadinehme/CADFit) on the meshes.

Both pipelines reconstruct each mesh; we score BOTH with the SAME metric — voxel
IoU of the reconstruction vs that pipeline's cleaned input (iou_refine.voxel_iou)
— because CADFit's self-reported manifold IoU is unreliable on multi-body AI
meshes (returns >1; its own iou.py documents the bug). Higher IoU = better.

  this repo:  ingest -> recognize -> export_cadquery (residual loop + revolve +
              contour router)  -> tessellated solid vs cleaned mesh
  CADFit:     run_pipeline.py (extract sketches -> analyze -> greedy IoU build
              -> residual refine) -> best_greedy_parallel[_iterative].stl vs
              its decimated_input.stl

CADFit is run as a subprocess in its clone (Windows-patched: signal shim +
PYTHONUTF8). Set CADFIT_DIR to override its location.

Run:  python -u scripts/diag_cadfit_compare.py [AI3 Mesh2 ...]
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh2param.config import (                                # noqa: E402
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.export_cadquery import export_cadquery  # noqa: E402
from mesh2param.pipeline.fit import fit_surfaces                # noqa: E402
from mesh2param.pipeline.ingest import ingest_mesh              # noqa: E402
from mesh2param.pipeline.iou_refine import voxel_iou            # noqa: E402
from mesh2param.pipeline.recognize import recognize_features    # noqa: E402
from mesh2param.pipeline.regularize import regularize_mesh      # noqa: E402
from mesh2param.pipeline.segment import segment_mesh            # noqa: E402
from mesh2param.pipeline.topology import build_topology         # noqa: E402

CADFIT_DIR = os.environ.get(
    "CADFIT_DIR", "c:/Users/Trenton/AppData/Local/Temp/CADFit")
IN_DIR = str(ROOT / "_cadfit_tmp" / "in")
OUT_DIR = str(ROOT / "_cadfit_tmp" / "out")


def _raw_mesh_path(part: str):
    for ext in ("*.3mf", "*.stl", "*.obj", "*.ply"):
        fs = glob.glob(str(ROOT / "models" / part / ext))
        if fs:
            return fs[0]
    return None


def _load(path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    return m.to_geometry() if isinstance(m, trimesh.Scene) else m


def _tess_to_mesh(tv, tf):
    return trimesh.Trimesh(vertices=np.asarray(tv, float).reshape(-1, 3),
                           faces=np.asarray(tf, np.int64).reshape(-1, 3),
                           process=False)


def _bbox_norm(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center + uniformly scale to a [-1,1] box. Makes voxel IoU invariant to
    scale/translation so the two pipelines' frames are comparable (CADFit
    normalizes; this repo normalizes to canonical_diag). Neither pipeline
    re-orients the part relative to its own cleaned input, so a reconstruction
    is compared in a consistent frame within each pipeline."""
    m = m.copy()
    c = 0.5 * (m.bounds[0] + m.bounds[1])
    m.apply_translation(-c)
    s = float((m.bounds[1] - m.bounds[0]).max())
    if s > 0:
        m.apply_scale(2.0 / s)
    return m


CADFIT_TIMEOUT = int(os.environ.get("CADFIT_TIMEOUT", "180"))


def run_cadfit(parts) -> dict:
    """Run CADFit PER MESH with a wall-clock timeout (CADFit's Unix SIGALRM
    build-timeouts are shimmed out on Windows, so a problematic build can hang;
    a per-mesh timeout isolates that). Returns per-part {out_stl, raw, dur,
    status}."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    res = {}
    for p in parts:
        raw = _raw_mesh_path(p)
        pin = Path(IN_DIR) / p
        pout = Path(OUT_DIR) / p
        pin.mkdir(parents=True, exist_ok=True)
        pout.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(pin / "*.stl")):
            os.remove(f)
        _load(raw).export(str(pin / f"{p}.stl"))
        status = "ok"
        t0 = time.time()
        try:
            subprocess.run([sys.executable, "run_pipeline.py", str(pin),
                            "--output-folder", str(pout)],
                           cwd=CADFIT_DIR, env=env, capture_output=True,
                           encoding="utf-8", errors="replace",
                           timeout=CADFIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            status = "timeout"
        dur = time.time() - t0
        d = pout / p
        out = None
        for name in ("best_greedy_parallel_iterative.stl",
                     "best_greedy_parallel.stl"):
            if (d / name).exists():
                out = str(d / name)
                break
        if out is None and status == "ok":
            status = "no-output"
        res[p] = {"out": out, "raw": raw, "dur": dur, "status": status}
    return res


def cadfit_iou(info) -> float:
    """voxel IoU of CADFit's reconstruction vs the raw input, both
    bbox-normalized (CADFit preserves orientation, so this is frame-fair)."""
    if not info or not info["out"] or not info["raw"]:
        return float("nan")
    try:
        return voxel_iou(_bbox_norm(_load(info["raw"])),
                         _bbox_norm(_load(info["out"])), res=64)
    except Exception:
        return float("nan")


def this_repo(part):
    """Build via the same single-body sequence as diag_ai_build (matches all
    prior cad_vol_err numbers); return (iou, vol_err_pct, n_feats)."""
    raw = _raw_mesh_path(part)
    c = ingest_mesh(raw, IngestConfig(remesh=False, scan_auto=True))
    dirty = bool(c.metadata.get("scan_mode"))
    seg = SegmentConfig(noise_robust=dirty)
    regions = segment_mesh(c, seg)
    fitted = fit_surfaces(regions, c, FitConfig())
    if dirty:
        try:
            regularize_mesh(c, regions, fitted, snap_angle_deg=12.0)
            regions = segment_mesh(c, seg)
            fitted = fit_surfaces(regions, c, FitConfig())
        except Exception:
            pass
    graph, edges = build_topology(regions, fitted, c, TopologyConfig())
    tree = recognize_features(regions, fitted, graph, edges, c,
                              RecognizeConfig())
    r = export_cadquery(tree, c, use_llm=False, num_variants=1)
    target = trimesh.Trimesh(vertices=c.vertices, faces=c.faces, process=False)
    iou = float("nan")
    if r.tess_vertices is not None and r.tess_faces is not None:
        try:
            iou = voxel_iou(_bbox_norm(target),
                            _bbox_norm(_tess_to_mesh(r.tess_vertices,
                                                     r.tess_faces)), res=64)
        except Exception:
            pass
    return iou, r.volume_error_pct, tree.num_features


def main() -> int:
    parts = sys.argv[1:] or ["Mesh2", "Mesh4", "AI1", "AI3", "AI11", "AI13"]
    print(f"Running CADFit (per-mesh, timeout {CADFIT_TIMEOUT}s) on "
          f"{len(parts)} meshes...", flush=True)
    cf = run_cadfit(parts)
    print(f"\n{'part':<8}{'thisRepo_IoU':>13}{'CADFit_IoU':>12}"
          f"{'CADFit_status':>15}{'thisRepo_volErr':>16}{'winner':>10}",
          flush=True)
    for p in parts:
        try:
            r_iou, r_err, _ = this_repo(p)
        except Exception as e:
            r_iou, r_err = float("nan"), float("nan")
            print(f"{p:<8} thisRepo ERR {str(e)[:40]}", flush=True)
        info = cf.get(p) or {}
        c_iou = cadfit_iou(info)
        win = "-"
        if r_iou == r_iou and c_iou == c_iou:
            win = "thisRepo" if r_iou > c_iou + 0.02 else (
                "CADFit" if c_iou > r_iou + 0.02 else "tie")
        print(f"{p:<8}{r_iou:>13.3f}{c_iou:>12.3f}"
              f"{info.get('status','?'):>15}{r_err:>15.1f}%{win:>10}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
