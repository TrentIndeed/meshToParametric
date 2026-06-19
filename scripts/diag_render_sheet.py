"""Visual diagnosis sheet: mesh vs reconstruction, for Claude (and humans) to SEE.

The pipeline's bottleneck in debugging is that the failure is geometric — a
mispositioned extrusion, an uncut pocket, a fillet on the wrong edge — and a
single IoU number can't tell you which. This renders a per-part PNG that makes
the failure obvious at a glance:

  rows  = two camera angles (isometric, top)
  cols  = [ cleaned mesh | reconstruction | overlay | error heatmap ]

The heatmap colours the reconstruction by SIGNED distance to the cleaned mesh:
RED where the recon sticks OUT past the true surface (extra material), BLUE
where it sits inside (so blue voids = the recon is MISSING material the mesh
has). Title carries the metrics (exact IoU, volume error, feature list, dims).

Headless by design: matplotlib Agg + Poly3DCollection — no VTK/OpenGL/display,
so it runs in the lean `mesh2param` container. Output is a flat PNG Claude reads.

Usage (in the mesh2param container, models + scripts + output dir mounted):
  python scripts/diag_render_sheet.py AI13 AI9 --out tests/visual_reports/diagnosis
"""
from __future__ import annotations

import os
import sys

# Make `mesh2param` importable when run as `python scripts/diag_render_sheet.py`
# (sys.path[0] is scripts/, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from mesh2param.config import (  # noqa: E402
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.ingest import ingest_mesh  # noqa: E402
from mesh2param.pipeline.segment import segment_mesh  # noqa: E402
from mesh2param.pipeline.fit import fit_surfaces  # noqa: E402
from mesh2param.pipeline.topology import build_topology  # noqa: E402
from mesh2param.pipeline.recognize import recognize_features  # noqa: E402
from mesh2param.pipeline.export_cadquery import export_cadquery  # noqa: E402


def _find(part: str) -> str | None:
    for ext in ("3mf", "stl", "obj", "ply"):
        hits = sorted(glob.glob(f"models/{part}/*.{ext}"))
        if hits:
            return hits[0]
    return None


def _decimate(verts, faces, n=4000):
    if len(faces) <= n:
        return np.asarray(verts, float), np.asarray(faces, np.int64)
    try:
        import fast_simplification
        v, f = fast_simplification.simplify(
            np.asarray(verts, np.float64), np.asarray(faces, np.int64), target_count=n)
        return v, f
    except Exception:
        return np.asarray(verts, float), np.asarray(faces, np.int64)


def build(part: str):
    """Run the full AI/scan pipeline and return (cleaned_tm, recon_tm, meta)."""
    path = _find(part)
    if path is None:
        raise FileNotFoundError(f"no mesh under models/{part}/")
    c = ingest_mesh(path, IngestConfig(remesh=False, scan_auto=True))
    dirty = bool((c.metadata or {}).get("scan_mode"))
    seg = SegmentConfig(noise_robust=dirty)
    regs = segment_mesh(c, seg)
    fitted = fit_surfaces(regs, c, FitConfig())
    if dirty:
        from mesh2param.pipeline.regularize import regularize_mesh
        regularize_mesh(c, regs, fitted, snap_angle_deg=12.0)
        regs = segment_mesh(c, seg)
        fitted = fit_surfaces(regs, c, FitConfig())
    g, e = build_topology(regs, fitted, c, TopologyConfig())
    tree = recognize_features(regs, fitted, g, e, c, RecognizeConfig())
    r = export_cadquery(tree, c, use_llm=False, num_variants=1)

    cleaned = trimesh.Trimesh(vertices=np.asarray(c.vertices),
                              faces=np.asarray(c.faces), process=False)
    rv = np.asarray(r.tess_vertices, float).reshape(-1, 3)
    rf = np.asarray(r.tess_faces, np.int64).reshape(-1, 3)
    recon = trimesh.Trimesh(vertices=rv, faces=rf, process=False)

    iou = None
    try:
        from mesh2param.pipeline.iou_refine import exact_iou_available, mesh_iou_exact
        if exact_iou_available():
            iou = mesh_iou_exact(cleaned, recon)
    except Exception:
        pass
    mv = abs(float(cleaned.volume)) or 1.0
    vol_err = abs(abs(float(recon.volume)) - mv) / mv * 100.0
    meta = {
        "part": part, "iou": iou, "vol_err": vol_err,
        "features": [f.type.name for f in tree.features],
        "ext": (np.asarray(c.bbox[1]) - np.asarray(c.bbox[0])).round(1).tolist(),
        "scan_mode": dirty,
    }
    return cleaned, recon, meta


def _signed_color(recon: trimesh.Trimesh, cleaned: trimesh.Trimesh, scale: float):
    """Per-face diverging colour: +dist OUTSIDE cleaned (red) / inside (blue)."""
    cen = recon.triangles.mean(axis=1)
    try:
        from trimesh.proximity import closest_point
        _, dist, _ = closest_point(cleaned, cen)
    except Exception:
        dist = np.zeros(len(cen))
    try:
        inside = cleaned.contains(cen)
    except Exception:
        inside = np.zeros(len(cen), bool)
    signed = np.where(inside, -dist, dist)
    t = np.clip(signed / (scale or 1.0) * 0.5 + 0.5, 0, 1)
    return plt.cm.coolwarm(t)


def _panel(ax, verts, faces, color, title, elev, azim, alpha=1.0, edge=None):
    polys = verts[faces]
    pc = Poly3DCollection(polys, alpha=alpha)
    pc.set_facecolor(color)
    pc.set_edgecolor(edge if edge is not None else (0, 0, 0, 0.08))
    pc.set_linewidth(0.15)
    ax.add_collection3d(pc)
    lo, hi = verts.min(0), verts.max(0)
    ctr, rng = (lo + hi) / 2, (hi - lo).max() / 2 or 1
    for s, c in zip("xyz", ctr):
        getattr(ax, f"set_{s}lim")(c - rng, c + rng)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def render(part: str, out_dir: str):
    cleaned, recon, m = build(part)
    cv, cf = _decimate(cleaned.vertices, cleaned.faces)
    rv, rf = _decimate(recon.vertices, recon.faces)
    cvv, rvv = np.asarray(cv), np.asarray(rv)
    scale = 0.05 * float(np.linalg.norm(np.asarray(m["ext"])))
    hcolor = _signed_color(
        trimesh.Trimesh(vertices=rv, faces=rf, process=False), cleaned, scale)

    GREY = (0.62, 0.66, 0.70, 1.0)
    BLUE = (0.30, 0.55, 0.95, 1.0)
    fig = plt.figure(figsize=(13, 6.6))
    iou_s = f"{m['iou']:.3f}" if m["iou"] is not None else "n/a"
    feats = ", ".join(f"{t}×{m['features'].count(t)}" for t in dict.fromkeys(m["features"]))
    fig.suptitle(
        f"{part}   IoU={iou_s}   vol_err={m['vol_err']:.1f}%   "
        f"ext={m['ext']}mm   feats: {feats}",
        fontsize=11, y=0.98)

    views = [("iso", 22, -60), ("top", 80, -90)]
    cols = ["cleaned mesh", "reconstruction", "overlay", "error (red=extra, blue=inside)"]
    for ri, (vname, elev, azim) in enumerate(views):
        for ci in range(4):
            ax = fig.add_subplot(2, 4, ri * 4 + ci + 1, projection="3d")
            ttl = f"{vname} · {cols[ci]}" if ri == 0 else cols[ci]
            if ci == 0:
                _panel(ax, cvv, cf, GREY, ttl, elev, azim)
            elif ci == 1:
                _panel(ax, rvv, rf, BLUE, ttl, elev, azim)
            elif ci == 2:
                _panel(ax, cvv, cf, (0.6, 0.6, 0.6, 0.18), ttl, elev, azim, alpha=0.18)
                _panel(ax, rvv, rf, (0.30, 0.55, 0.95, 0.55), "", elev, azim, alpha=0.55)
            else:
                _panel(ax, rvv, rf, hcolor, ttl, elev, azim)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{part}.png")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"{part}: IoU={iou_s} vol_err={m['vol_err']:.1f}% -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out", default="tests/visual_reports/diagnosis")
    a = ap.parse_args()
    for p in a.parts:
        try:
            render(p, a.out)
        except Exception as exc:
            import traceback
            print(f"{p}: FAILED {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
