"""Early-stage diagnosis sheet: raw -> cleaned/aligned -> segmentation -> topology.

The final mesh-vs-recon sheet (diag_render_sheet.py) shows WHAT came out; this
shows WHY, by rendering the early pipeline stages where most failures originate:

  cols = [ raw mesh | cleaned + axes (alignment) | segmentation | topology edges ]
  rows = two camera angles (isometric, top)

- raw: the file as loaded (original frame), with world axes — compare its
  orientation to the cleaned panel to SEE the alignment stage.
- cleaned: post-ingest (repair/alpha-wrap + canonical alignment), with axes — a
  well-aligned prismatic part should sit square to X/Y/Z, especially in top view.
- segmentation: every mesh face coloured by its FaceRegion — shatter (one face =
  one region) or over-merge (a fillet swallowed into a plane) jumps out.
- topology: region-boundary edges coloured by transition (RED sharp, GREEN
  fillet, BLUE chamfer, faint smooth) over the faint solid.

Headless (matplotlib Agg). Run in the mesh2param container:
  python scripts/diag_stage_sheet.py AI9 AI13 --out tests/visual_reports/diagnosis
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from mesh2param.config import (  # noqa: E402
    FitConfig, IngestConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.ingest import ingest_mesh  # noqa: E402
from mesh2param.pipeline.segment import segment_mesh  # noqa: E402
from mesh2param.pipeline.fit import fit_surfaces  # noqa: E402
from mesh2param.pipeline.topology import build_topology  # noqa: E402

EDGE_COLORS = {"SHARP": (0.90, 0.20, 0.20), "FILLET": (0.20, 0.75, 0.35),
               "CHAMFER": (0.25, 0.50, 0.95), "SMOOTH": (0.6, 0.6, 0.6)}


def _find(part):
    for ext in ("3mf", "stl", "obj", "ply"):
        hits = sorted(glob.glob(f"models/{part}/*.{ext}"))
        if hits:
            return hits[0]
    return None


def _sample_faces(n_faces, cap=9000):
    if n_faces <= cap:
        return np.arange(n_faces)
    rng = np.random.default_rng(0)
    return rng.choice(n_faces, cap, replace=False)


def _decimate(verts, faces, n=6000):
    if len(faces) <= n:
        return np.asarray(verts, float), np.asarray(faces, np.int64)
    try:
        import fast_simplification
        return fast_simplification.simplify(
            np.asarray(verts, np.float64), np.asarray(faces, np.int64), target_count=n)
    except Exception:
        return np.asarray(verts, float), np.asarray(faces, np.int64)


def _set_view(ax, verts, elev, azim, title, axes=False):
    lo, hi = verts.min(0), verts.max(0)
    ctr, rng = (lo + hi) / 2, (hi - lo).max() / 2 or 1
    for s, c in zip("xyz", ctr):
        getattr(ax, f"set_{s}lim")(c - rng, c + rng)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    if axes:  # world axes from origin to show alignment
        L = rng * 0.9
        for vec, col in ((( L, 0, 0), "r"), ((0, L, 0), "g"), ((0, 0, L), "b")):
            ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=col, lw=1.6)


def _faces_panel(ax, verts, faces, facecolor, elev, azim, title, alpha=1.0, axes=False):
    pc = Poly3DCollection(verts[faces], alpha=alpha)
    pc.set_facecolor(facecolor)
    pc.set_edgecolor((0, 0, 0, 0.06))
    pc.set_linewidth(0.1)
    ax.add_collection3d(pc)
    _set_view(ax, verts, elev, azim, title, axes=axes)


def build(part):
    path = _find(part)
    if path is None:
        raise FileNotFoundError(f"no mesh under models/{part}/")
    raw = trimesh.load(path, file_type=path.rsplit(".", 1)[-1].lower(), force="mesh")
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
    graph, edges = build_topology(regs, fitted, c, TopologyConfig())
    return raw, c, regs, edges, dirty


def render(part, out_dir):
    raw, c, regs, edges, dirty = build(part)
    cv = np.asarray(c.vertices, float)
    cf = np.asarray(c.faces, np.int64)
    rv, rf = _decimate(raw.vertices, raw.faces)
    cdv, cdf = _decimate(cv, cf)

    # Per-face region colour (on the FULL cleaned faces) + a render sample.
    try:
        import distinctipy
        cols = distinctipy.get_colors(max(1, len(regs)), rng=0)
    except Exception:
        cmap = plt.cm.tab20
        cols = [cmap(i % 20)[:3] for i in range(max(1, len(regs)))]
    fcol = np.tile(np.array([0.78, 0.5, 0.5, 1.0]), (len(cf), 1))  # unassigned=dusty red
    for ri, reg in enumerate(regs):
        idx = np.asarray(reg.face_indices, int)
        idx = idx[(idx >= 0) & (idx < len(cf))]
        fcol[idx] = (*cols[ri], 1.0)
    samp = _sample_faces(len(cf))

    n_sharp = sum(1 for e in edges if e.transition_type.value == "sharp")
    n_fil = sum(1 for e in edges if e.transition_type.value == "fillet")
    n_cham = sum(1 for e in edges if e.transition_type.value == "chamfer")
    ext = (np.asarray(c.bbox[1]) - np.asarray(c.bbox[0])).round(1).tolist()

    fig = plt.figure(figsize=(14.5, 7.2))
    fig.suptitle(
        f"{part}   scan_mode={dirty}   ext={ext}mm   "
        f"{len(regs)} regions   edges: sharp={n_sharp} fillet={n_fil} chamfer={n_cham}",
        fontsize=11, y=0.98)
    GREY = (0.62, 0.66, 0.70, 1.0)
    views = [("iso", 22, -60), ("top", 80, -90)]
    titles = ["raw mesh (orig frame)", "cleaned + axes (alignment)",
              f"segmentation ({len(regs)} regions)", "topology edges"]
    for ri, (vname, elev, azim) in enumerate(views):
        # raw
        ax = fig.add_subplot(2, 4, ri * 4 + 1, projection="3d")
        _faces_panel(ax, rv, rf, GREY, elev, azim,
                     f"{vname} · {titles[0]}" if ri == 0 else titles[0], axes=True)
        # cleaned + axes
        ax = fig.add_subplot(2, 4, ri * 4 + 2, projection="3d")
        _faces_panel(ax, cdv, cdf, GREY, elev, azim,
                     f"{vname} · {titles[1]}" if ri == 0 else titles[1], axes=True)
        # segmentation
        ax = fig.add_subplot(2, 4, ri * 4 + 3, projection="3d")
        _faces_panel(ax, cv, cf[samp], fcol[samp], elev, azim,
                     f"{vname} · {titles[2]}" if ri == 0 else titles[2])
        # topology
        ax = fig.add_subplot(2, 4, ri * 4 + 4, projection="3d")
        _faces_panel(ax, cv, cf[samp], (0.8, 0.82, 0.85, 0.12), elev, azim,
                     f"{vname} · {titles[3]}" if ri == 0 else titles[3], alpha=0.12)
        segs, scols = [], []
        for e in edges:
            ev = getattr(e, "edge_vertices", None)
            if ev is None or len(ev) < 2:
                continue
            ev = np.asarray(ev, float)
            col = EDGE_COLORS.get(e.transition_type.value.upper(), (0.5, 0.5, 0.5))
            for k in range(len(ev) - 1):
                segs.append([ev[k], ev[k + 1]])
                scols.append(col)
        if segs:
            lc = Line3DCollection(segs, colors=scols, linewidths=1.3)
            ax.add_collection3d(lc)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{part}_stages.png")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"{part}: {len(regs)} regions, edges sharp={n_sharp}/fil={n_fil}/cham={n_cham} -> {out}")
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
