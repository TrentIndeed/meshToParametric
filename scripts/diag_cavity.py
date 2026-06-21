"""SEE the cavity: cross-section a part along each axis at several depths and draw
the material polygon (with interior holes) at each slice. Reveals whether the big
concavity is a deep blind pocket, a through-tunnel, a hollow shell, or an open
box — which determines the cavity-recognition approach. Writes one sheet per part.
Run in the mesh2param Docker container."""
import glob
import logging
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon

sys.path.insert(0, ".")
logging.disable(logging.INFO)

import trimesh
from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.recognize import _section_material


def _draw(ax, poly, title):
    ax.set_title(title, fontsize=7)
    ax.set_aspect("equal"); ax.axis("off")
    if poly is None or poly.is_empty:
        ax.text(0.5, 0.5, "empty", ha="center", fontsize=6, transform=ax.transAxes)
        return
    geoms = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
    for g in geoms:
        xs, ys = g.exterior.xy
        ax.fill(xs, ys, facecolor="#9fb4c7", edgecolor="#234", linewidth=0.6)
        for ring in g.interiors:                 # holes = cavities at this slice
            xs, ys = ring.xy
            ax.fill(xs, ys, facecolor="white", edgecolor="#c33", linewidth=0.8)


for part in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["AI4"]):
    hits = sorted(glob.glob(f"models/{part}/*.3mf") + glob.glob(f"models/{part}/*.stl")
                  + glob.glob(f"models/{part}/*.obj") + glob.glob(f"models/{part}/*.ply"))
    if not hits:
        print(f"{part}: no mesh"); continue
    c = ingest_mesh(hits[0], IngestConfig(remesh=False, scan_auto=True))
    m = trimesh.Trimesh(vertices=c.vertices, faces=c.faces, process=False)
    lo, hi = np.asarray(c.bbox[0]), np.asarray(c.bbox[1])
    dims = hi - lo
    fracs = [0.06, 0.25, 0.5, 0.75, 0.94]
    fig, axes = plt.subplots(3, len(fracs), figsize=(len(fracs) * 2.3, 7))
    fig.suptitle(f"{part}  dims={np.round(dims,1).tolist()}  "
                 f"(white/red = cavity hole in the material slice)", fontsize=9)
    for ai, axis in enumerate(range(3)):
        for fi, fr in enumerate(fracs):
            t = lo[axis] + fr * dims[axis]
            poly = _section_material(m, axis, t)
            nh = 0
            if poly is not None and not poly.is_empty:
                gs = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
                nh = sum(len(g.interiors) for g in gs)
            _draw(axes[ai, fi], poly, f"{'XYZ'[axis]} @ {fr:.2f}  holes={nh}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    op = f"tests/visual_reports/diagnosis/{part}_cavity.png"
    fig.savefig(op, dpi=110); plt.close(fig)
    print(f"{part}: wrote {op}")
