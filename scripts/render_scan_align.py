"""Render the wrench scan BEFORE vs AFTER canonical-frame alignment as a
side-by-side PNG, so the re-orientation is visible (numbers in
diag_scan_align.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh

SCAN = ROOT / "models/Scan1/Spanner-stl.stl"
OUT = ROOT / "data" / "scan1_alignment.png"


def panel(ax, cleaned, title):
    v = cleaned.vertices
    f = cleaned.faces
    # cap triangles drawn for speed
    if len(f) > 12000:
        f = f[np.random.RandomState(0).choice(len(f), 12000, replace=False)]
    tris = v[f]
    col = Poly3DCollection(tris, alpha=0.9, linewidths=0.0)
    col.set_facecolor((0.55, 0.68, 0.86))
    col.set_edgecolor((0.3, 0.4, 0.55, 0.15))
    ax.add_collection3d(col)
    lo = v.min(axis=0); hi = v.max(axis=0)
    c = (lo + hi) / 2; r = float((hi - lo).max()) / 2 * 1.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ext = hi - lo
    ax.set_title(f"{title}\nextents {np.round(ext,1).tolist()} mm",
                 fontsize=10)
    ax.view_init(elev=22, azim=-60)


def main() -> int:
    base = ingest_mesh(str(SCAN), IngestConfig(remesh=False))
    aligned = ingest_mesh(str(SCAN), IngestConfig(
        remesh=False, align_canonical=True, denoise_scan=True))
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    panel(ax1, base, "BEFORE — raw scan pose")
    panel(ax2, aligned, "AFTER — canonical frame (flat on XY, thickness Z)")
    fig.suptitle("Scan1 (wrench): canonical-frame alignment", fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
