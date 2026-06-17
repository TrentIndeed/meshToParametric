"""Diagnose the AI-generated test parts (models/AI*/): do real Meshy meshes
fragment like simulated noise, or segment cleanly?

For each part: raw face count + surface wobble (does it trip the dirty-mesh
detector?), then ingest (denoise+align) and segment with default vs
noise_robust. The seg_def column is the key: sane counts (tens) => real AI
meshes don't fragment and regularization already works; hundreds => the
fragmentation is real and stage C needs region-growing work.

Run unbuffered:  python -u scripts/diag_ai_parts.py
"""
from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for n in ("trimesh", "matplotlib", "PIL", "urllib3", "pymeshlab"):
    logging.getLogger(n).setLevel(logging.ERROR)

import numpy as np
import trimesh

from mesh2param.config import IngestConfig, SegmentConfig
from mesh2param.pipeline.ingest import ingest_mesh, _surface_wobble
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.convert import _decimate_to_density


def _axis_alignment(cleaned, tol_deg: float = 5.0) -> float:
    """Area-weighted fraction of face area whose normal sits within tol_deg of
    a world axis. Higher = the part's dominant flats are closer to the world
    axes = a better canonical frame. The metric the align-before-decimate fix
    is meant to improve/hold on AI parts."""
    try:
        m = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces,
                            process=False)
        n = np.asarray(m.face_normals, dtype=float)
        a = np.asarray(m.area_faces, dtype=float)
        if len(n) == 0 or a.sum() <= 0:
            return 0.0
        # max |component| = cosine of angle to the nearest world axis.
        near = np.abs(n).max(axis=1) >= np.cos(np.radians(tol_deg))
        return float(a[near].sum() / a.sum())
    except Exception:
        return 0.0


def main() -> int:
    print(f"{'part':<16}{'rawFaces':>9}{'wobble':>8}{'source':>11}"
          f"{'faces':>8}{'seg_def':>9}{'seg_nr':>8}{'segProd':>8}"
          f"{'alignQ':>8}{'method':>9}", flush=True)
    for d in sorted(glob.glob(str(ROOT / "models/AI*")),
                    key=lambda x: int(Path(x).name[2:])):
        files = glob.glob(d + "/*.3mf")
        if not files:
            continue
        nm = Path(d).name
        try:
            m = trimesh.load(files[0])
            m = m.to_geometry() if isinstance(m, trimesh.Scene) else m
            rf, wob = len(m.faces), _surface_wobble(m)
            c = ingest_mesh(files[0], IngestConfig(remesh=False, scan_auto=True))
            src = c.metadata.get("cleanup_source", "-")
            nd = len(segment_mesh(c, SegmentConfig()))
            nr = len(segment_mesh(c, SegmentConfig(noise_robust=True)))
            # Production-style: convert.py re-decimates the ingest output to a
            # reference triangle density before segmenting (noise_robust on for
            # scan/AI). This is the seg count the live pipeline actually sees.
            dens = _decimate_to_density(c) or c
            sp = len(segment_mesh(dens, SegmentConfig(noise_robust=True)))
            aq = _axis_alignment(c)
            method = c.metadata.get("align_method", "-")
            print(f"{nm:<16}{rf:>9}{wob:>8.2f}{src:>11}{len(c.faces):>8}"
                  f"{nd:>9}{nr:>8}{sp:>8}{aq:>8.2f}{method:>9}", flush=True)
        except Exception as e:
            print(f"{nm:<16} ERR {str(e)[:60]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
