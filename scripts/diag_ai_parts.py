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


def main() -> int:
    print(f"{'part':<16}{'rawFaces':>9}{'wobble':>8}{'source':>11}"
          f"{'faces':>8}{'seg_def':>9}{'seg_nr':>8}", flush=True)
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
            print(f"{nm:<16}{rf:>9}{wob:>8.2f}{src:>11}{len(c.faces):>8}"
                  f"{nd:>9}{nr:>8}", flush=True)
        except Exception as e:
            print(f"{nm:<16} ERR {str(e)[:60]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
