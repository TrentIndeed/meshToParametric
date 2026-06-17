"""Why does the canonical-frame alignment get WORSE on the high-res mesh for
some AI parts (AI3/4/9) when the handoff predicted it should get better?

For a few AI parts, decimate the RAW mesh to several face counts, fix normals,
then run _frame_from_normals and report:
  - the top candidate normal directions (area fraction + angle to nearest axis),
  - the resulting frame's axis-aligned area fraction (alignQ).

If the frame / alignQ swings with resolution, the dominant-normal triple is
resolution-sensitive (the suspected mechanism). Isolates the decimation effect
only (no denoise/repair) as a first cut.

Run:  python -u scripts/diag_align_resolution.py
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

from mesh2param.pipeline.ingest import _decimate, _fix_normals
from mesh2param.pipeline.align import (
    _frame_from_normals, _candidate_axes,
)

PARTS = ["AI3", "AI4", "AI9", "AI13", "AI2"]
RES = [12_000, 30_000, 120_000]


def _area_aligned_frac(mesh: trimesh.Trimesh, tol_deg: float = 5.0) -> float:
    n = np.asarray(mesh.face_normals, dtype=float)
    a = np.asarray(mesh.area_faces, dtype=float)
    if len(n) == 0 or a.sum() <= 0:
        return 0.0
    near = np.abs(n).max(axis=1) >= np.cos(np.radians(tol_deg))
    return float(a[near].sum() / a.sum())


def _angle_to_axis(d: np.ndarray) -> float:
    """Angle (deg) between unit dir d and its nearest world axis."""
    return float(np.degrees(np.arccos(np.clip(np.abs(d).max(), 0, 1))))


def main() -> int:
    for part in PARTS:
        files = glob.glob(str(ROOT / f"models/{part}/*.3mf"))
        if not files:
            print(f"{part}: no file")
            continue
        m = trimesh.load(files[0])
        m = m.to_geometry() if isinstance(m, trimesh.Scene) else m
        print(f"\n=== {part}  (raw {len(m.faces)} faces) ===")
        for res in RES:
            dm = _decimate(m.copy(), res)
            dm = _fix_normals(dm)
            normals = np.asarray(dm.face_normals, float)
            areas = np.asarray(dm.area_faces, float)
            total = float(areas.sum())
            cand = _candidate_axes(normals, areas)
            R = _frame_from_normals(dm)
            # alignQ as-is (current pose) and after applying R
            frac_id = _area_aligned_frac(dm)
            if R is not None:
                T = np.eye(4); T[:3, :3] = R
                rot = dm.copy(); rot.apply_transform(T)
                frac_R = _area_aligned_frac(rot)
                rmethod = "normals"
            else:
                frac_R = float("nan")
                rmethod = "obb(None)"
            top = "  ".join(
                f"{(a/total):.2f}@{_angle_to_axis(d):.0f}deg"
                for a, d in cand[:4]
            )
            print(f"  {len(dm.faces):>7}f  frac_id={frac_id:.2f} "
                  f"frac_afterR={frac_R:.2f}  [{rmethod}]  topcand: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
