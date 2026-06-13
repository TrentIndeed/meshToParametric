"""Validate canonical-frame alignment + denoise on Scan1 (the wrench).

Compares ingest WITHOUT vs WITH the scan front-end (align + denoise):
bbox extents, how axis-aligned the dominant flat face is, and what the
extrusion engine reads (the wrench failure is the engine extruding a
jagged side profile on the wrong axis past the part's true width).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh2param.config import IngestConfig
from mesh2param.pipeline.extrusion import plan_extrusions
from mesh2param.pipeline.ingest import ingest_mesh

SCAN = ROOT / "models/Scan1/Spanner-stl.stl"
AXN = "XYZ"


def dominant_axis_alignment(cleaned):
    """Largest-area face-normal direction and its angle off the nearest
    world axis (0 deg = perfectly axis-aligned)."""
    n = cleaned.normals
    tri = cleaned.vertices[cleaned.faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    # fold normals to a hemisphere, area-weight, pick the dominant dir
    idx = np.argmax(np.abs(n), axis=1)
    sign = np.sign(n[np.arange(len(n)), idx]); sign[sign == 0] = 1
    folded = n * sign[:, None]
    keys = np.round(folded * 12).astype(int)
    acc = {}
    for k, a, nn in zip(map(tuple, keys), areas, folded):
        s = acc.setdefault(k, [0.0, np.zeros(3)])
        s[0] += a; s[1] += a * nn
    area_sum, vsum = max(acc.values(), key=lambda v: v[0])
    d = vsum / (np.linalg.norm(vsum) + 1e-12)
    off = float(np.degrees(np.arccos(min(1.0, np.max(np.abs(d))))))
    return d, off, area_sum / areas.sum()


def report(label, cfg):
    print(f"\n=== {label} ===")
    cleaned = ingest_mesh(str(SCAN), cfg)
    ext = cleaned.bbox[1] - cleaned.bbox[0]
    print(f"bbox extents (mm): {np.round(ext, 1).tolist()}  "
          f"thin/long ratio: {ext.min() / ext.max():.3f}")
    d, off, frac = dominant_axis_alignment(cleaned)
    print(f"dominant face normal: {np.round(d, 2).tolist()}  "
          f"({frac * 100:.0f}% of area)  off-axis: {off:.1f} deg")
    plan = plan_extrusions(cleaned)
    print(f"extrusion engine: {len(plan.segments)} segment(s), "
          f"coverage {plan.coverage_frac * 100:.0f}%")
    for s in plan.segments:
        print(f"   axis {AXN[s.axis_idx]} span {s.start:.1f}..{s.end:.1f} "
              f"(len {s.length:.1f}) area {s.area:.0f} "
              f"gain {s.gain_frac * 100:.0f}%")
    return cleaned


def main() -> int:
    base = dict(remesh=False)  # same for both; isolate the front-end
    report("BASELINE (no align, no denoise)", IngestConfig(**base))
    report("SCAN FRONT-END (align + denoise)",
           IngestConfig(**base, align_canonical=True, denoise_scan=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
