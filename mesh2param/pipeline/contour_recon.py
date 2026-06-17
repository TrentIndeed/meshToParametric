"""Segmentation-free reconstruction (CADFit-style) — for the Phase 5 A/B.

The question (docs/CADFIT_PORTING_PLAN.md Phase 5): is this repo's segmentation
stage load-bearing for build QUALITY, or can a CADFit-style path that never
segments match it? CADFit extracts prismatic structure directly and lets a
global IoU loop forgive the rest. This repo already has both halves:
  * `extrusion.plan_extrusions` decomposes a watertight mesh into prismatic
    extrusions via slicing + voxel occupancy — NO segment/fit/topology/recognize.
  * `iou_refine` / `export_cadquery._residual_refine_solid` patch the residual.

So this module is: plan_extrusions -> build each extrusion segment as a CadQuery
solid -> union -> residual-refine against the target. The result is a solid +
`cad_vol_err` produced WITHOUT segmentation, directly comparable to the full
segment->fit->topology->recognize->export path. `scripts/diag_seg_ab.py` runs
both and applies the decision gate.

Segments are built with `extrudeLinear` over a face lifted from the segment's
world-space profile — avoids CadQuery named-plane local-coordinate bookkeeping
(the profile (u,v) are already world coords of the two non-axis axes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReconResult:
    solid: object | None = None
    cad_volume: float = 0.0
    mesh_volume: float = 0.0
    volume_error_pct: float = 100.0
    status: str = "FAIL"
    n_segments: int = 0
    n_built: int = 0
    coverage_frac: float = 0.0
    info: dict = field(default_factory=dict)


def _segment_to_solid(seg, cq):
    """Build one extrusion segment as a CadQuery solid, or None.

    The segment's `profile` is a list of (u, v) where u, v are world coords on
    the two non-axis axes. Lift each to a 3D world point on the start plane,
    make a planar face, and extrude along the world axis by the segment length.
    """
    uv = seg.axes_uv
    ax = seg.axis_idx
    if len(seg.profile) < 3 or seg.length <= 1e-9:
        return None
    pts = []
    for (u, v) in seg.profile:
        p = [0.0, 0.0, 0.0]
        p[ax] = float(seg.start)
        p[uv[0]] = float(u)
        p[uv[1]] = float(v)
        pts.append(cq.Vector(*p))
    try:
        verts = pts + [pts[0]]          # explicit closure
        wire = cq.Wire.makePolygon(verts)
        face = cq.Face.makeFromWires(wire)
        axis_unit = np.zeros(3)
        axis_unit[ax] = 1.0
        vec = cq.Vector(*(axis_unit * float(seg.length)))
        return cq.Solid.extrudeLinear(face, vec)
    except Exception:
        logger.debug("segment extrude failed (axis=%d, npts=%d)",
                     ax, len(seg.profile), exc_info=True)
        return None


def reconstruct_segmentation_free(cleaned, *, residual: bool = True) -> ReconResult:
    """Reconstruct `cleaned` with NO segmentation: extrusion plan + residual
    loop. Returns a ReconResult with the solid and cad_vol_err."""
    from mesh2param.pipeline.extrusion import plan_extrusions
    from mesh2param.pipeline.export_cadquery import (
        _estimate_mesh_volume, _residual_refine_solid, _solid_volume,
    )
    mesh_vol = _estimate_mesh_volume(cleaned)
    res = ReconResult(mesh_volume=mesh_vol)
    try:
        import cadquery as cq
    except Exception:
        res.info["error"] = "cadquery unavailable"
        return res

    try:
        plan = plan_extrusions(cleaned)
    except Exception:
        logger.debug("plan_extrusions failed", exc_info=True)
        plan = None
    if plan is None or not plan.segments:
        res.info["error"] = "no extrusion segments"
        return res

    res.n_segments = len(plan.segments)
    res.coverage_frac = float(getattr(plan, "coverage_frac", 0.0))

    solid = None
    built = 0
    for seg in sorted(plan.segments, key=lambda s: -s.area):
        s = _segment_to_solid(seg, cq)
        if s is None:
            continue
        try:
            solid = s if solid is None else solid.fuse(s)
            built += 1
        except Exception:
            logger.debug("segment fuse failed", exc_info=True)
            continue
    res.n_built = built
    if solid is None:
        res.info["error"] = "no segment built"
        return res

    wp = cq.Workplane(obj=solid)
    if residual:
        try:
            wp, _ = _residual_refine_solid(wp, cleaned, mesh_vol)
        except Exception:
            logger.debug("residual refine failed in contour_recon",
                         exc_info=True)

    cad_vol = _solid_volume(wp)
    res.solid = wp
    res.cad_volume = cad_vol
    res.volume_error_pct = (abs(cad_vol - mesh_vol) / (mesh_vol + 1e-15) * 100
                            if mesh_vol > 0 else 100.0)
    res.status = "PASS" if res.volume_error_pct < 10.0 else "FAIL"
    return res
