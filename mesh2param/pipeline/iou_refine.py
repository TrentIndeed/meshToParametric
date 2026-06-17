"""IoU-driven residual refinement — a post-pass that closes the gap between a
reconstructed solid and the target mesh.

This is the single highest-value idea borrowed from CADFit
(github.com/ghadinehme/CADFit, `run_pipeline.py`): rather than trusting the
feed-forward feature recognition, BUILD the result, compare it to the target,
and patch the difference. CADFit iterates `over -> cut, under -> union` until
the residual volume falls below tolerance, scoring every step by volumetric
IoU. This repo's recognition pipeline has no such global feedback, so a
mis-recognized feature silently propagates; this module gives it a safety net.

CADFit scores IoU with exact mesh booleans (manifold3d + alpha-wrap). manifold3d
isn't installed here, so we use VOXEL IoU on a shared grid — consistent with
this repo's existing voxel-occupancy machinery (pipeline/extrusion.py) and
dependency-free. The proposed patches are axis-aligned bounding boxes of the
largest residual chunks: an `add` box (under-residual the recon is missing) or
a `cut` box (over-residual the recon has in excess). A box is only accepted if
it measurably improves IoU — the same accept-if-it-helps discipline used in the
alignment gate.

This is a CONSERVATIVE first version: box patches + a greedy loop. Richer
patches (extrude the residual's actual silhouette, revolve, fillet) are the
natural extension, mirroring CADFit's per-residual re-analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoxelField:
    """A boolean occupancy grid on a shared world-space lattice."""

    occ: np.ndarray            # (nx,ny,nz) bool
    origin: np.ndarray         # world coords of cell (0,0,0) center
    pitch: float

    def cell_centers(self) -> np.ndarray:
        ix, iy, iz = np.indices(self.occ.shape)
        pts = np.stack([ix, iy, iz], axis=-1).reshape(-1, 3)
        return self.origin + pts * self.pitch


@dataclass
class RefineOp:
    """One accepted residual patch."""

    op: str                    # "add" | "cut"
    box_min: np.ndarray        # world AABB of the patch
    box_max: np.ndarray
    iou_before: float
    iou_after: float

    def to_dict(self) -> dict:
        return {"op": self.op,
                "box_min": np.asarray(self.box_min, float).tolist(),
                "box_max": np.asarray(self.box_max, float).tolist(),
                "iou_before": float(self.iou_before),
                "iou_after": float(self.iou_after)}


@dataclass
class RefineResult:
    iou_initial: float
    iou_final: float
    ops: list[RefineOp] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"iou_initial": float(self.iou_initial),
                "iou_final": float(self.iou_final),
                "ops": [o.to_dict() for o in self.ops]}


def _shared_lattice(meshes, res: int):
    """Common world AABB + pitch over all meshes. res = cells along the
    longest extent."""
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    ext = hi - lo
    pad = ext.max() * 0.04 + 1e-9        # margin so surfaces aren't clipped
    lo = lo - pad
    hi = hi + pad
    pitch = float((hi - lo).max()) / max(8, int(res))
    shape = np.maximum(1, np.ceil((hi - lo) / pitch).astype(int))
    origin = lo + 0.5 * pitch
    return origin, pitch, tuple(shape.tolist())


def _occupancy(mesh, origin, pitch, shape) -> np.ndarray:
    """Boolean occupancy of `mesh` on the shared lattice.

    Uses trimesh's `voxelized(pitch).fill()` (the same backend pipeline/
    extrusion.py relies on — no rtree/embree needed), then snaps the filled
    voxel centers onto the shared integer lattice so two meshes voxelized
    independently land on a common grid for IoU.
    """
    grid = np.zeros(shape, dtype=bool)
    try:
        vox = mesh.voxelized(pitch).fill()
        pts = np.asarray(vox.points, dtype=float)
    except Exception:
        logger.debug("voxelization failed for occupancy", exc_info=True)
        return grid
    if len(pts) == 0:
        return grid
    idx = np.floor((pts - origin) / pitch + 0.5).astype(int)
    valid = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx = idx[valid]
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    uni = int(np.count_nonzero(a | b))
    return inter / uni if uni else 0.0


def voxel_iou(target_mesh, recon_mesh, res: int = 64) -> float:
    """Volumetric IoU between two meshes via a shared voxel grid."""
    origin, pitch, shape = _shared_lattice([target_mesh, recon_mesh], res)
    t = _occupancy(target_mesh, origin, pitch, shape)
    r = _occupancy(recon_mesh, origin, pitch, shape)
    return _iou(t, r)


def _largest_component_bbox(mask: np.ndarray, origin, pitch,
                            min_cells: int):
    """World AABB of the largest connected True-component of `mask`, or None
    if no component is at least `min_cells` big."""
    if not mask.any():
        return None
    try:
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        if n == 0:
            return None
        sizes = ndimage.sum(np.ones_like(labels), labels,
                            index=np.arange(1, n + 1))
        biggest = int(np.argmax(sizes)) + 1
        if sizes[biggest - 1] < min_cells:
            return None
        comp = labels == biggest
    except Exception:
        comp = mask
        if int(np.count_nonzero(comp)) < min_cells:
            return None
    idx = np.argwhere(comp)
    lo_cell = idx.min(0)
    hi_cell = idx.max(0)
    box_min = origin + (lo_cell - 0.5) * pitch
    box_max = origin + (hi_cell + 0.5) * pitch
    return box_min, box_max


def _apply_box(occ: np.ndarray, origin, pitch, box_min, box_max,
               op: str) -> np.ndarray:
    """Set (add) or clear (cut) the grid cells whose centers fall in the box."""
    ix, iy, iz = np.indices(occ.shape)
    centers = origin + np.stack([ix, iy, iz], -1) * pitch
    inside = np.all((centers >= box_min) & (centers <= box_max), axis=-1)
    out = occ.copy()
    if op == "add":
        out[inside] = True
    else:
        out[inside] = False
    return out


def refine_against_target(target_mesh, recon_mesh, *,
                          res: int = 64,
                          max_iters: int = 4,
                          min_gain: float = 0.005,
                          min_residual_frac: float = 0.01) -> RefineResult:
    """Greedily patch `recon_mesh` toward `target_mesh` with add/cut boxes.

    Each iteration: pick the larger of the under-residual (target minus recon)
    and over-residual (recon minus target); take its largest connected
    component's bounding box; apply it in voxel space ONLY if IoU improves by
    at least `min_gain`. Stops when no patch helps, the residual is below
    `min_residual_frac` of the target volume, or `max_iters` is hit. Returns
    the accepted ops + IoU trajectory (all in voxel space; meshes are not
    mutated). A caller can replay `ops` as CadQuery box add/cut operations.
    """
    origin, pitch, shape = _shared_lattice([target_mesh, recon_mesh], res)
    t = _occupancy(target_mesh, origin, pitch, shape)
    r = _occupancy(recon_mesh, origin, pitch, shape)

    iou0 = _iou(t, r)
    t_count = max(1, int(np.count_nonzero(t)))
    min_cells = max(1, int(min_residual_frac * t_count))
    result = RefineResult(iou_initial=iou0, iou_final=iou0)

    for _ in range(max_iters):
        under = t & ~r       # recon is MISSING this -> add
        over = r & ~t        # recon has this in EXCESS -> cut
        cur_iou = _iou(t, r)

        # Try the larger residual first.
        candidates = sorted(
            [("add", under), ("cut", over)],
            key=lambda kv: -int(np.count_nonzero(kv[1])),
        )
        applied = False
        for op, mask in candidates:
            if int(np.count_nonzero(mask)) < min_cells:
                continue
            bbox = _largest_component_bbox(mask, origin, pitch, min_cells)
            if bbox is None:
                continue
            box_min, box_max = bbox
            r_try = _apply_box(r, origin, pitch, box_min, box_max, op)
            iou_try = _iou(t, r_try)
            if iou_try >= cur_iou + min_gain:
                r = r_try
                result.ops.append(RefineOp(op, box_min, box_max,
                                           cur_iou, iou_try))
                logger.info("Residual refine: %s box, IoU %.3f -> %.3f",
                            op, cur_iou, iou_try)
                applied = True
                break
        if not applied:
            break

    result.iou_final = _iou(t, r)
    return result
