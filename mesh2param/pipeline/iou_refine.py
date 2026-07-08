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
    """One accepted residual patch.

    kind:
      "box"      — axis-aligned box [box_min, box_max] (fallback / chunky bits).
      "prism"    — `polygon` (Nx2 world coords on the two non-`axis` axes)
                   extruded along `axis` from `lo` to `hi` (world). Tracks
                   L/stepped prismatic residuals cleanly.
      "cylinder" — disk of `radius` about `center` (2D world on the two
                   non-`axis` axes) extruded along `axis` from `lo` to `hi`.
                   The key candidate for round holes/bosses (the
                   segmentation-free path's main weakness, Phase 5).
    """

    op: str                    # "add" | "cut"
    box_min: np.ndarray = None
    box_max: np.ndarray = None
    iou_before: float = 0.0
    iou_after: float = 0.0
    kind: str = "box"
    axis: int = 2
    polygon: np.ndarray = None     # (N,2) world coords on the two non-axis axes
    center: np.ndarray = None      # (2,) world coords on the two non-axis axes
    radius: float = 0.0
    lo: float = 0.0                # world coord along `axis`
    hi: float = 0.0

    def to_dict(self) -> dict:
        d = {"op": self.op, "kind": self.kind,
             "iou_before": float(self.iou_before),
             "iou_after": float(self.iou_after)}
        if self.kind == "box":
            d["box_min"] = np.asarray(self.box_min, float).tolist()
            d["box_max"] = np.asarray(self.box_max, float).tolist()
        else:
            d.update(axis=int(self.axis), lo=float(self.lo), hi=float(self.hi))
            if self.kind == "prism":
                d["polygon"] = np.asarray(self.polygon, float).tolist()
            else:
                d["center"] = np.asarray(self.center, float).tolist()
                d["radius"] = float(self.radius)
        return d


@dataclass
class RefineResult:
    iou_initial: float
    iou_final: float
    ops: list[RefineOp] = field(default_factory=list)
    # Box-only fallback op-set. The voxel-optimal op-set (`ops`, which may use
    # prisms/cylinders) is not always the REAL-volume-optimal one once built in
    # CadQuery (voxel quantization + boolean interactions). The export builder
    # builds both and keeps whichever has the lower real volume error, so a
    # shape op can never regress a part below its box-only result.
    box_ops: list[RefineOp] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"iou_initial": float(self.iou_initial),
                "iou_final": float(self.iou_final),
                "ops": [o.to_dict() for o in self.ops],
                "box_ops": [o.to_dict() for o in self.box_ops]}


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


def _alpha_wrap(mesh):
    """Guaranteed-watertight wrap of any mesh via pymeshlab's alpha-wrap
    (decimating huge meshes first for speed). Returns a trimesh, or None if the
    alpha-wrap plugin is unavailable (e.g. on Windows) / fails. This is what
    makes exact IoU reliable on dirty AI/scan meshes — the same trick CADFit's
    iou.py uses."""
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, np.float64),
            face_matrix=np.asarray(mesh.faces, np.int32)))
        if len(mesh.faces) > 50_000:
            try:
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=50_000, preservenormal=True)
            except Exception:
                pass
        ms.generate_alpha_wrap(alpha=pymeshlab.PercentageValue(1.0),
                               offset=pymeshlab.PercentageValue(0.1))
        out = ms.current_mesh()
        import trimesh
        return trimesh.Trimesh(vertices=out.vertex_matrix(),
                               faces=out.face_matrix(), process=True)
    except Exception:
        return None


def _to_manifold(mesh):
    """trimesh -> manifold3d.Manifold, or None. Tries the direct conversion
    first (fast, no distortion); only when that gives an empty/volume-
    inconsistent manifold (non-watertight / multi-body) does it fall back to an
    alpha-wrap. Keeps exact IoU honest on dirty meshes where direct conversion
    silently under-/over-volumes."""
    import manifold3d

    def _conv(m):
        try:
            mm = manifold3d.Mesh(
                vert_properties=np.asarray(m.vertices, np.float32),
                tri_verts=np.asarray(m.faces, np.uint32))
            man = manifold3d.Manifold(mm)
            return man if man.volume() > 1e-12 else None
        except Exception:
            return None

    direct = _conv(mesh)
    if direct is not None:
        try:
            raw = abs(float(mesh.volume))
            if raw > 0 and 0.9 <= direct.volume() / raw <= 1.1:
                return direct        # volume-consistent -> trust direct
        except Exception:
            return direct
    wrapped = _alpha_wrap(mesh)
    if wrapped is not None:
        w = _conv(wrapped)
        if w is not None:
            # Same volume-consistency check the direct path gets: the
            # alpha-wrap of a seam-cracked (T-junction) tessellation can
            # distort badly, and feeding that into the boolean silently
            # reports IoU≈0 for a perfectly sound solid (the pillow
            # block's stacked-band tess read 0.034 at +9% volume).
            # Inconsistent wrap -> None -> callers voxel-fallback.
            try:
                raw = abs(float(mesh.volume))
                if raw > 0 and not (0.85 <= w.volume() / raw <= 1.15):
                    return None
            except Exception:
                pass
            return w
    return direct


_EXACT_AVAILABLE = None


def exact_iou_available() -> bool:
    """True iff exact-boolean IoU is RELIABLE here — manifold3d present AND
    alpha-wrap works (so dirty/non-watertight meshes can be manifoldified).
    On Windows the alpha-wrap plugin doesn't load, so this is False and callers
    use the volume metric instead of optimizing a voxel-fallback signal. On
    Linux (the deployment target) it's True. Cached."""
    global _EXACT_AVAILABLE
    if _EXACT_AVAILABLE is None:
        try:
            import manifold3d  # noqa: F401
            import trimesh
            box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
            _EXACT_AVAILABLE = _alpha_wrap(box) is not None
        except Exception:
            _EXACT_AVAILABLE = False
    return bool(_EXACT_AVAILABLE)


def mesh_iou_exact(target_mesh, recon_mesh, res: int = 64) -> float:
    """Exact volumetric IoU via manifold3d booleans — SHAPE-AWARE and sharper
    than voxel IoU (no quantization). Used as the arbiter for keep/route
    decisions, where 'right volume, wrong shape' must be caught (voxel IoU and
    volume error both miss it at coarse resolution).

    Falls back to voxel_iou when manifold3d is unavailable, a mesh isn't a clean
    manifold, or the boolean volumes are inconsistent (manifold3d can return
    vol(A&B) > vol(A|B) on multi-body/self-intersecting meshes — its own docs
    note this; we detect IoU>1 and fall back).
    """
    try:
        import manifold3d
    except Exception:
        return voxel_iou(target_mesh, recon_mesh, res)
    ma, mb = _to_manifold(target_mesh), _to_manifold(recon_mesh)
    if ma is None or mb is None:
        return voxel_iou(target_mesh, recon_mesh, res)
    try:
        inter = manifold3d.Manifold.batch_boolean(
            [ma, mb], manifold3d.OpType.Intersect)
        uni = manifold3d.Manifold.batch_boolean(
            [ma, mb], manifold3d.OpType.Add)
        vi, vu = abs(inter.volume()), abs(uni.volume())
    except Exception:
        return voxel_iou(target_mesh, recon_mesh, res)
    if vu <= 1e-12:
        return voxel_iou(target_mesh, recon_mesh, res)
    iou = vi / vu
    if iou > 1.001 or iou < 0:        # manifold volume bug -> voxel fallback
        return voxel_iou(target_mesh, recon_mesh, res)
    return min(iou, 1.0)


def _box_cover(mask: np.ndarray, origin, pitch, min_cells: int,
               max_boxes: int = 10, min_fill: float = 0.6):
    """A set of world-space AABBs covering the True voxels of `mask` without
    much overshoot.

    A single bounding box of a non-box residual (L-shaped, stepped,
    distributed) overshoots badly — it adds/cuts volume that isn't in the
    residual, so the volume gate rejects it (this is exactly why AI11/AI13
    didn't improve with the v1 single-box patch). Instead we recursively
    subdivide: take a region's tight filled bbox; if it is densely filled
    (>= min_fill) or tiny, emit it; otherwise split along its longest axis at
    the median of filled cells and recurse. Disconnected chunks fall out of the
    recursion naturally. Capped at `max_boxes` so a pathological residual can't
    explode the op list.
    """
    idx0 = np.argwhere(mask)
    if len(idx0) < min_cells:
        return []
    boxes = []
    stack = [(idx0.min(0), idx0.max(0))]
    shape = np.array(mask.shape)
    while stack and len(boxes) < max_boxes:
        lo, hi = stack.pop()
        sub = mask[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
        local = np.argwhere(sub)
        if len(local) == 0:
            continue
        tlo = lo + local.min(0)
        thi = lo + local.max(0)
        cnt = len(local)
        dims = thi - tlo + 1
        vol = int(np.prod(dims))
        fill = cnt / max(1, vol)
        if (fill >= min_fill or vol <= 8 or cnt < min_cells
                or len(boxes) >= max_boxes - 1):
            box_min = origin + (tlo - 0.5) * pitch
            box_max = origin + (thi + 0.5) * pitch
            boxes.append((box_min, box_max))
            continue
        # Split along the longest axis at the median filled coordinate.
        ax = int(np.argmax(dims))
        coord = (local[:, ax] + lo[ax])
        mid = int(np.median(coord))
        if mid <= tlo[ax] or mid >= thi[ax]:
            box_min = origin + (tlo - 0.5) * pitch
            box_max = origin + (thi + 0.5) * pitch
            boxes.append((box_min, box_max))
            continue
        hi1 = thi.copy(); hi1[ax] = mid
        lo2 = tlo.copy(); lo2[ax] = mid + 1
        stack.append((tlo.copy(), np.minimum(hi1, shape - 1)))
        stack.append((lo2, thi.copy()))
    return boxes


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


def _largest_component(mask: np.ndarray, min_cells: int):
    """Largest connected True-component of `mask` (or None if all < min_cells)."""
    if not mask.any():
        return None
    try:
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        if n == 0:
            return None
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        biggest = int(sizes.argmax())
        if sizes[biggest] < min_cells:
            return None
        return labels == biggest
    except Exception:
        return mask if int(mask.sum()) >= min_cells else None


def _best_axis_footprint(comp: np.ndarray):
    """Pick the axis along which `comp` is most prismatic. Returns
    (axis, foot2d, lo_cell, hi_cell, recon_mask, axis_iou) — recon_mask is the
    footprint re-extruded over the axis extent (how a prism would fill it)."""
    best = None
    for a in range(3):
        foot = comp.any(axis=a)
        idx_a = np.argwhere(comp)[:, a]
        lo_c, hi_c = int(idx_a.min()), int(idx_a.max())
        recon = np.zeros_like(comp)
        m = np.moveaxis(recon, a, 0)
        m[lo_c:hi_c + 1] = foot           # view -> writes back into recon
        iou = (int((recon & comp).sum())
               / max(1, int((recon | comp).sum())))
        if best is None or iou > best[-1]:
            best = (a, foot, lo_c, hi_c, recon, iou)
    return best


def _footprint_polygon(foot: np.ndarray):
    """Largest boundary contour of a 2D footprint as (N,2) cell coords, or None."""
    try:
        from skimage import measure
        contours = measure.find_contours(foot.astype(float), 0.5)
        if not contours:
            return None
        poly = max(contours, key=len)        # (N,2) in (row,col) float
        if len(poly) < 4:
            return None
        # Decimate to keep the CadQuery wire light.
        step = max(1, len(poly) // 64)
        return poly[::step]
    except Exception:
        return None


def _shape_op(comp: np.ndarray, origin, pitch, op: str):
    """Turn a residual component into a clean prism/cylinder RefineOp +
    its voxel recon_mask, or (None, None) to fall back to a box-cover."""
    axis, foot, lo_c, hi_c, recon, axis_iou = _best_axis_footprint(comp)
    if axis_iou < 0.75:
        return None, None        # not prismatic along any axis -> use boxes
    uv = tuple(i for i in range(3) if i != axis)
    lo_w = float(origin[axis] + (lo_c - 0.5) * pitch)
    hi_w = float(origin[axis] + (hi_c + 0.5) * pitch)

    def _foot_rc_to_world(rc):
        # foot index (i along uv[0], j along uv[1]) -> world (u,v)
        return np.column_stack([origin[uv[0]] + rc[:, 0] * pitch,
                                origin[uv[1]] + rc[:, 1] * pitch])

    poly_rc = _footprint_polygon(foot)
    if poly_rc is None:
        return None, None
    poly_w = _foot_rc_to_world(poly_rc)

    def _extrude_foot2d(foot2d):
        """Re-extrude a 2D footprint mask over the axis extent into a 3D mask."""
        out = np.zeros_like(comp)
        m = np.moveaxis(out, axis, 0)
        m[lo_c:hi_c + 1] = foot2d
        return out

    # Circular? Fit a circle to the boundary (resolution-robust, unlike a
    # disk-area test): small radial RMS relative to the radius => a round
    # hole/boss => emit an exact CYLINDER (cleaner than a polygonal prism).
    try:
        from mesh2param.pipeline.contour_fit import fit_circle_2d
        c2, r2, rms = fit_circle_2d(poly_w)
        if np.isfinite(r2) and r2 > 1.5 * pitch and rms < 0.12 * r2:
            # Rasterize the ACTUAL disk (not the component) so the voxel
            # acceptance reflects what export will build.
            rr, cc = np.indices(foot.shape)
            cen = ((c2[0] - origin[uv[0]]) / pitch,
                   (c2[1] - origin[uv[1]]) / pitch)
            disk = ((rr - cen[0]) ** 2 + (cc - cen[1]) ** 2) <= (r2 / pitch) ** 2
            opd = RefineOp(op=op, kind="cylinder", axis=axis,
                           center=np.asarray(c2, float), radius=float(r2),
                           lo=lo_w, hi=hi_w)
            return opd, _extrude_foot2d(disk)
    except Exception:
        pass

    # Prism: rasterize the ACTUAL (decimated) polygon as the acceptance mask.
    try:
        from skimage.draw import polygon as _sk_polygon
        fill = np.zeros_like(foot)
        rr, cc = _sk_polygon(poly_rc[:, 0], poly_rc[:, 1], shape=foot.shape)
        fill[rr, cc] = True
        recon_actual = _extrude_foot2d(fill)
    except Exception:
        recon_actual = recon          # fall back to the component footprint
    opd = RefineOp(op=op, kind="prism", axis=axis, polygon=poly_w,
                   lo=lo_w, hi=hi_w)
    return opd, recon_actual


def _greedy_box_apply(t, r, mask, op, origin, pitch, min_cells, start_iou):
    """Apply a box-cover of `mask` to a COPY of r, keeping only boxes that
    strictly improve IoU. Returns (r_new, ops, iou)."""
    rr = r
    cur = start_iou
    ops: list[RefineOp] = []
    for box_min, box_max in _box_cover(mask, origin, pitch, min_cells):
        r_try = _apply_box(rr, origin, pitch, box_min, box_max, op)
        iou_try = _iou(t, r_try)
        if iou_try >= cur + 1e-4:
            rr = r_try
            ops.append(RefineOp(op, box_min=box_min, box_max=box_max,
                                iou_before=cur, iou_after=iou_try))
            cur = iou_try
    return rr, ops, cur


def refine_against_target(target_mesh, recon_mesh, *,
                          res: int = 64,
                          max_iters: int = 4,
                          min_gain: float = 0.005,
                          min_residual_frac: float = 0.01) -> RefineResult:
    """Greedily patch `recon_mesh` toward `target_mesh` with add/cut boxes.

    Each iteration: pick the larger of the under-residual (target minus recon)
    and over-residual (recon minus target); cover it with a set of tight boxes
    (`_box_cover`); apply each box in voxel space, keeping only those that
    strictly improve IoU. Stops when no box helps, the residual is below
    `min_residual_frac` of the target volume, or `max_iters` is hit. Returns
    the accepted ops + IoU trajectory (all in voxel space; meshes are not
    mutated). A caller replays `ops` as CadQuery box add/cut operations.

    `min_gain` gates whether a residual is worth attacking at all; within a
    cover, each box only needs to strictly improve IoU (small cover boxes each
    contribute little but together track a non-box residual far better than one
    overshooting bbox).
    """
    origin, pitch, shape = _shared_lattice([target_mesh, recon_mesh], res)
    t = _occupancy(target_mesh, origin, pitch, shape)
    r0 = _occupancy(recon_mesh, origin, pitch, shape)

    iou0 = _iou(t, r0)
    t_count = max(1, int(np.count_nonzero(t)))
    min_cells = max(1, int(min_residual_frac * t_count))

    def _run_loop(use_shapes: bool):
        r = r0.copy()
        ops: list[RefineOp] = []
        for _ in range(max_iters):
            under = t & ~r       # recon is MISSING this -> add
            over = r & ~t        # recon has this in EXCESS -> cut
            cur_iou = _iou(t, r)
            candidates = sorted(
                [("add", under), ("cut", over)],
                key=lambda kv: -int(np.count_nonzero(kv[1])),
            )
            applied = False
            for op, mask in candidates:
                if int(np.count_nonzero(mask)) < min_cells:
                    continue
                # Path 1 (shapes only): largest component as ONE prism/cylinder
                # (round holes/bosses cut cleanly) + box-cover the leftover.
                p1_r, p1_ops, p1_iou = r, [], cur_iou
                if use_shapes:
                    comp = _largest_component(mask, min_cells)
                    if comp is not None:
                        shape_op, recon = _shape_op(comp, origin, pitch, op)
                        if shape_op is not None and recon is not None:
                            r1 = (r | recon) if op == "add" else (r & ~recon)
                            iou1 = _iou(t, r1)
                            if iou1 >= cur_iou + 1e-4:
                                shape_op.iou_before = cur_iou
                                shape_op.iou_after = iou1
                                resid1 = ((t & ~r1) if op == "add"
                                          else (r1 & ~t))
                                r1b, ops1b, iou1b = _greedy_box_apply(
                                    t, r1, resid1, op, origin, pitch,
                                    min_cells, iou1)
                                p1_r, p1_ops, p1_iou = (
                                    r1b, [shape_op] + ops1b, iou1b)
                # Path 2: box-cover of the whole residual.
                p2_r, p2_ops, p2_iou = _greedy_box_apply(
                    t, r, mask, op, origin, pitch, min_cells, cur_iou)

                if p1_iou >= p2_iou and p1_iou > cur_iou + 1e-4:
                    r, cur_iou = p1_r, p1_iou
                    ops.extend(p1_ops)
                    applied = True
                elif p2_iou > cur_iou + 1e-4:
                    r, cur_iou = p2_r, p2_iou
                    ops.extend(p2_ops)
                    applied = True
                if applied:
                    break
            if not applied:
                break
        return ops, _iou(t, r)

    ops_shape, iou_shape = _run_loop(use_shapes=True)
    ops_box, iou_box = _run_loop(use_shapes=False)
    # `ops` = the voxel-best; the export builder also tries `box_ops` and keeps
    # whichever wins on REAL volume error (voxel IoU doesn't always predict it).
    if iou_shape >= iou_box:
        return RefineResult(iou_initial=iou0, iou_final=iou_shape,
                            ops=ops_shape, box_ops=ops_box)
    return RefineResult(iou_initial=iou0, iou_final=iou_box,
                        ops=ops_box, box_ops=ops_box)
