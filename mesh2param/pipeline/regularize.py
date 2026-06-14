"""Stage E (orientation): global plane regularization.

Per-primitive snapping (`fit._snap_params`) snaps each plane normal to the
nearest WORLD axis independently. That misses the relational intent: the
faces of a real part are mutually parallel / perpendicular, and a scan or
AI-generated mesh fits them a few degrees off — so "90 deg" corners come out
at 87 or 92. This pass recovers the design intent:

  1. Build the part's OWN orthogonal frame from its dominant plane normals
     (area-weighted), instead of assuming the world axes.
  2. Snap every plane normal onto that frame AS A SET, so all snapped planes
     are exactly parallel / perpendicular to each other (crisp 90 deg).
  3. (offsets) Merge near-coincident parallel planes to a shared offset and
     round offsets — coplanar faces become exactly coplanar.

Runs after fitting, on scan / dirty-mesh inputs (clean CAD is already exact,
so it's left untouched by the caller's gate). Returns the (mutated) fitted
list plus a report of what moved — for confidence scoring and the cleanup
visualization.
"""
from __future__ import annotations

import logging

import numpy as np

from mesh2param.models.surface import FittedSurface, SurfaceType

logger = logging.getLogger(__name__)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _consensus_frame(normals: np.ndarray, weights: np.ndarray,
                     family_deg: float = 15.0) -> np.ndarray:
    """Dominant mutually-orthogonal normal triple (rows e1,e2,e3).

    Normals are grouped into axis families by absolute alignment (a plane
    and its back face share an axis); the heaviest family is e1, the
    heaviest family roughly perpendicular to it gives e2, e3 = e1 x e2.
    """
    ct = np.cos(np.radians(family_deg))
    used = np.zeros(len(normals), dtype=bool)
    fams: list[tuple[float, np.ndarray]] = []
    for i in np.argsort(-weights):
        if used[i]:
            continue
        sim = np.abs(normals @ normals[i]) > ct
        signs = np.sign(normals[sim] @ normals[i])
        signs[signs == 0] = 1.0
        mean = _unit((weights[sim, None] * signs[:, None] * normals[sim]).sum(0))
        fams.append((float(weights[sim].sum()), mean))
        used |= sim
    if not fams:
        return np.eye(3)
    fams.sort(key=lambda x: -x[0])
    e1 = fams[0][1]
    e2 = None
    for _, d in fams[1:]:
        perp = d - (d @ e1) * e1
        if np.linalg.norm(perp) > np.sin(np.radians(30)):  # reasonably perp
            e2 = _unit(perp)
            break
    if e2 is None:
        t = np.array([1.0, 0, 0]) if abs(e1[0]) < 0.9 else np.array([0, 1.0, 0])
        e2 = _unit(t - (t @ e1) * e1)
    e3 = _unit(np.cross(e1, e2))
    return _snap_frame_to_world(np.array([e1, e2, e3]))


def _snap_frame_to_world(frame: np.ndarray, deg: float = 12.0) -> np.ndarray:
    """If the consensus frame sits within `deg` of the world axes (it should,
    after canonical-frame alignment), snap it exactly onto them — so the
    regularized faces come out axis-aligned, not tilted by the residual fit
    bias. A part that's genuinely off-axis keeps its own frame."""
    ct = np.cos(np.radians(deg))
    out = frame.copy()
    for i in range(3):
        for ax in range(3):
            e = np.zeros(3)
            e[ax] = 1.0
            if abs(float(frame[i] @ e)) > ct:
                out[i] = e * (1.0 if frame[i, ax] >= 0 else -1.0)
                break
    e1 = _unit(out[0])
    e2 = _unit(out[1] - (out[1] @ e1) * e1)
    e3 = _unit(np.cross(e1, e2))
    return np.array([e1, e2, e3])


def regularize_planes(fitted: list[FittedSurface],
                      weights: dict[int, float] | None = None,
                      snap_angle_deg: float = 25.0,
                      offset_merge: float = 0.0) -> tuple[list[FittedSurface], list[dict]]:
    """Snap plane normals onto the part's consensus orthogonal frame.

    weights: optional {region_id: area-or-face-count} so big faces define the
    frame. snap_angle_deg: only snap a normal already within this of a frame
    axis (genuinely oblique faces — drafts, chamfers — are left alone).
    offset_merge: if > 0, parallel planes whose offsets differ by less than
    this are merged to a shared (mean) offset (coplanar faces). Mutates and
    returns `fitted`; also returns a per-plane report.
    """
    planes = [s for s in fitted if s.surface_type == SurfaceType.PLANE]
    report: list[dict] = []
    if len(planes) < 2:
        return fitted, report

    def _normal_of(s: FittedSurface) -> np.ndarray:
        return _unit((s.snapped_params or {}).get("normal")
                     or s.params.get("normal", [0, 0, 1.0]))

    def _point_of(s: FittedSurface) -> np.ndarray:
        return np.asarray((s.snapped_params or {}).get("point")
                          or s.params.get("point", [0, 0, 0.0]), dtype=float)

    N = np.array([_normal_of(s) for s in planes])
    W = np.array([float((weights or {}).get(s.region_id, 1.0)) for s in planes])
    frame = _consensus_frame(N, W)
    axes = np.vstack([frame, -frame])  # 6 signed candidate axes

    snapped_axis: dict[int, int] = {}   # plane idx -> frame axis index (0..2)
    for pi, s in enumerate(planes):
        n = N[pi]
        dots = axes @ n
        k = int(np.argmax(dots))
        ang = float(np.degrees(np.arccos(np.clip(dots[k], -1.0, 1.0))))
        if ang <= snap_angle_deg:
            new_n = axes[k]
            sp = dict(s.snapped_params or {})
            sp["normal"] = new_n.tolist()
            # keep the plane through its measured centroid, just reoriented
            sp.setdefault("point", _point_of(s).tolist())
            s.snapped_params = sp
            snapped_axis[pi] = k % 3
            report.append({"region_id": s.region_id, "type": "plane",
                           "angle_moved": round(ang, 2), "snapped": True})
        else:
            report.append({"region_id": s.region_id, "type": "plane",
                           "angle_moved": round(ang, 2), "snapped": False})

    # Offset consensus: parallel snapped planes (same frame axis + sign) whose
    # offsets are within offset_merge collapse to one shared plane.
    if offset_merge > 0:
        groups: dict[tuple, list[int]] = {}
        for pi in snapped_axis:
            n = np.asarray(planes[pi].snapped_params["normal"], dtype=float)
            key = tuple(np.round(n, 3))
            groups.setdefault(key, []).append(pi)
        for key, idxs in groups.items():
            n = np.asarray(key, dtype=float)
            offs = np.array([_point_of(planes[pi]) @ n for pi in idxs])
            order = np.argsort(offs)
            cluster_start = 0
            for j in range(1, len(order) + 1):
                if (j == len(order)
                        or offs[order[j]] - offs[order[cluster_start]] > offset_merge):
                    members = order[cluster_start:j]
                    if len(members) > 1:
                        shared = float(offs[members].mean())
                        for m in members:
                            pi = idxs[m]
                            planes[pi].snapped_params["point"] = (shared * n).tolist()
                    cluster_start = j

    n_snap = sum(1 for r in report if r["snapped"])
    logger.info("Plane regularization: %d/%d planes snapped to consensus "
                "frame (max move %.1f deg)", n_snap, len(planes),
                max((r["angle_moved"] for r in report if r["snapped"]),
                    default=0.0))
    return fitted, report


def regularize_mesh(cleaned, regions, fitted, snap_angle_deg: float = 25.0,
                    max_move_frac: float = 0.05):
    """Project a dirty mesh's vertices onto its recovered consensus planes.

    Snapping the fitted PRIMITIVES alone is inert — recognition reads the raw
    mesh geometry (the extrusion engine reads vertices, not plane params). So
    this snaps each planar region's normal onto the part's consensus frame
    and then moves every vertex onto the intersection of the regularized
    planes of the regions it touches: interior face vertices flatten onto
    their plane, shared-edge vertices land on the exact edge line, corners on
    the exact corner. The result is a mesh whose flat faces are actually flat
    and whose corners are actually 90 deg — so the EXISTING recognition then
    produces a crisp model. Vertices touching any non-planar or un-snapped
    region are left alone (protects fillets / cylinders). Mutates `cleaned`.
    """
    import trimesh

    V = np.asarray(cleaned.vertices, dtype=float).copy()
    F = np.asarray(cleaned.faces)
    bbox = np.asarray(cleaned.bbox, dtype=float)
    bbox_diag = float(np.linalg.norm(bbox[1] - bbox[0])) or 1.0

    fit_by_region = {f.region_id: f for f in fitted}
    plane_regs = [r for r in regions
                  if r.surface_type == SurfaceType.PLANE
                  and r.region_id in fit_by_region]
    if len(plane_regs) < 2:
        return cleaned, []

    normals = np.array([_normal_of_fit(fit_by_region[r.region_id])
                        for r in plane_regs])
    weights = np.array([float(len(r.face_indices)) for r in plane_regs])
    frame = _consensus_frame(normals, weights)
    axes = np.vstack([frame, -frame])

    reg_plane: dict[int, tuple[np.ndarray, float]] = {}
    report: list[dict] = []
    for r, n in zip(plane_regs, normals):
        dots = axes @ n
        k = int(np.argmax(dots))
        ang = float(np.degrees(np.arccos(np.clip(dots[k], -1.0, 1.0))))
        snapped = ang <= snap_angle_deg
        if snapped:
            nn = axes[k]
            cen = np.asarray(r.vertices, dtype=float).mean(0)
            reg_plane[r.region_id] = (nn, float(nn @ cen))
        report.append({"region_id": r.region_id, "type": "plane",
                       "angle_moved": round(ang, 2), "snapped": snapped})

    # face -> its region id (segmentation assigns each face to one region)
    face_region = np.full(len(F), -1, dtype=np.int64)
    for r in regions:
        face_region[np.asarray(r.face_indices, dtype=np.int64)] = r.region_id

    # vertex -> incident faces
    vfaces: list[list[int]] = [[] for _ in range(len(V))]
    for fi, tri in enumerate(F):
        for v in tri:
            vfaces[v].append(fi)

    moved = 0
    for v in range(len(V)):
        rids = {int(face_region[fi]) for fi in vfaces[v]}
        # eligible only if EVERY incident region is a planar, snapped plane
        if not rids or any(rid not in reg_plane for rid in rids):
            continue
        Ns = np.array([reg_plane[rid][0] for rid in rids])
        ds = np.array([reg_plane[rid][1] for rid in rids])
        try:
            delta = -np.linalg.pinv(Ns) @ (Ns @ V[v] - ds)
        except Exception:
            continue
        if np.linalg.norm(delta) <= max_move_frac * bbox_diag:
            V[v] = V[v] + delta
            moved += 1

    cleaned.vertices = V
    try:
        cleaned.normals = np.asarray(
            trimesh.Trimesh(V, F, process=False).face_normals, dtype=float)
        cleaned.bbox = np.array([V.min(0), V.max(0)], dtype=float)
    except Exception:
        pass
    logger.info("Mesh regularization: %d/%d plane regions snapped, %d "
                "vertices projected onto consensus planes",
                len(reg_plane), len(plane_regs), moved)
    return cleaned, report


def _normal_of_fit(fp) -> np.ndarray:
    return _unit((fp.snapped_params or {}).get("normal")
                 or fp.params.get("normal", [0, 0, 1.0]))
