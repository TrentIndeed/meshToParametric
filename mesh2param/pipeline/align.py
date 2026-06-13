"""Canonical-frame alignment for scans.

The clean-CAD pipeline assumes the part is already axis-aligned: fit.py
snaps surface normals to the WORLD axes, the extrusion engine reads
silhouettes on the world axes, and "prismatic" reasoning is all
world-relative. A scan arrives in an arbitrary pose, so every one of
those misfires — most visibly the wrench, whose flat-profile silhouette
gets read on the wrong axis and extruded past its true width.

This rotates the part onto its NATURAL orthogonal frame so all of that
becomes meaningful. Two strategies, best first:

  1. Dominant-normal triple — for parts with clean planar faces, find
     the largest mutually-orthogonal set of face-normal directions and
     map them to X/Y/Z. Exact for prismatic parts.
  2. Oriented bounding box — fall back to the minimum-volume box frame
     when there isn't a clean orthogonal normal triple (flat / partly
     organic parts like the wrench). Aligns the part to its own extents.

See docs/SCAN_RECOVERY_PLAN.md (stage B).
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def _candidate_axes(normals: np.ndarray, areas: np.ndarray,
                    n_top: int = 10) -> list[tuple[float, np.ndarray]]:
    """Area-weighted dominant face-normal directions, folded to a
    hemisphere (so ±n count as one axis). Returns [(area, unit dir)]
    sorted by area."""
    # Fold each normal so its largest-magnitude component is positive —
    # maps +axis and -axis to the same representative.
    idx = np.argmax(np.abs(normals), axis=1)
    sign = np.sign(normals[np.arange(len(normals)), idx])
    sign[sign == 0] = 1.0
    folded = normals * sign[:, None]
    # Coarse spherical binning by rounded direction.
    keys = np.round(folded * 12).astype(np.int64)
    acc: dict[tuple, list] = {}
    for k, a, n in zip(map(tuple, keys), areas, folded):
        slot = acc.get(k)
        if slot is None:
            acc[k] = [float(a), a * n]
        else:
            slot[0] += float(a)
            slot[1] = slot[1] + a * n
    out = []
    for area_sum, vsum in acc.values():
        nrm = float(np.linalg.norm(vsum))
        if nrm < 1e-12:
            continue
        out.append((area_sum, vsum / nrm))
    out.sort(key=lambda t: -t[0])
    return out[:n_top]


def _frame_from_normals(mesh: trimesh.Trimesh,
                        perp_tol_deg: float = 18.0,
                        min_area_frac: float = 0.04):
    """A right-handed orthonormal frame from the dominant orthogonal
    normal triple, or None if the part lacks one."""
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    if len(normals) < 8 or areas.sum() <= 0:
        return None
    total = float(areas.sum())
    axes = _candidate_axes(normals, areas)
    if not axes or axes[0][0] < min_area_frac * total:
        return None
    a1 = axes[0][1]
    perp_cos = np.sin(np.radians(perp_tol_deg))
    # axis 2: the largest-area candidate roughly perpendicular to a1
    a2 = None
    for area_sum, d in axes[1:]:
        if area_sum < min_area_frac * total:
            break
        if abs(float(d @ a1)) < perp_cos:
            a2 = d
            break
    if a2 is None:
        return None
    a2 = a2 - (a2 @ a1) * a1
    n2 = float(np.linalg.norm(a2))
    if n2 < 1e-6:
        return None
    a2 = a2 / n2
    a3 = np.cross(a1, a2)
    a3 = a3 / (np.linalg.norm(a3) + 1e-12)
    R = np.array([a1, a2, a3])  # rows = frame axes -> world X/Y/Z
    if np.linalg.det(R) < 0:     # keep right-handed
        R[2] = -R[2]
    return R


def _reorder_axes_by_extent(mesh: trimesh.Trimesh) -> np.ndarray:
    """Permutation (as a rotation/reflection-free signed perm) that puts
    the LARGEST extent on X and the SMALLEST on Z — an intuitive, stable
    pose (length along X, thickness up Z). Returns a 3x3 transform."""
    ext = mesh.bounds[1] - mesh.bounds[0]
    order = np.argsort(-ext)  # large -> small
    P = np.zeros((3, 3))
    for new_axis, old_axis in enumerate(order):
        P[new_axis, old_axis] = 1.0
    if np.linalg.det(P) < 0:  # avoid a reflection
        P[2] = -P[2]
    return P


def align_to_canonical_frame(mesh: trimesh.Trimesh):
    """Rotate `mesh` onto its natural orthogonal frame.

    Returns (aligned_mesh, transform_4x4, method) where applying
    transform to the input yields the aligned mesh. `method` is
    "normals" or "obb". Pure rotation/permutation — no scaling.
    """
    R = _frame_from_normals(mesh)
    method = "normals"
    if R is None:
        try:
            T, _extents = trimesh.bounds.oriented_bounds(mesh)
            R = np.asarray(T)[:3, :3]
            method = "obb"
        except Exception:
            logger.warning("OBB alignment failed; leaving pose unchanged")
            return mesh, np.eye(4), "none"

    T = np.eye(4)
    T[:3, :3] = R
    aligned = mesh.copy()
    aligned.apply_transform(T)

    # Stable, intuitive axis order (length->X, thickness->Z).
    P = _reorder_axes_by_extent(aligned)
    Tp = np.eye(4)
    Tp[:3, :3] = P
    aligned.apply_transform(Tp)

    full = Tp @ T
    logger.info("Canonical-frame alignment via %s; extents %s -> %s",
                method,
                np.round(mesh.bounds[1] - mesh.bounds[0], 1).tolist(),
                np.round(aligned.bounds[1] - aligned.bounds[0], 1).tolist())
    return aligned, full, method
