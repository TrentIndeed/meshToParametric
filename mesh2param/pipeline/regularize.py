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


def _line_intersect_2d(p1, d1, p2, d2):
    """Intersection of lines p1+t·d1 and p2+s·d2 (2D), or None if parallel."""
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    diff = p2 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
    return p1 + t * d1


def _fit_circle_2d_local(pts):
    arr = np.asarray(pts, dtype=float)
    A = np.c_[2 * arr[:, 0], 2 * arr[:, 1], np.ones(len(arr))]
    sol, *_ = np.linalg.lstsq(A, (arr ** 2).sum(1), rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    return cx, cy, r


def _regularize_outline(pts_mm, axis_tol_deg: float = 12.0, min_run: float = 1.0):
    """Reduce a (possibly dense, noisy) 2D outline to axis-aligned straight
    runs + their corner intersections + per-corner fillet radii.

    Like feature_builder._regularize_profile but FILLET-OPTIONAL (sharp 90°
    corners are fine) and tolerant of dense sampling (Douglas-Peucker first, a
    generous axis tolerance). Returns (sharp_corner_pts, {idx: radius}) or
    (None, None) if it isn't an axis-aligned prismatic outline.
    """
    from shapely.geometry import Polygon
    P = np.asarray(pts_mm, dtype=float)
    try:
        s = Polygon(P).buffer(0).simplify(0.4, preserve_topology=True)
        if s.geom_type == "Polygon" and len(s.exterior.coords) >= 5:
            P = np.asarray(s.exterior.coords[:-1], dtype=float)
    except Exception:
        pass
    n = len(P)
    if n < 4:
        return None, None
    seg = np.roll(P, -1, axis=0) - P
    L = np.linalg.norm(seg, axis=1)
    d = seg / (L[:, None] + 1e-12)
    ct = np.cos(np.radians(axis_tol_deg))
    ax = []
    for i in range(n):
        if abs(d[i, 0]) > ct:
            ax.append((0, 1.0 if d[i, 0] >= 0 else -1.0))
        elif abs(d[i, 1]) > ct:
            ax.append((1, 1.0 if d[i, 1] >= 0 else -1.0))
        else:
            ax.append(None)
    straight = [ax[i] is not None and L[i] >= min_run for i in range(n)]
    if sum(straight) < 3:
        return None, None

    def _same(a, b):
        return a is not None and b is not None and a[0] == b[0] and a[1] == b[1]

    start = next((i for i in range(n) if straight[i]
                  and not (straight[(i - 1) % n] and _same(ax[(i - 1) % n], ax[i]))),
                 None)
    if start is None:
        return None, None
    runs, i, guard = [], start, 0
    while guard < 4 * n:
        guard += 1
        if straight[i]:
            a = ax[i]
            mem = [i]
            j = (i + 1) % n
            while guard < 4 * n and straight[j] and _same(ax[j], a):
                mem.append(j)
                j = (j + 1) % n
                guard += 1
            runs.append((a, mem))
            i = j
        else:
            i = (i + 1) % n
        if i == start:
            break
    # Merge consecutive runs on the SAME axis (one side split by a short
    # non-straight bit) so adjacent runs are always perpendicular — otherwise
    # two parallel runs have no corner intersection and the whole thing bails.
    merged = []
    for a, mem in runs:
        if merged and merged[-1][0][0] == a[0]:
            merged[-1] = (merged[-1][0], merged[-1][1] + mem)
        else:
            merged.append((a, mem))
    if len(merged) > 1 and merged[0][0][0] == merged[-1][0][0]:
        merged[0] = (merged[0][0], merged[-1][1] + merged[0][1])
        merged.pop()
    runs = merged
    if len(runs) < 3:
        return None, None
    run_lines = []
    for (axis, _sign), mem in runs:
        vidx = mem + [(mem[-1] + 1) % n]
        ctr = P[vidx].mean(axis=0)
        dvec = np.zeros(2)
        dvec[axis] = 1.0
        run_lines.append((ctr, dvec, mem))
    m = len(runs)
    sharp, fillets = [], {}
    for k in range(m):
        c1, d1, m1 = run_lines[k]
        c2, d2, m2 = run_lines[(k + 1) % m]
        X = _line_intersect_2d(c1, d1, c2, d2)
        if X is None:
            return None, None
        bend, j, g = [], (m1[-1] + 1) % n, 0
        while j != m2[0] and g < n:
            bend.append(j)
            j = (j + 1) % n
            g += 1
        if len(bend) >= 2:
            try:
                _, _, rr = _fit_circle_2d_local(P[bend])
                if 0.3 < rr < 60.0:
                    fillets[len(sharp)] = rr
            except np.linalg.LinAlgError:
                pass
        sharp.append((float(X[0]), float(X[1])))
    return (sharp, fillets) if len(sharp) >= 3 else (None, None)


def regularize_profile_2d(pts_mm: list, arc_samples: int = 8) -> list | None:
    """Stage F (sketch): clean a bumpy 2D outline into axis-aligned straight
    sides meeting at exact corners, with tangent arcs at the rounded ones.

    A scan / AI mesh's extruded base profile comes out wavy with corners that
    aren't quite 90 deg (user's "bumpy sketch / lines not 90 from a filleted
    corner"). This reuses `feature_builder._regularize_profile` (which snaps
    straight runs to their axis and finds the sharp corners + per-corner
    radii), then RECONSTRUCTS a point polygon: straight axis-aligned sides and,
    at each rounded corner, a tangent circular arc sampled to points. Returns
    the clean polygon, or None when the outline isn't a prismatic axis-aligned
    shape (caller keeps the original). Gate on scan/dirty meshes — clean CAD
    profiles are already exact.
    """
    # Reject genuinely ROUND / organic outlines: a circle (flange) sampled as
    # a polygon has near-axis segments only near its 4 cardinal points, so most
    # of the perimeter is NOT axis-aligned — regularizing it would turn a
    # circle into a polygon. Only proceed when axis-aligned sides dominate.
    P0 = np.asarray(pts_mm, dtype=float)
    if len(P0) >= 4:
        seg0 = np.roll(P0, -1, axis=0) - P0
        L0 = np.linalg.norm(seg0, axis=1)
        d0 = seg0 / (L0[:, None] + 1e-12)
        near_axis = np.abs(d0).max(1) > np.cos(np.radians(10.0))
        if L0.sum() > 0 and float(L0[near_axis].sum() / L0.sum()) < 0.6:
            return None

    sharp, fillets = _regularize_outline(pts_mm)
    if not sharp or len(sharp) < 3:
        return None
    S = np.asarray(sharp, dtype=float)
    n = len(S)
    side_len = np.linalg.norm(np.roll(S, -1, axis=0) - S, axis=1)
    out: list[list[float]] = []
    for i in range(n):
        cur = S[i]
        r = float((fillets or {}).get(i, 0.0))
        din = _unit(S[(i - 1) % n] - cur)     # toward previous side
        dout = _unit(S[(i + 1) % n] - cur)    # toward next side
        cosang = float(np.clip(din @ dout, -1.0, 1.0))
        half = np.arccos(cosang) / 2.0
        # tangent distance from the corner; cap so the arc fits both sides
        tdist = r / np.tan(half) if half > 1e-3 else 0.0
        max_t = 0.45 * min(side_len[(i - 1) % n], side_len[i])
        if r <= 0 or half <= 1e-3 or half >= np.pi / 2 - 1e-3 or tdist > max_t:
            out.append([float(cur[0]), float(cur[1])])
            continue
        t_in = cur + din * tdist
        t_out = cur + dout * tdist
        center = cur + _unit(din + dout) * (r / np.sin(half))
        a0 = np.arctan2(t_in[1] - center[1], t_in[0] - center[0])
        a1 = np.arctan2(t_out[1] - center[1], t_out[0] - center[0])
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi   # short way
        for s in range(arc_samples + 1):
            a = a0 + da * s / arc_samples
            out.append([float(center[0] + r * np.cos(a)),
                        float(center[1] + r * np.sin(a))])
    if len(out) < 3:
        return None

    # SELF-GATE: only return the regularized outline when it is STRICTLY
    # cleaner than the input AND preserves the shape. The mesh-level
    # regularization already flattens many base outlines (AI15's was 0.4 deg);
    # forcing a run/arc rebuild on an already-clean outline can make it WORSE.
    # So measure straight-side deviation from axis before/after and the area;
    # if it doesn't clearly improve, return None and the caller keeps the
    # original.
    def _straight_dev(p):
        Q = np.asarray(p, dtype=float)
        s = np.roll(Q, -1, axis=0) - Q
        ln = np.linalg.norm(s, axis=1)
        di = s / (ln[:, None] + 1e-12)
        ang = np.degrees(np.arccos(np.clip(np.abs(di).max(1), -1.0, 1.0)))
        lg = ln > 3.0
        return float(np.median(ang[lg])) if lg.any() else 0.0
    try:
        from shapely.geometry import Polygon
        a_in = abs(Polygon(pts_mm).buffer(0).area)
        a_out = abs(Polygon(out).buffer(0).area)
        if a_in > 0 and abs(a_out - a_in) / a_in > 0.03:
            return None
    except Exception:
        return None
    if _straight_dev(out) >= _straight_dev(pts_mm) - 0.05:
        return None
    return out
