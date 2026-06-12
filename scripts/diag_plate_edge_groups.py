"""Diagnostic: faithful replica of the PRODUCTION stitched grouping
(job_queue.py) on RackMount single-body, with per-union logging.

Answers: which pass merges the left side plate's inner and outer outline
rings into one group (12 groups in the contact-only mirror vs 11 in
production)?

Usage:
    python scripts/diag_plate_edge_groups.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh2param.config import (
    FitConfig, IngestConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

DIRECT_EPS = 2.0
AXIS_DOT_MIN = 0.5


def bbox_str(v):
    lo, hi = v.min(axis=0), v.max(axis=0)
    return (f"({lo[0]:.0f}..{hi[0]:.0f},{lo[1]:.0f}..{hi[1]:.0f},"
            f"{lo[2]:.0f}..{hi[2]:.0f})")


def main() -> int:
    cleaned = ingest_mesh(
        str(ROOT / "models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False),
    )
    sub = cleaned
    regions = segment_mesh(sub, SegmentConfig())
    fitted = fit_surfaces(regions, sub, FitConfig())
    _graph, edges = build_topology(regions, fitted, sub, TopologyConfig())
    topo_blends = identify_blends_from_topology(regions, fitted, edges, sub)

    # ---- record split + records_info (mirrors job_queue lines ~1175) ----
    records_info: list[dict] = []
    pending = []
    for rec in topo_blends:
        if rec.get("kind") != "fillet":
            continue
        fi = rec.get("face_indices")
        if fi is None or len(fi) == 0:
            continue
        fi_arr = np.asarray(fi, dtype=np.int64)
        verts = sub.vertices[sub.faces[fi_arr]].reshape(-1, 3)
        halves = [(rec, fi_arr, verts)]
        dims = verts.max(axis=0) - verts.min(axis=0)
        thin, best_gap = -1, 0.0
        vt = gaps = None
        gi = 0
        frac = 0.0
        if len(verts) >= 24:  # single-body path
            for ax in range(3):
                if float(dims[ax]) > 12.0:
                    continue
                vt_a = np.sort(verts[:, ax])
                gaps_a = np.diff(vt_a)
                gi_a = int(np.argmax(gaps_a))
                if float(gaps_a[gi_a]) > best_gap:
                    best_gap = float(gaps_a[gi_a])
                    thin = ax
                    vt, gaps, gi = vt_a, gaps_a, gi_a
                    frac = (gi_a + 1) / len(vt_a)
        if thin >= 0:
            rec_r = float(rec.get("radius", 0.0) or 0.0)
            void_split = float(gaps[gi]) >= 1.5 and (
                0.2 < frac < 0.8 or float(gaps[gi]) >= 2.5
            )
            fused_split = (
                not void_split and rec_r > 0
                and float(dims[thin]) >= max(5.0, 1.6 * 1.5708 * rec_r)
            )
            if void_split or fused_split:
                cent = sub.vertices[sub.faces[fi_arr]].mean(axis=1)[:, thin]
                if void_split:
                    lo_b = float(vt[gi]) + 1e-6
                    hi_b = float(vt[gi + 1]) - 1e-6
                    fA = fi_arr[cent <= lo_b]
                    fB = fi_arr[cent >= hi_b]
                else:
                    mid = 0.5 * (float(vt[0]) + float(vt[-1]))
                    fA = fi_arr[cent < mid - 1.2]
                    fB = fi_arr[cent > mid + 1.2]
                if len(fA) >= 3 and len(fB) < 3:
                    rec["face_indices"] = fA.tolist()
                    fB = fB[:0]
                elif len(fB) >= 3 and len(fA) < 3:
                    rec["face_indices"] = fB.tolist()
                    fA = fA[:0]
                if len(fA) >= 3 and len(fB) >= 3:
                    rec_b = dict(rec)
                    rec_b["face_indices"] = fB.tolist()
                    rec["face_indices"] = fA.tolist()
                    tag = id(rec)
                    rec["_split_parent"] = tag
                    rec["_split_side"] = 0
                    rec_b["_split_parent"] = tag
                    rec_b["_split_side"] = 1
                    pending.append(rec_b)
                    vA = sub.vertices[sub.faces[fA]].reshape(-1, 3)
                    vB = sub.vertices[sub.faces[fB]].reshape(-1, 3)
                    halves = [(rec, fA, vA), (rec_b, fB, vB)]
        for rec_h, fi_h, verts_h in halves:
            ad_raw = rec_h.get("axis_dir")
            ap_raw = rec_h.get("axis_point")
            axis_dir = None
            endpoints = None
            if ad_raw is not None:
                a = np.asarray(ad_raw, dtype=float)
                an = float(np.linalg.norm(a))
                if an >= 1e-9:
                    axis_dir = a / an
                    axis_point = (
                        np.asarray(ap_raw, dtype=float)
                        if ap_raw is not None else verts_h.mean(axis=0)
                    )
                    t = (verts_h - axis_point) @ axis_dir
                    endpoints = np.array([
                        axis_point + float(t.min()) * axis_dir,
                        axis_point + float(t.max()) * axis_dir,
                    ])
            records_info.append({
                "rec": rec_h, "verts": verts_h, "axis_dir": axis_dir,
                "endpoints": endpoints,
                "radius": float(rec_h.get("radius", 0.0) or 0.0),
            })

    # ---- per-record closed-loop classification ----
    for ri in records_info:
        axis_dir = ri["axis_dir"]
        endpoints = ri["endpoints"]
        radius = ri["radius"]
        verts = ri["verts"]
        ri["is_closed_loop"] = False
        if axis_dir is None or endpoints is None or radius <= 0:
            continue
        axis_point = 0.5 * (endpoints[0] + endpoints[1])
        rel = verts - axis_point
        along = rel @ axis_dir
        perp = rel - np.outer(along, axis_dir)
        arb = (np.array([1.0, 0, 0]) if abs(float(axis_dir[0])) < 0.9
               else np.array([0, 1.0, 0]))
        u = arb - axis_dir * float(arb @ axis_dir)
        un = float(np.linalg.norm(u))
        if un < 1e-9:
            continue
        u = u / un
        v = np.cross(axis_dir, u)
        angles = np.arctan2(perp @ v, perp @ u)
        if len(angles) >= 4:
            s = np.sort(angles)
            wrap = 2 * np.pi - (float(s[-1]) - float(s[0]))
            max_gap = float(max(np.diff(s).max(), wrap))
        else:
            max_gap = 2 * np.pi
        ri["is_closed_loop"] = max_gap < np.pi / 2

    for uid, ri in enumerate(records_info):
        ri["uid"] = uid

    parent = {ri["uid"]: ri["uid"] for ri in records_info}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # ---- pass 2a: direct contact ----
    allv = np.vstack([ri["verts"] for ri in records_info])
    allu = np.concatenate([
        np.full(len(ri["verts"]), ri["uid"], dtype=np.int64)
        for ri in records_info
    ])
    tree = cKDTree(allv)
    uid_split = {
        ri["uid"]: (ri["rec"].get("_split_parent"), ri["rec"].get("_split_side"))
        for ri in records_info
    }
    for i, j in tree.query_pairs(r=DIRECT_EPS):
        ua, ub = int(allu[i]), int(allu[j])
        if ua == ub or find(ua) == find(ub):
            continue
        pa, sa = uid_split.get(ua, (None, None))
        pb, sb = uid_split.get(ub, (None, None))
        if pa is not None and pa == pb and sa != sb:
            continue
        union(ua, ub)

    # ---- proto loop stamp + bimodal sides ----
    proto_members: dict[int, list[dict]] = {}
    for ri in records_info:
        proto_members.setdefault(find(ri["uid"]), []).append(ri)
        ri["bimodal_side"] = 0

    def proto_axis_loop(verts):
        if len(verts) < 8:
            return None, False
        c = verts - verts.mean(axis=0)
        _, _, vh = np.linalg.svd(c, full_matrices=False)
        axis = vh[2] / (np.linalg.norm(vh[2]) + 1e-12)
        along = c @ axis
        perp = c - np.outer(along, axis)
        arb = (np.array([1.0, 0, 0]) if abs(float(axis[0])) < 0.9
               else np.array([0, 1.0, 0]))
        u = arb - axis * float(arb @ axis)
        un = float(np.linalg.norm(u))
        if un < 1e-9:
            return axis, False
        u = u / un
        v = np.cross(axis, u)
        ang = np.arctan2(perp @ v, perp @ u)
        if len(ang) < 8:
            return axis, False
        s = np.sort(ang)
        wrap = 2 * np.pi - (float(s[-1]) - float(s[0]))
        return axis, float(max(np.diff(s).max(), wrap)) < np.pi / 2

    for root, mem in proto_members.items():
        pv = np.vstack([m["verts"] for m in mem])
        axis, is_loop = proto_axis_loop(pv)
        if is_loop and axis is not None:
            for m in mem:
                m["is_closed_loop"] = True
        if len(mem) < 2:
            continue
        lo, hi = pv.min(axis=0), pv.max(axis=0)
        dims = hi - lo
        thin = int(np.argmin(dims))
        if float(dims[thin]) > 30.0:
            continue
        rec_thin = np.array([float(m["verts"].mean(axis=0)[thin]) for m in mem])
        if is_loop:
            split_at = 0.5 * (float(lo[thin]) + float(hi[thin]))
        else:
            vt = np.sort(pv[:, thin])
            gaps = np.diff(vt)
            gi = int(np.argmax(gaps))
            frac_low = (gi + 1) / len(vt)
            if gaps[gi] < 1.5 or not (0.15 < frac_low < 0.85):
                continue
            split_at = 0.5 * (float(vt[gi]) + float(vt[gi + 1]))
        side_a = {i for i, v in enumerate(rec_thin) if v < split_at}
        side_b = set(range(len(mem))) - side_a
        if not side_a or not side_b:
            continue
        for i, m in enumerate(mem):
            m["bimodal_side"] = 0 if i in side_a else 1

    def group_snapshot(label):
        sub_groups: dict[tuple, list[dict]] = {}
        for ri in records_info:
            sub_groups.setdefault(
                (find(ri["uid"]), int(ri.get("bimodal_side", 0))), [],
            ).append(ri)
        print(f"\n=== {label}: {len(sub_groups)} groups ===")
        for (root, side), mem in sorted(sub_groups.items()):
            gv = np.vstack([m["verts"] for m in mem])
            print(f"  (root={root},side={side}) n_rec={len(mem)} "
                  f"bbox={bbox_str(gv)}")
        return sub_groups

    group_snapshot("AFTER direct contact + sides (pass 2a)")

    # ---- pass 2b: collinear continuation (with logging) ----
    n_ax = 0
    for i_idx in range(len(records_info)):
        A = records_info[i_idx]
        if A["axis_dir"] is None or A["endpoints"] is None:
            continue
        for j_idx in range(i_idx + 1, len(records_info)):
            B = records_info[j_idx]
            if B["axis_dir"] is None or B["endpoints"] is None:
                continue
            if find(A["uid"]) == find(B["uid"]):
                continue
            if A.get("is_closed_loop") and B.get("is_closed_loop"):
                continue
            align = abs(float(A["axis_dir"] @ B["axis_dir"]))
            if align < AXIS_DOT_MIN:
                continue
            r_min = min(A["radius"], B["radius"])
            eps_perp = max(1.5, 1.5 * r_min)
            eps_axial = max(4.0, 3.0 * r_min)
            # Cloud-centroid perp gate (mirrors job_queue): parallel
            # offset bands must not merge as "continuations".
            d_c = B["verts"].mean(axis=0) - A["verts"].mean(axis=0)
            perp_ok = True
            for ax_chk in (A["axis_dir"], B["axis_dir"]):
                perp_c = d_c - float(d_c @ ax_chk) * ax_chk
                if float(np.linalg.norm(perp_c)) >= eps_perp:
                    perp_ok = False
                    break
            if not perp_ok:
                continue
            for EA in A["endpoints"]:
                hit = False
                for EB in B["endpoints"]:
                    d = EB - EA
                    axial = float(d @ A["axis_dir"])
                    perp_vec = d - axial * A["axis_dir"]
                    perp_d = float(np.linalg.norm(perp_vec))
                    if perp_d < eps_perp and abs(axial) < eps_axial:
                        print(
                            f"AXIAL UNION uid{A['uid']} bbox={bbox_str(A['verts'])} "
                            f"<-> uid{B['uid']} bbox={bbox_str(B['verts'])} "
                            f"align={align:.2f} perp={perp_d:.2f} "
                            f"axial={axial:.2f} side={A['bimodal_side']}/"
                            f"{B['bimodal_side']}"
                        )
                        union(A["uid"], B["uid"])
                        n_ax += 1
                        hit = True
                        break
                if hit:
                    break

    print(f"\n{n_ax} axial unions")
    group_snapshot("AFTER axial continuation (pass 2b)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
