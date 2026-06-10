"""Headless driver: run the upstream pipeline + the stitched-stage
union-find logic on the rackmount mesh and report the cluster count.

Mirrors the slice of `viz_server/job_queue.py` that builds the
`stitched_clusters_payload`. Lets us verify the fix (ring-to-ring
through-hole faces stay separate, expected 12 clusters not 10) without
launching the web UI.

Usage:
    python scripts/test_stitched_rackmount.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh2param.config import (
    DecomposeConfig,
    FitConfig,
    IngestConfig,
    SegmentConfig,
    TopologyConfig,
)
from mesh2param.models.region import SurfaceType as _ST_pn
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("rackmount_test")

DIRECT_EPS = 2.0
AXIS_DOT_MIN = 0.5


def run() -> int:
    mesh_path = ROOT / "models" / "Mesh5" / "RackMount.3mf"
    logger.info("Loading %s", mesh_path)
    cleaned = ingest_mesh(str(mesh_path), IngestConfig(remesh=False))
    decomposition = decompose_bodies(cleaned, DecomposeConfig())
    bodies = decomposition.bodies
    logger.info("Decomposed into %d bodies", len(bodies))

    body_stage_data = []
    for bi, body in enumerate(bodies):
        sub_cleaned = extract_sub_mesh(cleaned, body.face_indices, body=body)
        regions = segment_mesh(sub_cleaned, SegmentConfig())
        fitted = fit_surfaces(regions, sub_cleaned, FitConfig())
        graph, edges_list = build_topology(regions, fitted, sub_cleaned, TopologyConfig())
        try:
            topology_blends = identify_blends_from_topology(
                regions, fitted, edges_list, sub_cleaned,
            )
        except Exception as e:
            logger.exception("identify_blends_from_topology failed body %d: %s", bi, e)
            topology_blends = []
        body_stage_data.append((body, sub_cleaned, regions, fitted, topology_blends))

    # Per-body geometry + fillet-record summary so we can see if
    # both side plates are symmetric in what topology produced.
    print()
    print("  PER-BODY SUMMARY:")
    for body_idx, _bst in enumerate(body_stage_data):
        body, _sub, _regs, _fits, topo_blends = _bst
        bmin, bmax = _sub.vertices.min(0), _sub.vertices.max(0)
        n_fil = sum(1 for r in topo_blends if r.get("kind") == "fillet")
        n_cham = sum(1 for r in topo_blends if r.get("kind") == "chamfer")
        print(
            f"    body {body_idx}: {len(_sub.faces)} faces, "
            f"bbox X[{bmin[0]:.0f},{bmax[0]:.0f}] "
            f"Y[{bmin[1]:.0f},{bmax[1]:.0f}] Z[{bmin[2]:.0f},{bmax[2]:.0f}], "
            f"{n_fil} fillet + {n_cham} chamfer regions, "
            f"{len(_regs)} regions, {len(_fits)} fits"
        )

    # Mirror the stitched stage in job_queue.run()
    records_info: list[dict] = []
    n_chamfers = 0
    for body_idx, _bst in enumerate(body_stage_data):
        body, _sub, _regs, _fits, topo_blends = _bst
        regs_lookup = {r.region_id: r for r in _regs}
        fits_lookup = {f.region_id: f for f in _fits}

        def _plane_normal_for(rid: int):
            f = fits_lookup.get(rid)
            if f is None:
                return None
            if f.surface_type == _ST_pn.PLANE:
                n = np.asarray(f.params.get("normal", [0, 0, 0]), dtype=float)
            else:
                reg = regs_lookup.get(rid)
                if reg is None or len(reg.face_indices) == 0:
                    return None
                n = _sub.normals[reg.face_indices].mean(axis=0)
            nrm = float(np.linalg.norm(n))
            if nrm < 0.5:
                return None
            return n / nrm

        for rec in topo_blends:
            kind = rec.get("kind")
            if kind == "chamfer":
                n_chamfers += 1
                continue
            if kind != "fillet":
                continue
            fi = rec.get("face_indices")
            if fi is None or len(fi) == 0:
                continue
            fi_arr = np.asarray(fi, dtype=np.int64)
            verts = _sub.vertices[_sub.faces[fi_arr]].reshape(-1, 3)
            ad_raw = rec.get("axis_dir")
            ap_raw = rec.get("axis_point")
            if ad_raw is not None:
                axis_dir = np.asarray(ad_raw, dtype=float)
                an = float(np.linalg.norm(axis_dir))
            else:
                axis_dir = None
                an = 0.0
            if axis_dir is not None and an >= 1e-9:
                axis_dir = axis_dir / an
                axis_point = (
                    np.asarray(ap_raw, dtype=float)
                    if ap_raw is not None
                    else verts.mean(axis=0)
                )
                t = (verts - axis_point) @ axis_dir
                t_min, t_max = float(t.min()), float(t.max())
                endpoints = np.array([
                    axis_point + t_min * axis_dir,
                    axis_point + t_max * axis_dir,
                ])
            else:
                axis_dir = None
                endpoints = None
            plane_normals_rec = []
            for prid in (rec.get("plane_region_ids") or []):
                n = _plane_normal_for(int(prid))
                if n is not None:
                    plane_normals_rec.append(n.tolist())
            records_info.append({
                "body_idx": body_idx,
                "rec": rec,
                "verts": verts,
                "axis_dir": axis_dir,
                "endpoints": endpoints,
                "radius": float(rec.get("radius", 0.0) or 0.0),
                "plane_normals": plane_normals_rec,
            })

    for uid, ri in enumerate(records_info):
        ri["uid"] = uid

    # Per-record category (mirror of job_queue.py change)
    for ri in records_info:
        axis_dir = ri["axis_dir"]
        endpoints = ri["endpoints"]
        radius = ri["radius"]
        verts = ri["verts"]
        if axis_dir is None or endpoints is None or radius <= 0:
            ri["category"] = "curve_edge"
            ri["is_closed_loop"] = False
            ri["axial_extent"] = 0.0
            continue
        axis_point = 0.5 * (endpoints[0] + endpoints[1])
        rel = verts - axis_point
        along = rel @ axis_dir
        axial_extent = float(along.max() - along.min())
        perp = rel - np.outer(along, axis_dir)
        arb = (
            np.array([1.0, 0.0, 0.0])
            if abs(float(axis_dir[0])) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = arb - axis_dir * float(arb @ axis_dir)
        un = float(np.linalg.norm(u))
        if un < 1e-9:
            ri["category"] = "curve_edge"
            ri["is_closed_loop"] = False
            ri["axial_extent"] = axial_extent
            continue
        u = u / un
        v = np.cross(axis_dir, u)
        angles = np.arctan2(perp @ v, perp @ u)
        if len(angles) >= 4:
            asorted = np.sort(angles)
            diffs = np.diff(asorted)
            wrap = (2.0 * np.pi) - (float(asorted[-1]) - float(asorted[0]))
            max_gap = float(max(diffs.max(), wrap))
        else:
            max_gap = 2.0 * np.pi
        is_closed_loop = max_gap < (np.pi / 2.0)
        ri["is_closed_loop"] = is_closed_loop
        ri["axial_extent"] = axial_extent
        if is_closed_loop:
            ri["category"] = "hole_edge" if axial_extent < 5.0 * radius else "looped"
        else:
            ri["category"] = "straight_edge" if axial_extent > 3.0 * radius else "curve_edge"

    n_records = len(records_info)
    logger.info("Records collected: %d (chamfers: %d)", n_records, n_chamfers)
    for ri in records_info:
        verts = ri["verts"]
        bmin, bmax = verts.min(0), verts.max(0)
        d = bmax - bmin
        ax = ri["axis_dir"]
        ax_s = f"({ax[0]:.2f},{ax[1]:.2f},{ax[2]:.2f})" if ax is not None else "NONE"
        logger.info(
            "  uid=%d body=%d n=%d r=%.2f axis=%s closed_loop=%s cat=%s "
            "axial=%.1f bbox=(%.1f,%.1f,%.1f)",
            ri["uid"], ri["body_idx"], len(verts), ri["radius"], ax_s,
            ri["is_closed_loop"], ri["category"], ri["axial_extent"],
            d[0], d[1], d[2],
        )

    parent = {ri["uid"]: ri["uid"] for ri in records_info}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    n_direct = 0
    if records_info:
        all_verts = np.vstack([ri["verts"] for ri in records_info])
        all_uids = np.concatenate([
            np.full(len(ri["verts"]), ri["uid"], dtype=np.int64)
            for ri in records_info
        ])
        tree = cKDTree(all_verts)
        for i, j in tree.query_pairs(r=DIRECT_EPS):
            ua = int(all_uids[i]); ub = int(all_uids[j])
            if ua != ub and _find(ua) != _find(ub):
                _union(ua, ub)
                n_direct += 1
    logger.info("Direct-contact unions: %d", n_direct)

    # Proto-cluster closed-loop promotion (the new pass)
    def _proto_axis_loop(verts):
        if len(verts) < 8:
            return None, False
        c = verts - verts.mean(0)
        try:
            _, _, vh = np.linalg.svd(c, full_matrices=False)
        except Exception:
            return None, False
        axis = vh[2] / (np.linalg.norm(vh[2]) + 1e-12)
        rel = c
        along = rel @ axis
        perp = rel - np.outer(along, axis)
        arb = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = arb - axis * float(arb @ axis)
        un = float(np.linalg.norm(u))
        if un < 1e-9:
            return axis, False
        u = u / un
        v = np.cross(axis, u)
        angles = np.arctan2(perp @ v, perp @ u)
        if len(angles) < 8:
            return axis, False
        a = np.sort(angles)
        diffs = np.diff(a)
        wrap = (2.0 * np.pi) - (float(a[-1]) - float(a[0]))
        max_gap = float(max(diffs.max(), wrap))
        return axis, (max_gap < (np.pi / 2.0))

    proto = {}
    proto_full_verts_by_root = {}
    for ri in records_info:
        proto.setdefault(_find(ri["uid"]), []).append(ri)
        ri["bimodal_side"] = 0
    logger.info("Proto-clusters after direct-contact: %d", len(proto))
    for root, members in proto.items():
        proto_verts = np.vstack([m["verts"] for m in members])
        proto_full_verts_by_root[root] = proto_verts
        proto_axis, is_loop = _proto_axis_loop(proto_verts)
        ax_log = (
            f"({proto_axis[0]:.2f},{proto_axis[1]:.2f},{proto_axis[2]:.2f})"
            if proto_axis is not None else "NONE"
        )
        bmin, bmax = proto_verts.min(0), proto_verts.max(0)
        bd = bmax - bmin
        logger.info(
            "  proto root=%d nrecs=%d nverts=%d is_loop=%s axis=%s "
            "uids=%s bbox=(%.1f,%.1f,%.1f)",
            root, len(members), len(proto_verts), is_loop, ax_log,
            sorted(m["uid"] for m in members),
            bd[0], bd[1], bd[2],
        )
        if is_loop and proto_axis is not None:
            for m in members:
                m["is_closed_loop"] = True
                if m.get("category") == "curve_edge":
                    m["category"] = "hole_edge"
            if len(members) >= 2:
                # Same-body filter — a plate is one body.
                if len({m["body_idx"] for m in members}) > 1:
                    logger.info("    cross-body proto, not a plate split case")
                    continue
                # Opposite-plate-face split. Same geometric rule as
                # in job_queue.py: smallest bbox dim is the
                # candidate plate-normal direction; if records sit
                # on both sides of its midplane, split them.
                proto_min_ = proto_verts.min(0)
                proto_max_ = proto_verts.max(0)
                proto_dims_ = proto_max_ - proto_min_
                thin_axis_ = int(np.argmin(proto_dims_))
                thin_dim_ = float(proto_dims_[thin_axis_])
                logger.info(
                    "    proto dims=(%.1f,%.1f,%.1f) thin_axis=%d thin_dim=%.1f",
                    proto_dims_[0], proto_dims_[1], proto_dims_[2],
                    thin_axis_, thin_dim_,
                )
                if thin_dim_ > 30.0:
                    logger.info("    thin dim > 30 mm, not a plate; no split")
                    continue
                rec_thin = np.array([
                    float(m["verts"].mean(0)[thin_axis_]) for m in members
                ])
                midplane = 0.5 * (
                    float(proto_min_[thin_axis_])
                    + float(proto_max_[thin_axis_])
                )
                logger.info(
                    "    thin-centroids: %s, midplane=%.2f",
                    ", ".join(
                        f"uid{m['uid']}={a:.2f}"
                        for m, a in zip(members, rec_thin)
                    ),
                    midplane,
                )
                side_a = {i for i, v in enumerate(rec_thin) if v < midplane}
                side_b = set(range(len(members))) - side_a
                if side_a and side_b:
                    for i, m in enumerate(members):
                        m["bimodal_side"] = 0 if i in side_a else 1
                    logger.info(
                        "    opposite-plate-face split root=%d: A=%s B=%s",
                        root,
                        sorted(m["uid"] for i, m in enumerate(members) if i in side_a),
                        sorted(m["uid"] for i, m in enumerate(members) if i in side_b),
                    )

    # Axial-continuation
    n_axial = 0
    for i in range(len(records_info)):
        A = records_info[i]
        if A["axis_dir"] is None or A["endpoints"] is None:
            continue
        for j in range(i + 1, len(records_info)):
            B = records_info[j]
            if B["axis_dir"] is None or B["endpoints"] is None:
                continue
            if _find(A["uid"]) == _find(B["uid"]):
                continue
            if A.get("is_closed_loop") and B.get("is_closed_loop"):
                continue
            align = abs(float(A["axis_dir"] @ B["axis_dir"]))
            if align < AXIS_DOT_MIN:
                continue
            r_min = min(A["radius"], B["radius"])
            eps_perp = max(1.5, 1.5 * r_min)
            eps_axial = max(4.0, 3.0 * r_min)
            stitched = False
            for EA in A["endpoints"]:
                for EB in B["endpoints"]:
                    d = EB - EA
                    axial = float(d @ A["axis_dir"])
                    perp_vec = d - axial * A["axis_dir"]
                    perp_d = float(np.linalg.norm(perp_vec))
                    if perp_d < eps_perp and abs(axial) < eps_axial:
                        _union(A["uid"], B["uid"])
                        n_axial += 1
                        stitched = True
                        break
                if stitched:
                    break
    logger.info("Axial-continuation unions: %d", n_axial)

    # Final groups: (root, bimodal_side) key
    subkey_to_group = {}
    uid_to_group = {}
    for ri in records_info:
        root = _find(ri["uid"])
        sk = (root, int(ri.get("bimodal_side", 0)))
        if sk not in subkey_to_group:
            subkey_to_group[sk] = len(subkey_to_group)
        uid_to_group[ri["uid"]] = subkey_to_group[sk]
    n_groups = len(subkey_to_group)
    counts = {}
    for gi in uid_to_group.values():
        counts[gi] = counts.get(gi, 0) + 1

    # Aggregate categories per group
    group_records = {gi: [] for gi in range(n_groups)}
    for ri in records_info:
        group_records[uid_to_group[ri["uid"]]].append(ri)

    def _coverage_and_inplane(verts):
        if len(verts) < 4:
            return 0.0, 0.0
        c = verts - verts.mean(0)
        try:
            _, _, vh = np.linalg.svd(c, full_matrices=False)
        except Exception:
            return 0.0, 0.0
        best_cov = 0.0
        best_inplane = 0.0
        for axis_cand in (vh[2], vh[0]):
            axis = axis_cand / (np.linalg.norm(axis_cand) + 1e-12)
            along = c @ axis
            perp = c - np.outer(along, axis)
            arb = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = arb - axis * float(arb @ axis)
            un = float(np.linalg.norm(u))
            if un < 1e-9:
                continue
            u = u / un
            v = np.cross(axis, u)
            angles = np.arctan2(perp @ v, perp @ u)
            if len(angles) < 4:
                continue
            s = np.sort(angles)
            diffs = np.diff(s)
            wrap = (2.0 * np.pi) - (float(s[-1]) - float(s[0]))
            max_gap = float(max(diffs.max(), wrap))
            coverage = (2.0 * np.pi) - max_gap
            if coverage > best_cov:
                best_cov = coverage
                pu = perp @ u; pv = perp @ v
                best_inplane = max(
                    float(pu.max() - pu.min()),
                    float(pv.max() - pv.min()),
                )
        return best_cov, best_inplane

    cat_per_group = {}
    cov_per_group = {}
    inplane_per_group = {}
    r_per_group = {}
    length_per_group = {}
    for gi, members in group_records.items():
        agg = np.vstack([m["verts"] for m in members])
        # Use full proto verts (pre-split) for coverage/inplane/length.
        root_for_cluster = _find(members[0]["uid"])
        agg_full = proto_full_verts_by_root.get(root_for_cluster, agg)
        cov_rad, inplane = _coverage_and_inplane(agg_full)
        cov_deg = cov_rad * 180.0 / float(np.pi)
        rs = [m["radius"] for m in members if m["radius"] > 0]
        r_med = float(np.median(rs)) if rs else 0.0
        r_max = float(max(rs)) if rs else 0.0
        if (cov_deg > 270.0 and r_med <= 8.0
                and r_max <= 10.0 and inplane <= 40.0):
            cat = "hole_edge"
        elif cov_deg > 135.0:
            cat = "curve_edge"
        else:
            cat = "straight_edge"
        # Fillet path length:
        #   - Bent (cov > 90°): length = coverage_radians * loop_radius
        #     where loop_radius = inplane / 2 (the half-diameter of the
        #     curved path in the perpendicular plane).
        #   - Straight: length = the cluster's principal-axis extent.
        if cov_deg > 90.0 and inplane > 0:
            length = float(cov_rad * inplane * 0.5)
        else:
            centered = agg_full - agg_full.mean(0)
            try:
                _, sv, _ = np.linalg.svd(centered, full_matrices=False)
                length = float(2.0 * sv[0] / np.sqrt(len(agg_full)))
            except Exception:
                length = 0.0
        cat_per_group[gi] = cat
        cov_per_group[gi] = cov_deg
        inplane_per_group[gi] = inplane
        r_per_group[gi] = r_med
        length_per_group[gi] = length

    print()
    print("=" * 70)
    print(f"RESULT: {n_records} records -> {n_groups} clusters")
    print(f"  direct-contact unions: {n_direct}")
    print(f"  axial-continuation unions: {n_axial}")
    print("  category breakdown:")
    cat_totals = {}
    for c in cat_per_group.values():
        cat_totals[c] = cat_totals.get(c, 0) + 1
    for c, n in sorted(cat_totals.items(), key=lambda kv: -kv[1]):
        print(f"    {n} {c}")
    print()
    print("  per-cluster (length = fillet path length in mm):")
    rows = []
    for gi in range(n_groups):
        verts = np.vstack([m["verts"] for m in group_records[gi]])
        centroid = verts.mean(0)
        d = verts.max(0) - verts.min(0)
        r = float(np.median([m["radius"] for m in group_records[gi] if m["radius"] > 0] or [0]))
        rows.append({
            "gi": gi,
            "cat": cat_per_group[gi],
            "len": length_per_group[gi],
            "cov": cov_per_group[gi],
            "inplane": inplane_per_group[gi],
            "r": r,
            "n_rec": counts[gi],
            "centroid": centroid,
            "bbox": d,
        })
    # Sort: hole_edge first by length, then curve_edge by length.
    cat_order = {"hole_edge": 0, "curve_edge": 1, "straight_edge": 2, "looped": 3}
    rows.sort(key=lambda r_: (cat_order.get(r_["cat"], 9), r_["len"]))
    print(f"    {'cluster':<8} {'cat':<14} {'length':>9} {'cov':>6} {'inplane':>9} {'r':>6} {'rec':>4}  centroid")
    for r_ in rows:
        c = r_["centroid"]
        print(
            f"    {r_['gi']:<8} {r_['cat']:<14} {r_['len']:>7.1f}mm "
            f"{r_['cov']:>5.0f}° {r_['inplane']:>7.1f}mm "
            f"{r_['r']:>6.2f} {r_['n_rec']:>4}  "
            f"({c[0]:>6.1f},{c[1]:>6.1f},{c[2]:>6.1f})"
        )
    # Length-grouping summary
    print()
    print("  by-category length groups (looking for the 4+4+4 split):")
    for cat in ["hole_edge", "curve_edge", "straight_edge"]:
        sub = [r_ for r_ in rows if r_["cat"] == cat]
        if not sub:
            continue
        lengths = sorted(r_["len"] for r_ in sub)
        print(f"    {cat}: {len(sub)} clusters, lengths = "
              + ", ".join(f"{l:.1f}" for l in lengths))
    # Ground-truth hole centers (the two big through-holes, r=15.48).
    hole_centers = [
        np.array([-40.4, 125.5, -133.4]),
        np.array([42.4, 125.5, -133.4]),
    ]
    print()
    print("  RECORDS whose FACES TOUCH each hole (min vertex dist < 18mm):")
    for hc in hole_centers:
        print(f"    hole @ ({hc[0]:.0f},{hc[1]:.0f},{hc[2]:.0f}):")
        for ri in records_info:
            v = ri["verts"]
            mind = float(np.min(np.linalg.norm(v - hc, axis=1)))
            if mind < 18.0:
                n_near = int(np.sum(np.linalg.norm(v - hc, axis=1) < 18.0))
                c = v.mean(0)
                print(
                    f"      uid={ri['uid']:>2} body={ri['body_idx']} "
                    f"r={ri['radius']:.2f} min_vtx_dist={mind:.1f} "
                    f"n_verts_near={n_near}/{len(v)} "
                    f"group={uid_to_group[ri['uid']]} "
                    f"cat={ri.get('category')} "
                    f"record_centroid=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) "
                    f"record_bbox={tuple(round(float(x),1) for x in (v.max(0)-v.min(0)))}"
                )
    print()
    print("  WHAT ARE CLUSTERS 4 & 5 (flagged hole_edge near Y=200)?")
    for gi in [4, 5]:
        for ri in group_records.get(gi, []):
            c = ri["verts"].mean(0)
            print(
                f"    cluster {gi}: uid={ri['uid']} body={ri['body_idx']} "
                f"r={ri['radius']:.2f} cat0={ri.get('category')} "
                f"centroid=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) "
                f"bbox={tuple(round(float(x),1) for x in (ri['verts'].max(0)-ri['verts'].min(0)))}"
            )
    print("=" * 70)
    print("EXPECTED: 12 clusters (was 10 before fix)")
    print(f"OBSERVED: {n_groups} clusters")
    if n_groups == 12:
        print(">>> PASS <<<")
        return 0
    print(">>> MISMATCH <<<")
    return 1


if __name__ == "__main__":
    sys.exit(run())
