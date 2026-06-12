"""Fillet-to-edge matcher: snap each stitched cluster to rebuilt edges.

Builds the rackmount's stitched fillet clusters (with vertices +
category) AND the rebuilt CAD solid, then matches each cluster to the
rebuilt-solid edge(s) it should fillet — by absolute-distance snapping
of the cluster's vertex cloud to each edge, gated by category:

  - hole_edge   -> prefer CIRCLE edges (the hole openings)
  - curve_edge  -> LINE / curved edges in the cluster's area
  - straight_edge -> LINE edges

A rebuilt edge belongs to a cluster if a good fraction of the edge's
sampled points come within SNAP_TOL of the cluster's fillet-surface
vertex cloud. The fillet surface sits ~(0.4 * tube_radius) from the
sharp rebuilt edge, so a few-mm snap tolerance is correct.

Usage:
    python scripts/match_fillets_to_edges.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)

from scipy.spatial import cKDTree

from mesh2param.config import (
    DecomposeConfig, FitConfig, IngestConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.export_cadquery import export_cadquery
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

SNAP_TOL_BASE = 3.5     # mm absolute snap floor
N_EDGE_SAMPLES = 15
MIN_EDGE_FRAC = 0.5     # >= this fraction of an edge's samples must be near


# ---------------------------------------------------------------------------
# 1. Build stitched clusters (vertices + category) — compact mirror of the
#    job_queue stitched stage, enough for matching.
# ---------------------------------------------------------------------------
def build_clusters(cleaned, single_body: bool = False):
    """Stitched blend clusters. single_body=True skips decomposition and
    builds clusters from the full-mesh topology — the multi-extrude
    (engine JOIN) path, where the part is one body with several
    extrusions instead of decomposed pieces."""
    decomp = decompose_bodies(
        cleaned, DecomposeConfig(enabled=not single_body),
    )
    body_stage = []
    body_results = []
    for body in decomp.bodies:
        sub = extract_sub_mesh(cleaned, body.face_indices, body=body)
        regions = segment_mesh(sub, SegmentConfig())
        fitted = fit_surfaces(regions, sub, FitConfig())
        graph, edges = build_topology(regions, fitted, sub, TopologyConfig())
        blends = identify_blends_from_topology(regions, fitted, edges, sub)
        tree = recognize_features(regions, fitted, graph, edges, sub)
        body_stage.append((sub, blends))
        body_results.append((body, sub, tree))

    records = []
    chamfer_records = []
    for bidx, (sub, blends) in enumerate(body_stage):
        for rec in blends:
            kind = rec.get("kind")
            if kind not in ("fillet", "chamfer"):
                continue
            fi = rec.get("face_indices")
            if fi is None or len(fi) == 0:
                continue
            verts = sub.vertices[sub.faces[np.asarray(fi)]].reshape(-1, 3)
            if kind == "chamfer":
                chamfer_records.append({
                    "body": bidx, "verts": verts,
                    "radius": float(rec.get("radius", 0) or 0),
                })
                continue
            ad = rec.get("axis_dir")
            axis = np.asarray(ad, float) if ad is not None else None
            if axis is not None and np.linalg.norm(axis) > 1e-9:
                axis = axis / np.linalg.norm(axis)
            else:
                axis = None
            # Plate-wrap split — mirrors job_queue: a band wrapping
            # both faces of a thin plate splits at the bimodal void so
            # each half joins its own face's group.
            import numpy as _np_split
            halves = [(verts, None, None)]
            dims = verts.max(axis=0) - verts.min(axis=0)
            # Search every plate-thickness-like axis for the widest void
            # (the record's own thinnest dim is wrong for side bands).
            thin, best_gap = -1, 0.0
            vt = gaps = None
            gi = 0
            frac = 0.0
            # Single-body records only — per-piece records are already
            # face-separated (mirrors job_queue).
            if single_body and len(verts) >= 24:
                for _ax in range(3):
                    if float(dims[_ax]) > 12.0:
                        continue
                    vt_a = _np_split.sort(verts[:, _ax])
                    gaps_a = _np_split.diff(vt_a)
                    gi_a = int(_np_split.argmax(gaps_a))
                    if float(gaps_a[gi_a]) > best_gap:
                        best_gap = float(gaps_a[gi_a])
                        thin = _ax
                        vt, gaps, gi = vt_a, gaps_a, gi_a
                        frac = (gi_a + 1) / len(vt_a)
            if thin >= 0:
                rec_r = float(rec.get("radius", 0) or 0)
                void_split = float(gaps[gi]) >= 1.5 and (
                    0.2 < frac < 0.8 or float(gaps[gi]) >= 2.5
                )
                fused_split = (
                    not void_split and rec_r > 0
                    and float(dims[thin]) >= max(5.0, 1.6 * 1.5708 * rec_r)
                )
                if void_split or fused_split:
                    if void_split:
                        # Split at the void's boundaries — the void is
                        # the dead zone.
                        vA = verts[verts[:, thin] <= float(vt[gi]) + 1e-6]
                        vB = verts[verts[:, thin] >= float(vt[gi + 1]) - 1e-6]
                    else:
                        mid = 0.5 * (float(vt[0]) + float(vt[-1]))
                        vA = verts[verts[:, thin] < mid - 1.2]
                        vB = verts[verts[:, thin] > mid + 1.2]
                    if len(vA) >= 9 and len(vB) >= 9:
                        halves = [(vA, id(rec), 0), (vB, id(rec), 1)]
                    elif len(vA) >= 9:
                        halves = [(vA, None, None)]  # drop stray minority
                    elif len(vB) >= 9:
                        halves = [(vB, None, None)]
            for vh, sp, ss in halves:
                records.append({
                    "body": bidx, "verts": vh, "axis": axis,
                    "radius": float(rec.get("radius", 0) or 0),
                    "split_parent": sp, "split_side": ss,
                })
    for i, r in enumerate(records):
        r["uid"] = i

    # union-find by direct contact
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if records:
        allv = np.vstack([r["verts"] for r in records])
        allu = np.concatenate([np.full(len(r["verts"]), r["uid"]) for r in records])
        t = cKDTree(allv)
        for i, j in t.query_pairs(r=2.0):
            a, b = int(allu[i]), int(allu[j])
            if a == b:
                continue
            ra_, rb_ = records[a], records[b]
            if (
                ra_.get("split_parent") is not None
                and ra_.get("split_parent") == rb_.get("split_parent")
                and ra_.get("split_side") != rb_.get("split_side")
            ):
                continue
            union(a, b)

    # proto closed-loop + opposite-plate split
    proto = {}
    for r in records:
        proto.setdefault(find(r["uid"]), []).append(r)
        r["side"] = 0

    def proto_axis_loop(verts):
        if len(verts) < 8:
            return None, False
        c = verts - verts.mean(0)
        _, _, vh = np.linalg.svd(c, full_matrices=False)
        axis = vh[2] / (np.linalg.norm(vh[2]) + 1e-12)
        perp = c - np.outer(c @ axis, axis)
        arb = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
        u = arb - axis * (arb @ axis)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(axis, u)
        ang = np.arctan2(perp @ v, perp @ u)
        if len(ang) < 8:
            return axis, False
        s = np.sort(ang)
        gap = max(np.diff(s).max(), 2 * np.pi - (s[-1] - s[0]))
        return axis, gap < np.pi / 2

    for root, mem in proto.items():
        pv = np.vstack([m["verts"] for m in mem])
        axis, loop = proto_axis_loop(pv)
        if len(mem) < 2:
            continue
        if len({m["body"] for m in mem}) > 1:
            continue
        dims = pv.max(0) - pv.min(0)
        thin = int(np.argmin(dims))
        if dims[thin] > 30:
            continue
        if loop and axis is not None:
            mid = 0.5 * (pv.min(0)[thin] + pv.max(0)[thin])
        else:
            # Non-loop: split only on a clear bimodal gap (two parallel
            # bands separated by plate thickness) — mirrors job_queue.
            vt = np.sort(pv[:, thin])
            gaps = np.diff(vt)
            gi = int(np.argmax(gaps))
            frac_low = (gi + 1) / len(vt)
            if gaps[gi] < 1.5 or not (0.15 < frac_low < 0.85):
                continue
            mid = 0.5 * (float(vt[gi]) + float(vt[gi + 1]))
        for m in mem:
            m["side"] = 0 if m["verts"].mean(0)[thin] < mid else 1

    groups = {}
    for r in records:
        groups.setdefault((find(r["uid"]), r["side"]), []).append(r)

    clusters = []
    for gid, ((root, side), mem) in enumerate(groups.items()):
        verts = np.vstack([m["verts"] for m in mem])
        full = np.vstack([m["verts"] for m in proto[root]])
        rs = [m["radius"] for m in mem if m["radius"] > 0]
        cov, inplane = coverage_inplane(full)
        cov_deg = cov * 180 / np.pi
        if cov_deg > 320 and inplane <= 40:
            cat = "hole_edge"
        elif cov_deg > 135:
            cat = "curve_edge"
        else:
            cat = "straight_edge"
        clusters.append({
            "gid": gid, "verts": verts, "category": cat,
            "radius": float(np.median(rs)) if rs else 0.0,
            "centroid": verts.mean(0), "coverage_deg": cov_deg,
            "inplane": inplane, "kind": "fillet",
        })

    # Chamfer clusters — mirror of job_queue's stitched chamfer block:
    # union-find by direct vertex contact, one cluster per physical
    # chamfer band, same apply path as fillets but kind="chamfer"
    # (RackMount's side-plate corner cuts live here).
    if chamfer_records:
        n_ch = len(chamfer_records)
        ch_parent = list(range(n_ch))

        def _ch_find(x):
            while ch_parent[x] != x:
                ch_parent[x] = ch_parent[ch_parent[x]]
                x = ch_parent[x]
            return x

        allv_ch = np.vstack([r["verts"] for r in chamfer_records])
        allu_ch = np.concatenate([
            np.full(len(r["verts"]), i) for i, r in enumerate(chamfer_records)
        ])
        for i, j in cKDTree(allv_ch).query_pairs(r=2.0):
            a, b = int(allu_ch[i]), int(allu_ch[j])
            ra, rb = _ch_find(a), _ch_find(b)
            if ra != rb:
                ch_parent[rb] = ra
        ch_groups = {}
        for i in range(n_ch):
            ch_groups.setdefault(_ch_find(i), []).append(i)
        gid0 = len(clusters)
        for off, mem_ch in enumerate(ch_groups.values()):
            cv = np.vstack([chamfer_records[i]["verts"] for i in mem_ch])
            radii = [chamfer_records[i]["radius"] for i in mem_ch
                     if chamfer_records[i]["radius"] > 0]
            setback = max(1.0, float(np.median(radii)) if radii else 1.0)
            clusters.append({
                "gid": gid0 + off, "verts": cv, "category": "curve_edge",
                "radius": setback, "centroid": cv.mean(0),
                "coverage_deg": 0.0, "inplane": 0.0, "kind": "chamfer",
            })
    return clusters, body_results, decomp


def coverage_inplane(verts):
    if len(verts) < 4:
        return 0.0, 0.0
    c = verts - verts.mean(0)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    best_cov, best_in = 0.0, 0.0
    for ax in (vh[2], vh[0]):
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        perp = c - np.outer(c @ ax, ax)
        arb = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
        u = arb - ax * (arb @ ax)
        un = np.linalg.norm(u)
        if un < 1e-9:
            continue
        u /= un
        v = np.cross(ax, u)
        ang = np.arctan2(perp @ v, perp @ u)
        if len(ang) < 4:
            continue
        s = np.sort(ang)
        gap = max(np.diff(s).max(), 2 * np.pi - (s[-1] - s[0]))
        cov = 2 * np.pi - gap
        if cov > best_cov:
            best_cov = cov
            pu, pv = perp @ u, perp @ v
            best_in = max(pu.max() - pu.min(), pv.max() - pv.min())
    return best_cov, best_in


# ---------------------------------------------------------------------------
# 2. Enumerate rebuilt-solid edges.
# ---------------------------------------------------------------------------
def enumerate_edges(solid):
    edges = []
    for e in solid.edges().vals():
        try:
            gt = e.geomType()
            sp, ep = e.startPoint(), e.endPoint()
        except Exception:
            continue
        s = np.array([sp.x, sp.y, sp.z])
        en = np.array([ep.x, ep.y, ep.z])
        length = float(np.linalg.norm(en - s))
        # sample points along the edge (param sampling handles circles)
        try:
            samples = []
            for t in np.linspace(0, 1, N_EDGE_SAMPLES):
                p = e.positionAt(t)
                samples.append([p.x, p.y, p.z])
            samples = np.array(samples)
        except Exception:
            samples = np.linspace(s, en, N_EDGE_SAMPLES)
        edges.append({
            "type": gt, "start": s, "end": en, "len": length,
            "samples": samples, "mid": samples[len(samples) // 2],
        })
    return edges


# ---------------------------------------------------------------------------
# 3. Match.
# ---------------------------------------------------------------------------
MIN_SPINE_LEN = 3.0  # ignore polygon-fragment edges shorter than this


def match(clusters, edges):
    results = []
    for c in clusters:
        ctree = cKDTree(c["verts"])
        # Robust fillet radius: topology's radius field is unreliable
        # (stores the sphere-fit / hole radius, not the fillet tube
        # radius). The cluster's SMALLEST PCA extent is the band
        # thickness — a stable upper bound on the tube radius. Taking
        # the min collapses garbage (19.9 -> 1.5) while leaving good
        # small radii untouched. This stops the snap tolerance from
        # blowing up on the two sphere-fit tab corners.
        cv = c["verts"] - c["verts"].mean(0)
        _, _, vh = np.linalg.svd(cv, full_matrices=False)
        principal = vh[0]
        smallest_ext = float(
            (cv @ vh[2]).max() - (cv @ vh[2]).min()
        )
        robust_r = min(c["radius"], smallest_ext) if c["radius"] > 0 else smallest_ext
        c["robust_radius"] = robust_r
        snap = max(SNAP_TOL_BASE, 0.6 * robust_r + 2.0)
        cand = []
        for ei, e in enumerate(edges):
            d, _ = ctree.query(e["samples"])
            frac = float(np.mean(d < snap))
            if frac < MIN_EDGE_FRAC:
                continue
            cand.append((ei, frac, float(d.mean()), e))
        if c["category"] == "hole_edge":
            # Hole-edge spine is the circular hole opening. Pick the
            # SINGLE nearest CIRCLE/ELLIPSE — the ring on this
            # cluster's own face. The other face's opening is also
            # in range (the cluster brushes it) but sits ~plate-
            # thickness farther, so nearest-by-mean-distance is the
            # correct one.
            circ = [x for x in cand if x[3]["type"] in ("CIRCLE", "ELLIPSE")]
            circ.sort(key=lambda x: x[2])
            chosen = circ[:1] if circ else []
        else:
            chosen = _select_crease(cand, snap, vh[2])
            chosen.sort(key=lambda x: (-x[3]["len"], x[2]))
        results.append((c, snap, chosen, principal))
    return results


def _select_crease(cand, snap, band_normal):
    """Select the connected crease edges for a curve/straight fillet.

    The crease is the LONG outline edges plus the SHORT corner arcs
    that bridge them — corner to corner. Excluded:
      - rim-thickness spanners: run along the band normal (across the
        plate), not in the outline plane.
      - junction connectors at an OPEN end of the U: short edges that
        dangle off a single spine edge instead of bridging two.
    """
    # In-plane edges only (drop edges running along the band normal).
    inplane = []
    for x in cand:
        e = x[3]
        d = e["end"] - e["start"]
        n = float(np.linalg.norm(d))
        if n < 1e-9 or e["len"] < 1.0 or x[2] > snap:
            continue
        d = d / n
        if abs(float(d @ band_normal)) < 0.6:
            inplane.append(x)
    if not inplane:
        return [x for x in cand if x[3]["len"] >= MIN_SPINE_LEN]
    max_len = max(x[3]["len"] for x in inplane)
    spine = [x for x in inplane if x[3]["len"] >= 0.25 * max_len]
    shorts = [x for x in inplane if x[3]["len"] < 0.25 * max_len]
    if not shorts or len(spine) < 2:
        return spine

    import collections

    def key(p):
        return tuple(np.round(p, 1).tolist())

    ep2short = collections.defaultdict(list)
    for i, x in enumerate(shorts):
        e = x[3]
        ep2short[key(e["start"])].append(i)
        ep2short[key(e["end"])].append(i)
    ep2spine = collections.defaultdict(set)
    for x in spine:
        e = x[3]
        ep2spine[key(e["start"])].add(x[0])
        ep2spine[key(e["end"])].add(x[0])

    # Connected components of short edges; keep a component only if it
    # touches >= 2 distinct spine edges (a corner bridge).
    seen = set()
    keep = []
    for i in range(len(shorts)):
        if i in seen:
            continue
        comp = []
        stack = [i]
        seen.add(i)
        touched = set()
        while stack:
            j = stack.pop()
            comp.append(j)
            e = shorts[j][3]
            for ep in (key(e["start"]), key(e["end"])):
                touched |= ep2spine.get(ep, set())
                for k in ep2short.get(ep, []):
                    if k not in seen:
                        seen.add(k)
                        stack.append(k)
        if len(touched) >= 2:
            keep.extend(comp)
    return spine + [shorts[i] for i in keep]


def run():
    cleaned = ingest_mesh(str(ROOT / "models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False))
    clusters, body_results, decomp = build_clusters(cleaned)
    tree0 = body_results[0][2]
    res = export_cadquery(tree0, cleaned, use_llm=False, num_variants=1,
                          body_results=body_results if decomp.is_multi_body else None)
    if res.solid is None:
        print("no solid")
        return 1
    edges = enumerate_edges(res.solid)
    print(f"{len(clusters)} clusters, {len(edges)} rebuilt edges\n")
    results = match(clusters, edges)
    n_matched = 0
    for c, snap, chosen, principal in sorted(results, key=lambda r: (r[0]["category"], -r[0]["verts"].shape[0])):
        cc = c["centroid"]
        tag = "OK" if chosen else "NO MATCH"
        if chosen:
            n_matched += 1
        print(f"[{tag}] cluster {c['gid']:>2} {c['category']:<13} "
              f"r={c['radius']:.1f}->{c.get('robust_radius',0):.1f} cov={c['coverage_deg']:.0f}° "
              f"centroid=({cc[0]:.0f},{cc[1]:.0f},{cc[2]:.0f}) snap={snap:.1f}mm "
              f"-> {len(chosen)} edge(s)")
        for ei, frac, md, e in chosen[:6]:
            em = e["mid"]
            print(f"        edge {ei:>3} {e['type']:<7} len={e['len']:>5.1f} "
                  f"on={frac*100:>3.0f}% meand={md:.2f} "
                  f"mid=({em[0]:.0f},{em[1]:.0f},{em[2]:.0f})")
    print(f"\n{n_matched}/{len(clusters)} clusters matched at least one edge")
    return 0


if __name__ == "__main__":
    sys.exit(run())
