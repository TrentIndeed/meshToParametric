"""Cross-body fillet radius reconciliation.

When the pipeline decomposes a part into per-plate bodies, each body's
segmentation produces its own fillet records with their own radii. Two
fillets that are actually the same underlying edge (running from one
plate across a bend into the next plate) get independent RANSAC fits,
so their radii disagree by fit noise even when the true geometry is
the same.

This module groups per-body fillet records into cross-body clusters
(same axis line, similar radius) and assigns every member of a cluster
a single `unified_radius`. The export stage reads `unified_radius`
instead of the raw `radius` when applying the CadQuery fillet
operation, so the joined edge comes out seamless instead of jagged.

Chamfers and single-body fillets pass through unchanged; their
`unified_radius` just mirrors `radius`.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def reconcile_blend_radii(body_records: list[list[dict]]) -> list[list[dict]]:
    """Enrich each blend record with a `unified_radius` field.

    Args:
        body_records: one list of blend dicts per body, in body order.

    Returns:
        The same structure (mutated in-place for convenience). Every
        record gains `unified_radius` and `cluster_id`. Records in the
        same cluster share the same `unified_radius`.
    """
    flat: list[tuple[int, dict]] = []
    for bi, recs in enumerate(body_records):
        for rec in recs:
            flat.append((bi, rec))

    n = len(flat)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Only fillets participate in cross-body clustering. A chamfer
    # between plates is geometrically different on each side, and we
    # don't want to force a unified radius on it.
    for i in range(n):
        bi_i, rec_i = flat[i]
        if rec_i.get("kind") != "fillet":
            continue
        for j in range(i + 1, n):
            bi_j, rec_j = flat[j]
            if rec_j.get("kind") != "fillet":
                continue
            if bi_i == bi_j:
                continue  # same body — not a cross-body edge
            if _same_underlying_edge(rec_i, rec_j):
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for k in range(n):
        clusters[find(k)].append(k)

    for root, members in clusters.items():
        if len(members) == 1:
            _, rec = flat[members[0]]
            rec["unified_radius"] = float(rec.get("radius", 0.0))
            rec["cluster_id"] = root
            continue
        # Arc-length-weighted average. Longer members carry more
        # weight because they represent more of the true edge.
        total_w = 0.0
        weighted_sum = 0.0
        for k in members:
            _, rec = flat[k]
            w = max(float(rec.get("length", 1.0)), 1e-6)
            total_w += w
            weighted_sum += w * float(rec.get("radius", 0.0))
        unified = weighted_sum / total_w
        max_dev = max(
            abs(float(flat[k][1].get("radius", 0.0)) - unified) for k in members
        )
        logger.info(
            "Reconciled fillet cluster: %d members across bodies %s → "
            "unified r=%.4f (max deviation before: %.4f)",
            len(members),
            sorted({flat[k][0] for k in members}),
            unified,
            max_dev,
        )
        for k in members:
            _, rec = flat[k]
            rec["unified_radius"] = float(unified)
            rec["cluster_id"] = root

    return body_records


def merge_blend_records_for_export(
    records: list[dict],
    axis_dot_tol: float = 0.97,
    perp_dist_abs: float = 2.0,
    radius_tol_pct: float = 0.35,
    perimeter_radius_tol_pct: float = 0.20,
) -> list[dict]:
    """Collapse fragmented fillet records into per-edge / per-loop records
    for CAD export.

    Mesh segmentation splits a single physical fillet into many region
    records:
      • Within one body, a single perimeter rolled-edge loop (a closed
        loop around the outer face of a plate) gets ONE record per
        straight side plus extra records at each 3D corner cap — a
        plate with 4 sides + 4 corner caps yields 8+ records that all
        belong to ONE CAD fillet operation at one radius.
      • Across bodies, a fillet that runs from one body onto another
        gets one record per body.

    Two-pass merge:
      1. Collinear pass: union records on the same straight axis line
         (parallel axes, collinear axis points, comparable radii).
      2. Same-body / same-radius pass: union records in the same body
         whose radii match within `perimeter_radius_tol_pct`. This
         catches perimeter loops, where the segments all share one
         radius but run around different faces of the body so their
         axes are not parallel.

    Drops:
      • SPHERE-fit records: their `axis_dir` is set to a plane
        neighbor's normal (not a real edge tangent) by
        identify_blends_from_topology, so the CAD matcher can't apply
        them to any sharp edge anyway.
      • Records with radius < 0.4 mm (mesh-noise slivers).

    The blend graph viz still shows ALL original records (cyan) — only
    the export pipeline sees the merged list.
    """
    if not records:
        return []

    # Minimum radius for CAD export: 1.0mm. Sub-mm "fillets" are almost
    # always mesh tessellation noise (sharp edges that picked up tiny
    # rounded faces from triangulation jitter). Real design intent for
    # mechanical parts rarely uses fillets below 1mm — manufacturing
    # tolerances make sub-mm fillets impractical, and automated CAD
    # reconstruction tools typically snap to common values (1, 2, 3,
    # 5mm). The blend graph viz still shows ALL records; only the
    # export stage applies this cutoff.
    MIN_EXPORT_RADIUS = 1.0
    keepable = [
        r for r in records
        if r.get("source_fit_type") != "sphere"
        and float(r.get("radius", 0.0) or 0.0) >= MIN_EXPORT_RADIUS
    ]
    if not keepable:
        return []

    n = len(keepable)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pass 1: collinear records (same straight axis line). Each record
    # remains a per-edge entry — perimeter-loop combining is left to the
    # CAD matcher's radius bucket pass downstream, which applies one
    # OCC fillet operation to all edges in a radius bucket.
    for i in range(n):
        for j in range(i + 1, n):
            if _same_axis_line(
                keepable[i], keepable[j],
                axis_dot_tol=axis_dot_tol,
                perp_dist_abs=perp_dist_abs,
                radius_tol_pct=radius_tol_pct,
            ):
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for k in range(n):
        clusters[find(k)].append(k)

    merged: list[dict] = []
    for members in clusters.values():
        if len(members) == 1:
            single = dict(keepable[members[0]])
            # Uniform membership tags so downstream debug viz can map a
            # merged record back to its source region(s).
            rid = int(single.get("region_id", -1))
            single.setdefault("merged_from", [rid])
            single["seed_region_id"] = rid
            merged.append(single)
            continue
        # Arc-length-weighted radius
        total_w = 0.0
        rad_sum = 0.0
        for k in members:
            w = max(float(keepable[k].get("length", 1.0)), 1e-6)
            total_w += w
            rad_sum += w * float(keepable[k].get("radius", 0.0))
        unified_r = rad_sum / total_w
        # Pick the longest member as the seed for axis_dir / axis_point;
        # mean-direction can flip-flop when records point opposite ways
        # along the same line.
        seed_idx = max(members, key=lambda k: float(keepable[k].get("length", 0.0)))
        seed = keepable[seed_idx]
        seed_dir = np.asarray(seed["axis_dir"], dtype=float)
        seed_dir = seed_dir / max(float(np.linalg.norm(seed_dir)), 1e-12)
        # Compute the bounding extent of all members along the seed axis
        # so the merged record's length covers the full physical edge.
        anchor = np.asarray(seed["axis_point"], dtype=float)
        ts: list[float] = []
        for k in members:
            ap = np.asarray(keepable[k]["axis_point"], dtype=float)
            t = float(np.dot(ap - anchor, seed_dir))
            ln = float(keepable[k].get("length", 0.0))
            ts.append(t - ln / 2.0)
            ts.append(t + ln / 2.0)
        t_min, t_max = min(ts), max(ts)
        merged_length = t_max - t_min
        merged_axis_point = anchor + 0.5 * (t_min + t_max) * seed_dir
        merged_rec = {
            "kind": seed.get("kind", "fillet"),
            "radius": unified_r,
            "unified_radius": unified_r,
            "axis_dir": seed_dir.tolist(),
            "axis_point": merged_axis_point.tolist(),
            "length": merged_length,
            "num_faces": sum(int(keepable[k].get("num_faces", 0)) for k in members),
            "merged_from": [int(keepable[k].get("region_id", -1)) for k in members],
            "seed_region_id": int(seed.get("region_id", -1)),
        }
        if "body_index" in seed:
            merged_rec["body_index"] = seed["body_index"]
        merged.append(merged_rec)

    logger.info(
        "merge_blend_records_for_export: %d records -> %d merged "
        "(dropped %d sphere/sliver before merging)",
        len(records), len(merged), len(records) - len(keepable),
    )
    return merged


def _same_axis_line(
    a: dict,
    b: dict,
    axis_dot_tol: float,
    perp_dist_abs: float,
    radius_tol_pct: float,
) -> bool:
    """Same physical edge, judged by absolute perp distance (mm).

    Used by merge_blend_records_for_export. Differs from
    _same_underlying_edge in that the perpendicular distance threshold
    is in absolute mm — the radius-relative version misfires on thin
    plates where two parallel edges on opposite faces sit within
    radius distance of each other but are NOT the same edge.
    """
    try:
        ad_a = np.asarray(a["axis_dir"], dtype=float)
        ad_b = np.asarray(b["axis_dir"], dtype=float)
        ap_a = np.asarray(a["axis_point"], dtype=float)
        ap_b = np.asarray(b["axis_point"], dtype=float)
    except (KeyError, TypeError):
        return False
    na = float(np.linalg.norm(ad_a))
    nb = float(np.linalg.norm(ad_b))
    if na < 1e-12 or nb < 1e-12:
        return False
    ad_a = ad_a / na
    ad_b = ad_b / nb
    if abs(float(np.dot(ad_a, ad_b))) < axis_dot_tol:
        return False
    delta = ap_b - ap_a
    along = float(np.dot(delta, ad_a))
    perp = delta - along * ad_a
    if float(np.linalg.norm(perp)) > perp_dist_abs:
        return False
    r_a = float(a.get("radius", 0))
    r_b = float(b.get("radius", 0))
    if max(r_a, r_b) < 1e-12:
        return False
    if abs(r_a - r_b) / max(r_a, r_b) > radius_tol_pct:
        return False
    return True


def _same_underlying_edge(
    a: dict,
    b: dict,
    axis_dot_tol: float = 0.998,        # parallel within ~3.6°
    perp_dist_ratio: float = 0.5,       # perp distance < 50% of radius
    radius_tol_pct: float = 0.25,       # radii within 25%
) -> bool:
    """True iff two fillet records describe the same underlying edge
    in world-space (parallel axes, coincident axis lines, comparable
    radii)."""
    try:
        ad_a = np.asarray(a["axis_dir"], dtype=float)
        ad_b = np.asarray(b["axis_dir"], dtype=float)
        ap_a = np.asarray(a["axis_point"], dtype=float)
        ap_b = np.asarray(b["axis_point"], dtype=float)
    except (KeyError, TypeError):
        return False

    na = float(np.linalg.norm(ad_a))
    nb = float(np.linalg.norm(ad_b))
    if na < 1e-12 or nb < 1e-12:
        return False
    ad_a = ad_a / na
    ad_b = ad_b / nb

    if abs(float(np.dot(ad_a, ad_b))) < axis_dot_tol:
        return False

    delta = ap_b - ap_a
    along = float(np.dot(delta, ad_a))
    perp = delta - along * ad_a
    pos_tol = max(float(a.get("radius", 0)), float(b.get("radius", 0))) * perp_dist_ratio
    if float(np.linalg.norm(perp)) > pos_tol:
        return False

    r_a = float(a.get("radius", 0))
    r_b = float(b.get("radius", 0))
    if max(r_a, r_b) < 1e-12:
        return False
    if abs(r_a - r_b) / max(r_a, r_b) > radius_tol_pct:
        return False

    return True
