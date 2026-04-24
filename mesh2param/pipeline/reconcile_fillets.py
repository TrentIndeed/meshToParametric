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
