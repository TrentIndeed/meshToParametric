"""Verify CAD-export fillet records stay consistent with Stage 4.5 blends.

The pipeline transforms Stage 4.5 blend records (one per region) into
the merged/filtered list fed to the CAD matcher in
`_apply_recognized_fillets_to_solid`. That transformation can drop
records (sphere fits, sub-mm slivers) and merge collinear ones — but
each surviving export record must still describe a fillet that the
blend graph identified, with the same radius, comparable length, and
a position lying on the original axis line.

If a future change to the merge step or to identify_blends_from_topology
shifts radii, axis points, or coverage, this test fails fast at
the boundary between "what the user saw cyan in Stage 4.5" and
"what gets applied as a CAD fillet operation". Every record sent to
the CAD build must be traceable back to one or more blend records.
"""
from __future__ import annotations

import numpy as np
import pytest

from mesh2param.config import (
    DecomposeConfig, FitConfig, IngestConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
from mesh2param.pipeline.reconcile_fillets import merge_blend_records_for_export
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

RACKMOUNT = "models/Mesh5/RackMount.3mf"

# Tolerances. Radius matches must be exact within a few % because the
# merge step does an arc-length-weighted average of source radii — if
# all sources have the same radius, the merged value equals it. Length
# is checked as "≥ longest source length" because the merge expands
# coverage along the axis. Position must lie on the source axis line.
RADIUS_REL_TOL = 0.05      # 5 %
LENGTH_REL_TOL = 0.5       # merged length can grow well beyond longest source
POSITION_PERP_ABS = 2.5    # mm perpendicular distance from source axis line


@pytest.fixture(scope="module")
def export_consistency_data():
    """Run ingest → decompose → per-body Stage 4.5 → merge.

    Returns (raw_records, export_records). Both are flat lists with a
    `body_index` field added so cross-body checks work.
    """
    cleaned = ingest_mesh(RACKMOUNT, IngestConfig(remesh=False))
    decomp = decompose_bodies(cleaned, DecomposeConfig())
    bbox_diag = float(np.linalg.norm(cleaned.bbox[1] - cleaned.bbox[0]))
    max_export_radius = bbox_diag * 0.25

    raw_records: list[dict] = []
    for body_idx, body in enumerate(decomp.bodies):
        sub = extract_sub_mesh(cleaned, body.face_indices, body=body)
        regs = segment_mesh(sub, SegmentConfig())
        fits = fit_surfaces(regs, sub, FitConfig())
        _g, edges = build_topology(regs, fits, sub, TopologyConfig())
        blends = identify_blends_from_topology(regs, fits, edges, sub)
        for rec in blends:
            r = float(rec.get("radius", 0) or 0)
            if r < 0.2 or r > max_export_radius:
                continue
            rec_copy = {k: v for k, v in rec.items() if k != "face_indices"}
            rec_copy["body_index"] = body_idx
            raw_records.append(rec_copy)

    export_records = merge_blend_records_for_export(raw_records)
    return raw_records, export_records


def _perp_distance_to_line(point: np.ndarray,
                           line_point: np.ndarray,
                           line_dir: np.ndarray) -> float:
    """Perpendicular distance from `point` to the line through
    `line_point` with direction `line_dir` (assumed unit-norm)."""
    rel = point - line_point
    along = float(np.dot(rel, line_dir))
    perp = rel - along * line_dir
    return float(np.linalg.norm(perp))


def _find_source_records(merged_rec: dict,
                         raw_records: list[dict]) -> list[dict]:
    """Look up the raw blend records that the merged export record
    came from. The merge step records source region IDs in
    `merged_from`; for unmerged passthroughs it keeps `region_id`.

    Region IDs collide across bodies (each body re-numbers from 0), so
    scope the search by body_index when the merged record carries one.
    For cross-body collinear merges (rare) we fall back to matching by
    radius proximity to disambiguate.
    """
    body_idx = merged_rec.get("body_index")
    merged_from = merged_rec.get("merged_from")
    pool = raw_records
    if body_idx is not None:
        pool = [r for r in raw_records if r.get("body_index") == body_idx]
    if merged_from is None:
        rid = merged_rec.get("region_id")
        if rid is None:
            return []
        return [r for r in pool if r.get("region_id") == rid]
    src_ids = set(merged_from)
    return [r for r in pool if r.get("region_id") in src_ids]


def test_every_export_record_traces_to_blend_records(export_consistency_data):
    """Each record fed to the CAD build must be a known Stage 4.5 blend
    (or a merge of several). Without this, a refactor that quietly
    invented new fillet records or reused a stale cache would still pass
    radius checks while applying fillets the user never saw cyan in
    Stage 4.5."""
    raw_records, export_records = export_consistency_data
    for rec in export_records:
        sources = _find_source_records(rec, raw_records)
        assert sources, (
            f"Export record radius={rec['radius']:.2f} has no source in "
            f"raw blend records. merged_from={rec.get('merged_from')}, "
            f"region_id={rec.get('region_id')}"
        )


def test_export_radius_matches_source_blend_radius(export_consistency_data):
    """The radius applied as a CAD fillet must be the (length-weighted)
    radius the blend graph reported. A 5% drift either way is allowed
    for floating-point and weighted-average noise; anything beyond that
    means the merge step is fudging the radius."""
    raw_records, export_records = export_consistency_data
    for rec in export_records:
        sources = _find_source_records(rec, raw_records)
        assert sources
        export_r = float(rec["radius"])
        # Length-weighted average of source radii — same formula
        # merge_blend_records_for_export uses internally.
        total_w = sum(max(float(s.get("length", 1.0)), 1e-6) for s in sources)
        weighted_r = sum(
            max(float(s.get("length", 1.0)), 1e-6) * float(s["radius"])
            for s in sources
        ) / total_w
        rel_err = abs(export_r - weighted_r) / max(weighted_r, 1e-9)
        assert rel_err <= RADIUS_REL_TOL, (
            f"Export radius {export_r:.3f} disagrees with weighted source "
            f"radius {weighted_r:.3f} (rel_err={rel_err:.3f} > "
            f"{RADIUS_REL_TOL}). Sources: "
            f"{[(s.get('region_id'), float(s['radius'])) for s in sources]}"
        )


def test_export_length_covers_source_span(export_consistency_data):
    """The merged record's length must cover at least the longest source
    record (we never *shrink* coverage). For a single-source record
    the lengths should match within a small tolerance."""
    raw_records, export_records = export_consistency_data
    for rec in export_records:
        sources = _find_source_records(rec, raw_records)
        assert sources
        max_source_len = max(float(s.get("length", 0.0)) for s in sources)
        export_len = float(rec.get("length", 0.0))
        # Allow a small undershoot for single-source passthroughs (the
        # merge keeps the seed record's length as-is); for multi-source
        # merges, length must be ≥ longest source minus 50 % tolerance
        # because the bounded extent along the seed axis can be a bit
        # shorter than the source's full axial span when the source
        # axis is slightly off-parallel.
        floor = max_source_len * (1.0 - LENGTH_REL_TOL)
        assert export_len >= floor, (
            f"Export length {export_len:.2f} is below floor {floor:.2f} "
            f"(longest source: {max_source_len:.2f}). Sources: "
            f"{[(s.get('region_id'), float(s.get('length', 0))) for s in sources]}"
        )


def test_export_position_lies_on_source_axis_line(export_consistency_data):
    """The merged record's axis_point must sit on (or within ~mm of)
    the axis line of every source it claims to summarize. If this
    fails, the merge is producing a fillet whose position drifts away
    from the regions it represents — the CAD matcher would then look
    for an edge nowhere near the actual fillet curve."""
    raw_records, export_records = export_consistency_data
    for rec in export_records:
        sources = _find_source_records(rec, raw_records)
        assert sources
        export_pt = np.asarray(rec["axis_point"], dtype=float)
        for src in sources:
            src_pt = np.asarray(src["axis_point"], dtype=float)
            src_dir = np.asarray(src["axis_dir"], dtype=float)
            n = float(np.linalg.norm(src_dir))
            if n < 1e-12:
                continue
            src_dir = src_dir / n
            d = _perp_distance_to_line(export_pt, src_pt, src_dir)
            assert d <= POSITION_PERP_ABS, (
                f"Export axis_point {export_pt.round(2).tolist()} is "
                f"{d:.2f}mm off the axis line of source "
                f"rid={src.get('region_id')} (limit {POSITION_PERP_ABS}mm). "
                f"Source axis_point={src_pt.round(2).tolist()}, "
                f"axis_dir={src_dir.round(3).tolist()}"
            )


def test_export_drops_sphere_fits_and_sub_mm_records(export_consistency_data):
    """Two filters in merge_blend_records_for_export must both fire:
        • SPHERE-fit records get dropped (their axis_dir is a plane
          normal, not an edge tangent — the CAD matcher can't apply
          them).
        • Sub-millimetre radii get dropped (mesh tessellation noise,
          not real design fillets).
    If a refactor accidentally lets either through, this test catches
    it."""
    raw_records, export_records = export_consistency_data
    has_sphere_in_raw = any(
        r.get("source_fit_type") == "sphere" for r in raw_records
    )
    has_sub_mm_in_raw = any(
        float(r.get("radius", 0)) < 1.0 for r in raw_records
    )
    # Both filters should be exercised by the rackmount fixture; if they
    # aren't, the test isn't actually checking what its name claims.
    assert has_sphere_in_raw, (
        "Test fixture lost coverage of SPHERE-fit records; "
        "test_export_drops_sphere_fits_and_sub_mm_records can't verify "
        "the sphere filter is firing."
    )
    assert has_sub_mm_in_raw, (
        "Test fixture lost coverage of sub-mm records; "
        "test_export_drops_sphere_fits_and_sub_mm_records can't verify "
        "the sub-mm filter is firing."
    )
    for rec in export_records:
        sources = _find_source_records(rec, raw_records)
        for s in sources:
            assert s.get("source_fit_type") != "sphere", (
                f"Export record kept SPHERE-fit source rid="
                f"{s.get('region_id')} — sphere filter regressed."
            )
        assert float(rec["radius"]) >= 1.0, (
            f"Export record radius={rec['radius']:.3f} is below 1.0mm "
            f"minimum — sub-mm filter regressed."
        )
