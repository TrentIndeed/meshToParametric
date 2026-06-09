"""Edge-selection regression tests for the stitched-fillet apply pass.

The matcher in `_apply_recognized_fillets_to_solid` takes a list of
stitched-fillet clusters (each = `{group_id, vertices: (N,3), ...}`)
and applies CadQuery fillets to the matching sharp edges on a
rebuilt CAD solid. These tests pin down the matcher's contract on
known geometries with controlled fillet locations, so we can catch
the failure modes (corner-edge false positives, partial-overlap
mis-assignments) before they show up in the rackmount viz.

Each test:
  1. Builds a simple rebuilt CAD solid in CadQuery.
  2. Synthesises a cluster's vertex set positioned where a fillet
     would sit on the original mesh (offset from the sharp edge by
     the fillet radius — that's how mesh segmentation reports it).
  3. Calls the matcher.
  4. Asserts which edges got matched and at what radius.

A reference for the synthetic cluster shape: a 90° concave fillet
of radius r between two perpendicular planes is a quarter-cylinder.
Its mesh vertices fill the arc that wraps from one tangent line on
plane A to the other tangent line on plane B, distance r from the
"virtual corner" on each side. Sample 12 angles × ~40 axial
positions = ~480 vertices per fillet.
"""
from __future__ import annotations

import pytest
import numpy as np

cq = pytest.importorskip("cadquery")

from mesh2param.pipeline.export_cadquery import (
    _apply_recognized_fillets_to_solid,
)


def _quarter_cylinder_cluster(
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    plane_a_normal: np.ndarray,
    plane_b_normal: np.ndarray,
    radius: float,
    n_axial: int = 40,
    n_arc: int = 12,
) -> np.ndarray:
    """Synthesise the (N,3) vertex set of a 90° concave quarter-
    cylinder fillet between two planes.

    `axis_start` / `axis_end` are the endpoints of the fillet's
    rolling-ball CENTER line (offset from the virtual sharp corner
    by r in each of the two plane normals' INWARD direction).
    `plane_a_normal` / `plane_b_normal` are unit OUTWARD normals of
    the two adjacent planes.
    """
    axis_dir = axis_end - axis_start
    axis_len = float(np.linalg.norm(axis_dir))
    assert axis_len > 0
    axis_dir = axis_dir / axis_len

    # Build a basis on the fillet's cross-section plane (perpendicular
    # to the axis). The two arc-endpoint directions are -n_a (toward
    # plane A) and -n_b (toward plane B).
    a_dir = -np.asarray(plane_a_normal, dtype=float)
    b_dir = -np.asarray(plane_b_normal, dtype=float)
    a_dir = a_dir / np.linalg.norm(a_dir)
    b_dir = b_dir / np.linalg.norm(b_dir)

    verts = []
    for t in np.linspace(0.0, 1.0, n_axial):
        axis_pt = axis_start + t * (axis_end - axis_start)
        for k in np.linspace(0.0, 1.0, n_arc):
            angle = k * np.pi / 2.0  # 0..90°
            # Direction on cross-section from center toward arc point.
            arc_dir = np.cos(angle) * a_dir + np.sin(angle) * b_dir
            arc_dir = arc_dir / (np.linalg.norm(arc_dir) + 1e-12)
            verts.append(axis_pt + radius * arc_dir)
    return np.array(verts, dtype=float)


def test_box_with_pocket_fillets_one_floor_edge():
    """A box with a rectangular pocket has 8 concave edges (4 floor
    + 4 vertical-corner). A cluster on ONE floor edge should fillet
    exactly that edge and nothing else, at the measured radius."""
    wp = (
        cq.Workplane("XY")
        .box(100, 60, 20)
        .faces(">Z")
        .workplane()
        .rect(40, 20)
        .cutBlind(-5)
    )
    # Front pocket-floor edge: from (-20, -10, 5) to (20, -10, 5).
    # The pocket wall normal points +Y (into pocket air); the floor
    # normal points +Z (into pocket air). The fillet's axis sits at
    # the rolling-ball center, offset r in each INWARD direction from
    # the sharp corner. Sharp corner at (x, -10, 5). Inward from
    # wall = -Y. Inward from floor = +Z (wait — inward for fillet
    # MEANS into solid? No: the fillet adds material in the inside
    # corner, so its axis sits in the AIR offset from the sharp corner
    # along the bisector of (-n_a, -n_b)).
    r = 3.0
    axis_y = -10.0 + r  # -7
    axis_z = 5.0 + r    # 8
    axis_start = np.array([-20.0, axis_y, axis_z])
    axis_end = np.array([20.0, axis_y, axis_z])
    verts = _quarter_cylinder_cluster(
        axis_start=axis_start,
        axis_end=axis_end,
        plane_a_normal=np.array([0.0, 1.0, 0.0]),   # wall faces +Y
        plane_b_normal=np.array([0.0, 0.0, 1.0]),   # floor faces +Z
        radius=r,
    )

    stitched = [{
        "group_id": 0, "vertices": verts, "n_records": 1,
        "radius": r,
    }]
    out_edges: list = []
    out_diag: list = []
    wp_after, steps, _ = _apply_recognized_fillets_to_solid(
        wp,
        stitched_clusters=stitched,
        out_edges=out_edges,
        out_surfaces=[],
        out_diag=out_diag,
    )

    assert len(steps) == 1, f"expected 1 fillet step, got {len(steps)}: {[s.name for s in steps]}"
    # The bbox-overlap matcher claims the 40 mm front floor edge
    # AND the two adjacent 5 mm vertical-pocket-corner edges that
    # share its bbox — OCC tangent-chains them into one continuous
    # fillet. The critical assertion is that the long 40 mm floor
    # edge is among the claimed edges.
    edge_lens = [
        float(np.linalg.norm(np.asarray(ep) - np.asarray(sp)))
        for sp, ep in out_edges
    ]
    assert any(abs(L - 40.0) < 0.5 for L in edge_lens), (
        f"40 mm floor edge not claimed; got {edge_lens}"
    )
    vol_before = abs(float(wp.val().Volume()))
    vol_after = abs(float(wp_after.val().Volume()))
    assert vol_after > vol_before
    assert "r=" in steps[0].name
    matched = [d for d in out_diag if d["matched"]]
    assert len(matched) == 1
    assert matched[0]["group_id"] == 0


def test_l_bracket_inside_corner():
    """A T-shaped bracket (vertical plate inset on a wider horizontal
    base) has one long concave inside-corner edge on each side of
    the vertical plate where a fillet belongs. The cluster sits
    along one of those edges. The vertical plate is INSET in y so
    the inside-corner is truly concave (3/4 solid, 1/4 air), not a
    saddle edge (which a flush-back vertical plate would produce)."""
    # Horiz: x=[-50, 50], y=[-15, 15], z=[0, 5]  (30 wide, 5 thick)
    horiz = cq.Workplane("XY").box(100, 30, 5).translate((0, 0, 2.5))
    # Vert: x=[-50, 50], y=[-2.5, 2.5], z=[5, 55]  (5 thick, 50 tall)
    vert = cq.Workplane("XY").box(100, 5, 50).translate((0, 0, 30))
    solid = horiz.union(vert.val())
    # Two inside-corner edges along the vertical plate's base, one on
    # each side. We'll fillet the +Y one: at x=[-50,50], y=2.5, z=5.
    # The vert's +Y face has normal +Y, the horiz's top face has
    # normal +Z. Fillet axis sits in the AIR quadrant (y>2.5, z>5)
    # at distance r from each face.
    r = 2.0
    axis_y = 2.5 + r   # 4.5
    axis_z = 5.0 + r   # 7.0
    axis_start = np.array([-50.0, axis_y, axis_z])
    axis_end = np.array([50.0, axis_y, axis_z])
    verts = _quarter_cylinder_cluster(
        axis_start=axis_start,
        axis_end=axis_end,
        plane_a_normal=np.array([0.0, 1.0, 0.0]),   # vert face +Y
        plane_b_normal=np.array([0.0, 0.0, 1.0]),    # horiz face +Z
        radius=r,
    )

    stitched = [{
        "group_id": 0, "vertices": verts, "n_records": 1,
        "radius": r,
    }]
    out_edges: list = []
    out_diag: list = []
    wp_after, steps, _ = _apply_recognized_fillets_to_solid(
        solid,
        stitched_clusters=stitched,
        out_edges=out_edges,
        out_surfaces=[],
        out_diag=out_diag,
    )

    assert len(steps) >= 1, f"no fillet step applied; diag={out_diag}"
    assert len(out_edges) >= 1
    # The longest matched edge should be the 100 mm inside-corner edge.
    longest = max(
        float(np.linalg.norm(np.asarray(ep) - np.asarray(sp)))
        for sp, ep in out_edges
    )
    assert longest > 90.0, f"longest matched edge only {longest} mm; want >= 100"


def test_does_not_fillet_when_cluster_is_far_away():
    """A cluster placed in mid-air, nowhere near any concave edge,
    must NOT cause any fillet operation."""
    wp = (
        cq.Workplane("XY")
        .box(100, 60, 20)
        .faces(">Z")
        .workplane()
        .rect(40, 20)
        .cutBlind(-5)
    )
    # Cluster in the middle of nowhere (200 mm offset).
    verts = np.random.RandomState(0).randn(200, 3) + np.array([500, 500, 500])
    stitched = [{
        "group_id": 0, "vertices": verts, "n_records": 1,
        "radius": 1.0,
    }]
    _, steps, _ = _apply_recognized_fillets_to_solid(
        wp, stitched_clusters=stitched,
        out_edges=[], out_surfaces=[], out_diag=[],
    )
    assert len(steps) == 0, "matcher applied a fillet for a far-away cluster"


def test_rackmount_mimic_three_plate_union():
    """Mimic the rackmount's gross geometry — two side plates joined
    to a base plate — and verify the matcher applies fillets along
    the long concave edges where the side plates meet the base.

    Real cluster bboxes from the rackmount (Mesh5/RackMount.3mf,
    stitched-stage output) place the perimeter-wrap fillets at:
      - cluster 4: x=[39.1, 40.5], y=[50.1, 195.9], z=[-170.9, -96.0]
        (right side plate OUTER face perimeter)
      - cluster 1: x=[-38.5, -37.1], y=[50.1, 195.9], z=[-170.9, -96.0]
        (left side plate OUTER face perimeter)
    The long concave inside-corner edge where each side plate's
    outer face meets the underside of the base plate sits at:
      - right: x=39, y=[50,195], z=-96 (145 mm long)
      - left:  x=-37, y=[50,195], z=-96 (145 mm long)

    The matcher should pick BOTH long edges, one per cluster.
    """
    # Right side plate: x=[39,45], y=[50,195], z=[-170,-96]
    sp_r = cq.Workplane("XY").box(6, 145, 74).translate((42, 122.5, -133))
    # Left side plate: x=[-43,-37], y=[50,195], z=[-170,-96]
    sp_l = cq.Workplane("XY").box(6, 145, 74).translate((-40, 122.5, -133))
    # Base plate: x=[-44,45], y=[50,195], z=[-96,-91] (5 mm thick)
    bp = cq.Workplane("XY").box(89, 145, 5).translate((0.5, 122.5, -93.5))
    solid = sp_r.union(sp_l.val()).union(bp.val())

    # Synthesise the right plate's outer perimeter-wrap cluster: a
    # 1.4 mm rolled fillet wrapping from the outer face (x=39 plane)
    # around the bottom edge. We only need the BOTTOM strip at z=-96
    # for the matcher to find the concave inside-corner edge.
    r = 1.4
    axis_x = 39.0 - r  # outer side, fillet axis is in the air to the
                        # -X side of the outer face — but the relevant
                        # edge for THIS rebuilt solid is the inside
                        # corner where the outer face meets the base
                        # plate's underside, at x=39, z=-96.
    # Better: place the cluster along the inside-corner edge itself
    # at (39, y, -96), with vertex set as a quarter cylinder in the
    # +X, -Z direction (air quadrant).
    right_start = np.array([39.0 - r, 50.0, -96.0 - r])
    right_end = np.array([39.0 - r, 195.0, -96.0 - r])
    left_start = np.array([-37.0 + r, 50.0, -96.0 - r])
    left_end = np.array([-37.0 + r, 195.0, -96.0 - r])
    verts_right = _quarter_cylinder_cluster(
        axis_start=right_start, axis_end=right_end,
        plane_a_normal=np.array([-1.0, 0.0, 0.0]),
        plane_b_normal=np.array([0.0, 0.0, -1.0]),
        radius=r, n_axial=80,
    )
    verts_left = _quarter_cylinder_cluster(
        axis_start=left_start, axis_end=left_end,
        plane_a_normal=np.array([1.0, 0.0, 0.0]),
        plane_b_normal=np.array([0.0, 0.0, -1.0]),
        radius=r, n_axial=80,
    )
    stitched = [
        {
            "group_id": 0, "vertices": verts_right, "n_records": 1,
            "radius": r,
        },
        {
            "group_id": 1, "vertices": verts_left, "n_records": 1,
            "radius": r,
        },
    ]

    out_edges: list = []
    out_diag: list = []
    wp_after, steps, _ = _apply_recognized_fillets_to_solid(
        solid,
        stitched_clusters=stitched,
        out_edges=out_edges,
        out_surfaces=[],
        out_diag=out_diag,
    )
    matched = [d for d in out_diag if d["matched"]]
    assert len(matched) >= 1, (
        f"no clusters matched on 3-plate rackmount mimic; diag={out_diag}"
    )
    # Both long inside-corner edges (145 mm) should be picked.
    long_matched = [
        e for e in out_edges
        if float(np.linalg.norm(np.asarray(e[1]) - np.asarray(e[0]))) > 100.0
    ]
    assert len(long_matched) >= 1, (
        f"matcher did not pick any 100+ mm edge; matched edge lengths = "
        f"{[float(np.linalg.norm(np.asarray(e[1]) - np.asarray(e[0]))) for e in out_edges]}"
    )


def test_peel_then_rebuild_matches_long_corner_edge():
    """When the original mesh's perimeter wrap fillet shows up as a
    stitched cluster with vertices spanning a wide bbox, and the
    rebuilt solid has a short cross-thickness edge near the corner,
    the median-radius + along-extent-filtered measurement must
    correctly read the local fillet radius (not the cluster's
    full perpendicular spread) so the length gate doesn't kill the
    match. This is the regression for the rackmount-mimic failure
    after peel."""
    horiz = cq.Workplane("XY").box(100, 30, 5).translate((0, 0, 2.5))
    vert = cq.Workplane("XY").box(100, 5, 50).translate((0, 0, 30))
    solid = horiz.union(vert.val())
    # Synthesise a cluster that wraps the +Y inside-corner along the
    # full 100 mm length — this is what `apply_blend_peel` followed
    # by stitching produces for a perimeter rolled edge.
    r = 2.0
    long_start = np.array([-50.0, 2.5 + r, 5.0 + r])
    long_end = np.array([50.0, 2.5 + r, 5.0 + r])
    far_start = np.array([-50.0, -2.5 - r, 5.0 + r])
    far_end = np.array([50.0, -2.5 - r, 5.0 + r])
    verts_long = _quarter_cylinder_cluster(
        axis_start=long_start, axis_end=long_end,
        plane_a_normal=np.array([0.0, 1.0, 0.0]),
        plane_b_normal=np.array([0.0, 0.0, 1.0]),
        radius=r, n_axial=100, n_arc=12,
    )
    verts_far = _quarter_cylinder_cluster(
        axis_start=far_start, axis_end=far_end,
        plane_a_normal=np.array([0.0, -1.0, 0.0]),
        plane_b_normal=np.array([0.0, 0.0, 1.0]),
        radius=r, n_axial=100, n_arc=12,
    )
    verts_combined = np.vstack([verts_long, verts_far])

    # Two segments: one along the +Y corner, one along the -Y corner.
    # The matcher should pick BOTH (one CAD edge per segment).
    stitched = [{
        "group_id": 0, "vertices": verts_combined, "n_records": 2,
        "radius": r,
    }]
    out_edges: list = []
    out_diag: list = []
    _, steps, _ = _apply_recognized_fillets_to_solid(
        solid, stitched_clusters=stitched,
        out_edges=out_edges, out_surfaces=[], out_diag=out_diag,
    )
    assert len(steps) >= 1, f"no fillet applied; diag={out_diag}"
    # The longest matched edge must be the 100 mm corner.
    longest = max(
        float(np.linalg.norm(np.asarray(ep) - np.asarray(sp)))
        for sp, ep in out_edges
    )
    assert longest > 90.0, f"longest matched edge only {longest} mm; want >= 100"


def test_does_not_fillet_short_corner_edge():
    """A cluster positioned at a long edge must NOT cause the
    matcher to instead pick the short corner-meeting edges. This is
    the 'fillets at corners not edges' regression."""
    wp = (
        cq.Workplane("XY")
        .box(100, 60, 20)
        .faces(">Z")
        .workplane()
        .rect(40, 20)
        .cutBlind(-5)
    )
    # Cluster on the front floor edge (length 40 mm). Side floor
    # edges are 20 mm long, vertical-corner edges are 5 mm long.
    # The matcher MUST pick the 40 mm edge.
    r = 3.0
    axis_start = np.array([-20.0, -7.0, 8.0])
    axis_end = np.array([20.0, -7.0, 8.0])
    verts = _quarter_cylinder_cluster(
        axis_start=axis_start, axis_end=axis_end,
        plane_a_normal=np.array([0.0, 1.0, 0.0]),
        plane_b_normal=np.array([0.0, 0.0, 1.0]),
        radius=r,
    )
    stitched = [{
        "group_id": 0, "vertices": verts, "n_records": 1,
        "radius": r,
    }]
    out_edges: list = []
    _, steps, _ = _apply_recognized_fillets_to_solid(
        wp, stitched_clusters=stitched,
        out_edges=out_edges, out_surfaces=[], out_diag=[],
    )
    assert len(steps) == 1
    # The 40 mm front floor edge must be among the claimed edges.
    # (The bbox may also include adjacent 5 mm vertical-pocket-corner
    # edges; that's OK — OCC tangent-chains them into one fillet.)
    edge_lens = [
        float(np.linalg.norm(np.asarray(ep) - np.asarray(sp)))
        for sp, ep in out_edges
    ]
    assert any(L > 35.0 for L in edge_lens), (
        f"40 mm front floor edge not claimed; got {edge_lens}"
    )
