"""Hole-recognition fixes locked by reference geometry.

Body1Mesh's reference STEP has 7 unique bores: 5 r=2.5 through-holes
along Y — two of them TANGENT (centers 1.4mm apart, walls fused into one
mesh region) — plus 2 blind holes drilled into the plate edge along Z.
BottomMount is a 13mm drafted plate whose 10 bolt holes all pierce the
large (Z) face.

These pin down three failure modes:
  * a fused tangent pair averaged into ONE r=2.67 hole (split recovery)
  * the same bore counted twice via coaxial fit fragments (dedup)
  * edge holes classified THROUGH and punched 138mm out the far side
    (blind classification by wall-height vs part-extent)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mesh2param.config import (
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.models.feature import FeatureType
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

MODELS = Path(__file__).parent.parent / "models"
HOLE_TYPES = {FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE}


@pytest.fixture(scope="module")
def mesh1_tree():
    cleaned = ingest_mesh(
        str(MODELS / "Mesh1" / "Body1Mesh.3mf"), IngestConfig(remesh=False),
    )
    regions = segment_mesh(cleaned, SegmentConfig())
    fitted = fit_surfaces(regions, cleaned, FitConfig())
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    return recognize_features(
        regions, fitted, graph, edges, cleaned, RecognizeConfig(),
    )


def _holes(tree):
    out = []
    for f in tree.features:
        if f.type not in HOLE_TYPES:
            continue
        sg = f.sketch_geometry or {}
        for c in sg.get("centers") or [sg.get("center")]:
            out.append((f.type, float(sg.get("radius", 0)), np.asarray(c),
                        np.asarray(f.direction, dtype=float), float(f.depth)))
    return out


def test_mesh1_tangent_pair_recovered(mesh1_tree):
    """The two tangent bores near (71-72, 26, 35-36) come out as TWO
    r~2.5 holes, not one averaged r=2.67 hole."""
    y_holes = [h for h in _holes(mesh1_tree)
               if abs(h[3][1]) > 0.9 and h[2][0] > 60 and h[2][2] > 25]
    assert len(y_holes) == 2, (
        f"expected the tangent pair as 2 holes, got {len(y_holes)}"
    )
    for _t, r, _c, _d, _depth in y_holes:
        assert abs(r - 2.5) < 0.2, f"tangent-pair radius {r} should be ~2.5"


def test_mesh1_hole_count_matches_reference(mesh1_tree):
    """7 unique bores: 5 along Y + 2 along Z (no doubled fragments)."""
    holes = _holes(mesh1_tree)
    along_y = [h for h in holes if abs(h[3][1]) > 0.9]
    along_z = [h for h in holes if abs(h[3][2]) > 0.9]
    assert len(along_y) == 5, f"expected 5 Y holes, got {len(along_y)}"
    assert len(along_z) == 2, f"expected 2 Z holes, got {len(along_z)}"


def test_mesh1_edge_holes_are_blind(mesh1_tree):
    """The Z edge holes are 27-47mm deep in a 138mm extent — they must be
    BLIND with their measured depth, not THROUGH (which punched a 138mm
    cut out the far side of the part)."""
    z_holes = [h for h in _holes(mesh1_tree) if abs(h[3][2]) > 0.9]
    for t, _r, _c, _d, depth in z_holes:
        assert t == FeatureType.BLIND_HOLE
        assert depth < 60, f"blind edge hole depth {depth} looks like a through-cut"


def test_mesh2_holes_match_reference():
    """BottomMount (13mm drafted plate with 4 countersunk holes): exactly
    the reference STEP's 13 bores — 9x r=2.5 + 4x r=4.0, all piercing the
    large Z face, none intersecting (no misplaced countersink "holes", no
    doubled fragments, none missing)."""
    cleaned = ingest_mesh(
        str(MODELS / "Mesh2" / "BottomMount.3mf"), IngestConfig(remesh=False),
    )
    regions = segment_mesh(cleaned, SegmentConfig())
    fitted = fit_surfaces(regions, cleaned, FitConfig())
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    tree = recognize_features(
        regions, fitted, graph, edges, cleaned, RecognizeConfig(),
    )
    holes = _holes(tree)
    assert len(holes) == 13, f"expected 13 holes, got {len(holes)}"
    small = [h for h in holes if abs(h[1] - 2.5) < 0.3]
    big = [h for h in holes if abs(h[1] - 4.0) < 0.3]
    assert len(small) == 9, f"expected 9 r=2.5 holes, got {len(small)}"
    assert len(big) == 4, f"expected 4 r=4.0 holes, got {len(big)}"
    for _t, _r, _c, d, _depth in holes:
        assert abs(d[2]) > 0.9, f"hole axis {d.tolist()} not on the Z large plane"
    # No two holes may intersect (the countersink cones must not surface
    # as offset overlapping bores).
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            _, r1, c1, _, _ = holes[i]
            _, r2, c2, _, _ = holes[j]
            d = float(np.linalg.norm(np.asarray(c1) - np.asarray(c2)))
            assert d >= r1 + r2, (
                f"holes intersect: r={r1:.2f}@{np.round(c1,1).tolist()} and "
                f"r={r2:.2f}@{np.round(c2,1).tolist()} (d={d:.2f})"
            )
