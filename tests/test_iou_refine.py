"""IoU residual-refinement post-pass (mesh2param.pipeline.iou_refine).

Synthetic recognition-error cases: a recon missing a boss (under-residual ->
add patch), a recon with an extra slab the target lacks (over-residual -> cut
patch), and a perfect recon (no patches). Validates voxel IoU and that each
accepted patch strictly improves IoU.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from mesh2param.pipeline.iou_refine import (  # noqa: E402
    refine_against_target, voxel_iou,
)


def _box(extents, center):
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b


def test_mesh_iou_exact_matches_geometry():
    from mesh2param.pipeline.iou_refine import mesh_iou_exact
    a = _box([10, 10, 10], [0, 0, 0])
    assert mesh_iou_exact(a, a.copy()) > 0.99
    # half-overlap: two equal boxes shifted by half -> exact IoU = 1/3
    b = _box([10, 10, 10], [5, 0, 0])
    iou = mesh_iou_exact(a, b)
    assert 0.30 < iou < 0.36, iou           # sharper than voxel's 0.2-0.45
    # disjoint
    far = _box([4, 4, 4], [100, 0, 0])
    assert mesh_iou_exact(a, far) < 1e-6
    # exact IoU never exceeds 1 (the multi-body manifold bug is guarded)
    assert mesh_iou_exact(a, b) <= 1.0


def test_voxel_iou_identical_is_one():
    b = _box([10, 10, 10], [0, 0, 0])
    assert voxel_iou(b, b.copy(), res=40) > 0.97


def test_voxel_iou_disjoint_is_zero():
    a = _box([4, 4, 4], [0, 0, 0])
    b = _box([4, 4, 4], [100, 0, 0])
    assert voxel_iou(a, b, res=40) < 1e-6


def test_voxel_iou_partial_overlap_between_0_and_1():
    a = _box([10, 10, 10], [0, 0, 0])
    b = _box([10, 10, 10], [5, 0, 0])  # half-overlap
    iou = voxel_iou(a, b, res=48)
    assert 0.2 < iou < 0.45  # ~1/3 for half-shifted equal boxes


def test_under_residual_gets_add_patch():
    # Target = base slab + a LARGE block on top; recon = base slab only. The
    # block is big relative to the base so the IoU gain is unambiguous (small
    # features are bounded by the voxel-quantization ceiling, ~0.95).
    base = _box([20, 20, 6], [0, 0, 3])
    block = _box([20, 20, 12], [0, 0, 12])
    target = trimesh.util.concatenate([base, block])
    recon = base.copy()

    res = refine_against_target(target, recon, res=56, max_iters=4)
    assert res.iou_final > res.iou_initial + 0.05
    assert any(o.op == "add" for o in res.ops), [o.op for o in res.ops]
    # Each accepted op strictly improved IoU.
    for o in res.ops:
        assert o.iou_after > o.iou_before


def test_over_residual_gets_cut_patch():
    # Target = slab; recon = slab + a LARGE extra block it shouldn't have.
    target = _box([20, 20, 6], [0, 0, 3])
    extra = _box([20, 20, 12], [0, 0, 12])
    recon = trimesh.util.concatenate([target.copy(), extra])

    res = refine_against_target(target, recon, res=56, max_iters=4)
    assert res.iou_final > res.iou_initial + 0.05
    assert any(o.op == "cut" for o in res.ops), [o.op for o in res.ops]
    for o in res.ops:
        assert o.iou_after > o.iou_before


def test_perfect_recon_no_ops():
    target = _box([12, 8, 5], [1, 2, 3])
    res = refine_against_target(target, target.copy(), res=48)
    assert res.ops == []
    assert res.iou_final > 0.97


def test_box_cover_splits_l_shaped_residual():
    # An L-shaped residual must be covered by MORE than one box (a single bbox
    # would overshoot into the empty corner).
    from mesh2param.pipeline.iou_refine import _box_cover
    mask = np.zeros((40, 40, 10), dtype=bool)
    mask[2:38, 2:12, 2:8] = True     # one limb
    mask[2:12, 2:38, 2:8] = True     # perpendicular limb -> L
    origin = np.zeros(3)
    boxes = _box_cover(mask, origin, pitch=1.0, min_cells=20, max_boxes=10)
    # Splits the non-convex shape into multiple candidate boxes (a single bbox
    # would engulf the empty corner). Aggregate tightness is enforced by the
    # loop's per-box IoU gate, proven in the end-to-end L test below — here we
    # just check the structural split and that no candidate escapes the bbox.
    assert len(boxes) >= 2, len(boxes)
    idx = np.argwhere(mask)
    gmin = origin + (idx.min(0) - 0.5)
    gmax = origin + (idx.max(0) + 0.5)
    for bmin, bmax in boxes:
        assert np.all(bmin >= gmin - 1e-6) and np.all(bmax <= gmax + 1e-6)


def test_l_shaped_residual_refines_better_than_bbox():
    # Target = base slab + an L-shaped wall on top; recon = base slab only.
    base = _box([40, 40, 6], [0, 0, 3])
    limb1 = _box([36, 8, 10], [0, -14, 11])
    limb2 = _box([8, 36, 10], [-14, 0, 11])
    target = trimesh.util.concatenate([base, limb1, limb2])
    recon = base.copy()
    res = refine_against_target(target, recon, res=56, max_iters=4)
    assert res.iou_final > res.iou_initial + 0.05
    # The L is covered by multiple add boxes, not one overshooting bbox.
    assert len([o for o in res.ops if o.op == "add"]) >= 2, [o.op for o in res.ops]


def test_cylindrical_residual_yields_cylinder_op():
    # Target = slab + a round boss; recon = slab. The round residual should be
    # recognized as a CYLINDER, not staircased into boxes.
    base = _box([40, 40, 6], [0, 0, 3])
    boss = trimesh.creation.cylinder(radius=9.0, height=14.0)  # axis = Z
    boss.apply_translation([0, 0, 13])
    target = trimesh.util.concatenate([base, boss])
    recon = base.copy()
    res = refine_against_target(target, recon, res=64, max_iters=4)
    assert res.iou_final > res.iou_initial + 0.03
    kinds = [(o.kind, o.op) for o in res.ops]
    assert any(k == "cylinder" and op == "add" for k, op in kinds), kinds


def test_round_hole_residual_yields_cylinder_cut():
    # Target = slab with a round hole; recon = solid slab (hole not cut).
    slab = _box([40, 40, 12], [0, 0, 0])
    drill = trimesh.creation.cylinder(radius=7.0, height=20.0)  # axis Z
    target = slab.difference(drill)
    if not getattr(target, "is_volume", True) or target.volume <= 0:
        pytest.skip("boolean backend unavailable")
    recon = slab.copy()
    res = refine_against_target(target, recon, res=72, max_iters=4)
    kinds = [(o.kind, o.op) for o in res.ops]
    assert any(k == "cylinder" and op == "cut" for k, op in kinds), kinds


def test_l_residual_yields_prism_or_boxes():
    # An L-shaped prismatic residual becomes ONE clean prism (preferred) or a
    # box-cover fallback — either way IoU improves.
    base = _box([40, 40, 6], [0, 0, 3])
    limb1 = _box([36, 8, 10], [0, -14, 11])
    limb2 = _box([8, 36, 10], [-14, 0, 11])
    target = trimesh.util.concatenate([base, limb1, limb2])
    recon = base.copy()
    res = refine_against_target(target, recon, res=56, max_iters=4)
    assert res.iou_final > res.iou_initial + 0.05
    kinds = [o.kind for o in res.ops if o.op == "add"]
    assert ("prism" in kinds) or (len(kinds) >= 2), kinds


def test_refinement_never_lowers_iou():
    base = _box([20, 20, 6], [0, 0, 3])
    boss = _box([8, 5, 7], [3, 0, 9])
    target = trimesh.util.concatenate([base, boss])
    recon = base.copy()
    res = refine_against_target(target, recon, res=56)
    assert res.iou_final >= res.iou_initial
