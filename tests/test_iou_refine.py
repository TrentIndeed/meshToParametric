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


def test_refinement_never_lowers_iou():
    base = _box([20, 20, 6], [0, 0, 3])
    boss = _box([8, 5, 7], [3, 0, 9])
    target = trimesh.util.concatenate([base, boss])
    recon = base.copy()
    res = refine_against_target(target, recon, res=56)
    assert res.iou_final >= res.iou_initial
