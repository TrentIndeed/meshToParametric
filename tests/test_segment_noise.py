"""Noise-robust segmentation: boundary-dominance fragment merge.

A noisy scan / AI mesh shatters a flat face into one big region plus tiny
slivers whose normals are too noisy to match. The merge absorbs a sliver
into the big face that owns most of its boundary edges — regardless of the
sliver's normal — and never across a true right-angle edge.
"""
from __future__ import annotations

import numpy as np

from mesh2param.config import SegmentConfig
from mesh2param.pipeline.segment import _merge_coplanar_fragments


def test_embedded_slivers_absorbed_despite_wild_normals():
    # Big +Z face (label 0, 20 faces) with 3 size-1 slivers (labels 1,2,3)
    # carrying WILD normals, each embedded (bordered only by +Z faces); plus
    # a separate big +X face (label 100).
    z = 20
    normals = np.array(
        [[0, 0, 1.0]] * z
        + [[0.6, 0.6, 0.5], [-0.5, 0.5, 0.7], [0.7, -0.3, 0.6]]   # wild slivers
        + [[1.0, 0, 0]] * 20)
    labels = np.array([0] * z + [1, 2, 3] + [100] * 20)
    adj = {i: set() for i in range(len(labels))}
    for i in range(z - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    for k, sl in enumerate((z, z + 1, z + 2)):       # each sliver embedded in +Z
        a, b = k * 2, k * 2 + 1
        adj[sl].update({a, b})
        adj[a].add(sl)
        adj[b].add(sl)
    base = z + 3
    for i in range(base, base + 19):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    adj[0].add(base)
    adj[base].add(0)               # one cross edge at the 90° join

    out = _merge_coplanar_fragments(
        labels.copy(), normals, adj,
        SegmentConfig(noise_robust=True, min_region_faces=10))
    # slivers gone (absorbed into +Z by boundary dominance), +Z and +X distinct
    assert len(np.unique(out)) == 2


def test_edge_sliver_does_not_cross_right_angle():
    # A sliver straddling the 90° edge: half its boundary on +Z, half on +X
    # (no dominant face) -> must NOT merge into either.
    normals = np.array([[0, 0, 1.0]] * 15 + [[5.0, 1, 5]] + [[1.0, 0, 0]] * 15)
    labels = np.array([0] * 15 + [1] + [100] * 15)
    adj = {i: set() for i in range(31)}
    for i in range(14):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    for i in range(16, 30):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    # sliger (15) borders equally: two +Z faces and two +X faces
    for nb in (0, 1, 16, 17):
        adj[15].add(nb)
        adj[nb].add(15)

    out = _merge_coplanar_fragments(
        labels.copy(), normals, adj,
        SegmentConfig(noise_robust=True, min_region_faces=10,
                      fragment_dominance_frac=0.6))
    assert out[15] not in (out[0], out[16])   # sliver kept its own region
