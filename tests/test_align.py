"""Canonical-frame alignment (scan front-end, stage B)."""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from mesh2param.pipeline.align import align_to_canonical_frame


def test_aligns_arbitrarily_rotated_box():
    box = trimesh.creation.box(extents=[40.0, 20.0, 10.0])
    R = trimesh.transformations.rotation_matrix(0.7, [1.0, 2.0, 3.0])
    box.apply_transform(R)
    box.apply_translation([13.0, -5.0, 22.0])

    aligned, T, method = align_to_canonical_frame(box)

    # box dims recovered (any order)
    ext = sorted((aligned.bounds[1] - aligned.bounds[0]).tolist(),
                 reverse=True)
    assert np.allclose(ext, [40.0, 20.0, 10.0], atol=1.0), ext
    # every face normal is now axis-aligned
    n = np.asarray(aligned.face_normals)
    off = np.degrees(np.arccos(np.clip(np.max(np.abs(n), axis=1), 0, 1)))
    assert float(off.max()) < 1.0
    # a clean box of orthogonal planes -> the normals strategy
    assert method == "normals"
    # intuitive pose: largest extent on X, smallest on Z
    e = aligned.bounds[1] - aligned.bounds[0]
    assert e[0] >= e[1] - 1e-6 >= 0 and e[1] >= e[2] - 1e-6


def test_transform_is_rigid():
    box = trimesh.creation.box(extents=[30.0, 18.0, 7.0])
    box.apply_transform(trimesh.transformations.rotation_matrix(1.1, [2, 1, 4]))
    aligned, T, _ = align_to_canonical_frame(box)
    R = T[:3, :3]
    # pure rotation/permutation: orthonormal, det +1, volume preserved
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    assert np.isclose(abs(aligned.volume), abs(box.volume), rtol=1e-3)


@pytest.mark.slow
def test_scan1_wrench_becomes_axis_aligned():
    """End-to-end: the wrench scan, which the engine reads as 0%
    coverage in its arbitrary pose, becomes a flat axis-aligned part the
    engine can read once the scan front-end runs."""
    from pathlib import Path

    from mesh2param.config import IngestConfig
    from mesh2param.pipeline.extrusion import plan_extrusions
    from mesh2param.pipeline.ingest import ingest_mesh

    scan = Path(__file__).parent.parent / "models/Scan1/Spanner-stl.stl"
    if not scan.exists():
        pytest.skip("Scan1 not present")
    cleaned = ingest_mesh(str(scan), IngestConfig(
        remesh=False, align_canonical=True, denoise_scan=True))
    ext = cleaned.bbox[1] - cleaned.bbox[0]
    # flat part: thinnest dim << longest, and it's on Z
    assert ext.min() / ext.max() < 0.15
    assert int(np.argmin(ext)) == 2
    # the dominant flat face is axis-aligned
    n = cleaned.normals
    off = np.degrees(np.arccos(np.clip(np.max(np.abs(n), axis=1), 0, 1)))
    assert float(np.median(off)) < 10.0
    # and the engine now reads the part (was 0% in the raw pose)
    plan = plan_extrusions(cleaned)
    assert plan.coverage_frac > 0.6
