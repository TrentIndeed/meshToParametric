"""Scan smoke tests: high-poly scanned meshes must decimate and run the
full pipeline WITHOUT crashing, within a time budget.

Scans are non-watertight organic geometry, so we do NOT assert
reconstruction accuracy (the prismatic pipeline can't model them well —
see docs/TODO.md). These tests guard that the decimation path keeps the
stages alive and bounded, which is what regressed before the face cap.

Marked slow: scan files are 0.3–2.5M faces and take 20–60s just to load.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mesh2param.config import (
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

MODELS = Path(__file__).parent.parent / "models"
SCANS = [
    ("Scan1", MODELS / "Scan1" / "Spanner-stl.stl"),
    ("Scan2", MODELS / "Scan2" / "Industrial Part.obj"),
    ("Scan3", MODELS / "Scan3" / "Cam-gear.stl"),
]
SCANS = [(n, p) for n, p in SCANS if p.exists()]


@pytest.mark.slow
@pytest.mark.parametrize("name,path", SCANS, ids=[n for n, _ in SCANS])
def test_scan_decimates_and_runs(name, path):
    cfg = IngestConfig(remesh=False)

    t0 = time.perf_counter()
    cleaned = ingest_mesh(path, cfg)
    ingest_s = time.perf_counter() - t0

    # Decimation must cap the face count near the budget.
    assert cleaned.num_faces <= cfg.max_faces * 1.1, (
        f"{name}: {cleaned.num_faces} faces exceeds cap {cfg.max_faces}"
    )
    # Poisson reconstruction (on by default) must leave scans watertight,
    # so the downstream volume metric and average-cross-section base
    # profile have a real volume to work with.
    assert cleaned.is_watertight, (
        f"{name}: not watertight after ingest (Poisson reconstruction failed)"
    )
    # Ingest (load + 2-stage decimate) must stay bounded even for 2.5M-face
    # scans — the clustering pre-pass keeps it well under two minutes.
    assert ingest_s < 120, f"{name}: ingest took {ingest_s:.0f}s"

    regions = segment_mesh(cleaned, SegmentConfig())
    assert len(regions) > 0

    fitted = fit_surfaces(regions, cleaned, FitConfig())
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())
    # At minimum a base extrusion is always produced.
    assert tree.num_features >= 1
