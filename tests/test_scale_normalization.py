"""Scale normalization: parts far outside the tuned mm regime are
scaled to a canonical size at ingest so absolute thresholds hold, and
unit_scale records how to get back to original units.

Uses Mesh1 rescaled to 4% and 1200% as stand-ins for an inch-unit
export and an oversized scan.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import trimesh

from mesh2param.config import FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.metrics import recognized_holes
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

MESH1 = Path(__file__).parent.parent / "models" / "Mesh1" / "Body1Mesh.3mf"


def _scaled_mesh1(scale: float) -> str:
    loaded = trimesh.load(str(MESH1))
    if isinstance(loaded, trimesh.Scene):
        loaded = list(loaded.geometry.values())[0]
    loaded.apply_scale(scale)
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp.close()
    loaded.export(tmp.name)
    return tmp.name


def _run_through_recognize(cleaned):
    regions = segment_mesh(cleaned, SegmentConfig())
    fitted = fit_surfaces(regions, cleaned, FitConfig())
    graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    return recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())


def test_in_range_part_untouched():
    cleaned = ingest_mesh(MESH1, IngestConfig(remesh=False))
    assert "unit_scale" not in cleaned.metadata
    assert 190 < cleaned.bbox_diagonal < 230  # Mesh1's native ~209


@pytest.mark.parametrize("scale", [0.04, 12.0], ids=["tiny", "huge"])
def test_out_of_range_part_normalized(scale):
    path = _scaled_mesh1(scale)
    config = IngestConfig(remesh=False)
    cleaned = ingest_mesh(path, config)

    unit_scale = cleaned.metadata.get("unit_scale")
    assert unit_scale is not None, "out-of-range part was not normalized"
    # Normalized diagonal hits the canonical size
    assert abs(cleaned.bbox_diagonal - config.canonical_diag) < 1.0
    # unit_scale converts back: original diag = normalized / unit_scale
    original_diag = cleaned.bbox_diagonal / unit_scale
    assert abs(original_diag - 209.2 * scale) < 209.2 * scale * 0.01


@pytest.mark.parametrize("scale", [0.04, 12.0], ids=["tiny", "huge"])
def test_holes_recognized_at_any_input_scale(scale):
    """The mm-tuned recognition thresholds must work after normalization.

    Without normalization a 4%-scale Mesh1 (bbox diag ~8mm, hole radius
    0.1mm) loses every hole to absolute minimum-radius checks.
    """
    path = _scaled_mesh1(scale)
    cleaned = ingest_mesh(path, IngestConfig(remesh=False))
    unit_scale = cleaned.metadata["unit_scale"]

    tree = _run_through_recognize(cleaned)
    holes = recognized_holes(tree)
    assert len(holes) >= 6, f"only {len(holes)} holes recognized at scale {scale}"

    # Converting a hole radius back to original units recovers the
    # scaled true radius (2.5 * scale), within fit tolerance.
    radii_original = sorted(h["radius"] / unit_scale for h in holes)
    true_r = 2.5 * scale
    close = [r for r in radii_original if abs(r - true_r) < true_r * 0.15]
    assert len(close) >= 4, (
        f"expected >=4 holes near r={true_r:.4g} in original units, "
        f"got {[round(r, 4) for r in radii_original]}"
    )
