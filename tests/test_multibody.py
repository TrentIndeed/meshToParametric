"""Multi-body end-to-end regression test (RackMount).

The multi-body fillet/apply path is the most-iterated, most-failure-prone
subsystem in the pipeline and previously had no end-to-end coverage — the
stitched-cluster builder lives as an inline block in job_queue and the
CAD apply is reached only via the server. This test exercises the
production export_cadquery multi-body path (decompose -> per-body
pipeline -> stitched clusters -> build + fillet apply) and LOCKS the
hard-won reconstruction quality so it can't silently regress:

  - 3 bodies decomposed,
  - all stitched fillet clusters apply (the U-wrap continuity work),
  - volume within tolerance of the mesh,
  - hole pattern grouping survives (one sketch for the 4 bottom holes).

Slow: full multi-body build + OCC fillet apply on RackMount.
"""

from __future__ import annotations

from pathlib import Path

import pytest

cq = pytest.importorskip("cadquery")

from mesh2param.config import IngestConfig  # noqa: E402
from mesh2param.pipeline.ingest import ingest_mesh  # noqa: E402
from mesh2param.pipeline.export_cadquery import export_cadquery  # noqa: E402
from mesh2param.pipeline.metrics import blend_apply_report  # noqa: E402

RACKMOUNT = Path(__file__).parent.parent / "models" / "Mesh5" / "RackMount.3mf"


def _build_multibody():
    """Decompose + per-body pipeline + stitched clusters via the validated
    mirror in scripts/match_fillets_to_edges (the same logic the server's
    stitched stage runs)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import match_fillets_to_edges as mfe

    cleaned = ingest_mesh(str(RACKMOUNT), IngestConfig(remesh=False))
    raw, body_results, decomp = mfe.build_clusters(cleaned)
    # Map the script's cluster dicts to the payload shape export_cadquery
    # consumes (group_id/vertices), matching the server stitched stage.
    clusters = [{
        "group_id": c["gid"],
        "vertices": c["verts"],
        "radius": c["radius"],
        "category": c["category"],
        "n_records": 0,
    } for c in raw]
    return cleaned, clusters, body_results, decomp


@pytest.mark.slow
def test_rackmount_multibody_reconstruction():
    cleaned, clusters, body_results, decomp = _build_multibody()

    # RackMount is a 3-body part (two side plates + base).
    assert decomp.is_multi_body
    assert len(body_results) == 3, f"expected 3 bodies, got {len(body_results)}"

    # The four side-plate U-wraps + hole-edge rim fillets get stitched.
    assert len(clusters) >= 10, f"only {len(clusters)} stitched clusters"

    res = export_cadquery(
        body_results[0][2], cleaned, use_llm=False, num_variants=1,
        body_results=body_results, stitched_clusters=clusters,
    )
    assert res.status == "PASS", f"export status {res.status}"

    # Volume within tolerance of the mesh (multi-body union + holes + fillets).
    assert res.volume_error_pct < 10.0, (
        f"volume error {res.volume_error_pct:.1f}% exceeds 10%"
    )

    # Every stitched fillet cluster must match an edge and apply — this is
    # the U-wrap continuity + per-cluster apply work. A drop here means a
    # wrap fragmented or went unmatched again.
    report = blend_apply_report(res.fillet_match_diag)
    assert report["recognized"] >= 10
    assert report["match_ratio"] == 1.0, (
        f"only {report['matched']}/{report['recognized']} fillet clusters "
        f"matched: {report['unmatched']}"
    )

    # Hole-pattern grouping: the 4 identical bottom-plate holes stay one
    # sketch (one Hole Pattern step), not 4 separate cuts.
    pattern_steps = [
        s.name for s in res.construction_steps
        if "Hole Pattern" in s.name and "Sketch" in s.name
    ]
    assert any("4" in s for s in pattern_steps), (
        f"4-hole bottom-plate pattern not grouped into one sketch; "
        f"pattern steps = {pattern_steps}"
    )


@pytest.mark.slow
def test_rackmount_fillets_are_whole_mm():
    """Applied fillet radii are whole millimetres >= 2mm (the design-intent
    rounding), not the sub-mm phantom radii the matcher produced before the
    curvature-radius work."""
    cleaned, clusters, body_results, decomp = _build_multibody()
    res = export_cadquery(
        body_results[0][2], cleaned, use_llm=False, num_variants=1,
        body_results=body_results, stitched_clusters=clusters,
    )
    fillet_steps = [s.name for s in res.construction_steps if s.name.startswith("Fillet")]
    assert fillet_steps, "no fillet steps applied"
    for name in fillet_steps:
        # name like "Fillet (r=2mm) x9"
        r = float(name.split("r=")[1].split("mm")[0])
        assert r >= 2.0 and r == round(r), f"non-whole-mm/sub-2mm fillet: {name}"
