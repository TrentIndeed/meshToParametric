"""Multi-extrude path gated against the REFERENCE STEP.

The user's demand after a batch of fabricated/phantom holes shipped:
"can't you check the output cad model to the step file or at least
compare cross sections to the mesh?" — this test does exactly that for
the single-body multi-extrude path (the server route for connected
solids): build RackMount end-to-end, then score the result against
models/Mesh5/RackMountStep.step by cross-section IoU, volume, and the
hole inventory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")

from mesh2param.config import IngestConfig  # noqa: E402
from mesh2param.pipeline.export_cadquery import export_cadquery  # noqa: E402
from mesh2param.pipeline.ingest import ingest_mesh  # noqa: E402
from mesh2param.pipeline.metrics import cross_section_fidelity  # noqa: E402

MODELS = Path(__file__).parent.parent / "models"


@pytest.fixture(scope="module")
def rackmount_result():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import match_fillets_to_edges as mfe

    cleaned = ingest_mesh(
        str(MODELS / "Mesh5" / "RackMount.3mf"), IngestConfig(remesh=False),
    )
    raw, body_results, _ = mfe.build_clusters(cleaned, single_body=True)
    clusters = [{
        "group_id": cl["gid"], "vertices": cl["verts"],
        "radius": cl["radius"], "category": cl["category"],
        "kind": cl.get("kind", "fillet"), "n_records": 0,
    } for cl in raw]
    _, _, tree = body_results[0]
    res = export_cadquery(
        tree, cleaned, use_llm=False, stitched_clusters=clusters,
    )
    ref = cq.importers.importStep(str(MODELS / "Mesh5" / "RackMountStep.step"))
    rv, rf = ref.val().tessellate(0.3)
    rv = np.array([[p.x, p.y, p.z] for p in rv])
    rf = np.array(rf)
    return cleaned, res, rv, rf


@pytest.mark.slow
def test_rackmount_one_part_n_extrudes(rackmount_result):
    """One feature tree, one timeline: base + 2 joins, no bodies."""
    _, res, _, _ = rackmount_result
    names = [s.name for s in res.construction_steps]
    assert "Extrude Base" in names
    assert sum(1 for n in names if n.startswith("Extrude Join")) == 2, names


@pytest.mark.slow
def test_rackmount_fidelity_vs_reference(rackmount_result):
    """Cross-section IoU against the reference STEP — the metric that
    cannot be gamed by volume matching. Phantom bores, fabricated holes
    and stuffed channels all crater the worst-section score."""
    _, res, rv, rf = rackmount_result
    assert res.status == "PASS", f"export status {res.status}"
    final = res.construction_steps[-1]
    fid = cross_section_fidelity(
        final.tess_vertices, final.tess_faces, rv, rf,
    )
    assert fid["mean_iou"] > 0.88, f"mean section IoU {fid['mean_iou']:.3f}"
    assert fid["worst_iou"] > 0.80, f"worst section IoU {fid['worst_iou']:.3f}"
    assert abs(res.volume_error_pct) < 12.0, (
        f"volume error {res.volume_error_pct:.1f}%"
    )


@pytest.mark.slow
def test_rackmount_hole_inventory(rackmount_result):
    """Exactly the reference's holes, verified GEOMETRICALLY: each side
    plate carries its r~14.5 bore (baked into the join profile — no
    duplicate drill step, no phantom offset bore), and the four tab
    holes drill as one pattern."""
    import trimesh

    _, res, _, _ = rackmount_result
    hole_steps = [
        s.name for s in res.construction_steps
        if "Hole" in s.name and not s.name.startswith("Sketch")
    ]
    assert len(hole_steps) == 1, f"hole steps: {hole_steps}"
    assert "r=4.0" in hole_steps[0] and "4" in hole_steps[0], hole_steps

    final = res.construction_steps[-1]
    solid = trimesh.Trimesh(
        vertices=final.tess_vertices, faces=final.tess_faces, process=True,
    )
    # Section through each side plate: exactly one bore ring at the
    # reference position (125.5, -133.4), r~14.5 — and no second ring.
    for x_plate in (42.5, -40.5):
        section = solid.section(
            plane_origin=[x_plate, 0, 0], plane_normal=[1, 0, 0],
        )
        assert section is not None
        rings = []
        for ring in section.discrete:
            ctr = ring.mean(axis=0)
            span = np.ptp(ring[:, 1])
            if 20 < span < 40:  # bore-sized loop
                rings.append((ctr[1], ctr[2], span / 2))
        assert len(rings) == 1, (
            f"plate x={x_plate}: expected exactly 1 bore ring, got {rings}"
        )
        cy, cz, r_est = rings[0]
        assert abs(cy - 125.5) < 2.0 and abs(cz - -133.4) < 2.0, rings
        assert abs(r_est - 14.5) < 1.5, rings
