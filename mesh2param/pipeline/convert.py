"""Single source of truth for the mesh->feature-tree pipeline.

The web server (viz_server/job_queue.py) and standalone scripts/tests MUST run
the SAME computation, or a standalone reproduction won't match what a user sees
in the app. This module holds that shared computation so the two can't drift:
both call `recognize_body` for a single body, and `convert_mesh` for the whole
end-to-end (ingest -> decompose -> per-body recognize -> assemble tree).

The server wraps these with its own visualization + pause/step orchestration,
but the GEOMETRY/FEATURE results come from here, identically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass


from mesh2param.config import (
    DecomposeConfig, FitConfig, IngestConfig, RecognizeConfig, SegmentConfig,
    TopologyConfig,
)
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.tree import FeatureTree

logger = logging.getLogger(__name__)


@dataclass
class BodyResult:
    tree: FeatureTree
    regions: list
    fitted: list
    graph: object
    edges: list
    blends: list


def recognize_body(sub_cleaned: CleanedMesh, scan_mode: bool) -> BodyResult:
    """Single-body recognition core, shared by server and scripts.

    segment -> fit -> (if dirty: regularize -> re-segment -> re-fit) ->
    topology -> blend identification -> recognize_features. This is EXACTLY
    what viz_server/job_queue._process_single_body runs (minus the viz), so a
    standalone reproduction calling this matches the app's feature tree.
    """
    from mesh2param.pipeline.fit import fit_surfaces
    from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
    from mesh2param.pipeline.recognize import recognize_features
    from mesh2param.pipeline.segment import segment_mesh
    from mesh2param.pipeline.topology import build_topology

    seg = SegmentConfig(noise_robust=scan_mode)
    regions = segment_mesh(sub_cleaned, seg)
    fitted = fit_surfaces(regions, sub_cleaned, FitConfig())
    if scan_mode:
        try:
            from mesh2param.pipeline.regularize import regularize_mesh
            regularize_mesh(sub_cleaned, regions, fitted, snap_angle_deg=12.0)
            regions = segment_mesh(sub_cleaned, seg)
            fitted = fit_surfaces(regions, sub_cleaned, FitConfig())
        except Exception:
            logger.exception("Regularization failed — using raw fit")
    graph, edges = build_topology(regions, fitted, sub_cleaned, TopologyConfig())
    try:
        blends = identify_blends_from_topology(regions, fitted, edges, sub_cleaned)
    except Exception:
        logger.exception("Blend identification failed")
        blends = []
    tree = recognize_features(
        regions, fitted, graph, edges, sub_cleaned,
        RecognizeConfig(), topo_blends=blends,
    )
    return BodyResult(tree, regions, fitted, graph, edges, blends)


def convert_mesh(
    mesh_path: str, ingest_config: IngestConfig | None = None,
):
    """End-to-end headless conversion matching the server's computation.

    ingest(scan_auto) -> connected-component decompose decision -> per-body
    recognize_body -> assemble the feature tree (single body => that body's
    tree; multi-body => combined). Returns (cleaned, tree, body_results).
    The server's _run_pipeline does the same; standalone scripts/tests call
    this so they match.
    """
    import trimesh

    from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
    from mesh2param.pipeline.ingest import ingest_mesh

    if ingest_config is None:
        ingest_config = IngestConfig(remesh=False, scan_auto=True)
    cleaned = ingest_mesh(mesh_path, ingest_config)
    scan_mode = bool(cleaned.metadata.get("scan_mode"))

    conn = trimesh.Trimesh(
        vertices=cleaned.vertices, faces=cleaned.faces, process=False)
    try:
        n_components = len(conn.split(only_watertight=False))
    except Exception:
        n_components = 1
    decomposition = decompose_bodies(
        cleaned, DecomposeConfig(enabled=n_components > 1))

    body_results: list[tuple] = []
    for body in decomposition.bodies:
        sub = extract_sub_mesh(cleaned, body.face_indices, body=body)
        br = recognize_body(sub, scan_mode)
        body_results.append((body, sub, br))

    if not body_results:
        return cleaned, None, []
    # Single body => its tree IS the part tree (mirrors job_queue line ~2287).
    tree = body_results[0][2].tree
    return cleaned, tree, body_results
