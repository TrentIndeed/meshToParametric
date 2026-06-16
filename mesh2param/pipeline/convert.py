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
    # The mesh the REGIONS live on (for segmentation viz). For dirty/AI parts
    # this is a DECIMATED copy whose flats segment cleanly; the tree's build
    # geometry comes from the full mesh. For clean CAD it is the full mesh.
    seg_mesh: CleanedMesh | None = None


def recognize_body(sub_cleaned: CleanedMesh, scan_mode: bool) -> BodyResult:
    """Single-body recognition core, shared by server and scripts.

    segment -> fit -> (if dirty: regularize -> re-segment -> re-fit) ->
    topology -> blend identification -> recognize_features. This is EXACTLY
    what viz_server/job_queue._process_single_body runs (minus the viz), so a
    standalone reproduction calling this matches the app's feature tree.

    DECOUPLING (dirty/AI only): the BUILD (component planner, base/join
    profiles, holes) uses the FULL-density mesh for accuracy, while
    SEGMENTATION + feature recognition run on a DECIMATED copy so noisy flats
    stay ONE region instead of shattering into hundreds (which spawned bogus
    chamfers/slots). The two share a coordinate frame, so the coordinate-based
    features line up with the full-density build.
    """
    from mesh2param.pipeline.fit import fit_surfaces
    from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
    from mesh2param.pipeline.recognize import recognize_features
    from mesh2param.pipeline.segment import segment_mesh
    from mesh2param.pipeline.topology import build_topology

    seg = SegmentConfig(noise_robust=scan_mode)

    # DECIMATE FIRST (scan/AI only). The expensive analysis — segment, fit,
    # regularize, topology — all runs on a mesh at the reference triangle
    # density, so it's FAST and the noisy flats stay ONE region instead of
    # shattering into hundreds. The full-density mesh is kept ONLY as
    # build_mesh, for the component planner's cross-section build. Previously
    # we segmented the full mesh first and threw that work away — slow, and it
    # left the viz showing the fragmented full-mesh segmentation.
    seg_mesh = sub_cleaned
    build_mesh = None
    if scan_mode:
        decimated = _decimate_to_density(sub_cleaned)
        if decimated is not None and len(decimated.faces) < len(sub_cleaned.faces):
            seg_mesh = decimated
            build_mesh = sub_cleaned

    regions = segment_mesh(seg_mesh, seg)
    fitted = fit_surfaces(regions, seg_mesh, FitConfig())
    if scan_mode:
        try:
            from mesh2param.pipeline.regularize import regularize_mesh
            regularize_mesh(seg_mesh, regions, fitted, snap_angle_deg=12.0)
            regions = segment_mesh(seg_mesh, seg)
            fitted = fit_surfaces(regions, seg_mesh, FitConfig())
        except Exception:
            logger.exception("Regularization failed — using raw fit")

    graph, edges = build_topology(regions, fitted, seg_mesh, TopologyConfig())
    try:
        blends = identify_blends_from_topology(regions, fitted, edges, seg_mesh)
    except Exception:
        logger.exception("Blend identification failed")
        blends = []
    # On dirty AI/scan parts the build is robust but the region-based detail
    # finders hallucinate fillets/chamfers/slots from surface noise — gate
    # them conservatively (build stays locked, weak detail dropped).
    rec_cfg = RecognizeConfig(conservative_detail=scan_mode)
    tree = recognize_features(
        regions, fitted, graph, edges, seg_mesh,
        rec_cfg, topo_blends=blends, build_mesh=build_mesh,
    )
    return BodyResult(tree, regions, fitted, graph, edges, blends, seg_mesh)


def _decimate_to_density(mesh: CleanedMesh, faces_per_cm2: float = 120.0,
                         min_faces: int = 4_000) -> CleanedMesh | None:
    """A decimated copy at the reference triangle density (faces per cm^2),
    scaled with surface area. Used ONLY for segmentation; the build keeps the
    full mesh. Returns None on failure."""
    try:
        import trimesh
        import numpy as np
        from mesh2param.pipeline.ingest import _decimate
        from mesh2param.models.mesh import CleanedMesh as _CM
        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                             process=False)
        area_cm2 = float(tm.area) / 100.0
        target = int(np.clip(faces_per_cm2 * area_cm2, min_faces,
                             len(mesh.faces)))
        if target >= len(mesh.faces):
            return None
        dm = _decimate(tm, target)
        dm.fix_normals()
        verts = np.asarray(dm.vertices, dtype=float)
        faces = np.asarray(dm.faces)
        return _CM(
            vertices=verts,
            faces=faces,
            normals=np.asarray(dm.face_normals, dtype=float),
            is_watertight=bool(dm.is_watertight),
            bbox=np.array([verts.min(0), verts.max(0)]),
            metadata=dict(mesh.metadata),
        )
    except Exception:
        logger.exception("Segmentation decimation failed")
        return None


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
