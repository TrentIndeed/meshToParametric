"""Stage 4.5: Region-based fillet / chamfer removal.

Uses the output of topology (regions + classified edges) to identify
every blend surface — a small cylinder region for a fillet, a narrow
plane region for a chamfer — and projects its vertices onto the
sharp intersection line of the two adjacent plane regions. The
result is a mesh whose blend surfaces have collapsed to sharp edges,
so downstream feature recognition sees clean plane-to-plane
transitions and doesn't fragment planes around the blend.

Why this replaces the pre-decompose triangle-based peel:

  - Works at the region level, not the triangle level, so it's
    O(regions) instead of O(triangles).
  - Uses the SAME adjacency graph the user sees in the topology
    viz — "the edge between region A and region B was classified
    FILLET, so peel B onto A's sharp edges."
  - Naturally symmetric: if the segmenter found a cylinder region
    on each side of a symmetric part, both sides get peeled.
  - No need to re-derive strip connectivity from dihedrals.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.region import FaceRegion, SurfaceType
from mesh2param.models.surface import FittedSurface, SurfaceEdge, TransitionType

logger = logging.getLogger(__name__)


def identify_blends_from_topology(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    surface_edges: list[SurfaceEdge],
    cleaned: CleanedMesh,
) -> list[dict]:
    """Identify fillet and chamfer blend regions using the topology graph.

    Does not modify the mesh — just returns a list of blend records. Each
    record describes one blend surface that sits between two planes, with
    enough information (axis, radius, face_indices) that downstream stages
    can peel the mesh or re-apply the blend as a CAD feature.

    A fillet blend is a small partial-arc cylinder adjacent to two planes
    via FILLET-classified edges. A chamfer blend is a narrow plane strip
    adjacent to two larger planes via CHAMFER-classified edges.

    Returns records of the form:
        {kind, radius, axis_dir, axis_point, length, num_faces,
         face_indices, region_id, plane_region_ids}
    """
    # trimesh import not needed here — no mesh modification.
    region_lookup = {r.region_id: r for r in regions}
    fitted_lookup = {f.region_id: f for f in fitted}

    # Build region-to-region edges indexed by BOTH endpoints so we can
    # quickly look up "what regions is region X adjacent to?"
    adj_by_region: dict[int, list[SurfaceEdge]] = defaultdict(list)
    for se in surface_edges:
        adj_by_region[se.surface_a_id].append(se)
        adj_by_region[se.surface_b_id].append(se)

    vertices = np.asarray(cleaned.vertices, dtype=float)
    faces = np.asarray(cleaned.faces, dtype=np.int64)
    bbox_diag = cleaned.bbox_diagonal

    records: list[dict] = []
    peeled_region_ids: set[int] = set()

    # A fillet blend region is a CYLINDER adjacent to TWO plane regions.
    # A chamfer blend region is a small PLANE adjacent to TWO larger plane
    # regions via CHAMFER edges.
    for region in regions:
        rid = region.region_id
        if rid in peeled_region_ids:
            continue
        f = fitted_lookup.get(rid)
        if f is None:
            continue

        neighbors = adj_by_region.get(rid, [])
        if not neighbors:
            continue

        kind: str | None = None
        # A fillet-like blend region is small, curved, adjacent to
        # plane regions. The EDGE-tag from topology is unreliable here
        # because topology's fillet-tag requires the cylinder region
        # to itself be a `_is_fillet_candidate` (partial-arc cylinder),
        # which fit.py sometimes doesn't produce — it'll classify the
        # arc as a sphere or cone instead, or merge the through-hole
        # cylinder so it spans 360°. Both make topology tag the
        # neighboring edges as CHAMFER or SHARP instead of FILLET.
        # So we use shape + size + plane-adjacency as the filter, and
        # skip the edge-tag check entirely.
        if f.surface_type in (
            SurfaceType.CYLINDER, SurfaceType.SPHERE, SurfaceType.CONE,
        ):
            # Small curved region likely to be a fillet. Exclude huge-
            # radius fit-noise (the fitter sometimes assigns a cylinder
            # with r=10000mm to a large plate face).
            radius_raw = f.params.get("radius", 0) or 0
            if radius_raw > 0 and radius_raw > bbox_diag * 0.08:
                continue
            # Cap region size — a real edge-fillet strip has on the
            # order of tens-to-hundreds of faces, not thousands.
            if region.num_faces > 300:
                continue
            kind = "fillet"
            neighbor_edges = list(neighbors)
        elif f.surface_type == SurfaceType.PLANE:
            # Narrow plane strip — possible chamfer between two planes.
            chamfer_edges = [e for e in neighbors
                             if e.transition_type == TransitionType.CHAMFER]
            if not chamfer_edges:
                continue
            if region.num_faces > 40:
                continue
            kind = "chamfer"
            neighbor_edges = list(neighbors)
        else:
            continue

        # The two adjacent plane regions are the other endpoints of the
        # fillet/chamfer edges.
        plane_rids: list[int] = []
        for e in neighbor_edges:
            other = e.surface_b_id if e.surface_a_id == rid else e.surface_a_id
            of = fitted_lookup.get(other)
            if of is not None and of.surface_type == SurfaceType.PLANE:
                plane_rids.append(other)
        # Need exactly two distinct plane neighbors to define a sharp line.
        unique_plane_rids = list(dict.fromkeys(plane_rids))[:2]
        if len(unique_plane_rids) < 2:
            continue

        p_rid_a, p_rid_b = unique_plane_rids[0], unique_plane_rids[1]
        pA = fitted_lookup[p_rid_a]
        pB = fitted_lookup[p_rid_b]
        n_A = np.asarray(pA.params.get("normal"), dtype=float)
        n_B = np.asarray(pB.params.get("normal"), dtype=float)
        # Some plane-fit outputs store d directly; compute a point on the
        # plane from the region's centroid projected onto the plane normal.
        rA = region_lookup.get(p_rid_a)
        rB = region_lookup.get(p_rid_b)
        if rA is None or rB is None:
            continue
        p_A = rA.vertices.mean(axis=0)
        p_B = rB.vertices.mean(axis=0)
        nA_norm = float(np.linalg.norm(n_A))
        nB_norm = float(np.linalg.norm(n_B))
        if nA_norm < 1e-8 or nB_norm < 1e-8:
            continue
        n_A = n_A / nA_norm
        n_B = n_B / nB_norm

        # Reject near-parallel planes (no well-defined intersection).
        if abs(float(np.dot(n_A, n_B))) > 0.95:
            continue

        # Plane-plane intersection line.
        line_dir = np.cross(n_A, n_B)
        ld_norm = float(np.linalg.norm(line_dir))
        if ld_norm < 1e-8:
            continue
        line_dir = line_dir / ld_norm

        # Line must be principal-axis aligned for prismatic parts.
        if float(np.max(np.abs(line_dir))) < 0.9:
            continue

        M_sys = np.array([n_A, n_B, line_dir])
        rhs = np.array([float(n_A @ p_A), float(n_B @ p_B), 0.0])
        try:
            line_point = np.linalg.solve(M_sys, rhs)
        except np.linalg.LinAlgError:
            continue

        # Line point inside mesh bbox + margin.
        margin = bbox_diag * 0.1
        m_min = cleaned.bbox[0]
        m_max = cleaned.bbox[1]
        if (
            line_point[0] < m_min[0] - margin or line_point[0] > m_max[0] + margin
            or line_point[1] < m_min[1] - margin or line_point[1] > m_max[1] + margin
            or line_point[2] < m_min[2] - margin or line_point[2] > m_max[2] + margin
        ):
            continue

        # Collect blend-region vertex IDs and project onto the sharp line.
        blend_face_ids = region.face_indices
        blend_vert_ids = np.unique(faces[blend_face_ids].flatten())
        pts = vertices[blend_vert_ids]
        rel = pts - line_point
        t_along = rel @ line_dir
        perp = rel - np.outer(t_along, line_dir)
        perp_dist = np.linalg.norm(perp, axis=1)
        radius = float(np.median(perp_dist))
        if kind == "fillet":
            max_radius = bbox_diag * 0.08
            if radius < 0.2 or radius > max_radius:
                continue
        else:  # chamfer
            max_size = bbox_diag * 0.05
            if radius < 0.2 or radius > max_size:
                continue

        length = float(t_along.max() - t_along.min())
        if length < radius * 1.5:
            continue

        records.append({
            "kind": kind,
            "radius": radius,
            "axis_dir": line_dir.astype(float),
            "axis_point": line_point.astype(float),
            "length": length,
            "num_faces": int(len(blend_face_ids)),
            "face_indices": np.asarray(blend_face_ids, dtype=np.int64).copy(),
            "region_id": rid,
            "plane_region_ids": [p_rid_a, p_rid_b],
        })
        peeled_region_ids.add(rid)

    n_fil = sum(1 for r in records if r["kind"] == "fillet")
    n_cha = sum(1 for r in records if r["kind"] == "chamfer")
    logger.info(
        "identify_blends_from_topology: %d fillet + %d chamfer region(s) "
        "(from %d surface edges)",
        n_fil, n_cha, len(surface_edges),
    )
    return records


def apply_blend_peel(
    cleaned: CleanedMesh,
    records: list[dict],
) -> CleanedMesh:
    """Project each blend region's vertices onto its plane-intersection
    line, drop the resulting degenerate triangles, and return a cleaned
    mesh that has sharp edges where the blends used to be.

    Records must include axis_dir, axis_point, and face_indices (as
    returned by identify_blends_from_topology).
    """
    import trimesh

    if not records:
        return cleaned

    vertices = np.asarray(cleaned.vertices, dtype=float).copy()
    faces = np.asarray(cleaned.faces, dtype=np.int64)
    bbox_diag = cleaned.bbox_diagonal

    # vertex_id -> list of candidate new positions (we average if a
    # vertex straddles two blends).
    vertex_new_pos: dict[int, list[np.ndarray]] = defaultdict(list)

    for rec in records:
        face_idx = rec.get("face_indices")
        if face_idx is None or len(face_idx) == 0:
            continue
        axis_dir = np.asarray(rec["axis_dir"], dtype=float)
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)
        axis_point = np.asarray(rec["axis_point"], dtype=float)

        blend_vert_ids = np.unique(faces[face_idx].flatten())
        pts = vertices[blend_vert_ids]
        rel = pts - axis_point
        t_along = rel @ axis_dir
        projections = axis_point + np.outer(t_along, axis_dir)
        for idx_local, vi in enumerate(blend_vert_ids):
            vertex_new_pos[int(vi)].append(projections[idx_local])

    moved_count = 0
    for vi, positions in vertex_new_pos.items():
        vertices[vi] = np.mean(positions, axis=0)
        moved_count += 1

    # Drop degenerate triangles and merge coincident vertices.
    t = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        t.merge_vertices()
    except Exception:
        pass
    try:
        areas = t.area_faces
        keep = areas > (bbox_diag * 1e-6) ** 2
        t.update_faces(keep)
        t.remove_unreferenced_vertices()
    except Exception:
        pass

    new_vertices = np.asarray(t.vertices, dtype=float)
    new_faces = np.asarray(t.faces, dtype=np.int64)
    new_normals = np.asarray(t.face_normals, dtype=float)
    new_bbox = np.array([new_vertices.min(axis=0), new_vertices.max(axis=0)])

    peeled = CleanedMesh(
        vertices=new_vertices,
        faces=new_faces,
        normals=new_normals,
        is_watertight=bool(t.is_watertight),
        bbox=new_bbox,
        metadata={
            **(cleaned.metadata or {}),
            "peeled_blend_count": len(records),
        },
    )
    logger.info(
        "apply_blend_peel: projected %d vertex(es) across %d blend(s), "
        "faces %d -> %d",
        moved_count, len(records), len(faces), len(new_faces),
    )
    return peeled


def detect_and_peel_blends_whole_mesh(
    cleaned: CleanedMesh,
) -> tuple[CleanedMesh, list[dict]]:
    """End-to-end: run segment / fit / topology on the whole mesh,
    identify every plane-to-plane blend (fillet cylinder or chamfer
    strip), and peel them all at once.

    This is the preferred path for the pre-decompose peel stage: it
    only touches mesh regions that the segmenter classified as planes
    connected by topology-classified blend edges, so the peel matches
    actual flat-face-to-flat-face fillets / chamfers and never acts on
    noisy curved strips in isolation.
    """
    # Local imports to avoid circulars at module load time.
    from mesh2param.pipeline.segment import segment_mesh
    from mesh2param.pipeline.fit import fit_surfaces
    from mesh2param.pipeline.topology import build_topology
    from mesh2param.config import SegmentConfig, FitConfig, TopologyConfig

    regions = segment_mesh(cleaned, SegmentConfig())
    fitted = fit_surfaces(regions, cleaned, FitConfig())
    _graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    records = identify_blends_from_topology(regions, fitted, edges, cleaned)
    peeled = apply_blend_peel(cleaned, records)
    return peeled, records
