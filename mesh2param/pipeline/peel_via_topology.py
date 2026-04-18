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
        if f.surface_type == SurfaceType.CYLINDER:
            # Fillet candidate — topology tagged at least one edge FILLET.
            # A partial-arc cylinder between two planes will typically have
            # two FILLET-tagged edges, but the segmenter sometimes splits
            # the arc into subregions where only one of them shows a
            # FILLET tag. Accept >=1 and we'll still require TWO distinct
            # plane neighbors further down.
            fillet_edges = [e for e in neighbors
                            if e.transition_type == TransitionType.FILLET]
            if not fillet_edges:
                continue
            kind = "fillet"
            # Include all plane-facing edges on this cylinder so we find
            # both adjacent planes (not just the FILLET-tagged side).
            neighbor_edges = list(neighbors)
        elif f.surface_type == SurfaceType.PLANE:
            # Chamfer candidate — topology tagged at least one CHAMFER edge.
            chamfer_edges = [e for e in neighbors
                             if e.transition_type == TransitionType.CHAMFER]
            if not chamfer_edges:
                continue
            # Must be narrow — real chamfer strip, not a main face.
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
