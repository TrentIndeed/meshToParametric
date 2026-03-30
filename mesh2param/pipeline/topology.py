"""Stage 4: Topology & Edge Classification.

Builds a surface adjacency graph from segmented regions and classifies
the transition type (sharp, fillet, chamfer, smooth) between adjacent
surfaces based on dihedral angles and surface properties.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx
import numpy as np

from mesh2param.config import TopologyConfig
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.region import FaceRegion, SurfaceType
from mesh2param.models.surface import FittedSurface, SurfaceEdge, TransitionType

logger = logging.getLogger(__name__)


def build_topology(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    cleaned: CleanedMesh,
    config: TopologyConfig | None = None,
) -> tuple[nx.Graph, list[SurfaceEdge]]:
    """Build surface adjacency graph and classify inter-surface edges.

    Args:
        regions: Output of Stage 2 (Surface Segmentation).
        fitted: Output of Stage 3 (Primitive Fitting).
        cleaned: Output of Stage 1 (for mesh geometry).
        config: Topology configuration. Uses defaults if None.

    Returns:
        Tuple of (surface adjacency graph, list of classified SurfaceEdge objects).
    """
    if config is None:
        config = TopologyConfig()

    bbox_diag = cleaned.bbox_diagonal

    # Map each mesh face to its region
    face_to_region = _build_face_to_region(regions, cleaned.num_faces)

    # Discover which regions share mesh edges
    region_adjacencies = _find_region_adjacencies(
        cleaned.faces, face_to_region,
    )

    # Build lookups
    fitted_lookup = {f.region_id: f for f in fitted}
    region_lookup = {r.region_id: r for r in regions}

    # Classify each inter-region boundary
    surface_edges = []
    for (rid_a, rid_b), info in region_adjacencies.items():
        dihedral_rad = _compute_mean_dihedral(info["face_pairs"], cleaned.normals)
        dihedral_deg = float(np.degrees(dihedral_rad))

        edge_vertices = _chain_edge_vertices(info["mesh_edges"], cleaned.vertices)

        transition = _classify_transition(
            rid_a, rid_b, dihedral_deg,
            fitted_lookup, region_lookup,
            bbox_diag, config,
        )

        surface_edges.append(SurfaceEdge(
            surface_a_id=rid_a,
            surface_b_id=rid_b,
            transition_type=transition,
            dihedral_angle=round(dihedral_deg, 2),
            edge_vertices=edge_vertices,
        ))

    graph = _build_graph(regions, fitted, surface_edges)

    logger.info(
        "Topology: %d surfaces, %d edges (sharp=%d, fillet=%d, chamfer=%d, smooth=%d)",
        graph.number_of_nodes(),
        len(surface_edges),
        sum(1 for e in surface_edges if e.transition_type == TransitionType.SHARP),
        sum(1 for e in surface_edges if e.transition_type == TransitionType.FILLET),
        sum(1 for e in surface_edges if e.transition_type == TransitionType.CHAMFER),
        sum(1 for e in surface_edges if e.transition_type == TransitionType.SMOOTH),
    )

    return graph, surface_edges


# ---------------------------------------------------------------------------
# Face-to-region mapping
# ---------------------------------------------------------------------------


def _build_face_to_region(
    regions: list[FaceRegion],
    n_faces: int,
) -> np.ndarray:
    """Map each mesh face index to its region_id (−1 if unassigned)."""
    mapping = np.full(n_faces, -1, dtype=np.int64)
    for region in regions:
        mapping[region.face_indices] = region.region_id
    return mapping


# ---------------------------------------------------------------------------
# Region adjacency discovery
# ---------------------------------------------------------------------------


def _find_region_adjacencies(
    faces: np.ndarray,
    face_to_region: np.ndarray,
) -> dict[tuple[int, int], dict]:
    """Find pairs of adjacent regions and their shared mesh edges.

    Returns dict keyed by (region_a, region_b) with a < b, containing:
        - "mesh_edges": list of (v0, v1) vertex-index pairs on the boundary
        - "face_pairs": list of (face_a, face_b) sharing each edge, where
          face_a belongs to region_a and face_b to region_b
    """
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            edge_to_faces[edge].append(fi)

    adjacencies: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"mesh_edges": [], "face_pairs": []}
    )

    for edge, face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue
        fa, fb = face_list
        ra, rb = int(face_to_region[fa]), int(face_to_region[fb])
        if ra < 0 or rb < 0 or ra == rb:
            continue

        key = (min(ra, rb), max(ra, rb))
        adjacencies[key]["mesh_edges"].append(edge)
        # Keep face_a from the lower-numbered region
        if ra < rb:
            adjacencies[key]["face_pairs"].append((fa, fb))
        else:
            adjacencies[key]["face_pairs"].append((fb, fa))

    return dict(adjacencies)


# ---------------------------------------------------------------------------
# Dihedral angle computation
# ---------------------------------------------------------------------------


def _compute_mean_dihedral(
    face_pairs: list[tuple[int, int]],
    normals: np.ndarray,
) -> float:
    """Compute the mean dihedral angle across shared boundary face pairs.

    Dihedral angle = pi - arccos(n1 . n2), where n1, n2 are outward face
    normals. This gives 90 deg for a box corner and 180 deg for coplanar faces.
    """
    if not face_pairs:
        return np.pi

    angles = []
    for fa, fb in face_pairs:
        n1 = normals[fa]
        n2 = normals[fb]
        dot = np.clip(float(np.dot(n1, n2)), -1.0, 1.0)
        angle = np.pi - np.arccos(dot)
        angles.append(angle)

    return float(np.mean(angles))


# ---------------------------------------------------------------------------
# Edge vertex chaining
# ---------------------------------------------------------------------------


def _chain_edge_vertices(
    mesh_edges: list[tuple[int, int]],
    vertices: np.ndarray,
) -> np.ndarray:
    """Chain shared boundary edges into an ordered polyline of vertex positions.

    Greedily walks from an endpoint (degree-1 vertex) along the boundary.
    Returns (N, 3) array of positions.
    """
    if not mesh_edges:
        return np.empty((0, 3), dtype=np.float64)

    # Build vertex adjacency along the boundary
    vert_adj: dict[int, list[int]] = defaultdict(list)
    for v0, v1 in mesh_edges:
        vert_adj[v0].append(v1)
        vert_adj[v1].append(v0)

    # Start from an endpoint (degree 1) if one exists, else arbitrary
    endpoints = [v for v, nbrs in vert_adj.items() if len(nbrs) == 1]
    start = endpoints[0] if endpoints else mesh_edges[0][0]

    visited: set[int] = set()
    chain: list[int] = [start]
    visited.add(start)
    current = start

    while True:
        found = False
        for nbr in vert_adj.get(current, []):
            if nbr not in visited:
                chain.append(nbr)
                visited.add(nbr)
                current = nbr
                found = True
                break
        if not found:
            break

    # If there are unvisited vertices (disconnected chains), append them
    remaining = set(vert_adj.keys()) - visited
    while remaining:
        start2 = remaining.pop()
        chain.append(start2)
        visited.add(start2)
        current = start2
        while True:
            found = False
            for nbr in vert_adj.get(current, []):
                if nbr not in visited:
                    chain.append(nbr)
                    visited.add(nbr)
                    remaining.discard(nbr)
                    current = nbr
                    found = True
                    break
            if not found:
                break

    return vertices[chain].copy()


# ---------------------------------------------------------------------------
# Transition classification
# ---------------------------------------------------------------------------


def _classify_transition(
    rid_a: int,
    rid_b: int,
    dihedral_deg: float,
    fitted_lookup: dict[int, FittedSurface],
    region_lookup: dict[int, FaceRegion],
    bbox_diag: float,
    config: TopologyConfig,
) -> TransitionType:
    """Classify the transition type between two adjacent surfaces."""
    fillet_a = _is_fillet_candidate(rid_a, fitted_lookup, region_lookup, bbox_diag, config)
    fillet_b = _is_fillet_candidate(rid_b, fitted_lookup, region_lookup, bbox_diag, config)
    chamfer_a = _is_chamfer_candidate(rid_a, fitted_lookup, region_lookup, bbox_diag, config)
    chamfer_b = _is_chamfer_candidate(rid_b, fitted_lookup, region_lookup, bbox_diag, config)

    # Near-tangent: fillet or smooth
    if dihedral_deg >= config.smooth_angle_threshold_deg:
        if fillet_a or fillet_b:
            return TransitionType.FILLET
        return TransitionType.SMOOTH

    # Intermediate zone between sharp and smooth thresholds
    if dihedral_deg > config.sharp_angle_threshold_deg:
        if fillet_a or fillet_b:
            return TransitionType.FILLET
        if chamfer_a or chamfer_b:
            return TransitionType.CHAMFER
        return TransitionType.SMOOTH

    # Sharp zone — only call it a chamfer if the dihedral is in the
    # chamfer range (100°-160°).  A 90° box corner with a small adjacent
    # plane is just a normal sharp edge, not a chamfer.
    if (chamfer_a or chamfer_b) and dihedral_deg >= 100.0:
        return TransitionType.CHAMFER

    return TransitionType.SHARP


def _cylinder_angular_span(
    fitted: FittedSurface,
    region: FaceRegion,
) -> float:
    """Compute the angular span (in degrees) of a cylinder region's vertices.

    A full hole subtends ~360°; a fillet arc subtends < 180°.
    Uses the largest gap in sorted angles to handle wraparound correctly.
    """
    axis_point = np.array(fitted.params["axis_point"])
    axis_dir = np.array(fitted.params["axis_dir"])
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)

    vecs = region.vertices - axis_point
    proj_along = np.outer(vecs @ axis_dir, axis_dir)
    proj_perp = vecs - proj_along

    # Build orthonormal basis in the perpendicular plane
    norms = np.linalg.norm(proj_perp, axis=1)
    valid = norms > 1e-10
    if valid.sum() < 3:
        return 360.0  # degenerate — assume full cylinder

    proj_perp = proj_perp[valid]
    ref = proj_perp[0] / np.linalg.norm(proj_perp[0])
    ref2 = np.cross(axis_dir, ref)

    angles = np.arctan2(proj_perp @ ref2, proj_perp @ ref)
    sorted_angles = np.sort(angles)

    # Find the largest angular gap between consecutive samples
    gaps = np.diff(sorted_angles)
    wraparound_gap = (2 * np.pi) - (sorted_angles[-1] - sorted_angles[0])
    max_gap = max(float(np.max(gaps)) if len(gaps) > 0 else 0.0, wraparound_gap)

    span_rad = (2 * np.pi) - max_gap
    return float(np.degrees(span_rad))


def _is_fillet_candidate(
    rid: int,
    fitted_lookup: dict[int, FittedSurface],
    region_lookup: dict[int, FaceRegion],
    bbox_diag: float,
    config: TopologyConfig,
) -> bool:
    """A fillet is a small-radius *partial-arc* cylinder, not a full hole.

    Distinguishes fillets from holes by checking:
    1. Small radius relative to bbox diagonal
    2. Angular span < 200° (fillets are partial arcs; holes are ~360°)
    """
    fitted = fitted_lookup.get(rid)
    region = region_lookup.get(rid)
    if fitted is None or fitted.surface_type != SurfaceType.CYLINDER:
        return False
    if region is None:
        return False

    radius = fitted.params.get("radius", 0)
    max_radius = config.fillet_radius_ratio * bbox_diag
    if radius <= 0 or radius >= max_radius:
        return False

    # A fillet arc spans < 200°; a hole spans ~360°
    span = _cylinder_angular_span(fitted, region)
    return span < 200.0


def _is_chamfer_candidate(
    rid: int,
    fitted_lookup: dict[int, FittedSurface],
    region_lookup: dict[int, FaceRegion],
    bbox_diag: float,
    config: TopologyConfig,
) -> bool:
    """A chamfer is a small narrow planar strip connecting two larger surfaces.

    Must be both narrow (small extent) and have few faces relative to
    the total mesh, to avoid flagging major part faces.
    """
    fitted = fitted_lookup.get(rid)
    region = region_lookup.get(rid)
    if fitted is None or fitted.surface_type != SurfaceType.PLANE:
        return False
    if region is None:
        return False

    extent = fitted.params.get("extent", float("inf"))
    max_width = config.chamfer_width_ratio * bbox_diag
    if extent >= max_width:
        return False

    # Also require the region to be small (chamfers are narrow strips)
    # A chamfer should be a thin strip, not a significant part face
    # Use a tight face-count heuristic: chamfer regions are typically tiny
    return region.num_faces <= 30


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    surface_edges: list[SurfaceEdge],
) -> nx.Graph:
    """Build a networkx surface adjacency graph.

    Nodes carry surface_type, area, and label attributes.
    Edges carry transition_type and dihedral_angle attributes.
    """
    G = nx.Graph()
    fitted_lookup = {f.region_id: f for f in fitted}

    for region in regions:
        G.add_node(
            region.region_id,
            surface_type=region.surface_type.value.upper(),
            area=float(region.num_faces),
            label=f"R{region.region_id}",
        )

    for edge in surface_edges:
        G.add_edge(
            edge.surface_a_id,
            edge.surface_b_id,
            transition_type=edge.transition_type.value.upper(),
            dihedral_angle=edge.dihedral_angle,
        )

    return G
