"""Stage 5: Feature Recognition.

Identifies manufacturing features (base extrusion, holes, pockets, bosses,
slots, fillets, chamfers) from the fitted surfaces, topology graph, and
edge classifications produced by the earlier pipeline stages.

The recognition strategy for prismatic parts:
1. Identify the base extrusion (largest planar region group).
2. Detect holes (full cylinders penetrating the body).
3. Detect pockets / slots (groups of inward-facing planes).
4. Detect bosses (groups of outward-facing planes above the base).
5. Detect fillets and chamfers (already hinted by Stage 4 edge types).
6. Build a dependency DAG and topologically sort.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx
import numpy as np

from mesh2param.config import RecognizeConfig
from mesh2param.models.feature import Feature, FeatureType
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.region import FaceRegion, SurfaceType
from mesh2param.models.surface import FittedSurface, SurfaceEdge, TransitionType
from mesh2param.models.tree import FeatureTree

logger = logging.getLogger(__name__)


def recognize_features(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    edges: list[SurfaceEdge],
    cleaned: CleanedMesh,
    config: RecognizeConfig | None = None,
) -> FeatureTree:
    """Recognize manufacturing features from surface data and topology.

    Args:
        regions: Output of Stage 2 (Surface Segmentation).
        fitted: Output of Stage 3 (Primitive Fitting).
        graph: Surface adjacency graph from Stage 4.
        edges: Classified surface edges from Stage 4.
        cleaned: Output of Stage 1 (for mesh geometry).
        config: Recognition configuration. Uses defaults if None.

    Returns:
        FeatureTree with ordered features and dependency graph.
    """
    if config is None:
        config = RecognizeConfig()

    logger.debug(
        "recognize_features called: %d regions, %d fitted, %d graph nodes, %d edges",
        len(regions), len(fitted), graph.number_of_nodes(), len(edges),
    )

    bbox_diag = cleaned.bbox_diagonal

    region_lookup = {r.region_id: r for r in regions}
    fitted_lookup = {f.region_id: f for f in fitted}

    # Track which regions have been assigned to features
    assigned: set[int] = set()
    features: list[Feature] = []

    # --- 1. Base extrusion: the largest group of connected planar faces ---
    try:
        base_feature, base_rids = _find_base_extrusion(
            regions, fitted, graph, cleaned,
        )
    except Exception:
        logger.exception("Stage 5 failed finding base extrusion: regions=%d", len(regions))
        raise
    features.append(base_feature)
    assigned.update(base_rids)
    base_idx = 0
    logger.debug("Base extrusion found: %d region IDs, depth=%.2f", len(base_rids), base_feature.depth)

    # --- 2. Holes: full cylinders ---
    try:
        hole_features = _find_holes(
            regions, fitted, graph, region_lookup, cleaned, config, assigned,
        )
    except Exception:
        logger.exception("Stage 5 failed finding holes")
        raise
    for hf, rids in hole_features:
        hf.dependencies = [base_idx]
        features.append(hf)
        assigned.update(rids)
    logger.debug("Found %d holes", len(hole_features))

    # --- 3. Pockets: inward-facing plane groups connected together ---
    try:
        pocket_features = _find_pockets(
            regions, fitted, graph, region_lookup, fitted_lookup,
            cleaned, config, assigned, base_rids,
        )
    except Exception:
        logger.exception("Stage 5 failed finding pockets")
        raise
    for pf, rids in pocket_features:
        pf.dependencies = [base_idx]
        features.append(pf)
        assigned.update(rids)
    logger.debug("Found %d pockets", len(pocket_features))

    # --- 4. Bosses: outward-facing plane groups above the base ---
    try:
        boss_features = _find_bosses(
            regions, fitted, graph, region_lookup, fitted_lookup,
            cleaned, config, assigned, base_rids,
        )
    except Exception:
        logger.exception("Stage 5 failed finding bosses")
        raise
    for bf, rids in boss_features:
        bf.dependencies = [base_idx]
        features.append(bf)
        assigned.update(rids)
    logger.debug("Found %d bosses", len(boss_features))

    # --- 5. Fillets and chamfers (from Stage 4 edge classification) ---
    try:
        fillet_chamfer = _find_fillets_and_chamfers(
            regions, fitted, edges, region_lookup, fitted_lookup,
            config, assigned,
        )
    except Exception:
        logger.exception("Stage 5 failed finding fillets/chamfers")
        raise
    for fc, rids in fillet_chamfer:
        # Fillets/chamfers depend on whichever features own their neighbor surfaces
        fc.dependencies = _resolve_dependencies(
            rids, features, region_lookup,
        )
        if not fc.dependencies:
            fc.dependencies = [base_idx]
        features.append(fc)
        assigned.update(rids)

    # --- 6. Remaining unassigned regions → absorb into base ---
    remaining = set(r.region_id for r in regions) - assigned
    if remaining:
        remaining_faces = np.concatenate(
            [region_lookup[rid].face_indices for rid in remaining]
        )
        base_feature.face_indices = np.concatenate(
            [base_feature.face_indices, remaining_faces]
        )
        assigned.update(remaining)

    # Build dependency DAG
    dep_graph = nx.DiGraph()
    for i, f in enumerate(features):
        dep_graph.add_node(i, type=f.type.value)
        for dep in f.dependencies:
            if dep < i:
                dep_graph.add_edge(dep, i)

    tree = FeatureTree(
        features=features,
        dependency_graph=dep_graph,
        metadata={
            "num_regions": len(regions),
            "num_assigned": len(assigned),
        },
    )

    logger.info(
        "Recognized %d features: %s",
        tree.num_features,
        ", ".join(f"{f.type.value}" for f in features),
    )

    return tree


# ---------------------------------------------------------------------------
# Base extrusion
# ---------------------------------------------------------------------------


def _find_base_extrusion(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    cleaned: CleanedMesh,
) -> tuple[Feature, set[int]]:
    """Find the base extrusion — the largest connected group of planes.

    For a prismatic part, the base is the dominant planar body that
    everything else attaches to or subtracts from.
    """
    fitted_lookup = {f.region_id: f for f in fitted}

    # Score each planar region by face count
    plane_regions = [
        r for r in regions
        if fitted_lookup.get(r.region_id) is not None
        and fitted_lookup[r.region_id].surface_type == SurfaceType.PLANE
    ]

    if not plane_regions:
        # Fallback: use the single largest region
        largest = max(regions, key=lambda r: r.num_faces)
        return (
            Feature(
                type=FeatureType.BASE_EXTRUSION,
                face_indices=largest.face_indices.copy(),
                depth=0.0,
                direction=np.array([0.0, 0.0, 1.0]),
                operation="NEW",
            ),
            {largest.region_id},
        )

    # Find connected components of planar regions in the adjacency graph
    plane_ids = {r.region_id for r in plane_regions}
    plane_subgraph = graph.subgraph(plane_ids)

    best_component: set[int] = set()
    best_face_count = 0
    for component in nx.connected_components(plane_subgraph):
        face_count = sum(
            r.num_faces for r in plane_regions if r.region_id in component
        )
        if face_count > best_face_count:
            best_face_count = face_count
            best_component = set(component)

    # If the best planar component is very small, include all planes
    total_faces = sum(r.num_faces for r in regions)
    if best_face_count < total_faces * 0.1:
        best_component = plane_ids

    # Collect face indices
    base_faces = np.concatenate(
        [r.face_indices for r in plane_regions if r.region_id in best_component]
    )

    # Estimate extrusion direction from the dominant plane normal
    largest_plane = max(
        (r for r in plane_regions if r.region_id in best_component),
        key=lambda r: r.num_faces,
    )
    f = fitted_lookup[largest_plane.region_id]
    direction = np.array(f.params.get("normal", [0, 0, 1]))

    # Estimate depth from bbox extent along the extrusion direction
    verts = cleaned.vertices
    proj = verts @ direction
    depth = float(proj.max() - proj.min())

    return (
        Feature(
            type=FeatureType.BASE_EXTRUSION,
            face_indices=base_faces,
            depth=depth,
            direction=direction,
            operation="NEW",
        ),
        best_component,
    )


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------


def _find_holes(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    region_lookup: dict[int, FaceRegion],
    cleaned: CleanedMesh,
    config: RecognizeConfig,
    assigned: set[int],
) -> list[tuple[Feature, set[int]]]:
    """Find cylindrical holes (through-holes and blind holes)."""
    bbox_diag = cleaned.bbox_diagonal
    fitted_lookup = {f.region_id: f for f in fitted}
    results = []

    for f in fitted:
        if f.region_id in assigned:
            continue
        if f.surface_type not in (SurfaceType.CYLINDER, SurfaceType.CONE):
            continue

        radius = f.params.get("radius", 0)
        # For cones with small half_angle (<5°), treat as cylinders.
        # Compute radius from the region's vertex distances to the axis.
        if f.surface_type == SurfaceType.CONE:
            half_angle = f.params.get("half_angle", 0)
            if half_angle > np.radians(45):
                continue  # wide cone, not a hole
            region = region_lookup.get(f.region_id)
            if region is None:
                continue
            verts = cleaned.vertices[cleaned.faces[region.face_indices].flatten()]
            axis_pt = np.array(f.params.get("axis_dir", [0, 0, 1]))
            # Estimate radius from bbox: smallest two dims averaged / 2
            bbox_size = verts.max(axis=0) - verts.min(axis=0)
            sorted_dims = sorted(bbox_size)
            radius = (sorted_dims[0] + sorted_dims[1]) / 4
        if radius <= 0 or radius > config.hole_max_radius_ratio * bbox_diag:
            continue

        region = region_lookup.get(f.region_id)
        if region is None:
            continue

        # Check angular span to confirm it's a full cylinder (hole), not a partial arc
        span = _cylinder_angular_span(f, region)
        if span < 270.0:
            continue  # partial arc, not a hole

        height = f.params.get("height", 0)
        axis_dir = np.array(f.params.get("axis_dir", [0, 0, 1]))

        # For cones, compute height from region vertex extent along axis
        if height <= 0 and region is not None:
            verts = cleaned.vertices[cleaned.faces[region.face_indices].flatten()]
            proj = verts @ axis_dir
            height = float(proj.max() - proj.min())

        # Snap hole axis to nearest principal direction if close (within 15°)
        # This fixes holes whose fitted axis is slightly off due to mesh noise
        principal_axes = [
            np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
            np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1]),
        ]
        for pa in principal_axes:
            if abs(np.dot(axis_dir, pa)) > np.cos(np.radians(15)):
                axis_dir = pa * np.sign(np.dot(axis_dir, pa))
                break

        # Determine through vs blind: compare hole height to part extent along axis
        part_proj = cleaned.vertices @ axis_dir
        part_extent = float(part_proj.max() - part_proj.min())
        is_through = height > part_extent * 0.8

        hole_type = FeatureType.THROUGH_HOLE if is_through else FeatureType.BLIND_HOLE

        results.append((
            Feature(
                type=hole_type,
                face_indices=region.face_indices.copy(),
                sketch_geometry={
                    "type": "circle",
                    "center": (
                        f.params.get("axis_point")
                        or region.vertices.mean(axis=0).tolist()
                    ),
                    "radius": radius,
                },
                depth=height,
                direction=-axis_dir,  # holes cut inward
                operation="REMOVE",
            ),
            {f.region_id},
        ))

    return results


def _cylinder_angular_span(fitted: FittedSurface, region: FaceRegion) -> float:
    """Compute angular span of a cylinder/cone region (degrees)."""
    axis_point = np.array(fitted.params.get("axis_point", fitted.params.get("apex", [0, 0, 0])))
    axis_dir = np.array(fitted.params.get("axis_dir", [0, 0, 1]))
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)

    vecs = region.vertices - axis_point
    proj_along = np.outer(vecs @ axis_dir, axis_dir)
    proj_perp = vecs - proj_along

    norms = np.linalg.norm(proj_perp, axis=1)
    valid = norms > 1e-10
    if valid.sum() < 3:
        return 360.0

    proj_perp = proj_perp[valid]
    ref = proj_perp[0] / np.linalg.norm(proj_perp[0])
    ref2 = np.cross(axis_dir, ref)

    angles = np.arctan2(proj_perp @ ref2, proj_perp @ ref)
    sorted_angles = np.sort(angles)
    gaps = np.diff(sorted_angles)
    wraparound_gap = (2 * np.pi) - (sorted_angles[-1] - sorted_angles[0])
    max_gap = max(float(np.max(gaps)) if len(gaps) > 0 else 0.0, wraparound_gap)

    return float(np.degrees((2 * np.pi) - max_gap))


# ---------------------------------------------------------------------------
# Pockets
# ---------------------------------------------------------------------------


def _find_pockets(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    region_lookup: dict[int, FaceRegion],
    fitted_lookup: dict[int, FittedSurface],
    cleaned: CleanedMesh,
    config: RecognizeConfig,
    assigned: set[int],
    base_rids: set[int],
) -> list[tuple[Feature, set[int]]]:
    """Find pockets — groups of unassigned planes that face inward.

    A pocket is a concavity: its dominant normal points opposite to
    the base surface normal (or perpendicular, for side walls).
    """
    bbox_diag = cleaned.bbox_diagonal

    # Get the base extrusion direction
    base_normals = []
    for rid in base_rids:
        f = fitted_lookup.get(rid)
        if f and f.surface_type == SurfaceType.PLANE:
            r = region_lookup.get(rid)
            if r:
                base_normals.extend([f.params["normal"]] * r.num_faces)
    if not base_normals:
        return []
    base_dir = np.mean(base_normals, axis=0)
    base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-15)

    # Find unassigned plane regions whose normal is roughly opposite to base
    inward_rids: set[int] = set()
    for r in regions:
        if r.region_id in assigned:
            continue
        f = fitted_lookup.get(r.region_id)
        if f is None or f.surface_type != SurfaceType.PLANE:
            continue
        normal = np.array(f.params.get("normal", [0, 0, 0]))
        dot = np.dot(normal, base_dir)
        # Inward-facing: normal opposes base dir or is perpendicular (side wall)
        if dot < 0.3:
            inward_rids.add(r.region_id)

    if not inward_rids:
        return []

    # Group connected inward-facing planes
    sub = graph.subgraph(inward_rids)
    results = []
    for component in nx.connected_components(sub):
        face_count = sum(region_lookup[rid].num_faces for rid in component)
        if face_count < config.min_feature_faces:
            continue

        faces = np.concatenate([region_lookup[rid].face_indices for rid in component])

        # Estimate pocket depth
        pocket_verts = np.concatenate([region_lookup[rid].vertices for rid in component])
        proj = pocket_verts @ base_dir
        depth = float(proj.max() - proj.min())

        if depth > config.pocket_max_depth_ratio * bbox_diag:
            continue

        # Narrow elongated pockets → SLOT
        pocket_all_verts = pocket_verts
        centered = pocket_all_verts - pocket_all_verts.mean(axis=0)
        if len(centered) >= 3:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            aspect = s[0] / (s[1] + 1e-15) if len(s) >= 2 else 1.0
        else:
            aspect = 1.0

        feat_type = FeatureType.SLOT if aspect > 4.0 else FeatureType.POCKET

        results.append((
            Feature(
                type=feat_type,
                face_indices=faces,
                depth=depth,
                direction=-base_dir,
                operation="REMOVE",
            ),
            set(component),
        ))

    return results


# ---------------------------------------------------------------------------
# Bosses
# ---------------------------------------------------------------------------


def _find_bosses(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    region_lookup: dict[int, FaceRegion],
    fitted_lookup: dict[int, FittedSurface],
    cleaned: CleanedMesh,
    config: RecognizeConfig,
    assigned: set[int],
    base_rids: set[int],
) -> list[tuple[Feature, set[int]]]:
    """Find bosses — groups of unassigned planes that protrude outward."""
    bbox_diag = cleaned.bbox_diagonal

    base_normals = []
    for rid in base_rids:
        f = fitted_lookup.get(rid)
        if f and f.surface_type == SurfaceType.PLANE:
            r = region_lookup.get(rid)
            if r:
                base_normals.extend([f.params["normal"]] * r.num_faces)
    if not base_normals:
        return []
    base_dir = np.mean(base_normals, axis=0)
    base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-15)

    # Find unassigned planes with normals roughly aligned with base (outward)
    outward_rids: set[int] = set()
    for r in regions:
        if r.region_id in assigned:
            continue
        f = fitted_lookup.get(r.region_id)
        if f is None or f.surface_type != SurfaceType.PLANE:
            continue
        normal = np.array(f.params.get("normal", [0, 0, 0]))
        dot = np.dot(normal, base_dir)
        if dot > 0.7:
            outward_rids.add(r.region_id)

    if not outward_rids:
        return []

    sub = graph.subgraph(outward_rids)
    results = []
    for component in nx.connected_components(sub):
        # Bosses should NOT be the largest faces (those are the base)
        face_count = sum(region_lookup[rid].num_faces for rid in component)
        if face_count < config.min_feature_faces:
            continue

        faces = np.concatenate([region_lookup[rid].face_indices for rid in component])

        boss_verts = np.concatenate([region_lookup[rid].vertices for rid in component])
        proj = boss_verts @ base_dir
        height = float(proj.max() - proj.min())

        if height > config.boss_max_height_ratio * bbox_diag:
            continue

        results.append((
            Feature(
                type=FeatureType.BOSS,
                face_indices=faces,
                depth=height,
                direction=base_dir,
                operation="ADD",
            ),
            set(component),
        ))

    return results


# ---------------------------------------------------------------------------
# Fillets and chamfers
# ---------------------------------------------------------------------------


def _find_fillets_and_chamfers(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    edges: list[SurfaceEdge],
    region_lookup: dict[int, FaceRegion],
    fitted_lookup: dict[int, FittedSurface],
    config: RecognizeConfig,
    assigned: set[int],
) -> list[tuple[Feature, set[int]]]:
    """Find fillet and chamfer features.

    Detection strategies:
    1. Cone surfaces → CHAMFER (countersinks around holes, beveled edges)
    2. Small narrow plane strips at ~135° to neighbors → CHAMFER (flat bevels)
    3. Small partial-arc cylinders → FILLET (rounded edges)
    4. Surfaces adjacent to chamfer/fillet/smooth edges → check type
    """
    results = []
    fillet_rids: set[int] = set()
    chamfer_rids: set[int] = set()

    # Strategy 1: All unassigned CONE surfaces are chamfers
    # (conical surfaces are countersinks or beveled transitions)
    for f in fitted:
        if f.region_id in assigned:
            continue
        if f.surface_type == SurfaceType.CONE:
            chamfer_rids.add(f.region_id)

    # Strategy 2: Small narrow PLANE strips adjacent to chamfer edges
    # (flat 45° bevels between two larger faces)
    for e in edges:
        if e.transition_type == TransitionType.CHAMFER:
            for rid in (e.surface_a_id, e.surface_b_id):
                if rid in assigned or rid in chamfer_rids:
                    continue
                f = fitted_lookup.get(rid)
                r = region_lookup.get(rid)
                if f is None or r is None:
                    continue
                # Small plane strips are chamfer faces
                if f.surface_type == SurfaceType.PLANE and r.num_faces <= 50:
                    extent = f.params.get("extent", float("inf"))
                    if extent < config.chamfer_width_ratio * r.vertices.ptp() * 2:
                        chamfer_rids.add(rid)

    # Strategy 3: Smooth/fillet edges involving small cylinders → FILLET
    for e in edges:
        if e.transition_type in (TransitionType.FILLET, TransitionType.SMOOTH):
            for rid in (e.surface_a_id, e.surface_b_id):
                if rid in assigned or rid in chamfer_rids:
                    continue
                f = fitted_lookup.get(rid)
                r = region_lookup.get(rid)
                if f is None or r is None:
                    continue
                # Small partial-arc cylinder = fillet
                if f.surface_type == SurfaceType.CYLINDER:
                    radius = f.params.get("radius", 0)
                    bbox_diag = np.linalg.norm(r.vertices.max(0) - r.vertices.min(0))
                    if radius > 0 and radius < bbox_diag * 0.3:
                        span = _cylinder_angular_span(f, r)
                        if span < 200:
                            fillet_rids.add(rid)

    # Build features
    for rid in fillet_rids:
        if rid in assigned:
            continue
        region = region_lookup.get(rid)
        f = fitted_lookup.get(rid)
        if region is None or f is None:
            continue
        results.append((
            Feature(
                type=FeatureType.FILLET,
                face_indices=region.face_indices.copy(),
                sketch_geometry={
                    "type": "arc",
                    "radius": f.params.get("radius", 0),
                },
                depth=0.0,
                operation="REMOVE",
            ),
            {rid},
        ))

    for rid in chamfer_rids:
        if rid in assigned:
            continue
        region = region_lookup.get(rid)
        f = fitted_lookup.get(rid)
        if region is None or f is None:
            continue
        # Compute chamfer width from the region's bounding box
        # For countersink cones, the smallest bbox dimension is the chamfer depth
        bbox_size = region.vertices.max(axis=0) - region.vertices.min(axis=0)
        width = float(sorted(bbox_size)[0])  # smallest dimension = chamfer depth
        if width < 0.1:
            width = float(sorted(bbox_size)[1])  # fallback to second smallest
        results.append((
            Feature(
                type=FeatureType.CHAMFER,
                face_indices=region.face_indices.copy(),
                sketch_geometry={
                    "type": "bevel",
                    "width": round(width, 2),
                },
                depth=round(width, 2),
                operation="REMOVE",
            ),
            {rid},
        ))

    return results


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


def _resolve_dependencies(
    feature_rids: set[int],
    existing_features: list[Feature],
    region_lookup: dict[int, FaceRegion],
) -> list[int]:
    """Find which existing features contain faces adjacent to this feature's regions."""
    deps: set[int] = set()
    for i, feat in enumerate(existing_features):
        feat_faces = set(feat.face_indices.tolist())
        for rid in feature_rids:
            region = region_lookup.get(rid)
            if region is None:
                continue
            region_faces = set(region.face_indices.tolist())
            # Simple heuristic: if any faces are nearby in index space,
            # it's likely a dependency. More robust: use mesh adjacency.
            # For now, depend on the base and any feature that shares
            # the same face index range.
            if feat_faces & region_faces:
                deps.add(i)
    # Always depend on at least the base (index 0)
    if not deps:
        deps.add(0)
    return sorted(deps)
