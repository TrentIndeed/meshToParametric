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
            config, assigned, graph, cleaned,
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
        # Reject tiny cylinders — these are edge radii/fillets, not holes.
        # Minimum 1.5mm radius for a real hole.
        if radius < 1.5:
            continue

        region = region_lookup.get(f.region_id)
        if region is None:
            continue

        # Check angular span to confirm it's a full cylinder (hole), not a partial arc
        span = _cylinder_angular_span(f, region)
        if span < 270.0:
            continue  # partial arc, not a hole

        # Reject cylinders that span the entire body along their axis —
        # these are edge fillets running the full length, not holes
        height_pre = f.params.get("height", 0)
        if height_pre > 0:
            axis_pre = np.array(f.params.get("axis_dir", [0, 0, 1]))
            part_proj_pre = cleaned.vertices @ axis_pre
            part_extent_pre = float(part_proj_pre.max() - part_proj_pre.min())
            if part_extent_pre > 0 and height_pre > part_extent_pre * 0.9:
                # Cylinder runs the full length → edge fillet, not a hole
                continue

        height = f.params.get("height", 0)
        axis_dir = np.array(f.params.get("axis_dir", [0, 0, 1]))

        # For cones, compute height from region vertex extent along axis
        if height <= 0 and region is not None:
            verts = cleaned.vertices[cleaned.faces[region.face_indices].flatten()]
            proj = verts @ axis_dir
            height = float(proj.max() - proj.min())

        # Snap hole axis to nearest principal direction. Reject holes whose
        # axis is more than 15° from any principal direction — these are
        # artifacts from cut-piece cap geometry, not real holes.
        principal_axes = [
            np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
            np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1]),
        ]
        best_dot = 0
        best_pa = None
        for pa in principal_axes:
            d = abs(np.dot(axis_dir, pa))
            if d > best_dot:
                best_dot = d
                best_pa = pa
        if best_pa is None or best_dot < np.cos(np.radians(15)):
            continue  # axis too far from any principal direction
        axis_dir = best_pa * np.sign(np.dot(axis_dir, best_pa))

        # Determine through vs blind. A cylindrical region that wraps
        # more than 270° of arc (already checked above) and has a
        # drill-hole-sized radius is almost always a through-hole.
        # Blind holes have a distinct bottom cap that gets segmented
        # as a separate planar region, leaving the cylindrical region
        # smaller. We already filter span >= 270 above, so at this point
        # we can safely classify as through for reasonable radii.
        is_through = radius < bbox_diag * 0.1  # drill holes are small

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
    graph: nx.Graph | None = None,
    cleaned: CleanedMesh | None = None,
) -> list[tuple[Feature, set[int]]]:
    """Find fillet and chamfer features.

    Detection strategies:
    1. Cone surfaces → CHAMFER — only if the cone is small relative to the
       part and actually sits between two principal (plane/cylinder) faces
       in the adjacency graph (not a standalone conical feature).
    2. Small narrow plane strips at ~135 deg to neighbors → CHAMFER (flat bevels)
    3. Small partial-arc cylinders → FILLET (rounded edges)
    4. Surfaces adjacent to chamfer/fillet/smooth edges → check type
    """
    results = []
    fillet_rids: set[int] = set()
    chamfer_rids: set[int] = set()

    # Part-scale reference for size checks
    bbox_diag = cleaned.bbox_diagonal if cleaned is not None else 100.0

    # Helper: check whether *rid* is adjacent (in the surface graph) to at
    # least two principal faces (PLANE or CYLINDER that are either assigned
    # to a feature or large enough to be principal).
    def _adjacent_to_two_principal(rid: int) -> bool:
        if graph is None or not graph.has_node(rid):
            return False
        principal_count = 0
        for nbr in graph.neighbors(rid):
            nbr_fit = fitted_lookup.get(nbr)
            if nbr_fit is None:
                continue
            if nbr_fit.surface_type in (SurfaceType.PLANE, SurfaceType.CYLINDER):
                principal_count += 1
                if principal_count >= 2:
                    return True
        return False

    # Strategy 1: Unassigned CONE surfaces → CHAMFER, but only if:
    #   a) The cone's bounding box is small relative to the part (< 10% of
    #      bbox diagonal), which rules out large conical features.
    #   b) The cone sits between two principal faces in the adjacency graph,
    #      which rules out standalone cone geometry (e.g. a conical boss).
    for f in fitted:
        if f.region_id in assigned:
            continue
        if f.surface_type != SurfaceType.CONE:
            continue
        r = region_lookup.get(f.region_id)
        if r is None:
            continue

        # (a) Size check: cone must be small relative to part
        cone_bbox_size = r.vertices.max(axis=0) - r.vertices.min(axis=0)
        cone_diag = float(np.linalg.norm(cone_bbox_size))
        if cone_diag > bbox_diag * 0.10:
            logger.debug(
                "Strategy 1: cone rid=%d too large (diag=%.2f vs part=%.2f), skipping",
                f.region_id, cone_diag, bbox_diag,
            )
            continue

        # (b) Adjacency check: must border at least 2 principal faces
        if not _adjacent_to_two_principal(f.region_id):
            logger.debug(
                "Strategy 1: cone rid=%d not adjacent to 2 principal faces, skipping",
                f.region_id,
            )
            continue

        logger.debug("Strategy 1: cone rid=%d accepted as chamfer (diag=%.2f)", f.region_id, cone_diag)
        chamfer_rids.add(f.region_id)

    # Strategy 2: Small narrow PLANE strips adjacent to chamfer edges
    # (flat 45 deg bevels between two larger faces)
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
                    if extent < config.chamfer_width_ratio * float(r.vertices.max() - r.vertices.min()) * 2:
                        chamfer_rids.add(rid)

    # Strategy 3: Topology-driven fillet blend detection.
    #
    # A fillet is a rolling-ball blend sitting tangent to two or more
    # principal faces. Geometrically that means: a non-planar region whose
    # boundary to neighboring regions is near-tangent (the stage-4 dihedral
    # classifier calls those SMOOTH or FILLET edges). It does not matter
    # whether the segmenter/fitter happened to fit the blend as a cylinder,
    # cone, or sphere — what matters is the topological role it plays.
    #
    # For each candidate region we reconstruct the cylinder axis from a
    # closed-form 2D circle fit on the vertex cross-section (project the
    # vertices perpendicular to the SVD length direction and fit a circle
    # by least squares). A closed-form 2D fit is far more stable than a
    # 6-parameter 3D cylinder fit on the handful of faces a fillet strip
    # contains, which is where the previous approach broke down.
    fillet_fit_lookup: dict[int, dict] = {}  # rid -> fitted cylinder params

    def _fit_strip_as_cylinder(pts: np.ndarray, face_normals: np.ndarray | None):
        """Fit a cylinder to a curved strip via normal-covariance + 2D circle.

        The cylinder axis is recovered from the *face normals* of the
        region, not from the vertex positions: on a cylinder all surface
        normals are perpendicular to the axis, so the axis is the
        direction along which the normal covariance is smallest. Using
        vertex SVD instead (as a previous version did) can pick the
        wrong axis when the region spans the length of the part but is
        actually a blend around a shorter direction.

        Returns (axis_point, axis_dir, radius, rms) or None if the fit
        is degenerate.
        """
        if pts is None or len(pts) < 4:
            return None
        if face_normals is None or len(face_normals) < 2:
            return None
        if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(face_normals)):
            return None
        mean_p = pts.mean(axis=0)
        rel = pts - mean_p
        if not np.all(np.isfinite(rel)):
            return None

        # Axis direction from face normals: for a cylinder with axis d,
        # every surface normal is perpendicular to d. So d is the
        # eigenvector of N^T N with the smallest eigenvalue.
        n_norms = np.linalg.norm(face_normals, axis=1)
        valid = n_norms > 1e-8
        if valid.sum() < 2:
            return None
        n_unit = face_normals[valid] / n_norms[valid, None]
        M = n_unit.T @ n_unit  # (3, 3)
        try:
            eig_vals, eig_vecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            return None
        # eigenvalues in ascending order
        axis_dir = eig_vecs[:, 0]
        # The smallest eigenvalue measures how non-perpendicular the
        # normals are to the axis. For a genuine cylindrical region it
        # should be near zero; for a saddle or corner cap it is large.
        # Rank the three eigenvalues — for a cylinder we expect roughly
        # one tiny eigenvalue and two comparable larger ones.
        lam = np.sort(np.abs(eig_vals))
        if lam[2] < 1e-10:
            return None
        # The smallest eigenvalue must be ~0 — i.e. all normals lie
        # in a plane. For a cylinder, that plane is perpendicular to
        # the axis.
        if lam[0] / lam[2] > 0.2:
            return None
        # The middle eigenvalue must be smaller than the largest. For
        # a partial-arc cylinder the normal spread within the plane is
        # anisotropic (middle ≪ max). For two perpendicular planes
        # meeting at a sharp interior edge the two directions are
        # evenly populated and middle/max ≈ 1 — that is the signature
        # of a phantom that we need to reject.
        if lam[1] / lam[2] > 0.85:
            return None
        n = float(np.linalg.norm(axis_dir))
        if n < 1e-12:
            return None
        axis_dir = axis_dir / n

        # Orthonormal 2D basis in the plane perpendicular to axis_dir.
        tmp = np.array([0.0, 0.0, 1.0]) if abs(axis_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u_vec = np.cross(axis_dir, tmp)
        nu = float(np.linalg.norm(u_vec))
        if nu < 1e-12:
            return None
        u_vec = u_vec / nu
        v_vec = np.cross(axis_dir, u_vec)

        x2 = rel @ u_vec
        y2 = rel @ v_vec
        # A cylindrical strip has a genuinely two-dimensional cross-section
        # in the perpendicular plane: a quarter-arc spans comparable width
        # in both directions. A strip that is effectively one-dimensional
        # in cross-section (all vertices on a line) is a flat band, not a
        # cylinder, and the Kasa circle fit below is rank-deficient on it.
        try:
            _, sv2, _ = np.linalg.svd(np.column_stack([x2, y2]), full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if sv2[0] < 1e-6:
            return None
        if sv2[1] / sv2[0] < 0.1:
            return None

        # Kasa algebraic circle fit gives a closed-form cheap estimate.
        # We then follow with a Levenberg-Marquardt geometric refinement
        # of the residuals sqrt((x-a)^2 + (y-b)^2) - r, which is what we
        # actually care about. If the geometric fit blows up (e.g. on a
        # near-line cross-section), we fall back to the algebraic answer
        # — both are reported in the RMS check below.
        A = np.column_stack([2 * x2, 2 * y2, np.ones_like(x2)])
        rhs = x2 ** 2 + y2 ** 2
        try:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        except Exception:
            return None
        a2d, b2d, c_val = sol
        r_sq = c_val + a2d * a2d + b2d * b2d
        if r_sq <= 0:
            return None
        radius_seed = float(np.sqrt(r_sq))

        from scipy.optimize import least_squares

        def _circle_residuals(params):
            ax_, bx_, rr = params
            return np.sqrt((x2 - ax_) ** 2 + (y2 - bx_) ** 2) - rr

        radius = radius_seed
        try:
            lsq = least_squares(
                _circle_residuals,
                np.array([a2d, b2d, radius_seed]),
                method="lm",
                max_nfev=200,
            )
            # Only accept the refined fit if it stays close to the
            # algebraic seed — a 10× explosion means LM wandered off.
            r_lm = float(abs(lsq.x[2]))
            if 0.25 * radius_seed <= r_lm <= 4.0 * radius_seed:
                a2d = float(lsq.x[0])
                b2d = float(lsq.x[1])
                radius = r_lm
        except Exception:
            pass
        if radius <= 0:
            return None

        # Recover the 3D axis point: lift the 2D circle center back through
        # the (u_vec, v_vec) basis relative to mean_p.
        axis_point = mean_p + a2d * u_vec + b2d * v_vec

        # RMS of perpendicular distance residuals.
        vecs = pts - axis_point
        proj_along = vecs @ axis_dir
        perp = vecs - np.outer(proj_along, axis_dir)
        perp_d = np.linalg.norm(perp, axis=1)
        rms = float(np.sqrt(np.mean((perp_d - radius) ** 2)))
        return axis_point, axis_dir, radius, rms

    for f in fitted:
        rid = f.region_id
        if rid in assigned or rid in chamfer_rids:
            continue
        if f.surface_type == SurfaceType.PLANE:
            continue  # planes are principal faces, not blends
        region = region_lookup.get(rid)
        if region is None or graph is None or not graph.has_node(rid):
            continue

        # Tangent-blend adjacency: the region must share at least two
        # near-tangent (SMOOTH or FILLET) boundaries with its neighbors.
        # Any neighbor counts — plane, cylinder, cone, whatever — because
        # a fillet's defining trait is that its boundary to the rest of
        # the part is smooth, not that those neighbors are planar.
        tangent_neighbors = 0
        for nbr in graph.neighbors(rid):
            edge_data = graph.get_edge_data(rid, nbr) or {}
            ttype = str(edge_data.get("transition_type", "")).upper()
            if ttype in ("SMOOTH", "FILLET"):
                tangent_neighbors += 1
        if tangent_neighbors < 2:
            continue

        pts = np.asarray(region.vertices, dtype=float)
        # Per-face normals for the region (defined on the full mesh).
        if cleaned is not None:
            face_normals = cleaned.normals[region.face_indices]
        else:
            face_normals = None

        # Connectivity check: a segmented region ought to be a single
        # connected patch (triangles linked via shared edges). When the
        # segmenter merges disjoint pieces of the mesh into one region
        # — e.g. the two front/back corners of a pocket — fitting a
        # single cylinder to the union is nonsense, and the resulting
        # "phantom fillet" ends up applied as a stripe that does not
        # exist in the reference mesh. We reject non-connected regions.
        if cleaned is not None and len(region.face_indices) > 1:
            face_to_idx = {
                int(fi): local_i
                for local_i, fi in enumerate(region.face_indices)
            }
            edge_to_local: dict[tuple[int, int], list[int]] = defaultdict(list)
            region_faces = cleaned.faces[region.face_indices]
            for local_i, tri in enumerate(region_faces):
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
                    key = (min(int(a), int(b)), max(int(a), int(b)))
                    edge_to_local[key].append(local_i)
            adj: dict[int, set[int]] = defaultdict(set)
            for locs in edge_to_local.values():
                if len(locs) < 2:
                    continue
                for i in range(len(locs)):
                    for j in range(i + 1, len(locs)):
                        adj[locs[i]].add(locs[j])
                        adj[locs[j]].add(locs[i])
            seen: set[int] = set()
            stack = [0]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(adj[cur] - seen)
            if len(seen) != len(region.face_indices):
                logger.debug(
                    "Strategy 3: rid=%d is %d/%d connected — region "
                    "is split across multiple pieces, skipping",
                    rid, len(seen), len(region.face_indices),
                )
                continue

        fit_res = _fit_strip_as_cylinder(pts, face_normals)
        if fit_res is None:
            continue
        axis_point, axis_dir, radius, rms = fit_res
        if radius <= 0.2 or radius > bbox_diag * 0.15:
            continue
        if rms > 0.35 * radius:
            logger.debug(
                "Strategy 3: rid=%d cylinder fit rms=%.2f vs radius=%.2f — "
                "not a clean cylindrical blend, skipping",
                rid, rms, radius,
            )
            continue

        # Snap axis direction so the stored axis_point sits at the
        # midpoint of the strip along its length — makes downstream edge
        # matching more predictable.
        vecs = pts - axis_point
        along = vecs @ axis_dir
        mid = 0.5 * (float(along.max()) + float(along.min()))
        axis_point = axis_point + mid * axis_dir
        strip_length = float(along.max() - along.min())

        fillet_rids.add(rid)
        fillet_fit_lookup[rid] = {
            "radius": float(radius),
            "axis_dir": [float(x) for x in axis_dir],
            "axis_point": [float(x) for x in axis_point],
            "length": strip_length,
            "rms": rms,
        }
        logger.debug(
            "Strategy 3: rid=%d accepted as fillet (r=%.2f, rms=%.2f, "
            "len=%.1f, %d faces, %d tangent neighbors)",
            rid, radius, rms, strip_length, region.num_faces,
            tangent_neighbors,
        )

    # Build features — use the fitted cylinder params from the topology
    # pass above, which reflect the actual rolling-ball blend geometry.
    for rid in fillet_rids:
        if rid in assigned:
            continue
        region = region_lookup.get(rid)
        if region is None:
            continue
        fit = fillet_fit_lookup.get(rid)
        if fit is None:
            # Fell through without a cylinder fit — skip silently.
            continue
        results.append((
            Feature(
                type=FeatureType.FILLET,
                face_indices=region.face_indices.copy(),
                sketch_geometry={
                    "type": "arc",
                    "radius": round(fit["radius"], 3),
                    "axis_dir": [round(float(x), 4) for x in fit["axis_dir"]],
                    "axis_point": [round(float(x), 4) for x in fit["axis_point"]],
                    "length": round(fit["length"], 3),
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
