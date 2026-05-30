"""Stage 2: Surface Segmentation.

Clusters mesh faces into regions belonging to a single analytic surface type
using region-growing with normal clustering, followed by curvature-based
classification.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from mesh2param.config import SegmentConfig
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.region import FaceRegion, SurfaceType

logger = logging.getLogger(__name__)


def segment_mesh(
    cleaned: CleanedMesh,
    config: SegmentConfig | None = None,
) -> list[FaceRegion]:
    """Segment a cleaned mesh into surface regions.

    Args:
        cleaned: Output of Stage 1 (Mesh Ingestion & Cleanup).
        config: Segmentation configuration. Uses defaults if None.

    Returns:
        List of FaceRegion objects, each containing face indices
        and a preliminary surface type label.
    """
    if config is None:
        config = SegmentConfig()

    logger.debug(
        "segment_mesh called: faces=%d, normals_shape=%s",
        len(cleaned.faces), cleaned.normals.shape,
    )

    try:
        adjacency = _build_face_adjacency(cleaned.faces)
        logger.debug("Face adjacency built: %d entries", len(adjacency))
    except Exception:
        logger.exception("Stage 2 failed building face adjacency: faces=%d", len(cleaned.faces))
        raise

    try:
        sharp_edges = _detect_sharp_edges(cleaned.normals, adjacency, config)
        logger.debug("Sharp edges detected: %d pairs", len(sharp_edges))
        labels = _region_growing(cleaned.normals, adjacency, config, sharp_edges)
        logger.debug("Region growing produced %d unique labels", len(np.unique(labels)))
        labels = _merge_small_regions(labels, adjacency, cleaned.normals, config.min_region_faces)
        labels = _split_flat_from_curved(labels, cleaned.normals, adjacency)
        # _split_flat_from_curved peels flat faces out of a mixed
        # region but leaves the remaining "curved" faces under one
        # label even when they form mesh-disconnected components
        # (e.g. after peeling the plate, the bore cylinder wall and
        # the perimeter rolled fillets end up in one label despite
        # being physically separated by the now-extracted plate).
        # Split labels into one-per-connected-component so each
        # feature gets its own highlight.
        labels = _split_disconnected_components(labels, adjacency)
        labels = _merge_small_regions(labels, adjacency, cleaned.normals, min(config.min_region_faces, 2))
        labels = _curvature_split(labels, cleaned, adjacency, config)
        labels = _split_cylinders_from_planes(labels, cleaned, adjacency)
        labels = _merge_adjacent_cylinder_fragments(labels, cleaned, adjacency)
        # Final pass: every spatially-contiguous cylindrical
        # feature (detected via local rank-2 normal structure)
        # becomes one region, regardless of how prior passes may
        # have split it. This specifically fixes the "inner bore
        # surface split into 4-5 regions" case where flood-fill
        # seeds pick different cardinal positions.
        labels = _unify_cylinder_features(labels, cleaned, adjacency)
        # Re-split: DBSCAN inside the unifier can bridge
        # mesh-disconnected components (e.g. the front rim fillet
        # of a through-hole and the back rim fillet, separated by
        # the plate's thickness but within DBSCAN's eps). They
        # share a label but are physically separate features —
        # they must be distinct regions so each can be rendered
        # and fitted independently. Only split components that
        # would each be substantial (>=50 faces) so we don't
        # fragment tabs where the bore wall and the local
        # through-hole rim are intentionally one region.
        labels = _split_disconnected_components(labels, adjacency, min_component_faces=50)
        # Final cleanup: peel axial "tangent-strip" faces that got
        # into curved regions. The first row of a rim fillet has
        # normals so close to the plate's normal (within 5–10°
        # of pure axial) that region growing absorbs them with
        # the rim, leaving each rim region with a ring of flat
        # faces tangent to the adjacent plate. Visually they
        # render as if the rim is "connected to the flat face
        # next to it". Peel those into their own region.
        labels = _peel_axial_tangents_from_curves(
            labels, cleaned.normals, adjacency,
        )
        # Peel rim fillets out of plate-spanning frame regions.
        labels = _peel_hole_rim_fillets(labels, cleaned, adjacency)
        # Peel small through-holes whose bore wall got fused with
        # an adjacent rolled perimeter edge (e.g. a hole drilled
        # through a plate corner). The general "is this round and
        # 360°?" heuristic in _peel_hole_rim_fillets fails when the
        # hole's bore wall normals only span a partial arc because
        # they share triangles with a curved edge feature. This pass
        # uses a DIFFERENT signal: a small flat island of plate-face
        # material that's spatially DISCONNECTED from the main plate
        # face implies a hole punched through the plate at that
        # island's location. Use the island's centroid as the hole
        # center and peel the bore wall + rim fillets around it.
        labels = _peel_isolated_flat_island_holes(
            labels, cleaned, adjacency,
        )
        # Split rolled-edge regions that connect two opposing flat
        # regions (e.g. the perimeter wrap of a cantilevered tab,
        # going from the tab's bottom face up to the parent body's
        # top face) into separate fillet-near-A + side-wall +
        # fillet-near-B sub-regions. This is what makes 4 mounting
        # tabs on the same body segment SYMMETRICALLY: without it,
        # some tabs' wraps split into chamfer/wall/transition by
        # coincidence of DBSCAN axis-detection thresholds, while
        # others stay as one giant region.
        labels = _split_rolled_edge_into_fillet_wall(
            labels, cleaned, adjacency,
        )
        # Split long curved regions (e.g. side-wall fillets running
        # the full plate length) into corner fillets at each end +
        # the middle wall. The end-faces' face normals have a
        # component along the LONG axis (because they curve toward
        # the perpendicular end-plane), while the middle-wall
        # faces' normals are perpendicular to the long axis.
        labels = _split_long_fillet_into_corners(
            labels, cleaned, adjacency,
        )
        # Pull borderline-tilted faces out of curved regions
        # back into adjacent flat regions when their neighborhood
        # is dominantly flat. Captures the general principle
        # "lots of flat-adjacent triangles = one flat plane",
        # fixing fan-triangulation patterns where slightly-tilted
        # slices got incorrectly pulled into the curved fillet
        # region instead of joining the flat surface.
        labels = _pull_axial_into_flat_regions(
            labels, cleaned.normals, adjacency,
        )
        # Final cleanup: small-region merge.
        labels = _merge_small_regions(
            labels, adjacency, cleaned.normals, 15,
        )
    except Exception:
        logger.exception(
            "Stage 2 failed during segmentation passes: faces=%d, labels_shape=%s",
            len(cleaned.faces), labels.shape if 'labels' in dir() else 'N/A',
        )
        raise

    regions = _build_regions(labels, cleaned, adjacency)
    logger.info("Segmented mesh into %d regions", len(regions))
    return regions


def _build_face_adjacency(faces: np.ndarray) -> dict[int, set[int]]:
    """Build face-face adjacency from shared edges.

    Two faces are adjacent if they share exactly two vertices (an edge).
    """
    logger.debug("Building face adjacency for %d faces", len(faces))
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            edge_to_faces[edge].append(fi)

    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge, face_list in edge_to_faces.items():
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                adjacency[face_list[i]].add(face_list[j])
                adjacency[face_list[j]].add(face_list[i])

    # Ensure every face has an entry even if isolated
    for fi in range(len(faces)):
        if fi not in adjacency:
            adjacency[fi] = set()

    return dict(adjacency)


def _detect_sharp_edges(
    normals: np.ndarray,
    adjacency: dict[int, set[int]],
    config: SegmentConfig,
) -> set[tuple[int, int]]:
    """Find face pairs separated by sharp dihedral angles.

    Returns a set of (face_a, face_b) pairs that should NOT be merged
    during region growing, because the angle between them indicates a
    feature boundary (e.g., where a flat face meets a hole wall).
    """
    sharp_angle = np.cos(np.radians(config.curved_angle_threshold_deg + 5))
    sharp_pairs: set[tuple[int, int]] = set()

    for fi, neighbors in adjacency.items():
        for ni in neighbors:
            if ni <= fi:
                continue  # avoid duplicates
            dot = float(np.dot(normals[fi], normals[ni]))
            if dot < sharp_angle:
                sharp_pairs.add((fi, ni))
                sharp_pairs.add((ni, fi))

    return sharp_pairs


def _region_growing(
    normals: np.ndarray,
    adjacency: dict[int, set[int]],
    config: SegmentConfig,
    sharp_edges: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Assign each face to a region via flood-fill on normal similarity.

    Seeds are chosen as the face with the most unassigned neighbors.
    Growing uses an adaptive angle threshold: tight for planar regions,
    looser for curved surfaces. Never crosses sharp edge boundaries.
    """
    n_faces = len(normals)
    labels = np.full(n_faces, -1, dtype=np.int64)
    assigned = np.zeros(n_faces, dtype=bool)
    if sharp_edges is None:
        sharp_edges = set()

    plane_thresh = np.cos(np.radians(config.plane_angle_threshold_deg))
    curved_thresh = np.cos(np.radians(config.curved_angle_threshold_deg))

    region_id = 0

    while not assigned.all():
        # Pick seed: unassigned face with the most unassigned neighbors
        seed = _pick_seed(assigned, adjacency)
        if seed is None:
            break

        # Flood-fill from seed
        queue = [seed]
        labels[seed] = region_id
        assigned[seed] = True
        region_normals = [normals[seed]]

        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, set()):
                if assigned[neighbor]:
                    continue

                # Never cross a sharp edge boundary
                if (current, neighbor) in sharp_edges:
                    continue

                # Adaptive threshold
                avg_normal = np.mean(region_normals, axis=0)
                avg_normal /= np.linalg.norm(avg_normal) + 1e-15

                dot = np.dot(normals[neighbor], avg_normal)
                dot_local = np.dot(normals[neighbor], normals[current])

                if dot >= plane_thresh or dot_local >= curved_thresh:
                    labels[neighbor] = region_id
                    assigned[neighbor] = True
                    queue.append(neighbor)
                    region_normals.append(normals[neighbor])

        region_id += 1

    return labels


def _pick_seed(assigned: np.ndarray, adjacency: dict[int, set[int]]) -> int | None:
    """Pick the unassigned face with the most unassigned neighbors."""
    unassigned = np.where(~assigned)[0]
    if len(unassigned) == 0:
        return None

    best = unassigned[0]
    best_count = -1
    for fi in unassigned:
        count = sum(1 for n in adjacency.get(fi, set()) if not assigned[n])
        if count > best_count:
            best_count = count
            best = fi

    return int(best)


def _merge_small_regions(
    labels: np.ndarray,
    adjacency: dict[int, set[int]],
    normals: np.ndarray,
    min_faces: int,
) -> np.ndarray:
    """Merge regions with fewer than min_faces into the most similar neighbor."""
    labels = labels.copy()
    changed = True

    while changed:
        changed = False
        unique_labels, counts = np.unique(labels, return_counts=True)

        for label, count in zip(unique_labels, counts):
            if label < 0 or count >= min_faces:
                continue

            face_mask = labels == label
            region_faces = np.where(face_mask)[0]
            avg_normal = normals[face_mask].mean(axis=0)
            norm_len = np.linalg.norm(avg_normal)
            if norm_len > 1e-15:
                avg_normal /= norm_len

            # Find neighboring regions
            neighbor_labels: dict[int, float] = {}
            for fi in region_faces:
                for ni in adjacency.get(fi, set()):
                    nl = labels[ni]
                    if nl != label and nl >= 0:
                        if nl not in neighbor_labels:
                            nl_mask = labels == nl
                            nl_avg = normals[nl_mask].mean(axis=0)
                            nl_norm = np.linalg.norm(nl_avg)
                            if nl_norm > 1e-15:
                                nl_avg /= nl_norm
                            neighbor_labels[nl] = float(np.dot(avg_normal, nl_avg))

            if not neighbor_labels:
                continue

            # Merge into the neighbor with the most similar normal,
            # but only if the similarity exceeds a minimum threshold.
            # Without this, distinct planar faces (e.g., box sides at 90°)
            # would cascade-merge into a single region.
            best_neighbor = max(neighbor_labels, key=neighbor_labels.get)
            merge_similarity_threshold = np.cos(np.radians(45.0))
            if neighbor_labels[best_neighbor] >= merge_similarity_threshold:
                labels[face_mask] = best_neighbor
                changed = True

    # Renumber labels to be contiguous 0..N-1
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)

    return labels


def _pull_axial_into_flat_regions(
    labels: np.ndarray,
    normals: np.ndarray,
    adjacency: dict[int, set[int]],
    iterations: int = 5,
) -> np.ndarray:
    """Reassign borderline-tilted faces from curved regions back
    into adjacent flat regions when their neighborhood is
    dominantly flat.

    General principle: a flat plane is a connected set of faces
    whose normals all point in (approximately) the same direction.
    Region growing with normal-similarity flood-fill correctly
    identifies most flat-plane faces, but at the boundary between
    a flat plate and an adjacent curved fillet, fan-triangulated
    "near-flat" faces (e.g. ~20° tilt) ambiguously satisfy BOTH
    the flat region's drift criterion AND the curved region's
    normal-similarity to its growing edge. They tend to land in
    the curved region, leaving the flat plate visually fragmented.

    This pass walks each face once per iteration. If the face is
    in a curved region (mean-normal magnitude < 0.95) AND has a
    mesh neighbor in a tightly-flat region (mean-normal magnitude
    ≥ 0.95) AND the face's normal is within 30° of that flat
    region's mean normal, the face joins the flat region.

    Iterates because the boundary moves inward by one face each
    pass — a fan slice next to a flat plate joins the plate, then
    its neighbor (one slice deeper into the fillet) is now next to
    the (extended) flat plate, and so on, until the slices are
    actually too tilted to merge.
    """
    labels = labels.copy()
    flat_threshold = 0.95
    cos_30 = np.cos(np.radians(30.0))

    for _ in range(iterations):
        # Cache per-region mean normal and magnitude.
        region_faces: dict[int, np.ndarray] = {}
        for lbl in np.unique(labels):
            region_faces[int(lbl)] = np.where(labels == lbl)[0]
        region_mean: dict[int, np.ndarray] = {}
        region_mag: dict[int, float] = {}
        for lbl, faces in region_faces.items():
            mn = normals[faces].mean(axis=0)
            mag = float(np.linalg.norm(mn))
            region_mag[lbl] = mag
            if mag > 1e-12:
                region_mean[lbl] = mn / mag
            else:
                region_mean[lbl] = mn

        changed = False
        new_labels = labels.copy()
        for fi in range(len(normals)):
            my_lbl = int(labels[fi])
            # Already in a flat region — skip
            if region_mag.get(my_lbl, 0.0) >= flat_threshold:
                continue
            face_n = normals[fi]
            hop1 = adjacency.get(fi, set())
            if not hop1:
                continue
            # 2-hop neighborhood normal magnitude — if it's near 1,
            # the LOCAL CONTEXT is flat (whole neighborhood points
            # one direction), even if this individual face is
            # slightly tilted from the fan layout. A rim-fillet face
            # has 2-hop neighborhood with rotating normals, mag ~0.6.
            hop2: set[int] = set()
            for h in hop1:
                hop2.update(adjacency.get(h, set()))
            nbrhd = list((hop1 | hop2) - {fi})
            if len(nbrhd) < 4:
                continue
            ctx_mean = normals[nbrhd].mean(axis=0)
            ctx_mag = float(np.linalg.norm(ctx_mean))
            if ctx_mag < 0.92:
                continue  # local context isn't flat
            ctx_dir = ctx_mean / ctx_mag

            # Find a mesh-adjacent flat region whose mean normal
            # matches the local context direction.
            best_lbl = None
            best_dot = -1.0
            for nb in hop1:
                nb_lbl = int(labels[nb])
                if nb_lbl == my_lbl:
                    continue
                if region_mag.get(nb_lbl, 0.0) < flat_threshold:
                    continue
                dot = float(np.dot(region_mean[nb_lbl], ctx_dir))
                if dot > 0.95 and dot > best_dot:
                    best_dot = dot
                    best_lbl = nb_lbl
            if best_lbl is None:
                continue
            # Final check: face's own normal isn't crazy-far from
            # the flat region (within 45°)
            if float(np.dot(face_n, region_mean[best_lbl])) < 0.7:
                continue
            new_labels[fi] = best_lbl
            changed = True
        labels = new_labels
        if not changed:
            break

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)
    return labels


def _peel_hole_rim_fillets(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Peel each through-hole's rim fillet into its own region
    when it's absorbed into a plate-spanning frame region.

    Detects through-holes by clustering radial-normal faces
    (perpendicular to a world axis) via DBSCAN. For each hole,
    finds nearby faces with rim-fillet normal signature
    (|n · axis| in [0.3, 0.95]) — those are the torus arc from
    the bore wall to the plate. Splits them by front vs back
    face side (sign of n · axis) and, for each side, reassigns
    the hole-local faces to their own new label — but only when
    the dominant source label is plate-spanning (its bbox is much
    larger than the hole). When the rim already sits in a
    hole-local region, this pass is a no-op.
    """
    from sklearn.cluster import DBSCAN

    labels = labels.copy()
    normals = cleaned.normals
    verts = cleaned.vertices
    n_faces = len(normals)

    centroids = np.array([
        verts[cleaned.faces[fi]].mean(axis=0) for fi in range(n_faces)
    ])
    next_label = int(labels.max()) + 1

    for axis_idx, axis_vec in enumerate(np.eye(3)):
        radial = np.where(np.abs(normals[:, axis_idx]) < 0.25)[0]
        if len(radial) < 10:
            continue
        perp_idx = [i for i in range(3) if i != axis_idx]
        proj = centroids[radial][:, perp_idx]
        db = DBSCAN(eps=3.0, min_samples=10).fit(proj)

        for cl in set(db.labels_):
            if cl == -1:
                continue
            mask = db.labels_ == cl
            hole_faces = radial[mask]
            hc = centroids[hole_faces]
            hole_bb = hc.max(axis=0) - hc.min(axis=0)
            # Cylindrical through-hole criteria (filter out plate
            # surfaces that happen to have faces with a small
            # component along this axis):
            #   1. Both perpendicular dims > 1mm (not a slit)
            #   2. Perpendicular dims are similar — the cluster
            #      is ROUND, not an elongated rectangle. Ratio
            #      short/long >= 0.6 admits circles with some
            #      tessellation noise but excludes plate top
            #      surfaces (ratio << 0.5 for 146x75 plates).
            #   3. Both perpendicular dims bounded (< 50mm).
            #      No real through-hole in this model is that
            #      wide. This is the gate that prevents the
            #      bottom plate's own flat top from being treated
            #      as a "hole" and having spurious rim regions
            #      peeled out of it.
            perp_dims = sorted([hole_bb[p] for p in perp_idx])
            if perp_dims[0] < 1.0:
                continue
            if perp_dims[0] / perp_dims[1] < 0.6:
                continue  # not round enough
            if perp_dims[1] > 50.0:
                continue  # too big to be a through-hole
            # Genuine 360° cylindrical feature: normals cover all
            # directions around the axis → their mean magnitude
            # is near zero. A partial rolled edge of the plate's
            # own perimeter covers only ~90-180° (mean magnitude
            # 0.6+) and must NOT be treated as a through-hole.
            mean_n = normals[hole_faces].mean(axis=0)
            if float(np.linalg.norm(mean_n)) > 0.35:
                continue
            hole_center = hc.mean(axis=0)
            hole_diam = perp_dims[1]

            # Rim fillet candidates: near the hole in the
            # perpendicular plane AND normal tilted (0.3–0.95 of
            # axis component). "Near" = within 1.2× hole radius.
            dyz = np.linalg.norm(
                centroids[:, perp_idx] - hole_center[perp_idx], axis=1
            )
            near_mask = dyz < hole_diam * 0.8
            axial = np.abs(normals[:, axis_idx])
            rim_mask = near_mask & (axial > 0.3) & (axial < 0.95)
            rim_faces = np.where(rim_mask)[0]
            if len(rim_faces) < 10:
                continue

            # Split front / back by sign of the axial component.
            signed_axial = normals[rim_faces, axis_idx]
            for side_sign, _name in [(1, "front"), (-1, "back")]:
                side_faces = rim_faces[
                    np.sign(signed_axial) == side_sign
                ]
                if len(side_faces) < 10:
                    continue

                # Check if the dominant source label is
                # plate-spanning (bbox in perpendicular plane
                # > 2× hole diameter). If not, the rim is
                # already local — nothing to peel.
                src_labels = Counter(
                    int(labels[fi]) for fi in side_faces
                )
                dominant_lbl, dominant_n = src_labels.most_common(1)[0]
                if dominant_n < len(side_faces) * 0.5:
                    # Rim already split; don't intervene
                    continue
                lbl_faces = np.where(labels == dominant_lbl)[0]
                lbl_centroids = centroids[lbl_faces]
                lbl_bb = lbl_centroids.max(axis=0) - lbl_centroids.min(axis=0)
                lbl_max_perp = max(lbl_bb[p] for p in perp_idx)
                if lbl_max_perp <= hole_diam * 2.0:
                    continue  # dominant source is already local

                # Reassign side_faces to a new label.
                for fi in side_faces:
                    labels[fi] = next_label
                next_label += 1

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)
    return labels


def _peel_isolated_flat_island_holes(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Peel small through-holes detected via isolated flat-face islands.

    For each candidate axis (X/Y/Z), find all faces whose normal is
    closely aligned with that axis. Spatially DBSCAN-cluster them in
    the perpendicular plane. A SMALL cluster (a few mm to ~25mm)
    that's spatially separated from larger plate-face clusters is a
    flat island — typically the donut-shaped flange around a small
    through-hole punched through a plate corner.

    For each such island, take its centroid as the hole center and:
      1. Peel BORE WALL faces: faces with |n·axis| < 0.25 within
         a small radius of the center → new region.
      2. Peel RIM FILLET faces (front + back side, by sign of
         n·axis): faces with 0.3 < |n·axis| < 0.95 within the same
         radius → two new regions.

    Skips islands when:
      • The bore wall already sits in a small (< 100 face) local
        region (already peeled by an earlier pass).
      • There aren't enough bore-wall candidate faces near the
        center (< 8) — false positive.

    This is the GENERAL fix for "tab embedded in rolled edge": the
    rolled edge gets the bore + rim if we don't intervene because
    they're geometrically continuous in the mesh.
    """
    from sklearn.cluster import DBSCAN

    labels = labels.copy()
    normals = cleaned.normals
    verts = cleaned.vertices
    n_faces = len(normals)
    centroids = np.array([
        verts[cleaned.faces[fi]].mean(axis=0) for fi in range(n_faces)
    ])
    next_label = int(labels.max()) + 1

    # Restrict to the body's THIN axis: a "hole punched through the
    # plate" goes along the plate's thinnest direction. A side plate
    # is thin in X (~6mm); its tab through-holes pierce in X. The
    # bottom plate is thin in Z (~10mm); its tabs pierce in Z. This
    # avoids the failure mode where the function processed a side
    # tab's perpendicular cluster (axis=X on the bottom plate) and
    # fragmented the tab's flat top + bottom into rim regions.
    bbox_size = cleaned.bbox[1] - cleaned.bbox[0]
    thin_axis = int(np.argmin(bbox_size))

    for axis_idx in [thin_axis]:
        axis_vec = np.eye(3)[axis_idx]
        flat = np.where(np.abs(normals @ axis_vec) > 0.85)[0]
        if len(flat) < 5:
            continue
        perp_idx = [i for i in range(3) if i != axis_idx]
        proj = centroids[flat][:, perp_idx]
        db = DBSCAN(eps=3.0, min_samples=4).fit(proj)
        clusters = [c for c in set(db.labels_) if c >= 0]
        if not clusters:
            continue

        # Compute each cluster's perp-plane bbox extent so we can
        # discriminate "main plate face" (huge) from "tab island"
        # (small, < 25mm).
        cluster_info: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        for cl in clusters:
            m = db.labels_ == cl
            ff = flat[m]
            cc = centroids[ff]
            bb = cc.max(axis=0) - cc.min(axis=0)
            extent = max(bb[p] for p in perp_idx)
            cluster_info.append((cl, ff, cc, extent))

        for cl, island_faces, island_centroids, extent in cluster_info:
            # Tab island: small in perp plane, but not tiny.
            if extent > 25.0 or extent < 4.0:
                continue
            # The cluster's perp-plane extent should be UNIFORM
            # (round-ish), not elongated: a tab annulus is round,
            # but a perimeter strip is long+thin.
            sub_bb = island_centroids.max(axis=0) - island_centroids.min(axis=0)
            perp_dims = sorted([sub_bb[p] for p in perp_idx])
            if perp_dims[0] / max(perp_dims[1], 0.01) < 0.6:
                continue
            # The cluster must bridge BOTH sides of the plate
            # (+axis-facing faces AND -axis-facing faces in the same
            # spatial cluster). This is the signature of a hole
            # punched through a thin region: DBSCAN bridges the
            # opposing flat faces because they share the same perp-
            # plane footprint.
            #
            # A standalone tab top (e.g. the bottom-plate's
            # cantilevered side tabs) has +Z-facing faces only and
            # does NOT bridge — its cluster is single-sided. We
            # skip those because they're already cleanly segmented
            # by earlier passes; intervening fragments them.
            side_signs = np.sign(normals[island_faces] @ axis_vec)
            n_pos = int((side_signs > 0).sum())
            n_neg = int((side_signs < 0).sum())
            if min(n_pos, n_neg) < 4:
                continue  # single-sided — leave alone

            # Don't peel an island that's ALREADY sitting in clean
            # local regions. We only need to act if the island's
            # faces are absorbed into giant plate-spanning regions
            # (the rolled-edge frame). When < 50% of island faces
            # are in large (>=200) regions, the existing
            # segmentation already gave the tab its own region(s)
            # and intervening would just fragment them.
            island_label_sizes = Counter(labels.tolist())
            in_big = int(sum(
                1 for fi in island_faces
                if island_label_sizes[int(labels[fi])] >= 200
            ))
            if in_big / len(island_faces) < 0.5:
                continue

            island_center = island_centroids.mean(axis=0)

            # Bore-wall candidates: faces (nearly) perpendicular to
            # axis (|n·axis|<0.15, tight to capture only the actual
            # cylinder wall and not the inner-edge fillet faces
            # which can have |n·axis| up to ~0.25), within a small
            # radius (extent * 0.7) of island center in perp plane,
            # AND currently sitting in a large plate-spanning
            # region (>= 200 faces). Faces already in clean local
            # regions (existing well-formed bore / rim / chamfer
            # regions, typically 30-100 faces) are left alone —
            # this pass is only meant to RESCUE faces absorbed
            # into a giant rolled-edge or frame region.
            label_sizes = Counter(labels.tolist())
            radial = np.where(np.abs(normals @ axis_vec) < 0.15)[0]
            if len(radial) == 0:
                continue
            d_perp = np.linalg.norm(
                centroids[radial][:, perp_idx]
                - island_center[perp_idx],
                axis=1,
            )
            bore_radius_max = extent * 0.7
            bore_mask = d_perp < bore_radius_max
            bore_candidates = radial[bore_mask]
            if len(bore_candidates) == 0:
                continue
            big_src = np.array(
                [label_sizes[int(labels[fi])] >= 200 for fi in bore_candidates],
                dtype=bool,
            )
            bore_candidates = bore_candidates[big_src]
            if len(bore_candidates) < 8:
                continue

            # Skip if bore is already local (dominant label is small
            # and centered here).
            src_labels = Counter(int(labels[fi]) for fi in bore_candidates)
            dom_lbl, dom_n = src_labels.most_common(1)[0]
            if dom_n / len(bore_candidates) < 0.5:
                # Already split across labels — leave alone
                pass
            else:
                lbl_face_count = int((labels == dom_lbl).sum())
                if lbl_face_count < 100:
                    continue  # already a local feature

            # Reassign bore-wall faces to a new label.
            for fi in bore_candidates:
                labels[fi] = next_label
            next_label += 1

            # Rim fillet candidates: tilted normal (anything between
            # the bore-wall threshold and the flat-plate threshold),
            # near hole center. Split front/back by sign of
            # (n · axis) and reassign each side independently so
            # each rim becomes its own region.
            #
            # Range 0.15-0.95 deliberately overlaps the bore-wall
            # 0.15 cutoff so the inner-edge fillet faces (|n·axis|
            # 0.15-0.3) become rim, not bore. Outer-edge fillet
            # faces (|n·axis| 0.85-0.95) become rim, not flat plate.
            tilted = np.where(
                (np.abs(normals @ axis_vec) > 0.15)
                & (np.abs(normals @ axis_vec) < 0.95)
            )[0]
            if len(tilted) == 0:
                continue
            d_perp_tilted = np.linalg.norm(
                centroids[tilted][:, perp_idx]
                - island_center[perp_idx],
                axis=1,
            )
            rim_mask = d_perp_tilted < bore_radius_max
            tilted_in_range = tilted[rim_mask]
            if len(tilted_in_range) == 0:
                continue
            # Same big-source restriction as for bore peeling.
            big_src_tilted = np.array(
                [label_sizes[int(labels[fi])] >= 200 for fi in tilted_in_range],
                dtype=bool,
            )
            tilted_in_range = tilted_in_range[big_src_tilted]
            for side in (1.0, -1.0):
                side_faces = tilted_in_range[
                    np.sign(normals[tilted_in_range] @ axis_vec) == side
                ]
                if len(side_faces) < 5:
                    continue
                for fi in side_faces:
                    labels[fi] = next_label
                next_label += 1

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[lbl] for lbl in labels], dtype=np.int64)
    return labels


def _split_rolled_edge_into_fillet_wall(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Split a curved region that connects two OPPOSING flat regions
    into fillet-near-A + side-wall + fillet-near-B sub-regions.

    A cantilevered protrusion (boss, tab, rib) has its outer
    perimeter rolled-edge connect from one flat face (e.g. the
    tab's bottom) around to the opposite flat face (the parent
    body's top). Region growing produces ONE region for the whole
    roll because the transition is smooth, but geometrically the
    roll has 3 distinct sub-regions:

        • fillet near flat A — face normals partly aligned with
          A's mean normal (the curve "pulling toward" A)
        • side wall — normals perpendicular to A's and B's normals,
          consistent direction throughout (low local rotation)
        • fillet near flat B — normals partly aligned with B's mean

    This pass enforces SYMMETRIC segmentation across multiple
    similar features on the same body. Without it, four identical
    mounting tabs on a base plate end up with different region
    counts because DBSCAN's axis voting and adjacent-cylinder
    fragmentation produce different cluster groupings depending on
    the tab's orientation relative to world axes.

    Conditions to split:
        1. Region has > 80 faces (substantial — likely a large roll)
        2. Region has mean-normal magnitude < 0.95 (truly curved)
        3. Region has TWO mesh-adjacent flat regions (mag > 0.95)
           with normals pointing in opposing directions
           (dot product < -0.7, i.e. > 135° apart)
        4. After splitting, all 3 sub-regions have >= 15 faces
           (otherwise this region was a simple fillet, not a roll)
    """
    labels = labels.copy()
    normals = cleaned.normals

    region_faces: dict[int, np.ndarray] = {}
    for lbl in np.unique(labels):
        region_faces[int(lbl)] = np.where(labels == lbl)[0]

    region_adj: dict[int, set[int]] = defaultdict(set)
    for fi, neighbors in adjacency.items():
        li = int(labels[fi])
        for nb in neighbors:
            ln = int(labels[nb])
            if li != ln:
                region_adj[li].add(ln)
                region_adj[ln].add(li)

    region_mean: dict[int, np.ndarray] = {}
    region_mag: dict[int, float] = {}
    for lbl, faces in region_faces.items():
        mn = normals[faces].mean(axis=0)
        mag = float(np.linalg.norm(mn))
        region_mag[lbl] = mag
        if mag > 1e-12:
            region_mean[lbl] = mn / mag
        else:
            region_mean[lbl] = mn

    next_label = int(labels.max()) + 1

    # Process largest curved regions first (most worth splitting);
    # iterate over a snapshot of the keys so adding new labels
    # mid-loop doesn't perturb iteration.
    curved_lbls = [
        l for l in region_faces
        if len(region_faces[l]) >= 80 and region_mag[l] < 0.95
    ]
    curved_lbls.sort(key=lambda l: -len(region_faces[l]))

    for lbl in curved_lbls:
        faces = region_faces[lbl]
        face_norms = normals[faces]

        flat_neighbors = [
            nb for nb in region_adj.get(lbl, set())
            if region_mag.get(nb, 0.0) >= 0.95
            and len(region_faces.get(nb, [])) >= 20
        ]
        if not flat_neighbors:
            continue

        # Try opposing pair first (3-piece roll: fillet-A +
        # side-wall + fillet-B). Otherwise try single flat neighbor
        # (2-piece roll: fillet + side-wall). The single-flat case
        # handles rolled edges that only cover one half of the
        # geometric wrap (the other half landed in a separate
        # region during region growing).
        opposing = None
        for i in range(len(flat_neighbors)):
            for j in range(i + 1, len(flat_neighbors)):
                a, b = flat_neighbors[i], flat_neighbors[j]
                if float(np.dot(region_mean[a], region_mean[b])) < -0.7:
                    opposing = (a, b)
                    break
            if opposing is not None:
                break

        if opposing is not None:
            a_lbl, b_lbl = opposing
            a_norm = region_mean[a_lbl]
            b_norm = region_mean[b_lbl]
            a_dots = face_norms @ a_norm
            b_dots = face_norms @ b_norm
            fillet_a_mask = a_dots > 0.5
            fillet_b_mask = b_dots > 0.5
            both = fillet_a_mask & fillet_b_mask
            if int(both.sum()) > 0:
                prefer_a = a_dots[both] >= b_dots[both]
                both_idx = np.where(both)[0]
                fillet_a_mask[both_idx] = prefer_a
                fillet_b_mask[both_idx] = ~prefer_a
            wall_mask = ~fillet_a_mask & ~fillet_b_mask

            if (
                int(fillet_a_mask.sum()) >= 20
                and int(fillet_b_mask.sum()) >= 20
                and int(wall_mask.sum()) >= 20
            ):
                wall_mean = face_norms[wall_mask].mean(axis=0)
                wall_mag = float(np.linalg.norm(wall_mean))
                if wall_mag >= 0.5:
                    wall_dir = wall_mean / wall_mag
                    if (
                        abs(float(np.dot(wall_dir, a_norm))) <= 0.5
                        and abs(float(np.dot(wall_dir, b_norm))) <= 0.5
                    ):
                        labels[faces[fillet_a_mask]] = next_label
                        next_label += 1
                        labels[faces[wall_mask]] = next_label
                        next_label += 1
                        continue  # done with this region

        # Single-flat-neighbor split: fillet (toward the flat) +
        # side-wall (perpendicular to it). Pick the flat neighbor
        # whose normal best matches the curved region's tilt
        # direction (i.e. dot(region_mean, flat_normal) is most
        # positive — the flat we're rolling INTO).
        best_flat = max(
            flat_neighbors,
            key=lambda nb: float(np.dot(region_mean[lbl], region_mean[nb])),
        )
        a_norm = region_mean[best_flat]
        if float(np.dot(region_mean[lbl], a_norm)) < 0.3:
            continue  # region isn't actually tilting toward this flat
        a_dots = face_norms @ a_norm
        fillet_mask = a_dots > 0.5
        wall_mask = ~fillet_mask
        if int(fillet_mask.sum()) < 20 or int(wall_mask.sum()) < 20:
            continue
        wall_mean = face_norms[wall_mask].mean(axis=0)
        wall_mag = float(np.linalg.norm(wall_mean))
        if wall_mag < 0.5:
            continue
        wall_dir = wall_mean / wall_mag
        if abs(float(np.dot(wall_dir, a_norm))) > 0.5:
            continue  # wall not perpendicular to flat → not a real roll

        labels[faces[fillet_mask]] = next_label
        next_label += 1
        # wall_mask faces keep `lbl`.

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[lbl] for lbl in labels], dtype=np.int64)
    return labels


def _split_long_fillet_into_corners(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Carve corner-fillet regions out of long curved wall regions.

    Region growing often merges a side wall with the perpendicular
    end-face strips it connects to via small (2-3mm radius) rounded
    corners. The merged region's bbox spans the full footprint of
    the wall + strips: long along the wall, modest along the strips,
    full plate-thickness in Z.

    The actual CORNER FILLET is a quarter-cylinder at each end of
    the wall, on the OUTER side (the side the wall normals point
    toward). It spans the full plate thickness in Z and has only
    ~2-3mm extent in the long-axis and mid-axis directions.

    This pass:
      1. Finds large curved regions (≥100 faces, mag<0.5) with a
         long-thin rectangular footprint (long axis ≥ 4× mid axis)
         and full plate-thickness in the short axis (≥5mm).
      2. Determines the OUTER mid-axis side from the dominant
         direction of face normals along that axis (e.g. for the
         left side wall, normals point -X, so outer is at mid_min).
      3. Carves out faces within ~2.5mm of each (long_end, outer)
         corner point in 2D — keeping the full Z range — into a
         new region per corner.
    """
    labels = labels.copy()
    normals = cleaned.normals
    next_label = int(labels.max()) + 1

    region_faces: dict[int, np.ndarray] = {}
    for lbl in np.unique(labels):
        region_faces[int(lbl)] = np.where(labels == lbl)[0]

    CORNER_RADIUS = 2.5  # mm — captures ~2-3mm fillet curl + a sliver of each adjacent flat
    MIN_CORNER_FACES = 12

    for lbl, faces in list(region_faces.items()):
        if len(faces) < 100:
            continue
        face_norms = normals[faces]
        mn = face_norms.mean(axis=0)
        mag = float(np.linalg.norm(mn))
        if mag >= 0.5:
            continue

        face_centroids = np.array([
            cleaned.vertices[cleaned.faces[fi]].mean(axis=0) for fi in faces
        ])
        bb_size = face_centroids.max(axis=0) - face_centroids.min(axis=0)
        bb_idx = np.argsort(bb_size)[::-1]
        long_axis = int(bb_idx[0])
        mid_axis = int(bb_idx[1])
        short_axis = int(bb_idx[2])

        if bb_size[long_axis] < 4.0 * bb_size[mid_axis]:
            continue
        if bb_size[short_axis] < 5.0:
            continue

        # Outer side along mid_axis: try first by face-normal direction
        # (works for a single-wall region like the left-side wall); fall
        # back to "whichever extreme is closest to the body's bbox edge"
        # for wrap regions where inner+outer walls cancel each other.
        body_min = float(cleaned.vertices[:, mid_axis].min())
        body_max = float(cleaned.vertices[:, mid_axis].max())
        reg_min = float(face_centroids[:, mid_axis].min())
        reg_max = float(face_centroids[:, mid_axis].max())
        n_mid_sum = float(face_norms[:, mid_axis].sum())
        if abs(n_mid_sum) >= 0.1 * len(faces):
            outer_mid = reg_max if n_mid_sum > 0 else reg_min
        else:
            d_min = abs(reg_min - body_min)
            d_max = abs(reg_max - body_max)
            outer_mid = reg_max if d_max < d_min else reg_min

        long_min = float(face_centroids[:, long_axis].min())
        long_max = float(face_centroids[:, long_axis].max())

        # Two corners: one at each long-axis end on the outer mid side.
        carved_any = False
        carved_mask_global = np.zeros(len(faces), dtype=bool)
        for long_target in (long_min, long_max):
            corner_mask = (
                (np.abs(face_centroids[:, long_axis] - long_target) < CORNER_RADIUS)
                & (np.abs(face_centroids[:, mid_axis] - outer_mid) < CORNER_RADIUS)
            )
            if int(corner_mask.sum()) < MIN_CORNER_FACES:
                continue
            labels[faces[corner_mask]] = next_label
            next_label += 1
            carved_any = True
            carved_mask_global |= corner_mask

        # If we carved corners, also split the leftover wrapper into
        # flat sub-regions by dominant face-normal direction. Without
        # this the corner fillet is adjacent to a single low-magnitude
        # "wrapper" region that doesn't pass the flat-plane test in
        # Stage 4.5's bridge-detection, so the corner never gets
        # tagged as a blend. Splitting the wrapper into +X/-X/+Y/-Y
        # buckets gives each corner a pair of true flat plane
        # neighbors to bridge between.
        if carved_any:
            leftover = ~carved_mask_global
            if int(leftover.sum()) >= 30:
                leftover_idx = np.where(leftover)[0]
                left_norms = face_norms[leftover_idx]
                # Bucket each face by signed dominant normal axis.
                # bucket = axis * 2 + (1 if positive else 0)
                abs_n = np.abs(left_norms)
                dom_axis = np.argmax(abs_n, axis=1)
                signs = (left_norms[np.arange(len(leftover_idx)), dom_axis] > 0).astype(np.int64)
                bucket = dom_axis * 2 + signs
                unique_buckets, counts = np.unique(bucket, return_counts=True)
                # Order buckets by count (largest stays as original label)
                order = np.argsort(counts)[::-1]
                for rank, ub_idx in enumerate(order):
                    ub = unique_buckets[ub_idx]
                    mask = bucket == ub
                    if int(mask.sum()) < 20:
                        continue
                    if rank == 0:
                        continue  # largest bucket keeps original lbl
                    target_faces = faces[leftover_idx[mask]]
                    labels[target_faces] = next_label
                    next_label += 1

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[lbl] for lbl in labels], dtype=np.int64)
    return labels


def _peel_axial_tangents_from_curves(
    labels: np.ndarray,
    normals: np.ndarray,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Peel nearly-axial faces out of curved regions.

    When a region classified as "curved" (mean-normal magnitude
    ≤ 0.9) contains a connected component of faces whose normals
    are nearly parallel to a principal axis (|n·axis| ≥ 0.95),
    those faces are the tangent edge of the fillet meeting a flat
    plate — NOT part of the rim fillet arc proper. Leaving them
    inside the rim region makes the rim visually "connect to the
    flat face next to it" and conflates two distinct features
    (flat tangent + fillet arc) under one highlight. This pass
    extracts them into their own region so each feature renders
    and fits independently.

    Only peels components of ≥10 mesh-connected axial faces
    (noise-level slivers stay in place).
    """
    labels = labels.copy()
    next_label = int(labels.max()) + 1
    # Tight threshold (~10° from axis = dot ≥ 0.985): catches
    # only the truly axis-aligned first row of a fillet — faces
    # whose normal is essentially the same as the adjacent
    # plate's. Looser thresholds (32° / 0.85) caught mid-fillet
    # faces too and split rolled-edge fillets into 2-3 segments
    # along their length. Rim-fillet vs plate separation for
    # through-holes is handled instead by _peel_hole_rim_fillets,
    # which is hole-targeted and doesn't fragment rolled edges.
    axis_cos_threshold = np.cos(np.radians(10.0))  # ≥ ~0.985

    for old_label in np.unique(labels):
        face_indices = np.where(labels == old_label)[0]
        if len(face_indices) < 20:
            continue
        region_normals = normals[face_indices]
        mean_n = region_normals.mean(axis=0)
        mean_mag = float(np.linalg.norm(mean_n))
        if mean_mag >= 0.9:
            continue  # already a tight planar cluster, nothing to peel
        if mean_mag <= 0.3:
            continue  # 360° rotating feature (bore wall) — its
            # cardinal-axis rows ARE part of the cylinder, not
            # tangent-to-plate strips. Leave them.

        # Flag faces whose normal is very close to a principal axis.
        axial_mask = np.max(np.abs(region_normals), axis=1) >= axis_cos_threshold
        if axial_mask.sum() < 10:
            continue

        axial_face_set = set(face_indices[axial_mask].tolist())
        # Connected components of axial faces within this region.
        visited: set[int] = set()
        for seed in axial_face_set:
            if seed in visited:
                continue
            component = {seed}
            queue = [seed]
            visited.add(seed)
            while queue:
                cur = queue.pop()
                for nb in adjacency.get(cur, ()):
                    if nb in axial_face_set and nb not in visited:
                        visited.add(nb)
                        component.add(nb)
                        queue.append(nb)
            if len(component) < 10:
                continue
            # Verify the component IS coherently pointing at one
            # axis (mean-normal magnitude ≥ 0.95 ≈ within ~18° of
            # one direction). This excludes axial-ish noise that
            # happens to cluster but covers multiple axis
            # directions, while still admitting genuine
            # near-tangent first-row strips of a fillet.
            comp_normals = normals[list(component)]
            comp_mag = float(np.linalg.norm(comp_normals.mean(axis=0)))
            if comp_mag < 0.95:
                continue
            for fi in component:
                labels[fi] = next_label
            next_label += 1

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)
    return labels


def _split_disconnected_components(
    labels: np.ndarray,
    adjacency: dict[int, set[int]],
    min_component_faces: int = 1,
) -> np.ndarray:
    """Split each label into one label per mesh-connected component,
    subject to a minimum component size.

    Earlier passes (specifically _split_flat_from_curved, which
    pulls out flat faces without re-labelling the remaining curved
    ones) can leave a single label referring to multiple
    mesh-disconnected face clusters. Downstream fits and visuals
    treat a label as a single region, so two physically separate
    features sharing a label would render as one highlight spanning
    both.

    `min_component_faces` gates how aggressive the split is — tiny
    disconnected slivers (< threshold) stay with the largest
    component's label so a multi-feature region (like a bottom
    plate with 4 tabs) doesn't explode into dozens of noise regions.
    """
    labels = labels.copy()
    next_label = int(labels.max()) + 1
    for old_label in np.unique(labels):
        face_indices = np.where(labels == old_label)[0]
        if len(face_indices) <= 1:
            continue
        # Find all components
        remaining = set(face_indices.tolist())
        components: list[set[int]] = []
        while remaining:
            seed = next(iter(remaining))
            component = {seed}
            queue = [seed]
            remaining.discard(seed)
            while queue:
                cur = queue.pop()
                for nb in adjacency.get(cur, ()):
                    if nb in remaining:
                        remaining.discard(nb)
                        component.add(nb)
                        queue.append(nb)
            components.append(component)
        if len(components) <= 1:
            continue
        # Largest component keeps old_label. Only components ≥
        # min_component_faces become their own region; smaller ones
        # stay under old_label (joined with the largest component,
        # which merges them visually under the same color).
        components.sort(key=len, reverse=True)
        for comp in components[1:]:
            if len(comp) >= min_component_faces:
                for fi in comp:
                    labels[fi] = next_label
                next_label += 1
            # else: keep old_label (same as components[0]) — tiny
            # slivers don't get their own region

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)
    return labels


def _split_flat_from_curved(
    labels: np.ndarray,
    normals: np.ndarray,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Split regions that contain both flat and curved faces.

    Region growing may merge flat side faces into adjacent cylinder regions
    because the face-to-face angle between the last cylinder face and the
    first flat face is within the curved threshold. This post-processing step
    detects such mixed regions and separates the flat faces into their own
    connected components.

    A face is considered "flat" if its normal is closely aligned with a
    principal axis (within ~8°). Flat faces that form a connected component
    within the region are split out.
    """
    labels = labels.copy()
    next_label = int(labels.max()) + 1
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        face_indices = np.where(mask)[0]
        if len(face_indices) < 4:
            continue

        # Check normal spread — only process regions with mixed normals
        region_normals = normals[face_indices]
        avg_n = region_normals.mean(axis=0)
        avg_norm = np.linalg.norm(avg_n)
        if avg_norm > 0.95:
            # Already a tight planar cluster — skip
            continue

        # Cylinder-parent guard: if the region's normals are rank-2
        # (a single rotating-normal arc like a bore wall), its
        # ±cardinal-direction strips are axis-aligned faces that
        # happen to land where the arc crosses a principal axis —
        # they're geometrically part of the cylinder and must NOT
        # be peeled off as "flat" sub-regions. Without this guard,
        # a 360° bore becomes 1 curved region + 4 cardinal strips.
        if len(region_normals) >= 3:
            centered_n = region_normals - avg_n
            _, s_vals, _ = np.linalg.svd(centered_n, full_matrices=False)
            if s_vals[0] > 1e-12:
                s_norm = s_vals / s_vals[0]
                if s_norm[1] > 0.3 and s_norm[2] < 0.2:
                    continue  # cylinder-like — preserve whole

        # Classify each face as flat or curved
        # A face is flat if its normal is within ~15° of a principal axis
        axis_threshold = np.cos(np.radians(15.0))
        is_flat = np.max(np.abs(region_normals), axis=1) >= axis_threshold

        n_flat = int(is_flat.sum())
        n_curved = len(face_indices) - n_flat
        if n_flat == 0 or n_curved == 0:
            # All flat or all curved — no split needed
            continue

        # Find connected components of flat faces within this region
        flat_faces = set(face_indices[is_flat].tolist())
        visited = set()

        for start in flat_faces:
            if start in visited:
                continue
            # BFS to find connected flat component
            component = []
            queue = [start]
            while queue:
                fi = queue.pop(0)
                if fi in visited:
                    continue
                visited.add(fi)
                component.append(fi)
                for ni in adjacency.get(fi, set()):
                    if ni in flat_faces and ni not in visited:
                        queue.append(ni)

            # Only peel substantial flat components. Tiny slivers
            # (<10 faces) are usually noise at the fillet-to-flat
            # boundary — splitting them creates dozens of 2-6 face
            # regions on complex bodies (the bottom plate's 4 tabs
            # was the motivating case). Leaving them with the
            # curved majority means they paint as part of the fillet
            # instead of as their own highlight.
            if len(component) >= 10:
                comp_normals = normals[component]
                comp_avg = comp_normals.mean(axis=0)
                comp_norm = np.linalg.norm(comp_avg)
                if comp_norm > 0.9:
                    for fi in component:
                        labels[fi] = next_label
                    next_label += 1

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)

    return labels


def _curvature_split(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
    config: SegmentConfig,
) -> np.ndarray:
    """Split regions with high curvature variance using k-means on (normal, curvature)."""
    logger.debug("Curvature split: %d labels in, threshold=%.6f", len(np.unique(labels)), config.curvature_variance_threshold)
    labels = labels.copy()
    try:
        vertex_curvatures = _estimate_vertex_curvatures(cleaned)
        face_curvatures = _vertex_to_face_curvatures(vertex_curvatures, cleaned.faces)
    except Exception:
        logger.exception("Curvature estimation failed: verts=%d, faces=%d", len(cleaned.vertices), len(cleaned.faces))
        raise

    next_label = int(labels.max()) + 1
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        count = int(mask.sum())
        if count < config.min_region_faces * 2:
            # Too small to meaningfully split
            continue

        region_curvatures = face_curvatures[mask]
        curv_var = float(np.var(region_curvatures))

        if curv_var < config.curvature_variance_threshold:
            continue

        # A uniform cylinder has all faces at the same curvature (=1/radius)
        # but with rotating normals. Don't split it — keep it as one region
        # so the cylinder fitter sees a full ring of faces.
        #
        # Protection 1 — uniform curvature: loose (cv < 0.6) because
        # per-face Gaussian curvature estimates are noisy on real
        # meshes; a true cylinder has cv ≈ 0 analytically but 0.3–0.5
        # in practice. At cv < 0.6 we avoid splitting real cylinders
        # whose noise is within expected bounds.
        curv_mean = float(np.mean(np.abs(region_curvatures)))
        if curv_mean > 1e-10:
            curv_cv = float(np.std(region_curvatures)) / curv_mean
            if curv_cv < 0.6:
                continue

        # Protection 2 — rank-2 normals: a cylinder/rolled-arc's
        # normals all lie in the plane perpendicular to the axis of
        # rotation. SVD of centered normals has BOTH a near-zero
        # third singular value AND a substantial second singular
        # value (real arc spread in the plane, not just plate
        # noise). Require s[2]/s[0] < 0.2 AND s[1]/s[0] > 0.3:
        # KMeans on a single rotating-normal feature would chop
        # the arc into cardinal quadrants (the "every 90°" split
        # we want to avoid), while a mostly-plate region with a
        # small tangent attached has low s[1]/s[0] and correctly
        # proceeds to splitting.
        region_normals = cleaned.normals[mask]
        if len(region_normals) >= 3:
            centered_n = region_normals - region_normals.mean(axis=0)
            _, s_vals, _ = np.linalg.svd(centered_n, full_matrices=False)
            if s_vals[0] > 1e-12:
                s_norm = s_vals / s_vals[0]
                if s_norm[1] > 0.3 and s_norm[2] < 0.2:
                    continue  # rank-2 rotating-normal arc → protect

        # Build feature vectors: (nx, ny, nz, curvature)
        features = np.column_stack([region_normals, region_curvatures])

        # Try 2 to max_clusters and pick the best silhouette
        region_indices = np.where(mask)[0]
        best_k = _find_best_k(features, config.max_kmeans_clusters)

        if best_k <= 1:
            continue

        km = KMeans(n_clusters=best_k, n_init=5, random_state=42)
        sub_labels = km.fit_predict(features)

        # Assign new labels
        for k in range(best_k):
            sub_mask = sub_labels == k
            if k == 0:
                # Keep the original label for cluster 0
                pass
            else:
                labels[region_indices[sub_mask]] = next_label
                next_label += 1

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)

    return labels


def _split_cylinders_from_planes(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Split cylindrical hole faces out of large planar regions.

    After region growing, small holes near edges often get absorbed into the
    dominant planar region. This function detects faces within planar regions
    whose normals deviate significantly from the region's dominant direction,
    clusters them spatially, and splits each cluster into its own region.
    """
    from sklearn.cluster import DBSCAN

    logger.debug("Splitting cylinders from planes: %d labels", len(np.unique(labels)))
    labels = labels.copy()
    next_label = int(labels.max()) + 1

    for label in np.unique(labels):
        mask = labels == label
        face_indices = np.where(mask)[0]
        if len(face_indices) < 20:
            continue

        region_normals = cleaned.normals[face_indices]
        avg_n = region_normals.mean(axis=0)
        avg_norm = np.linalg.norm(avg_n)
        if avg_norm < 0.8:
            continue  # not a planar region
        avg_n /= avg_norm

        # Find faces that deviate from the dominant normal (potential hole walls).
        # Use a loose threshold to catch cylindrical faces that tilt away from flat.
        dots = region_normals @ avg_n
        deviant_mask = dots < 0.92  # > ~23° from region normal
        deviant_indices = face_indices[deviant_mask]

        if len(deviant_indices) < 3:
            continue

        # Cluster deviant faces spatially — each cluster is a potential hole
        centroids = np.array([
            cleaned.vertices[cleaned.faces[fi]].mean(axis=0) for fi in deviant_indices
        ])

        # Adaptive eps based on part size
        part_size = float((cleaned.bbox[1] - cleaned.bbox[0]).max())
        eps = max(part_size * 0.05, 5.0)

        clustering = DBSCAN(eps=eps, min_samples=3).fit(centroids)

        for cl_label in set(clustering.labels_):
            if cl_label == -1:
                continue
            cl_mask = clustering.labels_ == cl_label
            cl_faces = deviant_indices[cl_mask]
            if len(cl_faces) < 3:
                continue

            # Assign these faces to a new region
            for fi in cl_faces:
                labels[fi] = next_label
            next_label += 1

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)

    return labels


def _unify_cylinder_features(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Find every connected cylindrical feature in the mesh and
    collapse it into a single region, overriding whatever label
    assignment the prior passes produced.

    A "cylinder face" is identified locally: its 2-hop mesh
    neighborhood has rank-2 normal structure (s[1] substantial,
    s[2] near zero) — i.e. normals that rotate in a single plane,
    the signature of any cylindrical surface (bore wall, rolled
    edge, rim fillet) regardless of direction. These faces are
    clustered spatially; each spatial cluster becomes one region.

    Fixes the "one through-hole ends up split across 4-5 regions"
    issue: flood-fill + per-region splitters fragment a single
    cylinder whenever the flood-fill picks seeds at different
    cardinal positions, and the existing fragment merger is too
    restrictive (sub-50-face candidates only). This pass works
    from the geometry outwards rather than trying to clean up
    after mesh-adjacency flood-fill.
    """
    from sklearn.cluster import DBSCAN

    labels = labels.copy()
    normals = cleaned.normals
    n_faces = len(normals)

    # 1. Identify cylinder-face candidates via 2-hop SVD rank-2.
    is_cylinder_face = np.zeros(n_faces, dtype=bool)
    for fi in range(n_faces):
        hop1 = adjacency.get(fi, set())
        hop2: set[int] = set()
        for f1 in hop1:
            hop2.update(adjacency.get(f1, set()))
        nbrhd = list(hop1 | hop2 | {fi})
        if len(nbrhd) < 6:
            continue
        nb_n = normals[nbrhd]
        centered = nb_n - nb_n.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        if s[0] < 1e-10:
            continue
        s_n = s / s[0]
        # Rank-2: enough in-plane spread (s[1] > 0.2) AND
        # little out-of-plane spread (s[2] < 0.3). Loose enough
        # to catch faces at the bore-rim junction where the 2-hop
        # neighborhood samples both the bore's arc AND the rim
        # fillet's arc, making strict rank-2 fail. Plates still
        # fail because s[1] ≈ 0 there.
        if s_n[1] > 0.2 and s_n[2] < 0.3:
            is_cylinder_face[fi] = True

    cyl_indices = np.where(is_cylinder_face)[0]
    if len(cyl_indices) < 4:
        return labels

    # 2. Cluster cylinder faces spatially (DBSCAN on centroids).
    # Each connected cylindrical feature forms one cluster.
    centroids = np.array([
        cleaned.vertices[cleaned.faces[fi]].mean(axis=0)
        for fi in cyl_indices
    ])
    bbox_diag = float(np.linalg.norm(cleaned.bbox[1] - cleaned.bbox[0]))
    # eps tuned to be larger than face size but smaller than the
    # distance between distinct features. 2% of bbox diagonal works
    # empirically for brackets and plates.
    eps = max(bbox_diag * 0.02, 2.0)
    clustering = DBSCAN(eps=eps, min_samples=3).fit(centroids)

    # For each DBSCAN cluster, find the cluster's dominant
    # rotation axis (direction of smallest variance in normals,
    # i.e. 3rd right singular vector) and filter to PURE-CYLINDER
    # faces whose normal is perpendicular to that axis. This
    # excludes rim-fillet (torus) faces that mingle with the bore
    # wall in the same spatial cluster but whose normals have an
    # axial component — they need to remain in their own region
    # so downstream fillet detection can see them.
    next_label = int(labels.max()) + 1
    face_centroid_cache: dict[int, np.ndarray] = {}

    def _face_centroid(fi: int) -> np.ndarray:
        c = face_centroid_cache.get(fi)
        if c is None:
            c = cleaned.vertices[cleaned.faces[fi]].mean(axis=0)
            face_centroid_cache[fi] = c
        return c

    for cl in set(clustering.labels_):
        if cl == -1:
            continue
        cluster_faces = cyl_indices[clustering.labels_ == cl]
        if len(cluster_faces) < 4:
            continue

        # Find cluster rotation axis. A real cylinder axis A has
        # face normals that ROTATE in the plane perpendicular to A,
        # spanning a wide range of angles (a quarter-cylinder spans
        # ~90°, a full cylinder ~360°). Wrong axes have normals
        # CLUMPED in the perpendicular plane.
        #
        # For each candidate world axis (X/Y/Z), project the cluster
        # normals onto the perpendicular plane and measure circular
        # spread (1 - |mean(e^{iθ})|). Pick the axis with highest
        # spread, falling back to SVD's 3rd direction for off-axis
        # bores.
        #
        # The previous "count |n·axis|<0.25" voting failed when a
        # spatial cluster mixed bore-wall faces (rotating around Z)
        # with adjacent flat tab-top (+Z) and base-plate (-Z) faces:
        # those flat faces also satisfy |n·X|<0.25 and |n·Y|<0.25,
        # tying the vote and arbitrarily picking X — which then
        # absorbed the tab top into the cylinder region.
        cn = normals[cluster_faces]

        def _axis_spread(ax: np.ndarray) -> float:
            proj = cn - (cn @ ax)[:, None] * ax
            mag = np.linalg.norm(proj, axis=1)
            signif = mag > 0.5
            if int(signif.sum()) < 4:
                return -1.0
            p = proj[signif] / mag[signif][:, None]
            if abs(ax[0]) < 0.5:
                u = np.cross(ax, np.array([1.0, 0.0, 0.0]))
            else:
                u = np.cross(ax, np.array([0.0, 1.0, 0.0]))
            u = u / np.linalg.norm(u)
            v = np.cross(ax, u)
            theta = np.arctan2(p @ v, p @ u)
            z = np.exp(1j * theta)
            return float(1.0 - abs(z.mean()))

        best_axis = None
        best_spread = -1.0
        for candidate in np.eye(3):
            spread = _axis_spread(candidate)
            if spread > best_spread:
                best_spread = spread
                best_axis = candidate
        # SVD fallback for non-axis-aligned bores
        centered_cn = cn - cn.mean(axis=0)
        _, _, vh = np.linalg.svd(centered_cn, full_matrices=False)
        svd_axis = vh[2]
        svd_spread = _axis_spread(svd_axis)
        if svd_spread > best_spread + 0.1:
            axis = svd_axis
        else:
            axis = best_axis

        # Keep only faces whose normal is perpendicular to axis
        # (|n · axis| < 0.25 ≈ > 75° from axis). Bore-wall
        # normals are radial, rim-fillet normals tilt toward the
        # axis — this discriminates them.
        axial_dots = np.abs(cn @ axis)
        pure_cyl_mask = axial_dots < 0.25
        pure_cyl_faces = cluster_faces[pure_cyl_mask]
        if len(pure_cyl_faces) < 4:
            continue

        # Cluster bbox for adjacency-based grow (only extends to
        # neighbors within the cluster's spatial footprint).
        cc = np.array([_face_centroid(int(fi)) for fi in pure_cyl_faces])
        c_min = cc.min(axis=0) - eps
        c_max = cc.max(axis=0) + eps

        # Grow by mesh adjacency: absorb mesh-neighbor faces that
        # are ALSO perpendicular-to-axis AND within the bbox.
        # Absorbs bore-wall faces the initial SVD rank check
        # missed (e.g. at the rim-wall junction) without
        # swallowing the rim fillet itself.
        absorbed = set(int(fi) for fi in pure_cyl_faces)
        frontier = set(absorbed)
        for _hop in range(2):
            next_frontier: set[int] = set()
            for fi in frontier:
                for nb in adjacency.get(fi, ()):
                    if nb in absorbed:
                        continue
                    nb_axial = abs(float(normals[nb] @ axis))
                    if nb_axial >= 0.25:
                        continue  # not a bore-wall normal
                    c = _face_centroid(int(nb))
                    if np.all(c >= c_min) and np.all(c <= c_max):
                        absorbed.add(int(nb))
                        next_frontier.add(int(nb))
            frontier = next_frontier

        existing_labels = Counter(int(labels[fi]) for fi in absorbed)
        best_existing, best_count = existing_labels.most_common(1)[0]
        if best_count / len(absorbed) >= 0.8:
            target_label = best_existing
        else:
            target_label = next_label
            next_label += 1
        for fi in absorbed:
            labels[fi] = target_label

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)
    return labels


def _merge_adjacent_cylinder_fragments(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Merge adjacent regions that together form cylindrical holes.

    After segmentation, hole walls often get split into many regions
    (e.g. 4 quarters for a cylinder). This pass finds small regions that
    are MESH-ADJACENT to each other and whose normals suggest they're
    part of a single curved surface (normals span a range of directions,
    suggesting they wrap around an axis). Merges them into one region.
    """
    labels = labels.copy()
    normals = cleaned.normals

    # Build region→faces map and collect region-level adjacency via shared faces
    region_faces: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        region_faces[int(label)] = np.where(labels == label)[0]

    # Build region-level adjacency from face adjacency
    region_adj: dict[int, set[int]] = defaultdict(set)
    for fi, neighbors in adjacency.items():
        li = int(labels[fi])
        for nb in neighbors:
            ln = int(labels[nb])
            if li != ln:
                region_adj[li].add(ln)
                region_adj[ln].add(li)

    # Identify "fragment" regions: small (<= 50 faces) AND curved-ish
    # (normals span a range of directions, indicating part of a curved surface)
    def _is_fragment(label: int) -> bool:
        faces = region_faces[label]
        if len(faces) > 50:
            return False
        if len(faces) < 2:
            return True  # tiny regions always candidates
        region_n = normals[faces]
        avg_n = region_n.mean(axis=0)
        avg_norm = np.linalg.norm(avg_n)
        if avg_norm < 1e-10:
            return True
        avg_n /= avg_norm
        # Normal spread: how much do the normals deviate from their mean?
        # High spread → curved surface fragment (quarter of a cylinder etc.)
        # Low spread → flat face
        dots = region_n @ avg_n
        spread = float(1.0 - dots.min())  # 0 = all aligned, 2 = opposing
        return spread > 0.05  # at least ~18° max deviation → curved

    fragments = {l for l in region_faces if _is_fragment(l)}
    if len(fragments) < 2:
        return labels

    # Group fragments by connected components in region adjacency
    # (only walking through fragment regions)
    visited = set()
    next_label = int(labels.max()) + 1
    for start in list(fragments):
        if start in visited:
            continue
        # BFS through fragment neighbors
        group = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            if node not in fragments:
                continue
            group.add(node)
            for nb in region_adj.get(node, set()):
                if nb not in visited and nb in fragments:
                    queue.append(nb)

        if len(group) < 2:
            continue

        # Verify this group looks like a cylindrical surface by checking
        # that the combined faces fit a cylinder. If not, skip.
        all_faces = np.concatenate([region_faces[l] for l in group])
        if len(all_faces) < 4 or len(all_faces) > 200:
            continue
        combined_normals = normals[all_faces]
        # Check normal variety: if normals span many directions, it's curved
        mean_n = combined_normals.mean(axis=0)
        mean_norm = np.linalg.norm(mean_n)
        if mean_norm < 0.01:
            normal_spread = 2.0  # pointing in opposite directions
        else:
            mean_n /= mean_norm
            dots = combined_normals @ mean_n
            normal_spread = float(1.0 - dots.min())
        # A cylinder quarter group has spread > 0.3, a full cylinder > 1.0
        if normal_spread < 0.2:
            continue  # too flat, not a cylinder

        logger.info(
            "Merging %d cylinder fragment regions: %d total faces, spread=%.2f",
            len(group), len(all_faces), normal_spread,
        )
        for l in group:
            labels[region_faces[l]] = next_label
        next_label += 1

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)

    return labels


def _find_best_k(features: np.ndarray, max_k: int) -> int:
    """Pick the best number of clusters using inertia elbow heuristic."""
    n_samples = len(features)
    max_k = min(max_k, n_samples)
    if max_k < 2:
        return 1

    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        km.fit(features)
        inertias.append(km.inertia_)

    if len(inertias) < 2:
        return 1

    # Simple elbow: find the k where the relative drop in inertia becomes small
    for i in range(1, len(inertias)):
        if inertias[i - 1] < 1e-15:
            return i
        relative_drop = (inertias[i - 1] - inertias[i]) / inertias[i - 1]
        if relative_drop < 0.3:
            return i

    return max_k


def _estimate_vertex_curvatures(cleaned: CleanedMesh) -> np.ndarray:
    """Estimate discrete mean curvature per vertex using the angle deficit method.

    For each vertex, the curvature is approximated by:
        H ≈ (2π - sum_of_angles_at_vertex) / mixed_area

    This is a simplified version — returns absolute curvature magnitude per vertex.
    """
    n_verts = len(cleaned.vertices)
    angle_sums = np.zeros(n_verts, dtype=np.float64)
    areas = np.zeros(n_verts, dtype=np.float64)

    for face in cleaned.faces:
        v0, v1, v2 = cleaned.vertices[face[0]], cleaned.vertices[face[1]], cleaned.vertices[face[2]]

        # Edges
        e01 = v1 - v0
        e02 = v2 - v0
        e12 = v2 - v1

        # Angles at each vertex of the triangle
        def _angle(a: np.ndarray, b: np.ndarray) -> float:
            cos_a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15)
            return float(np.arccos(np.clip(cos_a, -1, 1)))

        a0 = _angle(e01, e02)
        a1 = _angle(-e01, e12)
        a2 = _angle(-e02, -e12)

        angle_sums[face[0]] += a0
        angle_sums[face[1]] += a1
        angle_sums[face[2]] += a2

        # Triangle area contribution (1/3 of face area to each vertex)
        face_area = 0.5 * np.linalg.norm(np.cross(e01, e02))
        for vi in face:
            areas[vi] += face_area / 3.0

    # Gaussian curvature via angle deficit
    gauss_curvature = np.zeros(n_verts, dtype=np.float64)
    valid = areas > 1e-15
    gauss_curvature[valid] = (2.0 * np.pi - angle_sums[valid]) / areas[valid]

    return np.abs(gauss_curvature)


def _vertex_to_face_curvatures(
    vertex_curvatures: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Average vertex curvatures to get a per-face curvature estimate."""
    return vertex_curvatures[faces].mean(axis=1)


def _classify_surface_type(
    normals: np.ndarray, curvatures: np.ndarray
) -> SurfaceType:
    """Classify a region based on its curvature and normal statistics.

    - Near-zero mean curvature → PLANE
    - Normals spread in a ring (1 small SVD value) → CYLINDER
    - Linearly varying curvature → CONE
    - Normals spread in all 3D directions → SPHERE
    """
    mean_curv = float(np.mean(curvatures))
    std_curv = float(np.std(curvatures))

    # Normal spread: if normals are tightly clustered → planar
    avg_normal = normals.mean(axis=0)
    norm = np.linalg.norm(avg_normal)
    if norm > 1e-15:
        avg_normal /= norm
    dots = normals @ avg_normal
    normal_spread = float(1.0 - np.mean(dots))

    # Thresholds (empirically chosen for prismatic parts)
    if normal_spread < 0.01 and mean_curv < 0.5:
        return SurfaceType.PLANE

    if mean_curv < 0.1:
        if normal_spread < 0.1:
            # Slight scatter in normals: could be a true plane with mesh
            # noise OR a gentle 3D corner fillet (e.g. a 5-10mm radius
            # rounded transition tessellated coarsely). Use SVD on the
            # centered normals to disambiguate. A true plane has all
            # singular values near zero; a curved surface has at least
            # one large singular value, and a sphere-like 3D corner
            # fillet has TWO large ones (normals span a 2D patch).
            if len(normals) >= 3:
                centered = normals - normals.mean(axis=0)
                _, sv, _ = np.linalg.svd(centered, full_matrices=False)
                sv_n = sv / (sv[0] + 1e-15)
                if sv_n[1] < 0.5:
                    return SurfaceType.PLANE
            else:
                return SurfaceType.PLANE

    # Use SVD on normals to distinguish cylinder vs sphere.
    # Cylinder normals lie in a plane (perpendicular to the axis),
    # so SVD of centered normals will have one near-zero singular value.
    # Sphere normals spread in all 3 directions.
    if len(normals) >= 3:
        centered_n = normals - normals.mean(axis=0)
        _, s_vals, _ = np.linalg.svd(centered_n, full_matrices=False)
        s_vals = s_vals / (s_vals[0] + 1e-15)  # normalize
        # If smallest singular value is small relative to the largest,
        # normals are roughly coplanar → cylinder
        is_coplanar = s_vals[2] < 0.2
    else:
        is_coplanar = True

    if std_curv / (mean_curv + 1e-15) < 0.5:
        # Relatively uniform curvature
        if is_coplanar:
            return SurfaceType.CYLINDER
        else:
            return SurfaceType.SPHERE

    if std_curv / (mean_curv + 1e-15) < 1.0:
        return SurfaceType.CONE

    # High curvature variance — try SVD-based classification
    if is_coplanar:
        return SurfaceType.CYLINDER

    return SurfaceType.UNKNOWN


def _find_boundary_edges(
    face_indices: np.ndarray, faces: np.ndarray
) -> list[tuple[int, int]]:
    """Find edges that border exactly one face in the region (boundary edges)."""
    face_set = set(face_indices.tolist())
    edge_count: dict[tuple[int, int], int] = defaultdict(int)

    for fi in face_indices:
        face = faces[fi]
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            edge_count[edge] += 1

    # Boundary edges appear exactly once within the region
    return [(v0, v1) for (v0, v1), count in edge_count.items() if count == 1]


def _build_regions(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
) -> list[FaceRegion]:
    """Convert label array into a list of FaceRegion objects."""
    logger.debug("Building regions from %d unique labels", len(np.unique(labels)))
    try:
        vertex_curvatures = _estimate_vertex_curvatures(cleaned)
    except Exception:
        logger.exception("Vertex curvature estimation failed during region building")
        raise
    face_curvatures = _vertex_to_face_curvatures(vertex_curvatures, cleaned.faces)

    regions = []
    for region_id in range(int(labels.max()) + 1):
        mask = labels == region_id
        face_indices = np.where(mask)[0]

        if len(face_indices) == 0:
            continue

        region_normals = cleaned.normals[face_indices]
        region_curvatures = face_curvatures[face_indices]

        # Get unique vertices for this region
        region_vertex_indices = np.unique(cleaned.faces[face_indices].ravel())
        region_vertices = cleaned.vertices[region_vertex_indices]

        surface_type = _classify_surface_type(region_normals, region_curvatures)
        boundary_edges = _find_boundary_edges(face_indices, cleaned.faces)

        regions.append(
            FaceRegion(
                region_id=region_id,
                face_indices=face_indices,
                surface_type=surface_type,
                vertices=region_vertices,
                normals=region_normals,
                boundary_edges=boundary_edges,
            )
        )

    return regions
