"""Stage 2: Surface Segmentation.

Clusters mesh faces into regions belonging to a single analytic surface type
using region-growing with normal clustering, followed by curvature-based
classification.
"""

from __future__ import annotations

import logging
from collections import defaultdict

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

    adjacency = _build_face_adjacency(cleaned.faces)
    sharp_edges = _detect_sharp_edges(cleaned.normals, adjacency, config)
    labels = _region_growing(cleaned.normals, adjacency, config, sharp_edges)
    labels = _merge_small_regions(labels, adjacency, cleaned.normals, config.min_region_faces)
    labels = _split_flat_from_curved(labels, cleaned.normals, adjacency)
    # Use a smaller min_faces for the post-split merge so that small flat
    # strips (e.g. 2-4 face plane regions between cylinder holes) don't get
    # re-absorbed into the adjacent cylinder region they were just split from.
    labels = _merge_small_regions(labels, adjacency, cleaned.normals, min(config.min_region_faces, 2))
    labels = _curvature_split(labels, cleaned, adjacency, config)
    labels = _split_cylinders_from_planes(labels, cleaned, adjacency)
    labels = _merge_adjacent_cylinder_fragments(labels, cleaned, adjacency)

    regions = _build_regions(labels, cleaned, adjacency)
    logger.info("Segmented mesh into %d regions", len(regions))
    return regions


def _build_face_adjacency(faces: np.ndarray) -> dict[int, set[int]]:
    """Build face-face adjacency from shared edges.

    Two faces are adjacent if they share exactly two vertices (an edge).
    """
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

            if len(component) >= 2:
                # Split this flat component into its own region, but only
                # if the flat faces share a similar normal direction
                comp_normals = normals[component]
                comp_avg = comp_normals.mean(axis=0)
                comp_norm = np.linalg.norm(comp_avg)
                if comp_norm > 0.9:
                    # Reasonably tight cluster — split it out
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
    labels = labels.copy()
    vertex_curvatures = _estimate_vertex_curvatures(cleaned)
    face_curvatures = _vertex_to_face_curvatures(vertex_curvatures, cleaned.faces)

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

        # Build feature vectors: (nx, ny, nz, curvature)
        region_normals = cleaned.normals[mask]
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


def _merge_adjacent_cylinder_fragments(
    labels: np.ndarray,
    cleaned: CleanedMesh,
    adjacency: dict[int, set[int]],
) -> np.ndarray:
    """Merge small adjacent regions that together form cylindrical holes.

    After segmentation, hole walls often get split into many tiny regions
    (2-10 faces each). This function detects spatially clustered small regions
    with non-planar normals and merges them into unified hole regions.
    """
    from sklearn.cluster import DBSCAN

    labels = labels.copy()
    normals = cleaned.normals

    # Find small regions with non-axis-aligned normals (potential hole fragments)
    region_info = {}
    for label in np.unique(labels):
        face_idx = np.where(labels == label)[0]
        if len(face_idx) > 30:
            continue  # skip large regions
        avg_n = normals[face_idx].mean(axis=0)
        avg_norm = np.linalg.norm(avg_n)
        if avg_norm < 0.01:
            continue
        avg_n /= avg_norm
        # Non-planar: average normal not closely aligned with any axis
        if np.max(np.abs(avg_n)) < 0.85:
            center = np.mean([
                cleaned.vertices[cleaned.faces[fi]].mean(axis=0) for fi in face_idx
            ], axis=0)
            region_info[label] = {
                "faces": face_idx,
                "center": center,
                "normal": avg_n,
                "count": len(face_idx),
            }

    if len(region_info) < 2:
        return labels

    # Cluster these small regions spatially
    region_labels = list(region_info.keys())
    centers = np.array([region_info[l]["center"] for l in region_labels])

    part_size = float((cleaned.bbox[1] - cleaned.bbox[0]).min())  # smallest dim
    eps = max(part_size * 0.4, 8.0)

    clustering = DBSCAN(eps=eps, min_samples=2).fit(centers)

    next_label = int(labels.max()) + 1
    for cl in set(clustering.labels_):
        if cl == -1:
            continue
        cl_mask = clustering.labels_ == cl
        group_labels = [region_labels[i] for i in range(len(region_labels)) if cl_mask[i]]

        # Merge all faces in this spatial cluster into one region
        if len(group_labels) < 2:
            continue
        total_faces = sum(region_info[l]["count"] for l in group_labels)
        if total_faces < 4:
            continue

        for l in group_labels:
            for fi in region_info[l]["faces"]:
                labels[fi] = next_label
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
    vertex_curvatures = _estimate_vertex_curvatures(cleaned)
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
