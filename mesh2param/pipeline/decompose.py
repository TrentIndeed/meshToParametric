"""Stage 1.5: Feature-Aware Body Decomposition.

Decomposes a cleaned mesh into prismatic sub-bodies using cross-section
sweep analysis. Each body has a roughly constant cross-section along its
extrude direction.

Algorithm (Cross-Section Sweep):
1. For each principal axis, sweep a slicing plane and compute area at each position
2. Find transitions where the area changes significantly (>50% jump)
3. The axis with the clearest single transition defines the split plane
4. Faces are assigned to bodies based on which side of the split their centroid falls on
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from mesh2param.config import DecomposeConfig
from mesh2param.models.body import BodyDecomposition, PrismaticBody
from mesh2param.models.mesh import CleanedMesh

logger = logging.getLogger(__name__)


def decompose_bodies(
    cleaned: CleanedMesh,
    config: DecomposeConfig | None = None,
) -> BodyDecomposition:
    """Decompose a mesh into prismatic sub-bodies using cross-section analysis."""
    if config is None:
        config = DecomposeConfig()

    if not config.enabled:
        return _single_body(cleaned)

    import trimesh

    num_faces = len(cleaned.faces)
    mesh = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces, process=False)
    bbox = cleaned.bbox

    logger.debug("Decomposing %d faces via cross-section sweep", num_faces)

    # Step 1: Sweep each axis and find transitions
    best_axis = None
    best_split_pos = None
    best_score = 0  # higher = clearer split

    for axis in range(3):
        lo, hi = float(bbox[0][axis]), float(bbox[1][axis])
        extent = hi - lo
        if extent < 1.0:
            continue

        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0

        # Sample cross-section areas
        n_samples = 30
        positions = np.linspace(lo + extent * 0.03, hi - extent * 0.03, n_samples)
        areas = []
        for pos in positions:
            origin = [0.0, 0.0, 0.0]
            origin[axis] = pos
            try:
                section = mesh.section(plane_origin=origin, plane_normal=normal)
                if section:
                    s2d, _ = section.to_planar()
                    area = sum(p.area for p in s2d.polygons_closed)
                    areas.append(area)
                else:
                    areas.append(0)
            except Exception:
                areas.append(0)

        if not areas or max(areas) < 1:
            continue

        areas_arr = np.array(areas)

        # Find the single biggest area transition
        # Look for a jump where the area changes by >50%
        ratios = np.zeros(len(areas_arr) - 1)
        for i in range(len(areas_arr) - 1):
            a1, a2 = areas_arr[i], areas_arr[i + 1]
            if a1 > 0 and a2 > 0:
                ratios[i] = max(a2 / a1, a1 / a2)

        if len(ratios) == 0 or ratios.max() < 1.5:
            continue  # no significant transition on this axis

        best_idx = int(np.argmax(ratios))
        score = float(ratios[best_idx])

        # Verify it's a clean split: check that each side has a plateau
        left_areas = areas_arr[:best_idx + 1]
        right_areas = areas_arr[best_idx + 1:]

        left_stable = len(left_areas) >= 3 and (left_areas.std() / max(left_areas.mean(), 1)) < 0.15
        right_stable = len(right_areas) >= 3 and (right_areas.std() / max(right_areas.mean(), 1)) < 0.15

        if not (left_stable or right_stable):
            continue

        # Fine-tune split position: binary search between the two samples
        # to find the exact Z where the area jumps
        z_lo = float(positions[best_idx])
        z_hi = float(positions[best_idx + 1])
        a_lo = float(areas_arr[best_idx])
        a_hi = float(areas_arr[best_idx + 1])
        threshold = (a_lo + a_hi) / 2
        for _ in range(10):  # 10 iterations of binary search
            z_mid = (z_lo + z_hi) / 2
            origin = [0.0, 0.0, 0.0]
            origin[axis] = z_mid
            try:
                sec = mesh.section(plane_origin=origin, plane_normal=normal)
                if sec:
                    s2d, _ = sec.to_planar()
                    a_mid = sum(p.area for p in s2d.polygons_closed)
                else:
                    a_mid = 0
            except Exception:
                a_mid = 0
            if a_mid > threshold:
                z_hi = z_mid
            else:
                z_lo = z_mid
        # Place the split at the LOW side (just before the jump)
        split_pos = float(z_lo)

        logger.debug(
            "Axis %s: transition at %.1f (ratio %.1f), left_stable=%s right_stable=%s",
            'XYZ'[axis], split_pos, score, left_stable, right_stable,
        )

        if score > best_score:
            best_score = score
            best_axis = axis
            best_split_pos = split_pos

    if best_axis is None or best_score < 5.0:
        logger.info("No clear body split found (best_score=%.1f, need >5.0), single body", best_score)
        return _single_body(cleaned)

    logger.info(
        "Split on %s axis at %.1f (score %.1f)",
        'XYZ'[best_axis], best_split_pos, best_score,
    )

    # Step 2: Assign faces to bodies using cap-face flood fill.
    # Find faces whose normals are parallel to the split axis (cap faces).
    # Cap faces clearly belong to one side. Then flood-fill from each
    # cap to assign wall faces to the nearest cap's body.
    normals = cleaned.normals
    abs_normal_along_axis = np.abs(normals[:, best_axis])
    is_cap = abs_normal_along_axis > 0.85  # face normal is parallel to split axis

    face_centroids = cleaned.vertices[cleaned.faces].mean(axis=1)

    # Seed assignment: cap faces are clearly on one side
    assignment = np.full(num_faces, -1, dtype=np.int32)  # -1 = unassigned
    for fi in range(num_faces):
        if is_cap[fi]:
            assignment[fi] = 0 if face_centroids[fi, best_axis] < best_split_pos else 1

    # Build face adjacency for flood fill
    edge_to_faces = defaultdict(list)
    for fi in range(num_faces):
        tri = cleaned.faces[fi]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            key = (min(a, b), max(a, b))
            edge_to_faces[key].append(fi)

    adjacency = defaultdict(set)
    for faces_list in edge_to_faces.values():
        for i in range(len(faces_list)):
            for j in range(i + 1, len(faces_list)):
                adjacency[faces_list[i]].add(faces_list[j])
                adjacency[faces_list[j]].add(faces_list[i])

    # Flood fill from assigned cap faces to unassigned wall faces
    from collections import deque
    queue = deque()
    for fi in range(num_faces):
        if assignment[fi] >= 0:
            queue.append(fi)

    while queue:
        fi = queue.popleft()
        body_id = assignment[fi]
        for neighbor in adjacency.get(fi, []):
            if assignment[neighbor] < 0:
                assignment[neighbor] = body_id
                queue.append(neighbor)

    # Any still unassigned faces go to body 0
    assignment[assignment < 0] = 0

    body_a_mask = assignment == 0
    body_b_mask = assignment == 1

    body_a_faces = np.where(body_a_mask)[0].astype(np.int64)
    body_b_faces = np.where(body_b_mask)[0].astype(np.int64)

    if len(body_a_faces) < config.min_body_faces or len(body_b_faces) < config.min_body_faces:
        logger.info("One body too small (%d/%d faces), using single body",
                     len(body_a_faces), len(body_b_faces))
        return _single_body(cleaned)

    # Step 3: Create PrismaticBody objects
    bodies = []
    for i, face_indices in enumerate([body_a_faces, body_b_faces]):
        direction = np.zeros(3)
        direction[best_axis] = 1.0

        body_verts = cleaned.vertices[cleaned.faces[face_indices].flatten()]
        depth = float(body_verts[:, best_axis].max() - body_verts[:, best_axis].min())

        bodies.append(PrismaticBody(
            body_id=i,
            face_indices=face_indices,
            extrude_direction=direction,
            depth=depth,
            is_base=(i == 0),
            operation="BASE" if i == 0 else "UNION",
        ))

    # Order: larger body first (base)
    bodies.sort(key=lambda b: -len(b.face_indices))
    for i, b in enumerate(bodies):
        b.body_id = i
        b.is_base = (i == 0)
        b.operation = "BASE" if i == 0 else "UNION"

    logger.info(
        "Decomposed into %d bodies: %s",
        len(bodies),
        [(b.body_id, len(b.face_indices), f"depth={b.depth:.1f}") for b in bodies],
    )

    return BodyDecomposition(
        bodies=bodies,
        is_multi_body=True,
        metadata={"split_axis": 'XYZ'[best_axis], "split_pos": best_split_pos, "score": best_score},
    )


def extract_sub_mesh(cleaned: CleanedMesh, face_indices: np.ndarray) -> CleanedMesh:
    """Create a CleanedMesh containing only the specified faces."""
    sub_faces_orig = cleaned.faces[face_indices]
    sub_normals = cleaned.normals[face_indices]

    # Remap vertices to contiguous indices
    unique_verts, inverse = np.unique(sub_faces_orig.ravel(), return_inverse=True)
    sub_vertices = cleaned.vertices[unique_verts]
    sub_faces = inverse.reshape(-1, 3).astype(np.int64)

    sub_bbox = np.array([sub_vertices.min(axis=0), sub_vertices.max(axis=0)])

    return CleanedMesh(
        vertices=sub_vertices,
        faces=sub_faces,
        normals=sub_normals,
        is_watertight=False,
        bbox=sub_bbox,
        metadata={**getattr(cleaned, 'metadata', {}), "parent_face_indices": face_indices},
    )


def _single_body(cleaned: CleanedMesh) -> BodyDecomposition:
    """Return a single-body decomposition (backward compatible)."""
    all_faces = np.arange(len(cleaned.faces), dtype=np.int64)

    abs_normals = np.abs(cleaned.normals)
    dominant_axis = np.argmax(abs_normals, axis=1)
    axis_counts = [int(np.sum(dominant_axis == ax)) for ax in range(3)]
    best_axis = int(np.argmax(axis_counts))

    direction = np.zeros(3)
    direction[best_axis] = 1.0
    bbox_size = cleaned.bbox[1] - cleaned.bbox[0]
    depth = float(bbox_size[best_axis])

    return BodyDecomposition(
        bodies=[PrismaticBody(
            body_id=0,
            face_indices=all_faces,
            extrude_direction=direction,
            depth=depth,
            is_base=True,
            operation="BASE",
        )],
        is_multi_body=False,
    )
