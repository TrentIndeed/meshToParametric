"""Stage 1.5: Feature-Aware Body Decomposition.

Decomposes a cleaned mesh into prismatic sub-bodies using cross-section
sweep analysis followed by recursive virtual planar cuts.

Algorithm:
1. For each principal axis, sweep a slicing plane and compute cross-section area
2. Find transitions where the area changes >5x (indicates a body boundary)
3. Binary-search to find the exact split position
4. Use trimesh.slice_mesh_plane to physically cut the mesh into two pieces
5. Recursively attempt to split each piece further
6. Each piece is a standalone sub-mesh for per-body pipeline processing
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from mesh2param.config import DecomposeConfig
from mesh2param.models.body import BodyDecomposition, PrismaticBody
from mesh2param.models.mesh import CleanedMesh

logger = logging.getLogger(__name__)


def decompose_bodies(
    cleaned: CleanedMesh,
    config: DecomposeConfig | None = None,
) -> BodyDecomposition:
    """Decompose a mesh into prismatic sub-bodies."""
    if config is None:
        config = DecomposeConfig()

    if not config.enabled:
        return _single_body(cleaned)

    mesh = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces, process=False)

    # Recursively split the mesh
    pieces = _recursive_split(mesh, cleaned, config, depth=0, max_depth=3)

    if len(pieces) < 2:
        logger.info("No multi-body split found, single body")
        return _single_body(cleaned)

    # Clean each piece and build PrismaticBody objects
    from mesh2param.pipeline.ingest import clean_trimesh_piece
    from mesh2param.config import IngestConfig

    bodies = []
    for pi, (piece_trimesh, face_mask) in enumerate(pieces):
        piece_cleaned = clean_trimesh_piece(piece_trimesh, IngestConfig(remesh=False))
        direction, depth = _detect_extrude_dir(piece_cleaned)
        bodies.append(PrismaticBody(
            body_id=pi,
            face_indices=np.where(face_mask)[0].astype(np.int64),
            extrude_direction=direction,
            depth=depth,
            is_base=False,
            operation="UNION",
            metadata={"piece_mesh": piece_cleaned},
        ))

    # Order: largest piece first (base)
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
        metadata={"num_splits": len(pieces) - 1},
    )


def _recursive_split(mesh, cleaned, config, depth, max_depth):
    """Try to split a mesh, then recursively split each piece.

    Returns list of (trimesh_piece, face_mask_on_original) tuples.
    face_mask is a boolean array over cleaned.faces.
    """
    if depth >= max_depth:
        return [(mesh, np.ones(len(cleaned.faces), dtype=bool))]

    result = _find_best_split(mesh, cleaned)
    if result is None:
        return [(mesh, np.ones(len(cleaned.faces), dtype=bool))]

    best_axis, best_split_pos, best_score = result

    if best_score < 5.0:
        return [(mesh, np.ones(len(cleaned.faces), dtype=bool))]

    # Perform the cut
    plane_normal = [0.0, 0.0, 0.0]
    plane_normal[best_axis] = 1.0
    plane_origin = [0.0, 0.0, 0.0]
    plane_origin[best_axis] = best_split_pos

    try:
        piece_above = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal, plane_origin, cached_dots=None,
        )
        piece_below = trimesh.intersections.slice_mesh_plane(
            mesh, [-n for n in plane_normal], plane_origin, cached_dots=None,
        )
    except Exception:
        logger.warning("Planar cut failed at depth %d", depth)
        return [(mesh, np.ones(len(cleaned.faces), dtype=bool))]

    if piece_above is None or piece_below is None:
        return [(mesh, np.ones(len(cleaned.faces), dtype=bool))]

    if len(piece_above.faces) < config.min_body_faces or len(piece_below.faces) < config.min_body_faces:
        return [(mesh, np.ones(len(cleaned.faces), dtype=bool))]

    logger.info("Split at depth %d on %s axis at %.1f (score %.1f): %d + %d faces",
                depth, 'XYZ'[best_axis], best_split_pos, best_score,
                len(piece_above.faces), len(piece_below.faces))

    # Map to original mesh face indices
    orig_centroids = cleaned.vertices[cleaned.faces].mean(axis=1)
    above_mask = orig_centroids[:, best_axis] >= best_split_pos
    below_mask = ~above_mask

    # Recursively try to split each piece further
    print(f"[DECOMP] Trying recursive split on above piece ({len(piece_above.faces)} faces)", flush=True)
    above_results = _try_recursive_piece(piece_above, cleaned, above_mask, config, depth + 1, max_depth)
    print(f"[DECOMP] Trying recursive split on below piece ({len(piece_below.faces)} faces)", flush=True)
    below_results = _try_recursive_piece(piece_below, cleaned, below_mask, config, depth + 1, max_depth)

    return above_results + below_results


def _try_recursive_piece(piece_trimesh, cleaned, parent_mask, config, depth, max_depth):
    """Try to recursively split a piece. Returns list of (trimesh, mask) tuples."""
    if depth >= max_depth or len(piece_trimesh.faces) < config.min_body_faces * 2:
        return [(piece_trimesh, parent_mask)]

    piece_bbox = np.array([piece_trimesh.vertices.min(axis=0), piece_trimesh.vertices.max(axis=0)])
    piece_size = piece_bbox[1] - piece_bbox[0]
    # Don't split pieces that are already thin in any dimension —
    # they're likely a single plate/flange, not a multi-body piece
    min_dim = float(piece_size.min())
    if min_dim < 15.0:
        return [(piece_trimesh, parent_mask)]

    # Strategy 1: cross-section area sweep
    result = _find_best_split(piece_trimesh, None, bbox_override=piece_bbox)
    if result is not None and result[2] >= 5.0:
        logger.info("Sub-split (area sweep) depth=%d: axis=%s pos=%.1f score=%.1f",
                     depth, 'XYZ'[result[0]], result[1], result[2])
    elif depth <= 1:
        # Strategy 2: face-normal-based spatial split (only at first recursion
        # level to avoid over-splitting)
        result = _find_normal_based_split(piece_trimesh, config)
        if result is not None:
            logger.info("Sub-split (normal-based) depth=%d: axis=%s pos=%.1f score=%.1f",
                         depth, 'XYZ'[result[0]], result[1], result[2])

    if result is None or result[2] < 3.0:
        return [(piece_trimesh, parent_mask)]

    best_axis, best_split_pos, best_score = result

    plane_normal = [0.0, 0.0, 0.0]
    plane_normal[best_axis] = 1.0
    plane_origin = [0.0, 0.0, 0.0]
    plane_origin[best_axis] = best_split_pos

    try:
        sub_above = trimesh.intersections.slice_mesh_plane(
            piece_trimesh, plane_normal, plane_origin, cached_dots=None,
        )
        sub_below = trimesh.intersections.slice_mesh_plane(
            piece_trimesh, [-n for n in plane_normal], plane_origin, cached_dots=None,
        )
    except Exception:
        return [(piece_trimesh, parent_mask)]

    if sub_above is None or sub_below is None:
        return [(piece_trimesh, parent_mask)]

    if len(sub_above.faces) < config.min_body_faces or len(sub_below.faces) < config.min_body_faces:
        return [(piece_trimesh, parent_mask)]

    logger.info("Sub-split at depth %d on %s axis at %.1f (score %.1f): %d + %d faces",
                depth, 'XYZ'[best_axis], best_split_pos, best_score,
                len(sub_above.faces), len(sub_below.faces))

    # Split the parent mask based on original mesh centroids
    orig_centroids = cleaned.vertices[cleaned.faces].mean(axis=1)
    above_mask = parent_mask & (orig_centroids[:, best_axis] >= best_split_pos)
    below_mask = parent_mask & (orig_centroids[:, best_axis] < best_split_pos)

    # Try further splits on each sub-piece
    above_results = _try_recursive_piece(sub_above, cleaned, above_mask, config, depth + 1, max_depth)
    below_results = _try_recursive_piece(sub_below, cleaned, below_mask, config, depth + 1, max_depth)

    return above_results + below_results


def _find_best_split(mesh, cleaned=None, bbox_override=None):
    """Find the best split plane for a mesh via cross-section area sweep.

    Returns (axis, split_pos, score) or None.
    """
    if bbox_override is not None:
        bbox = bbox_override
    elif cleaned is not None:
        bbox = cleaned.bbox
    else:
        bbox = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])

    best_axis = None
    best_split_pos = None
    best_score = 0

    for axis in range(3):
        lo, hi = float(bbox[0][axis]), float(bbox[1][axis])
        extent = hi - lo
        if extent < 1.0:
            continue

        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0

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
        ratios = np.zeros(len(areas_arr) - 1)

        # Check for a wide zero-gap: a long run of zero-area samples flanked
        # by nonzero regions on both sides. This detects spatially separated
        # bodies (e.g. two flanges with a gap between them).
        zero_mask = areas_arr < 0.01
        nonzero_mask = ~zero_mask
        if nonzero_mask.any() and zero_mask.sum() >= 3:
            # Find the longest zero run
            zero_runs = []
            start = None
            for i in range(len(zero_mask)):
                if zero_mask[i]:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        zero_runs.append((start, i))
                        start = None
            if start is not None:
                zero_runs.append((start, len(zero_mask)))

            for zs, ze in zero_runs:
                gap_len = ze - zs
                has_left = nonzero_mask[:zs].any()
                has_right = nonzero_mask[ze:].any() if ze < len(nonzero_mask) else False
                gap_frac = gap_len / len(areas_arr)
                # Both sides must have substantial nonzero regions (>=3 samples each)
                left_nonzero = nonzero_mask[:zs].sum()
                right_nonzero = nonzero_mask[ze:].sum() if ze < len(nonzero_mask) else 0
                if not (has_left and has_right and gap_frac > 0.3 and left_nonzero >= 3 and right_nonzero >= 3):
                    continue
                # Verify the mesh is actually disconnected in the gap region.
                # A window/cutout creates a zero cross-section but the mesh
                # still has faces spanning the gap (the frame). Check if any
                # mesh faces have centroids inside the gap region.
                gap_lo = float(positions[zs])
                gap_hi = float(positions[ze - 1]) if ze > 0 else float(positions[zs])
                face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
                faces_in_gap = ((face_centroids[:, axis] > gap_lo) &
                                (face_centroids[:, axis] < gap_hi))
                if faces_in_gap.sum() > 5:
                    # Faces exist in the gap — it's a window, not a real split
                    continue
                if True:
                    # Split at the center of the zero gap
                    gap_mid = (zs + ze) // 2
                    split_pos_here = float(positions[gap_mid])
                    gap_score = gap_frac * 50  # strong signal
                    if gap_score > best_score:
                        best_score = gap_score
                        best_axis = axis
                        best_split_pos = split_pos_here
                    continue  # skip normal ratio analysis for this axis

        for i in range(len(areas_arr) - 1):
            a1, a2 = areas_arr[i], areas_arr[i + 1]
            if a1 > 0 and a2 > 0:
                ratios[i] = max(a2 / a1, a1 / a2)

        if len(ratios) == 0 or ratios.max() < 1.5:
            continue

        best_idx = int(np.argmax(ratios))
        score = float(ratios[best_idx])

        # Verify plateaus on each side
        left_areas = areas_arr[:best_idx + 1]
        right_areas = areas_arr[best_idx + 1:]
        left_stable = len(left_areas) >= 3 and (left_areas.std() / max(left_areas.mean(), 1)) < 0.15
        right_stable = len(right_areas) >= 3 and (right_areas.std() / max(right_areas.mean(), 1)) < 0.15
        if not (left_stable or right_stable):
            continue

        # Binary search for exact split position
        z_lo = float(positions[best_idx])
        z_hi = float(positions[best_idx + 1])
        a_lo = float(areas_arr[best_idx])
        a_hi = float(areas_arr[best_idx + 1])
        threshold = (a_lo + a_hi) / 2
        for _ in range(10):
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
        split_pos = float(z_lo)

        if score > best_score:
            best_score = score
            best_axis = axis
            best_split_pos = split_pos

    if best_axis is None:
        return None
    return (best_axis, best_split_pos, best_score)


def _find_normal_based_split(mesh, config):
    """Detect perpendicular extrusions by finding faces with the same dominant
    normal that are spatially separated (e.g. two side flanges on a base plate).

    For each axis, find faces whose normals are primarily along that axis.
    If those faces cluster into two spatially separated groups along the same
    axis, there's a gap in between where we can split.

    Returns (axis, split_pos, score) or None.
    """
    normals = mesh.face_normals
    abs_normals = np.abs(normals)
    dominant = np.argmax(abs_normals, axis=1)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    areas = mesh.area_faces
    bbox = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
    bbox_size = bbox[1] - bbox[0]
    total_area = areas.sum()

    best = None
    best_score = 0

    for ax in range(3):
        # Faces with dominant normal along this axis
        mask = dominant == ax
        ax_area = areas[mask].sum()

        # Need significant area (>20% of total) to be considered a body direction
        if ax_area < total_area * 0.20:
            continue

        # Check if these faces are spatially separated along this axis
        ax_centroids = centroids[mask, ax]
        ax_extent = bbox_size[ax]
        if ax_extent < 2.0:
            continue

        # Build a histogram of face positions along this axis
        n_bins = 20
        hist, bin_edges = np.histogram(ax_centroids, bins=n_bins,
                                        weights=areas[mask])

        # Look for a gap: a sequence of empty or near-empty bins between
        # two populated regions
        threshold = hist.max() * 0.05  # bins with <5% of peak are "empty"
        populated = hist > threshold

        # Find the largest gap (consecutive empty bins)
        gap_start = None
        best_gap = None
        best_gap_len = 0
        for i in range(n_bins):
            if not populated[i]:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_len = i - gap_start
                    # Check that there are populated bins on BOTH sides
                    has_left = any(populated[:gap_start])
                    has_right = any(populated[i:])
                    if has_left and has_right and gap_len > best_gap_len:
                        best_gap = (gap_start, i)
                        best_gap_len = gap_len
                    gap_start = None

        if best_gap is None or best_gap_len < 3:
            continue

        # Split position: center of the gap
        gap_center = (bin_edges[best_gap[0]] + bin_edges[best_gap[1]]) / 2
        gap_frac = best_gap_len / n_bins
        score = gap_frac * 20  # scale to be comparable to area-sweep scores

        logger.debug("Normal-based split candidate: axis=%s gap=%.0f%% at %.1f, score=%.1f",
                     'XYZ'[ax], gap_frac * 100, gap_center, score)

        if score > best_score:
            best_score = score
            best = (ax, float(gap_center), score)

    return best


def _detect_extrude_dir(piece_cleaned):
    """Find the dominant extrude axis from a body's face normals."""
    normals = piece_cleaned.normals
    abs_normals = np.abs(normals)
    dominant_axis = np.argmax(abs_normals, axis=1)
    axis_counts = [int(np.sum(dominant_axis == ax)) for ax in range(3)]
    ax = int(np.argmax(axis_counts))
    d = np.zeros(3)
    d[ax] = 1.0
    bbox_size = piece_cleaned.bbox[1] - piece_cleaned.bbox[0]
    depth = float(bbox_size[ax])
    return d, depth


def extract_sub_mesh(cleaned: CleanedMesh, face_indices: np.ndarray, body: PrismaticBody | None = None) -> CleanedMesh:
    """Get the sub-mesh for a body.

    If the body has a pre-cut piece_mesh (cleaned after planar cut), use that.
    Otherwise fall back to extracting faces from the original mesh.
    """
    if body is not None and "piece_mesh" in body.metadata:
        return body.metadata["piece_mesh"]

    # Fallback: extract faces from original mesh
    sub_faces_orig = cleaned.faces[face_indices]
    sub_normals = cleaned.normals[face_indices]
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
    )


def _single_body(cleaned: CleanedMesh) -> BodyDecomposition:
    """Return a single-body decomposition (backward compatible)."""
    all_faces = np.arange(len(cleaned.faces), dtype=np.int64)
    direction, depth = _detect_extrude_dir(cleaned)

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
