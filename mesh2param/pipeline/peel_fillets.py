"""Stage 1.75: Virtual fillet removal.

A rolling-ball fillet between two plane faces creates a curved strip
tangent to both. When the segmenter sees that strip it fits a cylinder
primitive there, which confuses every downstream step: the topology
classifier no longer sees a sharp edge between the two planes, the
feature recognizer fragments the planes into pieces, and the CAD
reconstruction ends up trying to re-fillet rounded material which
either fails or creates phantom geometry.

This module identifies those tangent strips directly from the triangle
mesh (cluster curved interior edges into strips, require cylindrical
normal distribution, require axis parallel to a principal direction)
and virtually "peels" them off by projecting every strip vertex onto
the intersection line of the two adjacent planes. The resulting mesh
has sharp edges where fillets used to be, which is what the segmenter
and topology stages want to see. The peeled fillet records are
returned alongside the peeled mesh so that the export stage can
re-inject them as FILLET features at the correct edges.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from mesh2param.models.mesh import CleanedMesh

logger = logging.getLogger(__name__)


def peel_fillets(
    cleaned: CleanedMesh,
    *,
    dihedral_min_deg: float = 162.0,
    dihedral_max_deg: float = 179.5,
    max_radius_ratio: float = 0.06,
    min_strip_faces: int = 5,
    normal_plane_ratio: float = 0.15,
    axis_alignment: float = 0.96,
) -> tuple[CleanedMesh, list[dict]]:
    """Virtually remove rolling-ball tangent strips from a mesh.

    Args:
        cleaned: Input mesh.
        dihedral_min_deg/dihedral_max_deg: range of dihedral angles that
            count as "curved fillet interior" edges. Above the upper
            bound the edge is flat (same region); below the lower bound
            it's a chamfer or sharper bevel, not a rolling ball. Default
            162° accepts up to 18° turns per edge — needed so coarser
            meshes where a single fillet arc is sampled by fewer triangles
            still form continuous strips. Still well above the ~135°
            dihedral at a chamfer's crossing edges, so chamfers aren't
            misclassified as fillets.
        max_radius_ratio: reject strips whose estimated radius is more
            than this fraction of the mesh bbox diagonal. Default 0.06
            (~14mm on a 230mm part) filters out the spurious large-radius
            "fillets" that the dihedral detector sometimes finds on
            through-slot interior arcs or pocket edges.
        (max_radius_ratio described above.)
        min_strip_faces: a strip must have at least this many faces.
        normal_plane_ratio: reject strips whose normal covariance
            smallest/largest eigenvalue ratio exceeds this — i.e.
            strips whose normals don't lie in a plane (spherical
            corner caps, not cylinders).
        axis_alignment: require the recovered cylinder axis to have
            max |component| at least this large — i.e. axis-aligned,
            not a diagonal 3D blend.

    Returns:
        (peeled_mesh, fillets) where peeled_mesh is a new CleanedMesh
        whose fillet-strip vertices have been projected onto the sharp
        intersection of the two adjacent planes, and fillets is a list
        of dicts with keys radius, axis_dir, axis_point, length,
        num_faces describing each peeled fillet.
    """
    import trimesh

    vertices = np.asarray(cleaned.vertices, dtype=float).copy()
    faces = np.asarray(cleaned.faces, dtype=np.int64)
    normals = np.asarray(cleaned.normals, dtype=float)
    n_faces = int(faces.shape[0])
    bbox_diag = cleaned.bbox_diagonal
    max_radius = bbox_diag * max_radius_ratio

    # --- 1. Build edge-to-faces map (triangle mesh adjacency). ---
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(n_faces):
        tri = faces[fi]
        for a, b in ((int(tri[0]), int(tri[1])),
                     (int(tri[1]), int(tri[2])),
                     (int(tri[0]), int(tri[2]))):
            edge_to_faces[(min(a, b), max(a, b))].append(fi)

    # --- 2. Strip graph: two faces are connected if they share an
    # edge whose dihedral is in the curved range, i.e. neither flat
    # (part of the same planar region) nor sharp (a chamfer or hard
    # edge). Using only curved edges gives clean strip boundaries
    # where the curvature actually stops. ---
    is_curved_face = np.zeros(n_faces, dtype=bool)
    face_adj: list[set[int]] = [set() for _ in range(n_faces)]
    for flist in edge_to_faces.values():
        if len(flist) != 2:
            continue
        f0, f1 = flist
        n1, n2 = normals[f0], normals[f1]
        dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        dihed = float(np.degrees(np.pi - np.arccos(dot)))
        if not (dihedral_min_deg < dihed < dihedral_max_deg):
            continue
        is_curved_face[f0] = True
        is_curved_face[f1] = True
        face_adj[f0].add(f1)
        face_adj[f1].add(f0)

    # --- 3. Connected components on the curved-face sub-graph. ---
    labels = np.full(n_faces, -1, dtype=np.int64)
    next_label = 0
    for fi in range(n_faces):
        if labels[fi] != -1 or not is_curved_face[fi]:
            continue
        stack = [fi]
        while stack:
            cur = stack.pop()
            if labels[cur] != -1:
                continue
            labels[cur] = next_label
            for nbr in face_adj[cur]:
                if labels[nbr] == -1:
                    stack.append(nbr)
        next_label += 1

    # --- 3.5. Merge fragmented strips ---
    # A single real fillet arc often breaks into 2-4 connected components
    # because a few triangles in the arc have dihedrals that cross the
    # 162°/179.5° thresholds (due to triangulation noise). Individually
    # those fragments are too short to pass length >= 2R; merged, they
    # form the true strip. Merge criterion: two components are the same
    # logical strip if they share a dominant PCA axis direction AND
    # their centroids lie roughly along each other's axis line.
    def _strip_axis(strip_f):
        verts = vertices[faces[strip_f].flatten()]
        centroid = verts.mean(axis=0)
        rel = verts - centroid
        try:
            _, _, vt = np.linalg.svd(rel, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, centroid
        return vt[0], centroid

    strip_axes: list[tuple[np.ndarray | None, np.ndarray]] = []
    strip_face_lists: list[np.ndarray] = []
    for sid in range(next_label):
        sf = np.where(labels == sid)[0]
        strip_face_lists.append(sf)
        if len(sf) < 3:
            strip_axes.append((None, np.zeros(3)))
            continue
        strip_axes.append(_strip_axis(sf))

    # Union-find to merge compatible strips.
    parent = list(range(next_label))
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    bbox_diag_for_merge = bbox_diag
    for i in range(next_label):
        ai, ci = strip_axes[i]
        if ai is None:
            continue
        for j in range(i + 1, next_label):
            aj, cj = strip_axes[j]
            if aj is None:
                continue
            # Axes must be parallel (or anti-parallel).
            if abs(float(np.dot(ai, aj))) < 0.95:
                continue
            # Centroid j must lie close to the line through ci along ai
            # (both components share the same underlying sharp edge).
            offset = cj - ci
            along = float(np.dot(offset, ai))
            perp_vec = offset - along * ai
            perp_dist = float(np.linalg.norm(perp_vec))
            # Allow merge if perp distance is small relative to the part
            # size (heuristic: 2% of bbox diag, ~4.6mm on a 230mm part).
            if perp_dist > bbox_diag_for_merge * 0.02:
                continue
            _union(i, j)

    merged_groups: dict[int, list[int]] = defaultdict(list)
    for sid in range(next_label):
        merged_groups[_find(sid)].append(sid)

    # Build final strip face lists by concatenating merged groups.
    final_strips: list[np.ndarray] = []
    for _root, members in merged_groups.items():
        combined = np.concatenate([strip_face_lists[m] for m in members])
        final_strips.append(combined)

    # --- 4. Per-strip: validate cylinder hypothesis, then peel. ---
    fillets_out: list[dict] = []
    # vertex_id -> list of candidate new positions (usually 1, sometimes
    # more if a vertex straddles two strips — we average).
    vertex_new_pos: dict[int, list[np.ndarray]] = defaultdict(list)

    for strip_faces in final_strips:
        if len(strip_faces) < min_strip_faces:
            continue

        strip_normals = normals[strip_faces]
        n_norms = np.linalg.norm(strip_normals, axis=1)
        good = n_norms > 1e-8
        if good.sum() < min_strip_faces:
            continue
        N_unit = strip_normals[good] / n_norms[good, None]

        # Cylindrical validity: normals must lie close to a plane
        # (smallest eigenvalue of Nᵀ N close to zero).
        M = N_unit.T @ N_unit
        try:
            eig_vals, eig_vecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(np.abs(eig_vals))
        lam = np.abs(eig_vals[order])
        if lam[2] < 1e-10:
            continue
        if lam[0] / lam[2] > normal_plane_ratio:
            continue
        axis_dir_est = eig_vecs[:, order[0]]
        axis_dir_est = axis_dir_est / (np.linalg.norm(axis_dir_est) + 1e-15)
        if float(np.max(np.abs(axis_dir_est))) < axis_alignment:
            continue

        # Boundary faces: one-ring neighbours of the strip that are not
        # themselves in the strip. These sit on the two adjacent planes.
        strip_set = {int(f) for f in strip_faces}
        boundary_faces: set[int] = set()
        for fid in strip_faces:
            tri = faces[fid]
            for a, b in ((int(tri[0]), int(tri[1])),
                         (int(tri[1]), int(tri[2])),
                         (int(tri[0]), int(tri[2]))):
                key = (min(a, b), max(a, b))
                for nbr in edge_to_faces[key]:
                    nbr_i = int(nbr)
                    if nbr_i != int(fid) and nbr_i not in strip_set:
                        boundary_faces.add(nbr_i)
        if len(boundary_faces) < 4:
            continue

        boundary_ids = np.array(sorted(boundary_faces))
        bn = normals[boundary_ids]
        bn_norms = np.linalg.norm(bn, axis=1)
        bn_good = bn_norms > 1e-8
        if bn_good.sum() < 4:
            continue
        bn_unit = bn[bn_good] / bn_norms[bn_good, None]
        bn_ids_good = boundary_ids[bn_good]

        # Two plane clusters: K=2 split on the unit normals themselves.
        # A plane-plane fillet's boundary faces fall into exactly two
        # normal directions (one per adjacent plane). Splitting by
        # *normal similarity* cleanly separates them even when the
        # boundary set contains extra faces from a neighbouring feature
        # at an unusual angle. The earlier angle-around-axis median
        # split was fooled by those stray neighbours — a long plate
        # edge with one tab corner at one end would get that corner's
        # face(s) attached to the wrong side, polluting the mean normal
        # and making the planarity check reject the strip.
        #
        # Implementation: two Lloyd-style iterations on unit normals.
        # Seed with the two most-opposite normals (smallest pairwise
        # dot product), then assign each normal to whichever seed it's
        # closer to (higher dot), recompute means, repeat.
        dots = bn_unit @ bn_unit.T
        i_seed, j_seed = np.unravel_index(np.argmin(dots), dots.shape)
        mu_a = bn_unit[i_seed].copy()
        mu_b = bn_unit[j_seed].copy()
        for _ in range(5):
            side_a = (bn_unit @ mu_a) >= (bn_unit @ mu_b)
            side_b = ~side_a
            if side_a.sum() < 2 or side_b.sum() < 2:
                break
            mu_a_new = bn_unit[side_a].mean(axis=0)
            mu_b_new = bn_unit[side_b].mean(axis=0)
            nma = float(np.linalg.norm(mu_a_new))
            nmb = float(np.linalg.norm(mu_b_new))
            if nma > 1e-8:
                mu_a = mu_a_new / nma
            if nmb > 1e-8:
                mu_b = mu_b_new / nmb
        if side_a.sum() < 2 or side_b.sum() < 2:
            continue

        ids_a = bn_ids_good[side_a]
        ids_b = bn_ids_good[side_b]
        n_A = normals[ids_a].mean(axis=0)
        n_A_norm = float(np.linalg.norm(n_A))
        n_B = normals[ids_b].mean(axis=0)
        n_B_norm = float(np.linalg.norm(n_B))
        if n_A_norm < 1e-8 or n_B_norm < 1e-8:
            continue
        n_A = n_A / n_A_norm
        n_B = n_B / n_B_norm
        # Need n_A and n_B to be distinct — a real fillet sits between
        # two planes whose normals differ meaningfully.
        if abs(float(np.dot(n_A, n_B))) > 0.9:
            continue

        # Both boundary sides must themselves be planar. A rolling-ball
        # fillet sits between two *planes*; if one side is actually a
        # cylinder (e.g. the inner wall of a hole — normals span 360°
        # radially around the hole axis), peeling corrupts the hole
        # geometry because we end up projecting hole-rim vertices onto
        # a nonsense "sharp edge line" computed from the cylinder's
        # mean normal.
        #
        # The test: project all boundary-face unit normals against their
        # mean direction. A plane's normals all agree, so the minimum
        # dot product is near 1. A cylinder wall's normals span the
        # full radial arc, so the mean normal is near zero and individual
        # dot products include negatives.
        #
        # We allow some noise — every plate-edge fillet in a triangulated
        # mesh has a few "boundary" faces near corners or at intersections
        # with other features that are slightly off-plane. Requiring the
        # mean direction to be well-defined (|mean| > 0.85) and the bulk
        # of normals (90th percentile) to be within 30 degrees of it
        # accepts real plate fillets while still rejecting hole-rim
        # fillets (whose boundary normals span a full disc).
        # Reject the extreme case where one boundary side is actually a
        # cylinder wall (hole rim): its normals span the full radial arc
        # around the hole axis, so their mean unit vector magnitude is
        # near zero. A real plane, even with mesh noise, keeps mean
        # magnitude > 0.6. This lax threshold accepts real plate fillets
        # but still rejects hole-rim torus-like strips.
        def _side_not_cylinder(side_ids):
            if len(side_ids) < 2:
                return True  # too small to decide — let it through
            nn = normals[side_ids]
            nn_n = np.linalg.norm(nn, axis=1)
            ok = nn_n > 1e-8
            if int(ok.sum()) < 2:
                return True
            u = nn[ok] / nn_n[ok, None]
            return float(np.linalg.norm(u.mean(axis=0))) > 0.6

        if not _side_not_cylinder(ids_a) or not _side_not_cylinder(ids_b):
            continue

        # Take the mean vertex of each side's faces as a plane point.
        verts_a_ids = np.unique(faces[ids_a].flatten())
        verts_b_ids = np.unique(faces[ids_b].flatten())
        p_A = vertices[verts_a_ids].mean(axis=0)
        p_B = vertices[verts_b_ids].mean(axis=0)

        # Intersection line of the two planes.
        line_dir = np.cross(n_A, n_B)
        ld_norm = float(np.linalg.norm(line_dir))
        if ld_norm < 1e-8:
            continue
        line_dir = line_dir / ld_norm

        # The final sharp-edge axis should be roughly aligned to a
        # principal direction on a prismatic part. We allow more slack
        # here than on axis_dir_est (the normal-eigenvector) because
        # line_dir = n_A x n_B is sensitive to mesh noise in the
        # boundary-face normals. Strongly-diagonal line_dirs (>40° off
        # axis) typically come from through-slot interior arcs where
        # the "boundary planes" are actually slot walls — not a real
        # plate-to-plate fillet.
        if float(np.max(np.abs(line_dir))) < 0.75:
            continue
        # Solve (n_A · p = n_A · p_A, n_B · p = n_B · p_B, line_dir · p = 0)
        M_sys = np.array([n_A, n_B, line_dir])
        rhs = np.array([float(n_A @ p_A), float(n_B @ p_B), 0.0])
        try:
            line_point = np.linalg.solve(M_sys, rhs)
        except np.linalg.LinAlgError:
            continue

        # Sanity check: the sharp edge's closest point to the strip
        # must be roughly inside the mesh. Strips whose plane-intersection
        # point ends up far outside the bbox come from numerically
        # ill-conditioned fits (nearly parallel planes) and will project
        # strip vertices to nonsense positions.
        mesh_min = cleaned.bbox[0]
        mesh_max = cleaned.bbox[1]
        margin = bbox_diag * 0.1
        if (
            line_point[0] < mesh_min[0] - margin
            or line_point[0] > mesh_max[0] + margin
            or line_point[1] < mesh_min[1] - margin
            or line_point[1] > mesh_max[1] + margin
            or line_point[2] < mesh_min[2] - margin
            or line_point[2] > mesh_max[2] + margin
        ):
            continue

        # Strip vertices and their distances to the sharp line.
        strip_vertex_ids = np.unique(faces[strip_faces].flatten())
        pts = vertices[strip_vertex_ids]
        rel = pts - line_point
        t_along = rel @ line_dir
        perp = rel - np.outer(t_along, line_dir)
        perp_dist = np.linalg.norm(perp, axis=1)
        # For a true fillet, the perpendicular distance from a strip
        # vertex to the sharp edge line is equal to the fillet radius
        # at the tangent lines (distance R) and up to R/sin(half-angle)
        # at the far side of the arc. The minimum over the strip is
        # therefore a robust estimate of R.
        # Use median of the strip-vertex-to-sharp-line perpendicular
        # distances as the radius estimate. Using min (naive) gets
        # corrupted by any single vertex that sits on or near the
        # computed sharp line, which happens when the two boundary
        # plane fits are slightly off — one strip ended up with
        # min=0.24 but median=2.13 on a real 2mm fillet. The median
        # is robust to those outliers and still converges to the true
        # radius for a clean cylindrical strip since every arc vertex
        # sits at perpendicular distance R from the sharp edge.
        radius = float(np.median(perp_dist))
        if radius < 0.2 or radius > max_radius:
            continue
        # Length of the fillet along the sharp edge. A real rolling-ball
        # fillet strip is always much longer along the edge axis than
        # wide across it — a 90° fillet arc of radius R has arc width
        # about (pi/2)*R ≈ 1.6R, so a strip whose length isn't at least
        # 2R is almost certainly a corner blend patch (where multiple
        # fillets meet) rather than a single edge fillet. Peeling such
        # a patch collapses a ~R x R x R 3D blend onto a short line,
        # which moves plate-face vertices badly. Require length >= 2R
        # to reject those cases.
        length = float(t_along.max() - t_along.min())
        if length < radius * 2.0:
            continue

        # Record new vertex positions: each strip vertex projects onto
        # the sharp edge line.
        projections = np.outer(t_along, line_dir) + line_point
        for idx_local, vi in enumerate(strip_vertex_ids):
            vertex_new_pos[int(vi)].append(projections[idx_local])

        fillets_out.append({
            "radius": radius,
            "axis_dir": line_dir.astype(float),
            "axis_point": line_point.astype(float),
            "length": length,
            "num_faces": int(len(strip_faces)),
            "face_indices": np.asarray(strip_faces, dtype=np.int64).copy(),
        })

    # --- 5. Apply the vertex moves. ---
    moved_count = 0
    for vi, positions in vertex_new_pos.items():
        vertices[vi] = np.mean(positions, axis=0)
        moved_count += 1

    # --- 6. Clean up: drop degenerate triangles, merge duplicate verts. ---
    # A strip-collapsed triangle's three vertices are all on the sharp
    # edge line, so the triangle has (nearly) zero area. Remove it.
    t = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Merge vertices that ended up coincident (tolerance scaled to part size).
    try:
        merge_tol = max(1e-5, bbox_diag * 1e-5)
        t.merge_vertices(merge_tex_norm=False) if False else t.merge_vertices()
    except Exception:
        pass
    # Drop zero-area triangles.
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
        metadata={**(cleaned.metadata or {}), "peeled_fillets": len(fillets_out)},
    )

    logger.info(
        "peel_fillets: %d fillets peeled, %d vertex moves, "
        "faces %d -> %d",
        len(fillets_out), moved_count, n_faces, len(new_faces),
    )
    return peeled, fillets_out


def peel_chamfers(
    cleaned: CleanedMesh,
    *,
    diag_threshold: float = 0.25,
    min_cluster_faces: int = 5,
    normal_eps: float = 0.08,
    max_size_ratio: float = 0.03,
    axis_alignment: float = 0.95,
    cluster_normal_spread: float = 0.05,
    bisector_tolerance: float = 0.12,
) -> tuple[CleanedMesh, list[dict]]:
    """Virtually remove flat bevel (chamfer) strips from a mesh.

    A chamfer between two plane faces is a single flat surface tilted
    roughly 45° from each plane. The face normals inside the chamfer
    cluster share a single diagonal direction, not a curved sweep (which
    is what distinguishes it from a rolling-ball fillet and requires a
    different detector). Like fillets, chamfers corrupt the segmenter:
    the chamfer surface gets fit as its own plane, the topology graph
    inserts a plane-to-plane transition where a sharp edge should be,
    and downstream recognition re-fits the chamfer as an extra feature.

    This peeler finds diagonal-normal face clusters, identifies the two
    adjacent planes (one on each side of the chamfer), computes their
    intersection line, and projects each chamfer vertex onto that line
    — same peel operation as for fillets, just a different way of
    gathering the strip.

    Args:
        cleaned: Input mesh.
        diag_threshold: how far off-axis a normal component must be to
            count as "diagonal". A value of 0.25 corresponds to ~14.5°.
            A chamfer surface has two components above this threshold
            and one below, covering 30°-60° bevel angles.
        min_cluster_faces: minimum faces in a single-normal cluster to
            be considered a real chamfer (not noise).
        normal_eps: DBSCAN epsilon on the unit normal sphere. 0.20 keeps
            same-normal faces together while separating different bevels.
        max_size_ratio: reject chamfers whose size (cluster bbox short
            axis) exceeds this fraction of the mesh bbox diagonal.
        axis_alignment: require the recovered sharp-edge line direction
            to be roughly aligned to a principal axis.

    Returns:
        (peeled_mesh, chamfers) — same shape as peel_fillets' return.
        Each chamfer record carries radius=size (the chamfer width),
        axis_dir (the sharp-edge direction), axis_point (on the line),
        length, num_faces.
    """
    import trimesh
    from sklearn.cluster import DBSCAN

    vertices = np.asarray(cleaned.vertices, dtype=float).copy()
    faces = np.asarray(cleaned.faces, dtype=np.int64)
    normals = np.asarray(cleaned.normals, dtype=float)
    n_faces = int(faces.shape[0])
    bbox_diag = cleaned.bbox_diagonal
    max_size = bbox_diag * max_size_ratio

    # --- 1. Find diagonal-normal candidate faces (potential chamfers). ---
    abs_normals = np.abs(normals)
    sorted_abs = np.sort(abs_normals, axis=1)
    diagonal_mask = (
        (sorted_abs[:, 1] > diag_threshold)
        & (sorted_abs[:, 2] > diag_threshold)
        & (sorted_abs[:, 0] < diag_threshold)
    )
    if int(diagonal_mask.sum()) < min_cluster_faces:
        return cleaned, []

    diag_indices = np.where(diagonal_mask)[0]
    diag_normals = normals[diag_indices]

    # --- 2. Cluster diagonal faces by normal direction. ---
    clustering = DBSCAN(eps=normal_eps, min_samples=3).fit(diag_normals)
    labels = clustering.labels_
    unique_labels = sorted(set(labels) - {-1})

    # --- 3. Adjacency map (all mesh faces). ---
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(n_faces):
        tri = faces[fi]
        for a, b in ((int(tri[0]), int(tri[1])),
                     (int(tri[1]), int(tri[2])),
                     (int(tri[0]), int(tri[2]))):
            edge_to_faces[(min(a, b), max(a, b))].append(fi)

    # Build face-to-neighbour adjacency among diagonal faces, limited to
    # edges within the SAME normal cluster. This is critical: DBSCAN by
    # normal alone lumps every chamfer with the same orientation across
    # the whole part into one huge cluster (e.g. all "top-front 45°"
    # chamfers from different tabs share a normal). We need to split
    # each normal cluster into spatially-connected components so each
    # physical chamfer strip is peeled independently.
    diag_face_to_cluster: dict[int, int] = {
        int(diag_indices[i]): int(labels[i])
        for i in range(len(diag_indices))
        if labels[i] != -1
    }

    def _split_cluster_by_connectivity(cluster_faces: np.ndarray) -> list[np.ndarray]:
        """Split a set of faces into connected components via shared edges."""
        member = set(int(f) for f in cluster_faces)
        face_adj: dict[int, list[int]] = defaultdict(list)
        for fid in cluster_faces:
            tri = faces[int(fid)]
            for a, b in ((int(tri[0]), int(tri[1])),
                         (int(tri[1]), int(tri[2])),
                         (int(tri[0]), int(tri[2]))):
                for nbr in edge_to_faces[(min(a, b), max(a, b))]:
                    nbr_i = int(nbr)
                    if nbr_i != int(fid) and nbr_i in member:
                        face_adj[int(fid)].append(nbr_i)
        comps: list[list[int]] = []
        visited: set[int] = set()
        for f0 in member:
            if f0 in visited:
                continue
            stack = [f0]
            comp = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                for nb in face_adj[cur]:
                    if nb not in visited:
                        stack.append(nb)
            comps.append(np.array(comp, dtype=np.int64))
        return comps

    # Expand unique_labels into spatially-connected sub-clusters
    split_clusters: list[np.ndarray] = []
    for lbl in unique_labels:
        cluster_face_idx = diag_indices[labels == lbl]
        if len(cluster_face_idx) < min_cluster_faces:
            continue
        comps = _split_cluster_by_connectivity(cluster_face_idx)
        split_clusters.extend(comps)

    chamfers_out: list[dict] = []
    vertex_new_pos: dict[int, list[np.ndarray]] = defaultdict(list)

    # --- 4. For each connected strip, walk to boundary faces and peel. ---
    for cluster_face_idx in split_clusters:
        if len(cluster_face_idx) < min_cluster_faces:
            continue

        # Tight normal variation check: a real flat chamfer has a single
        # shared normal direction across ALL its faces. Curved fillet
        # arcs sneak into DBSCAN as small spatial clusters whose normals
        # span a range around the arc. Require cluster_normal_spread so
        # we only peel actual planar bevels, not fillet fragments.
        cluster_norms = normals[cluster_face_idx]
        cluster_norms_n = np.linalg.norm(cluster_norms, axis=1)
        okmask = cluster_norms_n > 1e-8
        if int(okmask.sum()) < min_cluster_faces:
            continue
        u_cluster = cluster_norms[okmask] / cluster_norms_n[okmask, None]
        mean_n = u_cluster.mean(axis=0)
        mean_n_norm = float(np.linalg.norm(mean_n))
        if mean_n_norm < 1e-6:
            continue
        mean_n = mean_n / mean_n_norm
        dots = u_cluster @ mean_n
        # Every face normal must be within cos(arcsin(spread)) of the mean.
        # With spread=0.05, faces within ~3° of the shared normal — tight.
        if float(np.min(dots)) < 1.0 - cluster_normal_spread:
            continue
        chamfer_normal = mean_n
        strip_set = set(int(f) for f in cluster_face_idx)

        # Boundary faces: neighbours of cluster faces not in the cluster.
        boundary_faces: set[int] = set()
        for fid in cluster_face_idx:
            tri = faces[fid]
            for a, b in ((int(tri[0]), int(tri[1])),
                         (int(tri[1]), int(tri[2])),
                         (int(tri[0]), int(tri[2]))):
                for nbr in edge_to_faces[(min(a, b), max(a, b))]:
                    if nbr != int(fid) and int(nbr) not in strip_set:
                        boundary_faces.add(int(nbr))
        if len(boundary_faces) < 4:
            continue

        bn_ids = np.array(sorted(boundary_faces))
        bn = normals[bn_ids]
        bn_norms = np.linalg.norm(bn, axis=1)
        ok = bn_norms > 1e-8
        if int(ok.sum()) < 4:
            continue
        bn_unit = bn[ok] / bn_norms[ok, None]
        bn_ids = bn_ids[ok]

        # Expected: the two adjacent planes of a chamfer have normals at
        # ~45° from the chamfer's own normal. K=2 cluster the boundary
        # normals using the most-opposite pair as seed.
        dots = bn_unit @ bn_unit.T
        i_seed, j_seed = np.unravel_index(np.argmin(dots), dots.shape)
        mu_a = bn_unit[i_seed].copy()
        mu_b = bn_unit[j_seed].copy()
        for _ in range(5):
            side_a = (bn_unit @ mu_a) >= (bn_unit @ mu_b)
            side_b = ~side_a
            if side_a.sum() < 2 or side_b.sum() < 2:
                break
            na = bn_unit[side_a].mean(axis=0)
            nb = bn_unit[side_b].mean(axis=0)
            la = float(np.linalg.norm(na))
            lb = float(np.linalg.norm(nb))
            if la > 1e-8:
                mu_a = na / la
            if lb > 1e-8:
                mu_b = nb / lb
        if side_a.sum() < 2 or side_b.sum() < 2:
            continue

        # Each side's mean normal. Reject if they aren't distinct planes.
        if abs(float(np.dot(mu_a, mu_b))) > 0.9:
            continue

        # A real chamfer's flat normal sits at the angular bisector of
        # the two adjacent plane normals (the chamfer bevels exactly
        # halfway between them). Check: chamfer_normal should have
        # equal dot products with both mu_a and mu_b. The difference
        # |dot(n, mu_a) - dot(n, mu_b)| must be small. This rejects
        # cylinder walls, where a single "cluster normal" is NOT the
        # bisector of the two plate faces on either side.
        dot_a = float(chamfer_normal @ mu_a)
        dot_b = float(chamfer_normal @ mu_b)
        if abs(dot_a - dot_b) > bisector_tolerance:
            continue
        # And the normal must be roughly diagonal between the two plane
        # normals — dot with each plane's normal ≈ cos(bevel_half_angle).
        # For a 45° chamfer between perpendicular planes, both dots are
        # ~0.707. Must be < 0.95 (otherwise it's coplanar with a plane).
        if abs(dot_a) > 0.95 or abs(dot_b) > 0.95:
            continue

        ids_a = bn_ids[side_a]
        ids_b = bn_ids[side_b]
        verts_a_ids = np.unique(faces[ids_a].flatten())
        verts_b_ids = np.unique(faces[ids_b].flatten())
        p_A = vertices[verts_a_ids].mean(axis=0)
        p_B = vertices[verts_b_ids].mean(axis=0)

        # Intersection line direction and a point on it.
        line_dir = np.cross(mu_a, mu_b)
        ld_norm = float(np.linalg.norm(line_dir))
        if ld_norm < 1e-8:
            continue
        line_dir = line_dir / ld_norm
        if float(np.max(np.abs(line_dir))) < axis_alignment:
            continue

        M_sys = np.array([mu_a, mu_b, line_dir])
        rhs = np.array([float(mu_a @ p_A), float(mu_b @ p_B), 0.0])
        try:
            line_point = np.linalg.solve(M_sys, rhs)
        except np.linalg.LinAlgError:
            continue

        # Sanity check: line_point must be inside or near the mesh.
        mesh_min = cleaned.bbox[0]
        mesh_max = cleaned.bbox[1]
        margin = bbox_diag * 0.1
        if (
            line_point[0] < mesh_min[0] - margin
            or line_point[0] > mesh_max[0] + margin
            or line_point[1] < mesh_min[1] - margin
            or line_point[1] > mesh_max[1] + margin
            or line_point[2] < mesh_min[2] - margin
            or line_point[2] > mesh_max[2] + margin
        ):
            continue

        # Measure chamfer size from strip vertex distances to the line.
        strip_vertex_ids = np.unique(faces[cluster_face_idx].flatten())
        pts = vertices[strip_vertex_ids]
        rel = pts - line_point
        t_along = rel @ line_dir
        perp = rel - np.outer(t_along, line_dir)
        perp_dist = np.linalg.norm(perp, axis=1)
        size = float(np.median(perp_dist))
        if size < 0.2 or size > max_size:
            continue
        length = float(t_along.max() - t_along.min())
        if length < size * 2.0:
            continue

        # Project every strip vertex onto the sharp edge line.
        projections = np.outer(t_along, line_dir) + line_point
        for idx_local, vi in enumerate(strip_vertex_ids):
            vertex_new_pos[int(vi)].append(projections[idx_local])

        chamfers_out.append({
            "radius": size,
            "axis_dir": line_dir.astype(float),
            "axis_point": line_point.astype(float),
            "length": length,
            "num_faces": int(len(cluster_face_idx)),
            "face_indices": np.asarray(cluster_face_idx, dtype=np.int64).copy(),
            "kind": "chamfer",
        })

    # --- 5. Apply vertex moves. ---
    moved_count = 0
    for vi, positions in vertex_new_pos.items():
        vertices[vi] = np.mean(positions, axis=0)
        moved_count += 1

    # --- 6. Clean up degenerate triangles. ---
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
        metadata={**(cleaned.metadata or {}), "peeled_chamfers": len(chamfers_out)},
    )

    logger.info(
        "peel_chamfers: %d chamfers peeled, %d vertex moves, faces %d -> %d",
        len(chamfers_out), moved_count, n_faces, len(new_faces),
    )
    return peeled, chamfers_out


def peel_blends(cleaned: CleanedMesh) -> tuple[CleanedMesh, list[dict]]:
    """Peel both fillets and chamfers from a mesh in two passes.

    Fillets first (so the fillet strips don't leak into the chamfer
    diagonal-normal classifier), then chamfers on the already-peeled
    mesh. Returns the combined list of records; each carries a "kind"
    field ("fillet" or "chamfer") so the export stage can re-apply
    them differently if needed.
    """
    fmesh, fillets = peel_fillets(cleaned)
    for f in fillets:
        f.setdefault("kind", "fillet")
    cmesh, chamfers = peel_chamfers(fmesh)
    return cmesh, fillets + chamfers


def detect_blend_candidates(
    cleaned: CleanedMesh,
    *,
    dihedral_min_deg: float = 168.0,
    dihedral_max_deg: float = 179.5,
    diag_threshold: float = 0.25,
) -> dict:
    """Identify per-face candidates for visualization only — no peeling.

    Returns a dict with:
        fillet_face_mask: bool array, faces incident to a curved
            interior edge (potential fillet strip member).
        chamfer_face_mask: bool array, faces whose normal has two
            components above diag_threshold and one below (diagonal
            normal — potential chamfer bevel surface).

    Useful for pre-peel visualization so the user can see what peel
    is about to consider before it runs.
    """
    faces = np.asarray(cleaned.faces, dtype=np.int64)
    normals = np.asarray(cleaned.normals, dtype=float)
    n_faces = int(faces.shape[0])

    # Fillet candidates: faces on at least one curved-dihedral edge.
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(n_faces):
        tri = faces[fi]
        for a, b in ((int(tri[0]), int(tri[1])),
                     (int(tri[1]), int(tri[2])),
                     (int(tri[0]), int(tri[2]))):
            edge_to_faces[(min(a, b), max(a, b))].append(fi)

    fillet_mask = np.zeros(n_faces, dtype=bool)
    for flist in edge_to_faces.values():
        if len(flist) != 2:
            continue
        f0, f1 = flist
        n1, n2 = normals[f0], normals[f1]
        dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        dihed = float(np.degrees(np.pi - np.arccos(dot)))
        if dihedral_min_deg < dihed < dihedral_max_deg:
            fillet_mask[f0] = True
            fillet_mask[f1] = True

    # Chamfer candidates: faces whose normals are diagonal (2 components
    # significant, 1 small).
    abs_normals = np.abs(normals)
    sorted_abs = np.sort(abs_normals, axis=1)
    chamfer_mask = (
        (sorted_abs[:, 1] > diag_threshold)
        & (sorted_abs[:, 2] > diag_threshold)
        & (sorted_abs[:, 0] < diag_threshold)
    )

    return {
        "fillet_face_mask": fillet_mask,
        "chamfer_face_mask": chamfer_mask,
    }
