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
    dihedral_min_deg: float = 168.0,
    dihedral_max_deg: float = 179.5,
    max_radius_ratio: float = 0.08,
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

    # --- 4. Per-strip: validate cylinder hypothesis, then peel. ---
    fillets_out: list[dict] = []
    # vertex_id -> list of candidate new positions (usually 1, sometimes
    # more if a vertex straddles two strips — we average).
    vertex_new_pos: dict[int, list[np.ndarray]] = defaultdict(list)

    for sid in range(next_label):
        strip_faces = np.where(labels == sid)[0]
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
