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
        # A fillet-like blend region is small, curved, adjacent to
        # plane regions. The EDGE-tag from topology is unreliable here
        # because topology's fillet-tag requires the cylinder region
        # to itself be a `_is_fillet_candidate` (partial-arc cylinder),
        # which fit.py sometimes doesn't produce — it'll classify the
        # arc as a sphere or cone instead, or merge the through-hole
        # cylinder so it spans 360°. Both make topology tag the
        # neighboring edges as CHAMFER or SHARP instead of FILLET.
        # So we use shape + size + plane-adjacency as the filter, and
        # skip the edge-tag check entirely.
        # Collect plane neighbors. We treat both true PLANE fits AND
        # flat-but-misfit CYLINDER/CONE/SPHERE fits as "plane
        # equivalents" — these are regions whose face normals are
        # near-uniform (mean-normal magnitude > 0.95), i.e. flat
        # plates that the fitter occasionally classifies as a
        # cylinder with absurd radius. Counting them lets fillets
        # that border such misfit plates still pass the
        # "must-bridge-to-a-flat" gate.
        plane_nbr_fits: list = []
        for e in neighbors:
            other = e.surface_b_id if e.surface_a_id == rid else e.surface_a_id
            of = fitted_lookup.get(other)
            if of is None:
                continue
            other_region = region_lookup.get(other)
            if other_region is None or len(other_region.face_indices) == 0:
                continue
            other_mean_n = cleaned.normals[other_region.face_indices].mean(axis=0)
            other_mag = float(np.linalg.norm(other_mean_n))
            is_flat = (
                of.surface_type == SurfaceType.PLANE
                or other_mag > 0.95
            )
            if is_flat:
                # Build a synthetic plane-like fit using the mean
                # normal as `normal` so downstream _pn() works.
                if of.surface_type == SurfaceType.PLANE:
                    plane_nbr_fits.append(of)
                else:
                    syn = type(of)(
                        region_id=of.region_id,
                        surface_type=SurfaceType.PLANE,
                        params={
                            "normal": (other_mean_n / other_mag).tolist(),
                            "point": other_region.vertices.mean(axis=0).tolist(),
                        },
                        rms_error=0.0,
                    )
                    plane_nbr_fits.append(syn)

        def _pn(fit):
            n = np.asarray(fit.params.get("normal", [0, 0, 1]), dtype=float)
            nn = np.linalg.norm(n)
            return n / nn if nn > 1e-12 else n

        if f.surface_type in (
            SurfaceType.CYLINDER, SurfaceType.SPHERE, SurfaceType.CONE,
        ):
            # Reject regions that are actually FLAT but got misfit
            # as a curved primitive (mean-normal magnitude > 0.95
            # means all face normals point ~the same direction —
            # a plate face). The fitter occasionally classifies a
            # flat plate as a cylinder with absurd radius (10000+
            # mm), and without this guard it gets tagged as a
            # giant fillet.
            mean_n = cleaned.normals[region.face_indices].mean(axis=0)
            if float(np.linalg.norm(mean_n)) > 0.95:
                continue
            # A fillet is a LONG THIN CYLINDER STRIP. The positive
            # signature is geometric, not topological:
            #   - small radius (mm-scale, not plate-scale)
            #   - reasonable face count (tens to a few hundred)
            #   - axis doesn't pierce any adjacent plane (hole bores,
            #     boss walls, and countersinks pierce)
            #   - length much larger than radius (strip, not ring)
            # We intentionally don't require ≥2 non-parallel planes:
            # a rolled-over edge on a thin plate sits between two
            # parallel plate faces, and a through-hole rim fillet only
            # has one plane neighbor (the plate face — the other side
            # is the hole cylinder).
            # Skip the early radius check entirely. Cylinder/sphere
            # fits to large rim fillets and perimeter rolled edges
            # often produce huge or wildly-wrong radii (the fitter
            # is approximating a torus/partial-arc surface with the
            # closest-available primitive). The downstream radius
            # check (line ~360) uses median perp-distance from the
            # actual edge line, which is geometrically correct, so
            # rejecting here on the raw fit radius drops legitimate
            # fillets unnecessarily.
            #
            # Face-count cap raised from 300 → no fixed limit. Small
            # mesh tabs have <100-face fillets, but a side plate's
            # main-hole rim fillet can run 500+ faces. The bridge-
            # test below + plane-neighbor-count rule handle all the
            # discrimination work either way.
            radius_raw = f.params.get("radius", 0) or 0
            # Get axis for cylinder/cone (used downstream for
            # line_dir fallback). Spheres don't have an axis.
            axis = None
            if f.surface_type in (SurfaceType.CYLINDER, SurfaceType.CONE):
                axis = np.asarray(
                    f.params.get("axis_dir") or f.params.get("axis")
                    or [0, 0, 1], dtype=float,
                )
                an = np.linalg.norm(axis)
                if an < 1e-12:
                    continue
                axis = axis / an

            # POSITIVE FILLET SIGNATURE: the curved region must
            # BRIDGE adjacent flat surfaces. Run this for ALL
            # curved fits (cylinder, cone, AND sphere) so big
            # rim fillets fitted as spheres are still detected.
            #
            # For each plane neighbor, check whether the region's
            # face normals form a BOUNDED arc tangent to that
            # plane:
            #   • some faces near-aligned with n_p (dot > 0.866)
            #     — the fillet sits TANGENT to this plate at one
            #     side
            #   • angular spread bounded ≤ 1.0 (~90°) — a real
            #     fillet sweeps at most a quarter-circle; side
            #     walls wrap 180°+ and fail.
            # This rejects bore walls (radial normals never
            # aligned with any plate), side walls (normals
            # uniform, fail spread test), countersink cones
            # (normals at uniform half-angle, fail alignment).
            face_norms = cleaned.normals[region.face_indices]
            _big_fillet = False
            if len(plane_nbr_fits) == 0:
                continue
            # Plane-count restriction targets cantilever side
            # walls: small (<150 face) curved regions wrapping a
            # 3D corner of a tab, with 3+ plane neighbors (top,
            # end, side faces). Two exemptions:
            #   • Large regions (≥150 faces) — perimeter rolled
            #     edges of plates legitimately touch 4-5 planes.
            #   • Long thin strips (longest bbox dim ≥ 50× the
            #     shortest dim) — these are fillets running
            #     along an edge (e.g. the top-edge fillet on a
            #     side plate runs the full plate length but is
            #     only a fraction of a mm thick). Cantilever
            #     walls are blocky (aspect ratio ≈ 3-4).
            if not _big_fillet:
                if (
                    f.surface_type in (SurfaceType.CYLINDER, SurfaceType.CONE)
                    and len(plane_nbr_fits) > 2
                    and region.num_faces < 150
                ):
                    region_verts = cleaned.vertices[
                        np.unique(cleaned.faces[region.face_indices].ravel())
                    ]
                    bb_size = region_verts.max(axis=0) - region_verts.min(axis=0)
                    bb_sorted = sorted(bb_size, reverse=True)
                    aspect_ratio = bb_sorted[0] / max(bb_sorted[2], 0.01)
                    # Two exemptions: long thin strips (aspect ≥ 50)
                    # AND highly-curved cylinders (mag < 0.4) that
                    # wrap most of the way around an axis. The
                    # latter catches base-plate side fillets where
                    # the rolled edge continues from the side plate
                    # onto the base plate (a curved strip with many
                    # plane neighbors). Cantilever side walls have
                    # mag in 0.5-0.8 and don't pass this exemption.
                    mean_mag = float(np.linalg.norm(face_norms.mean(axis=0)))
                    if aspect_ratio < 50 and mean_mag >= 0.4:
                        continue
                bridge_ok = False
                for pn in plane_nbr_fits:
                    n_p = _pn(pn)
                    dots = face_norms @ n_p
                    if dots.size == 0:
                        continue
                    spread = float(dots.max() - dots.min())
                    if spread > 1.0:
                        continue
                    if (dots > 0.866).any():
                        bridge_ok = True
                        break
                # Corner fillet fallback: small-medium curved
                # regions (30-100 faces) with directional normals
                # (mag in 0.6-0.95 — clearly tilted, not flat or
                # full-cylinder) that have ≥2 plane neighbors are
                # tab/perimeter corner fillets. The bridge test
                # rejects them because their face normals don't
                # reach the +Z/-Z directions of the neighbor
                # plates — they tilt toward a SMALL end-face plane
                # (which got filtered out by the size threshold).
                # Accept them as fillets in this case.
                if not bridge_ok:
                    mean_mag2 = float(np.linalg.norm(face_norms.mean(axis=0)))
                    if (
                        f.surface_type in (SurfaceType.CYLINDER, SurfaceType.CONE)
                        and 30 <= region.num_faces <= 100
                        and 0.6 <= mean_mag2 <= 0.95
                        and len(plane_nbr_fits) >= 2
                    ):
                        bridge_ok = True
                if not bridge_ok:
                    continue
                kind = "fillet"
                neighbor_edges = list(neighbors)
                bis_ni = bis_nj = None  # chamfer-only, unused for fillets
        elif f.surface_type == SurfaceType.PLANE:
            # Narrow plane strip — possible chamfer between two planes.
            chamfer_edges = [e for e in neighbors
                             if e.transition_type == TransitionType.CHAMFER]
            if not chamfer_edges:
                continue
            if region.num_faces > 40:
                continue
            if len(plane_nbr_fits) < 2:
                continue
            # Find a non-parallel pair so we can build the bisector.
            bis_ni = bis_nj = None
            for i in range(len(plane_nbr_fits)):
                ni = _pn(plane_nbr_fits[i])
                for j in range(i + 1, len(plane_nbr_fits)):
                    nj = _pn(plane_nbr_fits[j])
                    if abs(float(np.dot(ni, nj))) < 0.9:
                        bis_ni, bis_nj = ni, nj
                        break
                if bis_ni is not None:
                    break
            if bis_ni is None:
                continue
            # Chamfer face normal should roughly bisect its plane
            # neighbors — thin flat strips alongside a rounded edge
            # fail this because their normal matches the plate.
            cn = _pn(f)
            bis = bis_ni + bis_nj
            bis_n = np.linalg.norm(bis)
            if bis_n < 1e-9:
                continue
            bis = bis / bis_n
            if abs(float(np.dot(cn, bis))) < 0.85:
                continue
            kind = "chamfer"
            neighbor_edges = list(neighbors)
        else:
            continue

        # Resolve an edge line for the blend — either the exact plane-
        # plane intersection (when we have two non-parallel plane
        # neighbors) or a fallback using the cylinder axis (when
        # adjacency is ambiguous: rolled-over edges, rim fillets).
        p_rid_a = p_rid_b = None  # used later in record
        plane_rids: list[int] = []
        for e in neighbor_edges:
            other = e.surface_b_id if e.surface_a_id == rid else e.surface_a_id
            of = fitted_lookup.get(other)
            if of is not None and of.surface_type == SurfaceType.PLANE:
                plane_rids.append(other)
        unique_plane_rids = list(dict.fromkeys(plane_rids))

        # Try the two-non-parallel-planes path.
        line_dir = line_point = None
        for i in range(len(unique_plane_rids)):
            for j in range(i + 1, len(unique_plane_rids)):
                pA = fitted_lookup[unique_plane_rids[i]]
                pB = fitted_lookup[unique_plane_rids[j]]
                n_A = np.asarray(pA.params.get("normal"), dtype=float)
                n_B = np.asarray(pB.params.get("normal"), dtype=float)
                nA_norm = float(np.linalg.norm(n_A))
                nB_norm = float(np.linalg.norm(n_B))
                if nA_norm < 1e-8 or nB_norm < 1e-8:
                    continue
                n_A = n_A / nA_norm
                n_B = n_B / nB_norm
                if abs(float(np.dot(n_A, n_B))) > 0.95:
                    continue
                ld = np.cross(n_A, n_B)
                ld_norm = float(np.linalg.norm(ld))
                if ld_norm < 1e-8:
                    continue
                ld = ld / ld_norm
                rA = region_lookup.get(unique_plane_rids[i])
                rB = region_lookup.get(unique_plane_rids[j])
                if rA is None or rB is None:
                    continue
                p_A = rA.vertices.mean(axis=0)
                p_B = rB.vertices.mean(axis=0)
                M_sys = np.array([n_A, n_B, ld])
                rhs = np.array([float(n_A @ p_A), float(n_B @ p_B), 0.0])
                try:
                    lp = np.linalg.solve(M_sys, rhs)
                except np.linalg.LinAlgError:
                    continue
                line_dir = ld
                line_point = lp
                p_rid_a = unique_plane_rids[i]
                p_rid_b = unique_plane_rids[j]
                break
            if line_dir is not None:
                break

        # Fallback: rolled-over plate edge or rim fillet — use the
        # cylinder axis as the edge line (the "sharp line" a fillet
        # would collapse to lies along the cylinder axis for this
        # kind of blend). Cylinder/cone use their fitted axis;
        # sphere fits (which appear when the fitter approximates a
        # torus rim with a sphere) use the dominant plane
        # neighbor's normal — for a rim fillet around a hole, the
        # fillet's "edge axis" is perpendicular to the plate.
        if line_dir is None:
            if f.surface_type in (SurfaceType.CYLINDER, SurfaceType.CONE):
                axis_p = np.asarray(
                    f.params.get("axis_dir") or f.params.get("axis")
                    or [0, 0, 1], dtype=float,
                )
                an = np.linalg.norm(axis_p)
                if an < 1e-12:
                    continue
                line_dir = axis_p / an
                ap = f.params.get("axis_point")
                line_point = (
                    np.asarray(ap, dtype=float) if ap is not None
                    else region.vertices.mean(axis=0)
                )
            elif f.surface_type == SurfaceType.SPHERE and plane_nbr_fits:
                # Sphere rim fillet: use the plane neighbor's
                # normal as the fillet's edge-axis direction.
                # This is the axis a CAD fillet operation would
                # use (perpendicular to the plate it sits on).
                line_dir = _pn(plane_nbr_fits[0])
                line_point = region.vertices.mean(axis=0)
            else:
                continue

        # Re-anchor line_point to the closest point on the line
        # to the region's vertex centroid. This is the FULL
        # projection (not just along line_dir): drop the
        # perpendicular component as well so line_point ends up
        # at the actual nearest point on the line, which
        # guarantees it sits near the region in 3D rather than
        # wherever the plane-plane intersection landed.
        if line_dir is not None:
            region_centroid = region.vertices.mean(axis=0)
            rel = region_centroid - line_point
            t_anchor = float(rel @ line_dir)
            line_point = line_point + t_anchor * line_dir
            # Now line_point is on the line, closest to centroid.

        # Collect blend-region vertex IDs and project onto the sharp line.
        blend_face_ids = region.face_indices
        blend_vert_ids = np.unique(faces[blend_face_ids].flatten())
        pts = vertices[blend_vert_ids]
        rel = pts - line_point
        t_along = rel @ line_dir
        perp = rel - np.outer(t_along, line_dir)
        perp_dist = np.linalg.norm(perp, axis=1)
        # Prefer the fitter's radius (known-stable from least-squares
        # fit of the blend region itself) over the distance-from-line
        # estimate. The latter gets badly inflated when line_point
        # lands far from the true axis, which happens for fillets
        # with fragmented plane neighbors or diagonal axes.
        # For SPHERE-fitted rim fillets, the fit radius equals the
        # underlying torus's MAJOR radius (≈ hole radius), not the
        # fillet's minor radius. Use perp_dist instead so downstream
        # CAD reconstruction sees the actual fillet radius.
        fit_radius = float(f.params.get("radius", 0) or 0)
        if f.surface_type == SurfaceType.SPHERE:
            radius = float(np.median(perp_dist))
        elif fit_radius > 0.2:
            radius = fit_radius
        else:
            radius = float(np.median(perp_dist))
        # Sanity floor only — drop the upper-radius cap. Big main-
        # hole rim fillets and perimeter rolled edges have fit radii
        # 10-25 mm and were getting rejected, hiding the side-plate
        # fillets entirely. The bridge-test + plane-neighbor count
        # already discriminate fillets from non-fillets; an absolute
        # radius cap added no signal beyond that.
        if radius < 0.2:
            continue
        length = float(t_along.max() - t_along.min())
        if length < 1.0:
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
            "plane_region_ids": [
                rid_ for rid_ in (p_rid_a, p_rid_b) if rid_ is not None
            ],
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


def apply_blend_peel(
    cleaned: CleanedMesh,
    records: list[dict],
) -> CleanedMesh:
    """Project each blend region's vertices onto its plane-intersection
    line, drop the resulting degenerate triangles, and return a cleaned
    mesh that has sharp edges where the blends used to be.

    Records must include axis_dir, axis_point, and face_indices (as
    returned by identify_blends_from_topology).
    """
    import trimesh

    if not records:
        return cleaned

    vertices = np.asarray(cleaned.vertices, dtype=float).copy()
    faces = np.asarray(cleaned.faces, dtype=np.int64)
    bbox_diag = cleaned.bbox_diagonal

    # vertex_id -> list of candidate new positions (we average if a
    # vertex straddles two blends).
    vertex_new_pos: dict[int, list[np.ndarray]] = defaultdict(list)

    for rec in records:
        face_idx = rec.get("face_indices")
        if face_idx is None or len(face_idx) == 0:
            continue
        axis_dir = np.asarray(rec["axis_dir"], dtype=float)
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)
        axis_point = np.asarray(rec["axis_point"], dtype=float)

        blend_vert_ids = np.unique(faces[face_idx].flatten())
        pts = vertices[blend_vert_ids]
        rel = pts - axis_point
        t_along = rel @ axis_dir
        projections = axis_point + np.outer(t_along, axis_dir)
        for idx_local, vi in enumerate(blend_vert_ids):
            vertex_new_pos[int(vi)].append(projections[idx_local])

    moved_count = 0
    for vi, positions in vertex_new_pos.items():
        vertices[vi] = np.mean(positions, axis=0)
        moved_count += 1

    # Drop degenerate triangles and merge coincident vertices.
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
        metadata={
            **(cleaned.metadata or {}),
            "peeled_blend_count": len(records),
        },
    )
    logger.info(
        "apply_blend_peel: projected %d vertex(es) across %d blend(s), "
        "faces %d -> %d",
        moved_count, len(records), len(faces), len(new_faces),
    )
    return peeled


def detect_and_peel_blends_whole_mesh(
    cleaned: CleanedMesh,
) -> tuple[CleanedMesh, list[dict]]:
    """End-to-end: run segment / fit / topology on the whole mesh,
    identify every plane-to-plane blend (fillet cylinder or chamfer
    strip), and peel them all at once.

    This is the preferred path for the pre-decompose peel stage: it
    only touches mesh regions that the segmenter classified as planes
    connected by topology-classified blend edges, so the peel matches
    actual flat-face-to-flat-face fillets / chamfers and never acts on
    noisy curved strips in isolation.
    """
    # Local imports to avoid circulars at module load time.
    from mesh2param.pipeline.segment import segment_mesh
    from mesh2param.pipeline.fit import fit_surfaces
    from mesh2param.pipeline.topology import build_topology
    from mesh2param.config import SegmentConfig, FitConfig, TopologyConfig

    regions = segment_mesh(cleaned, SegmentConfig())
    fitted = fit_surfaces(regions, cleaned, FitConfig())
    _graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
    records = identify_blends_from_topology(regions, fitted, edges, cleaned)
    peeled = apply_blend_peel(cleaned, records)
    return peeled, records
