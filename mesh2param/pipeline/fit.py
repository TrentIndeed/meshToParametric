"""Stage 3: Primitive Fitting.

Fits exact analytic surfaces (plane, cylinder, cone, sphere) to each
segmented region via SVD, RANSAC, and least-squares refinement.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from scipy.optimize import least_squares

from mesh2param.config import FitConfig
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.region import FaceRegion, SurfaceType
from mesh2param.models.surface import FittedSurface

logger = logging.getLogger(__name__)

# Serialize fit_surfaces across threads. scipy's least_squares calls
# into MKL/BLAS routines via minpack, which on Windows crashes with
# an access violation when two threads hit _fit_sphere / _fit_cone
# at the same time (the viz server runs one body per worker thread,
# so 3-body parts trigger it reliably). The lock only wraps this
# stage — segmentation and topology continue to run in parallel.
_FIT_LOCK = threading.Lock()

# Ordered by simplicity for alternative-type fallback
_FITTERS = {
    SurfaceType.PLANE: "_fit_plane",
    SurfaceType.SPHERE: "_fit_sphere",
    SurfaceType.CYLINDER: "_fit_cylinder",
    SurfaceType.CONE: "_fit_cone",
}


def fit_surfaces(
    regions: list[FaceRegion],
    cleaned: CleanedMesh,
    config: FitConfig | None = None,
) -> list[FittedSurface]:
    """Fit analytic primitives to each segmented region.

    Args:
        regions: Output of Stage 2 (Surface Segmentation).
        cleaned: Output of Stage 1 (for bbox diagonal).
        config: Fitting configuration. Uses defaults if None.

    Returns:
        List of FittedSurface objects with exact parameters and error metrics.
    """
    if config is None:
        config = FitConfig()

    logger.debug(
        "fit_surfaces called: %d regions, bbox_diagonal=%.4f",
        len(regions), cleaned.bbox_diagonal,
    )

    bbox_diag = cleaned.bbox_diagonal
    rms_tol = config.rms_tolerance_ratio * bbox_diag
    inlier_thresh = config.ransac_inlier_threshold_ratio * bbox_diag
    logger.debug("Fit tolerances: rms_tol=%.6f, inlier_thresh=%.6f", rms_tol, inlier_thresh)

    # Serialize the scipy-heavy body of this stage. See _FIT_LOCK
    # comment at module top for why (MKL/BLAS minpack crash when
    # two threads run _fit_sphere / _fit_cone simultaneously on
    # Windows). The lock is held only while this call runs its
    # per-region fits; concurrent callers queue up and run one at
    # a time.
    with _FIT_LOCK:
        results = []
        for region in regions:
            points = region.vertices
            normals = region.normals
            logger.debug(
                "Fitting region %d: type=%s, points=%d, normals=%d",
                region.region_id, region.surface_type.value, len(points), len(normals),
            )

            try:
                fitted = _fit_region(region, points, normals, config, rms_tol, inlier_thresh)
            except Exception:
                logger.exception(
                    "Stage 3 failed fitting region %d (type=%s, points=%d)",
                    region.region_id, region.surface_type.value, len(points),
                )
                raise

            # Try alternative types if RMS exceeds tolerance
            if config.try_alternative_types and fitted.rms_error > rms_tol:
                fitted = _try_alternatives(region, points, normals, config, rms_tol, inlier_thresh, fitted)

            # Check if a non-plane fit should really be a plane:
            # If the plane fit has near-zero RMS AND near-zero normal spread,
            # prefer plane over cylinder/sphere/cone.
            if fitted.surface_type != SurfaceType.PLANE:
                plane_params, plane_rms = _fit_plane(points, normals, config, inlier_thresh)
                avg_n = normals.mean(axis=0)
                n_norm = np.linalg.norm(avg_n)
                if n_norm > 1e-15:
                    avg_n /= n_norm
                n_spread = float(1.0 - np.mean(normals @ avg_n))
                # Accept plane if RMS is essentially zero and normals are consistent
                if plane_rms < 0.01 and n_spread < 0.15:
                    fitted = FittedSurface(
                        region_id=region.region_id, surface_type=SurfaceType.PLANE,
                        params=plane_params, rms_error=plane_rms,
                    )

            # Dirty-mesh fit robustness: noise creates APPARENT curvature, so a
            # flat patch mis-fits as a cylinder / cone / sphere (user-observed:
            # "fitting did some cylindric shape instead of flat sides"). Prefer
            # the plane whenever it fits COMPARABLY — the curved primitive then
            # didn't earn its complexity (Occam). A GENUINE curved surface
            # (bore, real fillet) fits a plane MUCH worse, so the ratio guard
            # preserves it. Gated to dirty inputs; clean CAD keeps the strict
            # logic above.
            if (cleaned.metadata.get("scan_mode")
                    and fitted.surface_type in (
                        SurfaceType.CYLINDER, SurfaceType.CONE,
                        SurfaceType.SPHERE)):
                plane_params, plane_rms = _fit_plane(
                    points, normals, config, inlier_thresh)
                # (a) the plane fits COMPARABLY (curved didn't earn complexity)
                prefer = plane_rms <= fitted.rms_error * 1.25 + 1e-6
                # (b) GEOMETRIC flatness (the real discriminator for cylinders):
                # a cylinder whose radius exceeds its own patch extent is
                # effectively flat (the box face the user expects), whereas a
                # GENUINE bore/fillet has radius <= its region extent. This
                # cleanly flattens AI15's huge-radius mis-fits while preserving
                # AI3's tiny bore (radius << its region). Generous rms guard:
                # a truly-flat region's plane fits fine, so 3x is safe here.
                if (not prefer and fitted.surface_type == SurfaceType.CYLINDER):
                    rad = float(fitted.params.get("radius", 0.0) or 0.0)
                    ext = points.max(axis=0) - points.min(axis=0)
                    region_diag = float(np.linalg.norm(ext)) or 1.0
                    if (rad > 1.0 * region_diag
                            and plane_rms <= fitted.rms_error * 3.0 + 1e-6):
                        prefer = True
                # (c) NORMAL CONSISTENCY: tightly-aligned face normals ARE a flat
                # face — a genuine cylinder/cone/sphere fans its normals around
                # its axis. Catches flats mis-fit as CONE/SPHERE (the cylinder-
                # only radius guard above misses those), e.g. the carriage
                # block's front/back faces after the opposite-face split.
                if not prefer:
                    an = normals.mean(axis=0)
                    nl = float(np.linalg.norm(an))
                    if nl > 1e-9:
                        nsp = float(1.0 - np.mean(normals @ (an / nl)))
                        if nsp < 0.04 and plane_rms <= fitted.rms_error * 4.0 + 1e-6:
                            prefer = True
                # (d) RANK-1 normals = a PLANE by definition. A noisy flat wall
                # has coherence ~0.9 (normal spread too high for the 0.04 gate
                # above), yet its normals are still RANK 1 — one direction plus
                # ISOTROPIC noise — whereas a genuine cylinder/cone fans its
                # normals along an ARC (RANK 2, a large 2nd singular value). So
                # when the 2nd singular value of the (unit) normals is small
                # relative to the 1st, the patch is flat and must be a plane.
                # This catches the wall-split's freed walls that mis-fit as
                # CONE/CYLINDER/SPHERE (coherence ~0.90), while a real bore
                # (rank 2) keeps its curved fit.
                if not prefer:
                    try:
                        sv = np.linalg.svd(normals, compute_uv=False)
                        ratio = float(sv[1] / sv[0]) if sv[0] > 1e-9 else 1.0
                    except Exception:
                        ratio = 1.0
                    if ratio < 0.45 and plane_rms <= fitted.rms_error * 8.0 + 1e-6:
                        prefer = True
                if prefer:
                    logger.info(
                        "Region %d: %s -> plane (rms %.4f vs %.4f)",
                        region.region_id, fitted.surface_type.value,
                        fitted.rms_error, plane_rms)
                    fitted = FittedSurface(
                        region_id=region.region_id,
                        surface_type=SurfaceType.PLANE,
                        params=plane_params, rms_error=plane_rms,
                    )

            # Disambiguate sphere vs cylinder: both can fit hole surfaces with
            # near-zero RMS, but cylinder is the correct type for holes.
            if config.try_alternative_types and fitted.surface_type in (SurfaceType.SPHERE, SurfaceType.CYLINDER):
                fitted = _disambiguate_sphere_cylinder(
                    region, points, normals, config, inlier_thresh, fitted,
                )

            # Disambiguate sphere vs cone: a countersink (truncated cone) fits
            # as a sphere almost as well, and different triangulation of the
            # same cone can flip the type. Check normal-spread geometry: cone
            # normals span a 2D band (sphere spans 3D).
            if config.try_alternative_types and fitted.surface_type in (SurfaceType.SPHERE, SurfaceType.CONE):
                fitted = _disambiguate_sphere_cone(
                    region, points, normals, config, inlier_thresh, fitted,
                )

            # Snap parameters
            fitted.snapped_params = _snap_params(fitted.params, fitted.surface_type, config)

            results.append(fitted)
            logger.info(
                "Region %d: %s, RMS=%.6f (tol=%.6f)",
                region.region_id, fitted.surface_type.value, fitted.rms_error, rms_tol,
            )

        # Post-process: merge adjacent cylinder regions with matching params.
        # Segmentation often splits a single cylindrical hole into 4-5 rectangular
        # clusters due to KMeans on rotating normals. This merges them back.
        results = _merge_coincident_cylinders(results, regions, cleaned, bbox_diag)

    logger.debug("fit_surfaces completed: %d surfaces fitted", len(results))
    return results


def _merge_coincident_cylinders(
    fitted: list[FittedSurface],
    regions: list[FaceRegion],
    cleaned: CleanedMesh,
    bbox_diag: float,
) -> list[FittedSurface]:
    """Merge cylinder fits whose axis/radius/center are nearly identical.

    When a cylindrical hole is split into multiple regions by segmentation
    (e.g. KMeans clustering on the normal directions around the cylinder),
    each piece gets its own cylinder fit with similar parameters. This
    merges them back into a single fit representing the whole cylinder.
    """
    region_lookup = {r.region_id: r for r in regions}
    cyl_fits = [f for f in fitted if f.surface_type == SurfaceType.CYLINDER]
    other_fits = [f for f in fitted if f.surface_type != SurfaceType.CYLINDER]

    if len(cyl_fits) < 2:
        return fitted

    # Group cylinders by similar params
    pos_tol = bbox_diag * 0.003  # 0.3% of bbox — cylinders must share the same axis line
    radius_tol_ratio = 0.08  # 8% radius tolerance
    axis_dot_tol = 0.98  # within ~11° axis alignment

    groups: list[list] = []
    used = set()
    for i, ci in enumerate(cyl_fits):
        if i in used:
            continue
        axis_i = np.array(ci.params.get("axis_dir", [0, 0, 1]))
        axis_i = axis_i / (np.linalg.norm(axis_i) + 1e-15)
        center_i = np.array(ci.params.get("axis_point", [0, 0, 0]))
        radius_i = ci.params.get("radius", 0)
        if radius_i <= 0:
            continue

        group = [ci]
        used.add(i)
        for j in range(i + 1, len(cyl_fits)):
            if j in used:
                continue
            cj = cyl_fits[j]
            axis_j = np.array(cj.params.get("axis_dir", [0, 0, 1]))
            axis_j = axis_j / (np.linalg.norm(axis_j) + 1e-15)
            center_j = np.array(cj.params.get("axis_point", [0, 0, 0]))
            radius_j = cj.params.get("radius", 0)
            if radius_j <= 0:
                continue
            # Axis match (parallel, same direction or opposite)
            axis_match = abs(float(np.dot(axis_i, axis_j))) > axis_dot_tol
            # Radius match
            radius_match = abs(radius_i - radius_j) < max(radius_i, radius_j) * radius_tol_ratio
            # Center match: cylinders must share the same axis LINE (not just
            # a parallel axis), AND overlap axially within a few radii.
            delta = center_j - center_i
            axial = float(np.dot(delta, axis_i))
            perp = delta - axial * axis_i
            perp_dist = float(np.linalg.norm(perp))
            # Must be on the same axis line (perpendicular distance tiny)
            # AND axially close (within 3x the radius — same physical hole)
            center_match = perp_dist < pos_tol and abs(axial) < radius_i * 3.0
            if axis_match and radius_match and center_match:
                group.append(cj)
                used.add(j)
        groups.append(group)

    # Merge each group into one fit
    merged_cyls = []
    for group in groups:
        if len(group) == 1:
            merged_cyls.append(group[0])
            continue
        # Pick the group member with the most region faces as the "primary"
        primary = max(
            group,
            key=lambda f: len(region_lookup.get(f.region_id).face_indices)
            if region_lookup.get(f.region_id) else 0,
        )
        # Collect all face indices from this group into the primary's region
        primary_region = region_lookup.get(primary.region_id)
        if primary_region is None:
            merged_cyls.append(primary)
            continue
        all_face_idx = set(primary_region.face_indices.tolist())
        for f in group:
            if f.region_id == primary.region_id:
                continue
            r = region_lookup.get(f.region_id)
            if r is not None:
                all_face_idx.update(r.face_indices.tolist())
        merged_face_idx = np.array(sorted(all_face_idx), dtype=np.int64)
        # Update the primary's region in place so downstream stages (which share
        # this regions list) see the merged faces. Compute the derived fields
        # first and only assign all three together — a partial update would
        # leave face_indices inconsistent with vertices/normals.
        if len(merged_face_idx) and merged_face_idx[-1] >= len(cleaned.faces):
            logger.warning(
                "Skipping cylinder merge for region %d: face index %d out of range (%d faces)",
                primary.region_id, int(merged_face_idx[-1]), len(cleaned.faces),
            )
            merged_cyls.extend(group)
            continue
        merged_vertices = cleaned.vertices[cleaned.faces[merged_face_idx].flatten()]
        merged_normals = cleaned.normals[merged_face_idx]
        primary_region.face_indices = merged_face_idx
        primary_region.vertices = merged_vertices
        primary_region.normals = merged_normals
        logger.info(
            "Merged %d cylinder regions into region %d: %d total faces",
            len(group), primary.region_id, len(merged_face_idx),
        )
        merged_cyls.append(primary)

    return other_fits + merged_cyls


def _fit_region(
    region: FaceRegion,
    points: np.ndarray,
    normals: np.ndarray,
    config: FitConfig,
    rms_tol: float,
    inlier_thresh: float,
) -> FittedSurface:
    """Fit the region's declared surface type."""
    stype = region.surface_type

    if stype == SurfaceType.UNKNOWN or stype == SurfaceType.TORUS:
        # Unknown type — try all types and pick the best
        best = None
        for try_type in [SurfaceType.PLANE, SurfaceType.CYLINDER, SurfaceType.SPHERE, SurfaceType.CONE]:
            try:
                fitter = _get_fitter(try_type)
                params, rms = fitter(points, normals, config, inlier_thresh)
                if best is None or rms < best.rms_error:
                    best = FittedSurface(
                        region_id=region.region_id, surface_type=try_type,
                        params=params, rms_error=rms,
                    )
            except Exception:
                continue
        if best is not None:
            return best
        stype = SurfaceType.PLANE

    fitter = _get_fitter(stype)
    params, rms = fitter(points, normals, config, inlier_thresh)

    return FittedSurface(
        region_id=region.region_id,
        surface_type=stype,
        params=params,
        rms_error=rms,
    )


def _try_alternatives(
    region: FaceRegion,
    points: np.ndarray,
    normals: np.ndarray,
    config: FitConfig,
    rms_tol: float,
    inlier_thresh: float,
    current_best: FittedSurface,
) -> FittedSurface:
    """Try fitting with alternative surface types and keep the best.

    Uses hysteresis on the RMS comparison: the current type wins ties
    within ~15%. Without this, two geometrically-identical features
    (e.g. four countersinks on symmetric tabs) can flip between
    SPHERE and CONE on floating-point noise, breaking symmetry in
    the downstream region graph.
    """
    best = current_best
    # Alternative must beat current by a meaningful margin, unless
    # the current fit is clearly bad (RMS well above tolerance).
    hysteresis = 0.85 if current_best.rms_error < rms_tol * 1.5 else 1.0

    for stype, fname in _FITTERS.items():
        if stype == current_best.surface_type:
            continue

        try:
            fitter = _get_fitter(stype)
            params, rms = fitter(points, normals, config, inlier_thresh)
            if rms < best.rms_error * hysteresis:
                best = FittedSurface(
                    region_id=region.region_id,
                    surface_type=stype,
                    params=params,
                    rms_error=rms,
                )
                if rms < rms_tol:
                    break
                # Once we switch, further alternatives must beat the
                # new best by the same margin.
                hysteresis = 0.85
        except Exception:
            continue

    return best


def _disambiguate_sphere_cylinder(
    region: FaceRegion,
    points: np.ndarray,
    normals: np.ndarray,
    config: FitConfig,
    inlier_thresh: float,
    current: FittedSurface,
) -> FittedSurface:
    """Disambiguate sphere vs cylinder when both fit well.

    A sphere can perfectly wrap a short cylinder segment. To distinguish:
    - Cylinder normals lie in a plane (perpendicular to the axis) — the normal
      directions span a 1D arc, so SVD of normals has 1 small singular value.
    - Sphere normals spread in all 3D directions — SVD has no small singular value.

    Also compare: for a true cylinder, the cylinder fit radius should be much
    smaller than the sphere fit radius (the sphere inflates to wrap the cylinder).
    """
    # Cheap early-exit, CYLINDER only: a coplanar-normal signature is
    # exactly the disambiguation's own "-> cylinder" path below, so
    # short-circuiting it can't change the answer and avoids re-fitting
    # both a cylinder and a sphere (the dominant cost). A SPHERE-fit
    # region is NOT safe to keep here — it can still flip to cylinder via
    # the radius-ratio test regardless of normal spread — so it falls
    # through to the full comparison.
    centered_n0 = normals - normals.mean(axis=0)
    if len(centered_n0) >= 3 and current.surface_type == SurfaceType.CYLINDER:
        _, sv0, _ = np.linalg.svd(centered_n0, full_matrices=False)
        if (sv0 / (sv0[0] + 1e-15))[2] < 0.2:
            return current  # clearly coplanar normals -> cylinder

    # Fit both types
    try:
        cyl_params, cyl_rms = _fit_cylinder(points, normals, config, inlier_thresh)
        sph_params, sph_rms = _fit_sphere(points, normals, config, inlier_thresh)
    except Exception:
        return current

    # Check radius ratio first — for a cylinder, the sphere fit inflates its
    # radius to wrap around, so sphere_radius >> cylinder_radius.
    # For a true sphere, both radii are similar.
    cyl_radius = cyl_params.get("radius", 0)
    sph_radius = sph_params.get("radius", 0)

    if cyl_radius > 0 and sph_radius > 0:
        ratio = sph_radius / cyl_radius
        if ratio > 1.5:
            logger.debug(
                "Region %d: sph_r/cyl_r=%.1f → cylinder (cyl_r=%.3f, sph_r=%.3f)",
                region.region_id, ratio, cyl_radius, sph_radius,
            )
            return FittedSurface(region_id=region.region_id, surface_type=SurfaceType.CYLINDER,
                                 params=cyl_params, rms_error=cyl_rms)

    # Also use SVD on normals: cylinder normals lie in a plane (one small SV),
    # sphere normals spread in all 3 directions.
    centered_n = normals - normals.mean(axis=0)
    if len(centered_n) >= 3:
        _, s_vals, _ = np.linalg.svd(centered_n, full_matrices=False)
        s_norm = s_vals / (s_vals[0] + 1e-15)
        if s_norm[2] < 0.2:
            logger.debug(
                "Region %d: normal SVD [%.3f, %.3f, %.3f] → cylinder (coplanar normals)",
                region.region_id, *s_norm,
            )
            return FittedSurface(region_id=region.region_id, surface_type=SurfaceType.CYLINDER,
                                 params=cyl_params, rms_error=cyl_rms)

    # Both radius ratio and SVD suggest sphere
    if sph_rms <= cyl_rms:
        return FittedSurface(region_id=region.region_id, surface_type=SurfaceType.SPHERE,
                             params=sph_params, rms_error=sph_rms)

    return current


def _disambiguate_sphere_cone(
    region: FaceRegion,
    points: np.ndarray,
    normals: np.ndarray,
    config: FitConfig,
    inlier_thresh: float,
    current: FittedSurface,
) -> FittedSurface:
    """Disambiguate sphere vs cone when both fit well.

    A cone section and a sphere cap have similar RMS on short segments,
    so raw RMS alone flip-flops between symmetric features whose
    triangulation differs slightly. Geometric tiebreaker:
    - Cone normals span a 2D fan (one principal direction along the
      axis, one wrapping around it) — 3rd SVD singular value is small.
    - Sphere normals spread uniformly in 3D.
    We prefer cone whenever that 2D-fan signature is clear, because
    countersinks (the main case here) are overwhelmingly more common
    than spherical caps in mechanical parts.
    """
    # Cheap early-exit, CONE only: a clear 2D-fan signature is exactly
    # the disambiguation's own "-> cone" path below, so short-circuiting
    # it can't change the answer and skips the cone+sphere re-fits (the
    # dominant cost). A SPHERE-fit region can still flip to cone via the
    # RMS comparison, so it falls through to the full check.
    centered_n0 = normals - normals.mean(axis=0)
    if len(centered_n0) >= 3 and current.surface_type == SurfaceType.CONE:
        _, sv0, _ = np.linalg.svd(centered_n0, full_matrices=False)
        sn0 = sv0 / (sv0[0] + 1e-15)
        if sn0[2] < 0.35 and sn0[1] > 0.3:
            return current  # clear 2D fan -> cone

    try:
        cone_params, cone_rms = _fit_cone(points, normals, config, inlier_thresh)
        sph_params, sph_rms = _fit_sphere(points, normals, config, inlier_thresh)
    except Exception:
        return current

    # If either fit is clearly bad, defer.
    if cone_rms > inlier_thresh * 3 or sph_rms > inlier_thresh * 3:
        return current

    centered_n = normals - normals.mean(axis=0)
    if len(centered_n) < 3:
        return current
    _, s_vals, _ = np.linalg.svd(centered_n, full_matrices=False)
    s_norm = s_vals / (s_vals[0] + 1e-15)

    # Cone signature: third singular value noticeably smaller than
    # second (normals fan in a plane around the axis).
    if s_norm[2] < 0.35 and s_norm[1] > 0.3:
        logger.debug(
            "Region %d: normal SVD [%.3f, %.3f, %.3f] → cone (fan signature)",
            region.region_id, *s_norm,
        )
        return FittedSurface(
            region_id=region.region_id, surface_type=SurfaceType.CONE,
            params=cone_params, rms_error=cone_rms,
        )

    # Otherwise keep whatever has smaller RMS, with slight cone bias
    # (countersinks dominate the use case).
    if cone_rms <= sph_rms * 1.1:
        return FittedSurface(
            region_id=region.region_id, surface_type=SurfaceType.CONE,
            params=cone_params, rms_error=cone_rms,
        )
    return FittedSurface(
        region_id=region.region_id, surface_type=SurfaceType.SPHERE,
        params=sph_params, rms_error=sph_rms,
    )


def _get_fitter(stype: SurfaceType):
    """Return the fitting function for a surface type."""
    mapping = {
        SurfaceType.PLANE: _fit_plane,
        SurfaceType.SPHERE: _fit_sphere,
        SurfaceType.CYLINDER: _fit_cylinder,
        SurfaceType.CONE: _fit_cone,
    }
    return mapping.get(stype, _fit_plane)


# ---------------------------------------------------------------------------
# Plane fitting: SVD
# ---------------------------------------------------------------------------

def _fit_plane(
    points: np.ndarray,
    normals: np.ndarray,
    config: FitConfig,
    inlier_thresh: float,
) -> tuple[dict, float]:
    """Fit a plane via SVD on the centered point cloud.

    Normal = smallest singular vector of centered points.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    if len(points) < 3:
        # Underdetermined: a plane needs 3+ non-collinear points. Aggressive
        # decimation can leave a 1-2 point sliver region; SVD then returns
        # fewer than 3 rows and vh[2] is out of bounds. Fall back to the mean
        # face normal and mark the fit as bad (inf RMS) so downstream gating
        # drops it, rather than crashing the whole build.
        normal = (np.asarray(normals, dtype=float).mean(axis=0)
                  if normals is not None and len(normals) > 0
                  else np.array([0.0, 0.0, 1.0]))
        nn = float(np.linalg.norm(normal))
        normal = normal / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
        return ({"point": centroid.tolist(), "normal": normal.tolist(),
                 "extent": 0.0}, float("inf"))
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[2]  # row corresponding to smallest singular value

    # Orient normal to agree with the majority of face normals
    if normals is not None and len(normals) > 0:
        avg_face_normal = normals.mean(axis=0)
        if np.dot(normal, avg_face_normal) < 0:
            normal = -normal

    # Compute the plane extent (max distance from centroid projected onto plane)
    proj = centered - np.outer(centered @ normal, normal)
    extent = float(np.max(np.linalg.norm(proj, axis=1))) * 2

    # RMS: point-to-plane distance
    dists = np.abs(centered @ normal)
    rms = float(np.sqrt(np.mean(dists ** 2)))

    params = {
        "point": centroid.tolist(),
        "normal": normal.tolist(),
        "extent": round(extent, 4),
    }
    return params, rms


# ---------------------------------------------------------------------------
# Sphere fitting: algebraic (Pratt) + geometric refinement
# ---------------------------------------------------------------------------

def _fit_sphere(
    points: np.ndarray,
    normals: np.ndarray,
    config: FitConfig,
    inlier_thresh: float,
) -> tuple[dict, float]:
    """Algebraic sphere fit (Pratt method) + geometric least-squares."""
    n = len(points)
    if n < 4:
        return {"center": [0, 0, 0], "radius": 0}, float("inf")

    # Algebraic fit: solve Ax = 0 where A = [x^2+y^2+z^2, x, y, z, 1]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r2 = x ** 2 + y ** 2 + z ** 2
    A = np.column_stack([r2, x, y, z, np.ones(n)])
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    coeffs = vh[-1]  # a, b, c, d, e  where a(x²+y²+z²) + bx + cy + dz + e = 0

    a = coeffs[0]
    if abs(a) < 1e-15:
        # Degenerate — fallback
        center = points.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(points - center, axis=1)))
    else:
        center = -coeffs[1:4] / (2 * a)
        radius_sq = np.dot(center, center) - coeffs[4] / a
        radius = float(np.sqrt(max(radius_sq, 0)))

    # Geometric refinement via least-squares
    def residuals(p):
        c = p[:3]
        r = p[3]
        return np.linalg.norm(points - c, axis=1) - r

    x0 = np.array([*center, radius])
    try:
        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(points)):
            raise ValueError("non-finite inputs")
        # method="lm" (Levenberg-Marquardt / minpack) crashed
        # scipy with Windows access-violation on some degenerate
        # inputs (e.g. narrow tangent-strip regions fitted as
        # spheres during sphere/cylinder disambiguation). Switched
        # to method="trf" which handles singular Jacobians by
        # trust-region reflective iteration instead of calling
        # into unsafe C minpack.
        result = least_squares(residuals, x0, method="trf")
        center = result.x[:3]
        radius = abs(result.x[3])
    except Exception:
        logger.debug("Sphere least-squares refinement failed for %d points, using algebraic fit", n)

    dists = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    rms = float(np.sqrt(np.mean(dists ** 2)))

    params = {
        "center": center.tolist(),
        "radius": round(float(radius), 4),
    }
    return params, rms


# ---------------------------------------------------------------------------
# Cylinder fitting: RANSAC + least-squares
# ---------------------------------------------------------------------------

def _fit_cylinder(
    points: np.ndarray,
    normals: np.ndarray,
    config: FitConfig,
    inlier_thresh: float,
) -> tuple[dict, float]:
    """RANSAC cylinder fit: sample 2 points + normals to estimate axis.

    Then least-squares refinement of axis direction, point-on-axis, and radius.
    """
    n = len(points)
    if n < 6:
        return {"axis_point": [0, 0, 0], "axis_dir": [0, 0, 1], "radius": 0, "height": 0}, float("inf")

    rng = np.random.RandomState(42)

    use_normals = normals is not None and len(normals) >= n

    # Sample all hypothesis pairs up front, then score every hypothesis
    # against every point with batched matrix ops. The previous one-
    # hypothesis-per-Python-iteration loop spent its time in thousands
    # of tiny numpy calls (cross/norm/median per iteration).
    iters = config.ransac_iterations
    i1 = rng.randint(0, n, size=iters)
    i2 = rng.randint(0, n - 1, size=iters)
    i2[i2 >= i1] += 1  # distinct indices

    if use_normals:
        axis_dirs = np.cross(normals[i1], normals[i2])
    else:
        axis_dirs = points[i2] - points[i1]
    norms = np.linalg.norm(axis_dirs, axis=1)
    valid = norms > 1e-10
    axis_dirs = axis_dirs[valid] / norms[valid, None]
    anchors = points[i1[valid]]

    if len(axis_dirs) == 0:
        return {"axis_point": [0, 0, 0], "axis_dir": [0, 0, 1], "radius": 0, "height": 0}, float("inf")

    p_sq = np.einsum("ij,ij->i", points, points)
    best_params = None
    best_inliers = -1
    best_rms = float("inf")
    # Chunk hypotheses so the (N x chunk) score matrices stay ~150 MB max
    chunk = max(1, int(2e7 // max(n, 1)))
    for s in range(0, len(axis_dirs), chunk):
        D = axis_dirs[s:s + chunk]                     # (C, 3)
        A = anchors[s:s + chunk]                       # (C, 3)
        ad = np.einsum("ij,ij->i", A, D)               # (C,)
        a_sq = np.einsum("ij,ij->i", A, A)             # (C,)
        t = points @ D.T - ad[None, :]                 # (N, C) axial coord
        d_sq = p_sq[:, None] - 2.0 * (points @ A.T) + a_sq[None, :] - t ** 2
        np.maximum(d_sq, 0.0, out=d_sq)
        dist = np.sqrt(d_sq)                           # (N, C) radial distance
        radius = np.median(dist, axis=0)               # (C,)
        resid = np.abs(dist - radius[None, :])
        inlier_mask = resid < inlier_thresh
        n_inliers = inlier_mask.sum(axis=0)
        sum_resid = np.where(inlier_mask, resid, 0.0).sum(axis=0)
        mean_resid = np.where(
            n_inliers > 0, sum_resid / np.maximum(n_inliers, 1), np.inf,
        )
        n_inliers = np.where(radius > 1e-10, n_inliers, -1)

        b = int(np.lexsort((mean_resid, -n_inliers))[0])
        if n_inliers[b] > best_inliers or (
            n_inliers[b] == best_inliers and mean_resid[b] < best_rms
        ):
            best_inliers = int(n_inliers[b])
            best_rms = float(mean_resid[b])
            best_params = (A[b].copy(), D[b].copy(), float(radius[b]))

    if best_params is None or best_inliers < 0:
        return {"axis_point": [0, 0, 0], "axis_dir": [0, 0, 1], "radius": 0, "height": 0}, float("inf")

    axis_point, axis_dir, radius = best_params

    # Least-squares refinement on inliers
    axis_point, axis_dir, radius = _refine_cylinder(points, axis_point, axis_dir, radius)

    # Compute height from projection range along axis
    vecs = points - axis_point
    proj = vecs @ axis_dir
    height = float(proj.max() - proj.min())

    # Recompute axis_point to be at the center of the projected range
    mid = (proj.max() + proj.min()) / 2.0
    axis_point = axis_point + mid * axis_dir

    # RMS
    proj_along = np.outer(vecs @ axis_dir, axis_dir)
    proj_perp = vecs - proj_along
    dists_from_axis = np.linalg.norm(proj_perp, axis=1)
    rms = float(np.sqrt(np.mean((dists_from_axis - radius) ** 2)))

    params = {
        "axis_point": axis_point.tolist(),
        "axis_dir": axis_dir.tolist(),
        "radius": round(float(radius), 4),
        "height": round(height, 4),
    }
    return params, rms


def _refine_cylinder(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refine cylinder parameters via least-squares."""
    # Parameterize: axis_point (3), axis_dir as (theta, phi) spherical (2), radius (1)
    theta = float(np.arccos(np.clip(axis_dir[2], -1, 1)))
    phi = float(np.arctan2(axis_dir[1], axis_dir[0]))

    def residuals(p):
        ap = p[:3]
        th, ph = p[3], p[4]
        r = p[5]
        ad = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        vecs = points - ap
        proj_along = np.outer(vecs @ ad, ad)
        proj_perp = vecs - proj_along
        return np.linalg.norm(proj_perp, axis=1) - r

    x0 = np.array([*axis_point, theta, phi, radius])
    try:
        if not np.all(np.isfinite(x0)):
            raise ValueError("non-finite x0")
        result = least_squares(residuals, x0, method="trf", max_nfev=200)
        p = result.x
        axis_point = p[:3]
        th, ph = p[3], p[4]
        axis_dir = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        radius = abs(p[5])
    except Exception:
        logger.debug(
            "Cylinder least-squares refinement failed: axis_point=%s, radius=%.4f, %d points",
            axis_point.tolist(), radius, len(points),
        )

    return axis_point, axis_dir, radius


# ---------------------------------------------------------------------------
# Cone fitting: RANSAC + Levenberg-Marquardt
# ---------------------------------------------------------------------------

def _fit_cone(
    points: np.ndarray,
    normals: np.ndarray,
    config: FitConfig,
    inlier_thresh: float,
) -> tuple[dict, float]:
    """RANSAC cone fit: sample 3 points + normals, estimate apex and half-angle."""
    n = len(points)
    if n < 6:
        return {"apex": [0, 0, 0], "axis_dir": [0, 0, 1], "half_angle": 0}, float("inf")

    rng = np.random.RandomState(42)

    # Limit iterations for cones (more expensive)
    iters = min(config.ransac_iterations, 500)

    # Sample all triplets up front; axes via one batched SVD; score all
    # hypotheses against all points with matrix ops (same math as the
    # previous per-iteration loop: axis from the best-fit line through 3
    # samples, then a closed-form linear fit perp_dist = slope*t + b).
    idx = np.array([rng.choice(n, 3, replace=False) for _ in range(iters)])
    tri = points[idx]                                   # (K, 3, 3)
    centroids = tri.mean(axis=1)                        # (K, 3)
    _, _, vh = np.linalg.svd(tri - centroids[:, None, :])
    axis_dirs = vh[:, 0, :]                             # (K, 3)

    p_sq = np.einsum("ij,ij->i", points, points)
    best_params = None
    best_rms = float("inf")
    chunk = max(1, int(2e7 // max(n, 1)))
    for s in range(0, len(axis_dirs), chunk):
        D = axis_dirs[s:s + chunk]                      # (C, 3)
        Cc = centroids[s:s + chunk]                     # (C, 3)
        cd = np.einsum("ij,ij->i", Cc, D)
        c_sq = np.einsum("ij,ij->i", Cc, Cc)
        t = points @ D.T - cd[None, :]                  # (N, C)
        perp_sq = p_sq[:, None] - 2.0 * (points @ Cc.T) + c_sq[None, :] - t ** 2
        np.maximum(perp_sq, 0.0, out=perp_sq)
        perp = np.sqrt(perp_sq)                         # (N, C)

        # Closed-form least squares of perp = slope * t + intercept
        t_mean = t.mean(axis=0)
        perp_mean = perp.mean(axis=0)
        t_c = t - t_mean[None, :]
        var_t = np.einsum("ij,ij->j", t_c, t_c) / n
        cov = np.einsum("ij,ij->j", t_c, perp - perp_mean[None, :]) / n
        ok = np.sqrt(np.maximum(var_t, 0.0)) > 1e-10
        slope = np.divide(cov, var_t, out=np.zeros_like(cov), where=ok)
        ok &= np.abs(slope) > 1e-10
        intercept = perp_mean - slope * t_mean
        t_apex = np.divide(-intercept, slope, out=np.zeros_like(slope), where=ok)

        # Residuals vs the cone surface: expected radius |t - t_apex|*|slope|
        expected_r = np.abs(t - t_apex[None, :]) * np.abs(slope)[None, :]
        rms = np.sqrt(np.mean((perp - expected_r) ** 2, axis=0))
        rms[~ok] = np.inf

        b = int(np.argmin(rms))
        if rms[b] < best_rms:
            best_rms = float(rms[b])
            apex = Cc[b] + t_apex[b] * D[b]
            best_params = (apex, D[b].copy(), float(np.arctan(np.abs(slope[b]))))

    if best_params is None:
        return {"apex": [0, 0, 0], "axis_dir": [0, 0, 1], "half_angle": 0}, float("inf")

    apex, axis_dir, half_angle = best_params

    # Levenberg-Marquardt refinement
    apex, axis_dir, half_angle = _refine_cone(points, apex, axis_dir, half_angle)

    # Recompute RMS
    vecs = points - apex
    t = vecs @ axis_dir
    expected_r = np.abs(t) * np.tan(half_angle)
    actual_r = np.linalg.norm(vecs - np.outer(t, axis_dir), axis=1)
    rms = float(np.sqrt(np.mean((actual_r - expected_r) ** 2)))

    params = {
        "apex": apex.tolist(),
        "axis_dir": axis_dir.tolist(),
        "half_angle": round(float(np.degrees(half_angle)), 4),
    }
    return params, rms


def _refine_cone(
    points: np.ndarray,
    apex: np.ndarray,
    axis_dir: np.ndarray,
    half_angle: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refine cone via Levenberg-Marquardt."""
    theta = float(np.arccos(np.clip(axis_dir[2], -1, 1)))
    phi = float(np.arctan2(axis_dir[1], axis_dir[0]))

    def residuals(p):
        ap = p[:3]
        th, ph = p[3], p[4]
        ha = p[5]
        ad = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        vecs = points - ap
        t = vecs @ ad
        expected_r = np.abs(t) * np.tan(ha)
        actual_r = np.linalg.norm(vecs - np.outer(t, ad), axis=1)
        return actual_r - expected_r

    x0 = np.array([*apex, theta, phi, half_angle])
    try:
        if not np.all(np.isfinite(x0)):
            raise ValueError("non-finite x0")
        result = least_squares(residuals, x0, method="trf", max_nfev=200)
        p = result.x
        apex = p[:3]
        th, ph = p[3], p[4]
        axis_dir = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        half_angle = abs(p[5])
    except Exception:
        logger.debug(
            "Cone least-squares refinement failed: apex=%s, half_angle=%.4f, %d points",
            apex.tolist(), half_angle, len(points),
        )

    return apex, axis_dir, half_angle


# ---------------------------------------------------------------------------
# Parameter snapping
# ---------------------------------------------------------------------------

def compute_residuals(
    points: np.ndarray,
    fitted: FittedSurface,
) -> np.ndarray:
    """Compute signed point-to-surface distances for a fitted surface.

    Returns (N,) array of signed residuals.
    """
    params = fitted.params
    stype = fitted.surface_type

    if stype == SurfaceType.PLANE:
        normal = np.array(params["normal"])
        point = np.array(params["point"])
        return (points - point) @ normal

    elif stype == SurfaceType.SPHERE:
        center = np.array(params["center"])
        radius = params["radius"]
        return np.linalg.norm(points - center, axis=1) - radius

    elif stype == SurfaceType.CYLINDER:
        axis_point = np.array(params["axis_point"])
        axis_dir = np.array(params["axis_dir"])
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)
        radius = params["radius"]
        vecs = points - axis_point
        proj_along = np.outer(vecs @ axis_dir, axis_dir)
        proj_perp = vecs - proj_along
        return np.linalg.norm(proj_perp, axis=1) - radius

    elif stype == SurfaceType.CONE:
        apex = np.array(params["apex"])
        axis_dir = np.array(params["axis_dir"])
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)
        half_angle = np.radians(params["half_angle"])
        vecs = points - apex
        t = vecs @ axis_dir
        expected_r = np.abs(t) * np.tan(half_angle)
        actual_r = np.linalg.norm(vecs - np.outer(t, axis_dir), axis=1)
        return actual_r - expected_r

    return np.zeros(len(points))


def _snap_params(
    params: dict,
    surface_type: SurfaceType,
    config: FitConfig,
) -> dict:
    """Snap fitted parameters to round values and principal axes."""
    snapped = {}
    for key, val in params.items():
        if isinstance(val, (int, float)):
            snapped[key] = _snap_scalar(val, config.snap_tolerance)
        elif isinstance(val, list) and all(isinstance(v, (int, float)) for v in val):
            if "dir" in key or "normal" in key:
                snapped[key] = _snap_direction(val, config.axis_snap_angle_deg)
            else:
                snapped[key] = [_snap_scalar(v, config.snap_tolerance) for v in val]
        else:
            snapped[key] = val
    return snapped


def _snap_scalar(val: float, tolerance: float) -> float:
    """Snap a scalar to the nearest 'round' value if within tolerance.

    Round values: integers, halves (0.5), quarters (0.25).
    """
    candidates = [
        round(val),
        round(val * 2) / 2,
        round(val * 4) / 4,
    ]
    for c in candidates:
        if abs(val - c) < tolerance:
            return c
    return val


def _snap_direction(direction: list[float], angle_threshold_deg: float) -> list[float]:
    """Snap a direction vector to a principal axis if within threshold."""
    d = np.array(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-15:
        return direction
    d = d / norm

    principal_axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([-1, 0, 0]),
        np.array([0, -1, 0]),
        np.array([0, 0, -1]),
    ]

    threshold = np.cos(np.radians(angle_threshold_deg))

    for axis in principal_axes:
        if np.dot(d, axis) >= threshold:
            return axis.tolist()

    return d.tolist()
