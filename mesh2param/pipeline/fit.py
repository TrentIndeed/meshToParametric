"""Stage 3: Primitive Fitting.

Fits exact analytic surfaces (plane, cylinder, cone, sphere) to each
segmented region via SVD, RANSAC, and least-squares refinement.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import least_squares

from mesh2param.config import FitConfig
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.region import FaceRegion, SurfaceType
from mesh2param.models.surface import FittedSurface

logger = logging.getLogger(__name__)

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

        # Disambiguate sphere vs cylinder: both can fit hole surfaces with
        # near-zero RMS, but cylinder is the correct type for holes.
        if config.try_alternative_types and fitted.surface_type in (SurfaceType.SPHERE, SurfaceType.CYLINDER):
            fitted = _disambiguate_sphere_cylinder(
                region, points, normals, config, inlier_thresh, fitted,
            )

        # Snap parameters
        fitted.snapped_params = _snap_params(fitted.params, fitted.surface_type, config)

        results.append(fitted)
        logger.info(
            "Region %d: %s, RMS=%.6f (tol=%.6f)",
            region.region_id, fitted.surface_type.value, fitted.rms_error, rms_tol,
        )

    logger.debug("fit_surfaces completed: %d surfaces fitted", len(results))
    return results


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
    """Try fitting with alternative surface types and keep the best."""
    best = current_best

    for stype, fname in _FITTERS.items():
        if stype == current_best.surface_type:
            continue

        try:
            fitter = _get_fitter(stype)
            params, rms = fitter(points, normals, config, inlier_thresh)
            if rms < best.rms_error:
                best = FittedSurface(
                    region_id=region.region_id,
                    surface_type=stype,
                    params=params,
                    rms_error=rms,
                )
                if rms < rms_tol:
                    break
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
        result = least_squares(residuals, x0, method="lm")
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

    best_params = None
    best_inliers = 0
    best_rms = float("inf")
    rng = np.random.RandomState(42)

    use_normals = normals is not None and len(normals) >= n

    for _ in range(config.ransac_iterations):
        # Sample 2 points
        idx = rng.choice(n, 2, replace=False)

        if use_normals:
            # Axis direction from cross product of two face normals
            n1, n2 = normals[idx[0]], normals[idx[1]]
            axis_dir = np.cross(n1, n2)
        else:
            # Fallback: direction between points
            axis_dir = points[idx[1]] - points[idx[0]]

        norm = np.linalg.norm(axis_dir)
        if norm < 1e-10:
            continue
        axis_dir = axis_dir / norm

        # Project all points onto the plane perpendicular to axis
        axis_point = points[idx[0]]
        vecs = points - axis_point
        proj_along = np.outer(vecs @ axis_dir, axis_dir)
        proj_perp = vecs - proj_along
        dists_from_axis = np.linalg.norm(proj_perp, axis=1)

        # Radius = median of distances (robust)
        radius = float(np.median(dists_from_axis))
        if radius < 1e-10:
            continue

        # Count inliers
        residuals = np.abs(dists_from_axis - radius)
        inlier_mask = residuals < inlier_thresh
        n_inliers = int(inlier_mask.sum())

        if n_inliers > best_inliers or (n_inliers == best_inliers and np.mean(residuals[inlier_mask]) < best_rms):
            best_inliers = n_inliers
            best_params = (axis_point.copy(), axis_dir.copy(), radius)
            best_rms = float(np.mean(residuals[inlier_mask])) if n_inliers > 0 else float("inf")

    if best_params is None:
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
        result = least_squares(residuals, x0, method="lm", max_nfev=200)
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

    best_params = None
    best_rms = float("inf")
    rng = np.random.RandomState(42)

    # Limit iterations for cones (more expensive)
    iters = min(config.ransac_iterations, 500)

    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]

        # Estimate axis as direction of best-fit line through sampled points
        centroid = (p1 + p2 + p3) / 3
        centered = np.array([p1 - centroid, p2 - centroid, p3 - centroid])
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_dir = vh[0]

        # Project points onto axis to estimate apex
        vecs = points - centroid
        t = vecs @ axis_dir
        perp_dists = np.linalg.norm(vecs - np.outer(t, axis_dir), axis=1)

        # Fit a linear model: perp_dist = |slope| * t + intercept
        # This gives us the half angle: tan(half_angle) = slope
        if np.std(t) < 1e-10:
            continue

        A = np.column_stack([t, np.ones(n)])
        try:
            result_lr = np.linalg.lstsq(A, perp_dists, rcond=None)
            slope, intercept = result_lr[0]
        except Exception:
            continue

        if abs(slope) < 1e-10:
            continue

        half_angle = float(np.arctan(abs(slope)))
        # Apex is where perp_dist = 0: t_apex = -intercept / slope
        t_apex = -intercept / slope
        apex = centroid + t_apex * axis_dir

        # Compute residuals
        vecs_apex = points - apex
        t_from_apex = vecs_apex @ axis_dir
        expected_r = np.abs(t_from_apex) * np.tan(half_angle)
        actual_r = np.linalg.norm(vecs_apex - np.outer(t_from_apex, axis_dir), axis=1)
        residuals = np.abs(actual_r - expected_r)
        rms = float(np.sqrt(np.mean(residuals ** 2)))

        if rms < best_rms:
            best_rms = rms
            best_params = (apex.copy(), axis_dir.copy(), half_angle)

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
        result = least_squares(residuals, x0, method="lm", max_nfev=200)
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
