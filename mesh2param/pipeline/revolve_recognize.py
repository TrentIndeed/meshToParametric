"""Body-of-revolution recognition (Phase 4 — REVOLVE feature).

The prismatic pipeline (segment -> fit -> extrude) cannot model turned parts
(shafts, flanges, motor housings) whose radius VARIES along an axis. CADFit
handles these via a revolve operator; this is the analogous capability here.

Detection has to work on noisy AI/Meshy meshes whose "circular" sections are
bumpy (radial CoV 0.1-0.25), so a strict circle fit fails. The robust signal is
that a body of revolution is AXISYMMETRIC: its cross-section perpendicular to
the axis has a flat radius profile r(theta) with NO coherent low-order harmonic.
A rounded polygon (hex standoff, L-bracket) instead has a strong 4/6/...-fold
harmonic. So per section we bin r(theta) and measure the strongest harmonic of
orders 2-13: small => round (revolve), large => polygon (prismatic). This
cleanly separates the real revolves (MotorMountPlate, PipeFlange) from the
hex standoff and brackets.

`try_revolve_tree` returns a single-REVOLVE FeatureTree when the body is
axisymmetric AND extrude can't model it (radius varies along the axis, or it's
a tall round body), else None. The outer-silhouette radius per station is the
revolve profile; inner bores/bolt-holes are left to the residual loop. The
CadQuery build lives in export_cadquery._build_revolve_result.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _candidate_axes(mesh) -> list[np.ndarray]:
    """World axes (the part is canonicalised) + the PCA principal direction."""
    axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    try:
        v = np.asarray(mesh.vertices, float)
        d = v - v.mean(0)
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        axes.append(vt[0])
    except Exception:
        pass
    return axes


def _section_metrics(pts2d: np.ndarray, n_bins: int = 72):
    """For a section's 2D boundary points, return
    (outer_radius, max_low_harmonic, radial_noise, center_offset) — or None.

    outer_radius: 95th-percentile distance from the section centroid (the
        silhouette radius; robust to noise spikes and to inner hole points).
    max_low_harmonic: strongest harmonic (orders 2..13) of the binned
        radius profile r(theta), normalised by mean radius. Low => round.
    radial_noise: std/mean of the binned radius (overall bumpiness).
    center_offset: |section centroid| relative to the axis, over radius.
    """
    if len(pts2d) < 16:
        return None
    c = pts2d.mean(0)
    rel = pts2d - c
    r = np.hypot(rel[:, 0], rel[:, 1])
    th = np.arctan2(rel[:, 1], rel[:, 0])
    rmean_all = r.mean()
    if rmean_all <= 1e-6:
        return None
    bins = ((th + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    rb = np.array([np.median(r[bins == k]) if (bins == k).any() else np.nan
                   for k in range(n_bins)])
    if np.isnan(rb).sum() > n_bins // 3:
        return None
    idx = np.arange(n_bins)
    good = ~np.isnan(rb)
    rb = np.interp(idx, idx[good], rb[good], period=n_bins)
    rm = rb.mean()
    if rm <= 1e-6:
        return None
    spectrum = np.abs(np.fft.rfft(rb - rm)) / n_bins / rm
    max_harm = float(spectrum[2:14].max()) if len(spectrum) > 2 else 9.0
    noise = float(rb.std() / rm)
    outer = float(np.percentile(r, 95))
    offset = float(np.linalg.norm(c)) / max(outer, 1e-6)
    return outer, max_harm, noise, offset


def detect_body_of_revolution(cleaned, *, n_slices: int = 20,
                              harmonic_tol: float = 0.05,
                              max_noise: float = 0.22,
                              min_axisym_frac: float = 0.7):
    """Detect a (possibly noisy) solid of revolution.

    Returns (origin(3), axis_dir(3), profile) where profile is an (M,2) array of
    (height_along_axis, outer_radius) samples, or None. A body qualifies when a
    sufficient fraction of its cross-sections are AXISYMMETRIC (low harmonic,
    bounded noise, centred on the axis) AND it genuinely needs a revolve — it's
    tall (aspect >= min_aspect) or its radius varies along the axis
    (min_radial_var), so a plain extrude can't reproduce it.
    """
    try:
        import trimesh
    except Exception:
        return None
    try:
        mesh = trimesh.Trimesh(vertices=np.asarray(cleaned.vertices, float),
                               faces=np.asarray(cleaned.faces), process=False)
    except Exception:
        return None
    if len(mesh.faces) < 8:
        return None
    centroid = np.asarray(mesh.centroid, float)

    best = None
    for axis in _candidate_axes(mesh):
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        proj = (np.asarray(mesh.vertices, float) - centroid) @ axis
        lo, hi = float(proj.min()), float(proj.max())
        span = hi - lo
        if span <= 1e-6:
            continue
        tmp = (np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9
               else np.array([0, 1.0, 0]))
        u = np.cross(axis, tmp); u /= np.linalg.norm(u) + 1e-12
        w = np.cross(axis, u)

        profile = []
        harmonics = []
        n_valid = n_axisym = 0
        for t in np.linspace(0.04, 0.96, n_slices):
            h = lo + t * span
            try:
                sec = mesh.section(plane_origin=centroid + axis * h,
                                   plane_normal=axis)
                if sec is None:
                    continue
                rel = np.asarray(sec.vertices, float) - centroid
                pts2 = np.column_stack([rel @ u, rel @ w])
            except Exception:
                continue
            m = _section_metrics(pts2)
            if m is None:
                continue
            outer, max_harm, noise, offset = m
            n_valid += 1
            if (max_harm <= harmonic_tol and noise <= max_noise
                    and offset <= 0.25):
                n_axisym += 1
                harmonics.append(max_harm)
                profile.append((h, outer))
        if n_valid < n_slices // 2 or len(profile) < 4:
            continue
        frac = n_axisym / max(1, n_valid)
        if frac < min_axisym_frac:
            continue
        prof = np.array(sorted(profile))
        if float(prof[:, 1].max()) <= 1e-6:
            continue
        # No aspect/variation gate: the strict harmonic test already admits
        # only genuinely round bodies, and a flat round flange (a degenerate
        # revolve) is exactly what we want — its prismatic build is buggy.
        score = -float(np.median(harmonics))   # rounder (lower harmonic) = better
        if best is None or score > best[0]:
            best = (score, centroid.copy(), axis.copy(), prof)

    if best is None:
        return None
    _, origin, axis, profile = best
    logger.info("Body of revolution: axis=%s, %d profile samples, r=%.1f..%.1f",
                np.round(axis, 2).tolist(), len(profile),
                float(profile[:, 1].min()), float(profile[:, 1].max()))
    return origin, axis, profile


def try_revolve_tree(cleaned):
    """Return a single-REVOLVE FeatureTree if `cleaned` is a body of revolution
    that needs a revolve, else None."""
    res = detect_body_of_revolution(cleaned)
    if res is None:
        return None
    origin, axis, profile = res
    from mesh2param.models.feature import Feature, FeatureType
    from mesh2param.models.tree import FeatureTree
    feat = Feature(
        type=FeatureType.REVOLVE,
        face_indices=np.arange(len(cleaned.faces)),
        sketch_geometry={
            "type": "revolve",
            "axis_origin": np.asarray(origin, float).tolist(),
            "axis_dir": np.asarray(axis, float).tolist(),
            "profile": np.asarray(profile, float).tolist(),  # [[height, radius]]
        },
        direction=np.asarray(axis, float),
        operation="NEW",
    )
    tree = FeatureTree(features=[feat])
    tree.metadata["revolve"] = True
    return tree
