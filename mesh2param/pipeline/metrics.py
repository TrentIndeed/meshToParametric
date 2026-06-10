"""Per-feature accuracy metrics.

Compares recognized features against reference CAD geometry (cylindrical
faces extracted from a reference STEP). Aggregate volume error hides
feature-level failures — a hole fitted at 6.4mm instead of 4mm barely
moves total volume but makes the part unusable. These metrics surface
that directly.

This module is pure numpy so it can run in-process anywhere; extracting
the reference cylinders requires cadquery/OCP and is done by the caller
(see tests/test_step_comparison.py for the subprocess pattern).
"""

from __future__ import annotations

import logging

import numpy as np

from mesh2param.models.feature import FeatureType
from mesh2param.models.tree import FeatureTree

logger = logging.getLogger(__name__)

HOLE_TYPES = (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)


def recognized_holes(tree: FeatureTree) -> list[dict]:
    """Flatten the tree's hole features into per-bore records.

    Grouped symmetric holes (sketch_geometry["centers"]) expand into one
    record per center, all sharing the group radius.
    """
    holes: list[dict] = []
    for feat in tree.features:
        if feat.type not in HOLE_TYPES:
            continue
        sg = feat.sketch_geometry or {}
        if sg.get("type") != "circle":
            continue
        radius = float(sg.get("radius", 0.0))
        axis = np.asarray(feat.direction, dtype=float)
        norm = float(np.linalg.norm(axis))
        if radius <= 0 or norm < 1e-12:
            continue
        axis = axis / norm
        centers = sg.get("centers") or ([sg["center"]] if sg.get("center") is not None else [])
        for c in centers:
            holes.append({
                "type": feat.type.value,
                "center": np.asarray(c, dtype=float),
                "radius": radius,
                "axis": axis,
            })
    return holes


def match_holes_to_reference(
    holes: list[dict],
    ref_cylinders: list[dict],
    axis_dot_min: float = 0.95,
    axis_dist_max: float = 2.0,
) -> dict:
    """Match recognized holes to reference cylindrical faces.

    Args:
        holes: output of recognized_holes().
        ref_cylinders: list of {"radius", "axis_dir", "axis_point"} dicts
            for every cylindrical face in the reference STEP (bores,
            fillet barrels, counterbores — all of them; matching picks
            the right one per hole).
        axis_dot_min: minimum |cos| between hole axis and ref axis.
        axis_dist_max: maximum distance (model units) between the hole
            center and the reference axis line.

    Returns a report dict:
        matched: list of {hole, ref_radius, radius_error, axis_dist}
        unmatched: list of holes with no colocated reference cylinder
        max_radius_error / mean_radius_error over matched holes
    """
    matched: list[dict] = []
    unmatched: list[dict] = []

    refs = []
    for rc in ref_cylinders:
        d = np.asarray(rc["axis_dir"], dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            continue
        refs.append({
            "radius": float(rc["radius"]),
            "axis_dir": d / n,
            "axis_point": np.asarray(rc["axis_point"], dtype=float),
        })

    for hole in holes:
        # Candidacy is positional only: the hole center must sit on (or
        # near) the reference cylinder's axis line. The hole's own axis
        # direction is deliberately NOT a filter — a recognition bug
        # that assigns the wrong axis should show up as a bad axis_dot
        # on a matched hole, not as a silently unmatched hole.
        candidates = []
        for ref in refs:
            v = hole["center"] - ref["axis_point"]
            perp = v - np.dot(v, ref["axis_dir"]) * ref["axis_dir"]
            axis_dist = float(np.linalg.norm(perp))
            if axis_dist > axis_dist_max:
                continue
            candidates.append((abs(hole["radius"] - ref["radius"]), axis_dist, ref))
        if not candidates:
            unmatched.append({
                "type": hole["type"],
                "center": hole["center"].tolist(),
                "radius": hole["radius"],
            })
            continue
        # Several reference cylinders share a bore's axis (the bore, its
        # rim fillet, a counterbore). Pick the closest radius among the
        # colocated ones — large fitted-radius errors still surface
        # because no reference cylinder sits near the bad value.
        candidates.sort(key=lambda c: (c[0], c[1]))
        radius_error, axis_dist, ref = candidates[0]
        axis_dot = abs(float(np.dot(hole["axis"], ref["axis_dir"])))
        matched.append({
            "type": hole["type"],
            "center": hole["center"].tolist(),
            "radius": hole["radius"],
            "ref_radius": ref["radius"],
            "radius_error": radius_error,
            "axis_dist": axis_dist,
            "axis_dot": axis_dot,
            "axis_ok": axis_dot >= axis_dot_min,
        })

    radius_errors = [m["radius_error"] for m in matched]
    return {
        "n_holes": len(holes),
        "n_matched": len(matched),
        "n_unmatched": len(unmatched),
        "n_axis_wrong": sum(1 for m in matched if not m["axis_ok"]),
        "matched": matched,
        "unmatched": unmatched,
        "max_radius_error": max(radius_errors) if radius_errors else 0.0,
        "mean_radius_error": float(np.mean(radius_errors)) if radius_errors else 0.0,
    }


def blend_apply_report(fillet_match_diag: list[dict]) -> dict:
    """Fillet clusters recognized vs actually matched to a CAD edge.

    Takes CadQueryResult.fillet_match_diag (one record per stitched
    cluster, with a ``matched`` flag). A low match ratio means
    recognized blends silently vanished from the output — historically
    the user only discovered this by inspecting the STEP in CAD.
    """
    diag = fillet_match_diag or []
    recognized = len(diag)
    matched = sum(1 for d in diag if d.get("matched"))
    return {
        "recognized": recognized,
        "matched": matched,
        "match_ratio": (matched / recognized) if recognized else None,
        "unmatched": [d for d in diag if not d.get("matched")],
    }
