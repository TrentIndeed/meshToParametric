"""Stage 6b: CadQuery Export.

Alternative offline export path that reconstructs the recognized part as a
CadQuery solid and exports a STEP file. No Onshape API calls required.

Two construction strategies:
- **LLM-guided** (default when Claude CLI is available): sends the feature tree
  and mesh geometry to Claude, which returns an optimal CadQuery construction
  script with proper build order and best practices.
- **Heuristic fallback**: cross-section profile → extrude → cut holes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from mesh2param.models.feature import Feature, FeatureType
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.tree import FeatureTree

logger = logging.getLogger(__name__)


@dataclass
class ConstructionStep:
    """A single step in the CAD construction process, with tessellated geometry."""

    name: str
    tess_vertices: np.ndarray  # (V, 3)
    tess_faces: np.ndarray  # (F, 3)
    volume: float = 0.0


@dataclass
class CadQueryResult:
    """Output of the CadQuery export pipeline."""

    solid: object | None = None  # cq.Workplane, lazy-imported
    tess_vertices: np.ndarray | None = None  # (V, 3) tessellated solid verts
    tess_faces: np.ndarray | None = None  # (F, 3) tessellated solid faces
    construction_steps: list[ConstructionStep] = field(default_factory=list)
    script: str = ""
    step_path: str = ""
    mesh_volume: float = 0.0
    cad_volume: float = 0.0
    volume_error_pct: float = 0.0
    status: str = "PASS"

    def to_dict(self) -> dict:
        return {
            "mesh_volume": round(self.mesh_volume, 2),
            "cad_volume": round(self.cad_volume, 2),
            "volume_error_pct": round(self.volume_error_pct, 2),
            "status": self.status,
            "step_path": self.step_path,
            "script_lines": self.script.count("\n") + 1,
            "construction_steps": len(self.construction_steps),
        }


def export_cadquery(
    feature_tree: FeatureTree,
    cleaned: CleanedMesh,
    output_path: str | Path | None = None,
    use_llm: bool = True,
) -> CadQueryResult:
    """Reconstruct the part as a CadQuery solid and optionally export STEP.

    When use_llm=True and the Claude CLI is available, sends the feature tree
    to Claude for optimal construction order and best-practice CadQuery code.
    Falls back to heuristic construction if the CLI is unavailable.

    Args:
        feature_tree: Output of Stage 5 (Feature Recognition).
        cleaned: Output of Stage 1 (for mesh geometry and volume).
        output_path: Path for the .step file. None skips file export.
        use_llm: Try LLM-guided construction first (default True).

    Returns:
        CadQueryResult with the solid, generated script, and validation.
    """
    logger.debug(
        "export_cadquery called: %d features, use_llm=%s, output_path=%s",
        feature_tree.num_features, use_llm, output_path,
    )
    mesh_volume = _estimate_mesh_volume(cleaned)

    # Extract base profile (needed by both LLM and heuristic paths)
    base_feature = feature_tree.features[0]
    extrude_dir = np.array(base_feature.direction)
    depth = base_feature.depth

    try:
        profiles, extrude_start, hole_features, chamfer_features, fillet_features = \
            _extract_profile_and_holes(feature_tree, cleaned, extrude_dir)
    except Exception:
        logger.exception(
            "Stage 6b failed extracting profile: extrude_dir=%s, depth=%.2f",
            extrude_dir.tolist(), depth,
        )
        raise

    if profiles is None or len(profiles) == 0:
        logger.error("Failed to extract base profile from mesh")
        return CadQueryResult(mesh_volume=mesh_volume, status="FAIL")

    # Get construction order — LLM-guided or default
    build_order = None
    if use_llm:
        build_order = _llm_plan_build_order(feature_tree, cleaned)

    # Build the CadQuery solid step-by-step:
    # 1. Sketch profile
    # 2. Extrude base
    # 3+. Cut holes
    # N+. Apply chamfers and fillets
    logger.debug(
        "Building CadQuery solid: %d profiles, depth=%.2f, %d holes, %d chamfers, %d fillets",
        len(profiles), depth, len(hole_features), len(chamfer_features), len(fillet_features),
    )
    try:
        solid, script, steps = _build_stepwise(
            profiles, depth, extrude_dir, extrude_start,
            hole_features, chamfer_features, fillet_features,
            build_order=build_order, cleaned=cleaned,
        )
    except Exception:
        logger.exception(
            "Stage 6b failed during CadQuery solid construction: profiles=%d, holes=%d",
            len(profiles), len(hole_features),
        )
        raise

    # Compute CAD volume
    try:
        import cadquery as cq
        cad_volume = abs(float(solid.val().Volume()))
    except Exception:
        logger.debug("Could not compute CAD volume from solid, defaulting to 0.0")
        cad_volume = 0.0

    volume_error = abs(cad_volume - mesh_volume) / (mesh_volume + 1e-15) * 100
    status = "PASS" if volume_error < 10.0 else "FAIL"

    # Final tessellation
    tess_verts, tess_faces = _tessellate_solid(solid)

    # Export STEP
    step_path = ""
    if output_path is not None:
        try:
            import cadquery as cq
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cq.exporters.export(solid, str(output_path))
            step_path = str(output_path)
            logger.info("Exported STEP to %s", step_path)
        except Exception:
            logger.exception("STEP export failed: output_path=%s", output_path)
            raise

    logger.info(
        "CadQuery export: mesh_vol=%.1f, cad_vol=%.1f, error=%.2f%%, status=%s",
        mesh_volume, cad_volume, volume_error, status,
    )

    result = CadQueryResult(
        solid=solid,
        tess_vertices=tess_verts,
        tess_faces=tess_faces,
        construction_steps=steps,
        script=script,
        step_path=step_path,
        mesh_volume=mesh_volume,
        cad_volume=cad_volume,
        volume_error_pct=volume_error,
        status=status,
    )

    # Visual verification via LLM (compares CAD result to original mesh)
    if use_llm and tess_verts is not None:
        try:
            verification = _llm_visual_verify(result, cleaned)
            if verification and not verification.get("pass", True):
                logger.warning(
                    "LLM visual verification FAILED: %s",
                    verification.get("issues", "unknown"),
                )
                result.status = "VISUAL_MISMATCH"
        except Exception as e:
            logger.info("Visual verification skipped: %s", e)

    return result


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------


def _extract_profile_and_holes(
    feature_tree: FeatureTree,
    cleaned: CleanedMesh,
    extrude_dir: np.ndarray,
) -> tuple[list[list[tuple[float, float]]] | None, float, list[Feature]]:
    """Extract ALL 2D cross-section profiles and collect hole features.

    Slices the mesh and returns every polygon (main body + leg extensions
    + bosses) so the builder can extrude and union them all.

    Returns (list_of_profiles, extrude_start, holes).
    """
    logger.debug(
        "_extract_profile_and_holes: extrude_dir=%s, %d features",
        extrude_dir.tolist(), feature_tree.num_features,
    )
    mesh = trimesh.Trimesh(
        vertices=cleaned.vertices, faces=cleaned.faces, process=False,
    )

    abs_dir = np.abs(extrude_dir)
    primary_axis = int(np.argmax(abs_dir))
    axes = [0, 1, 2]
    axes.remove(primary_axis)
    ax_u, ax_v = axes

    all_proj = cleaned.vertices[:, primary_axis]
    extrude_start = float(all_proj.min())

    normal = [0.0, 0.0, 0.0]
    normal[primary_axis] = 1.0

    # Collect profiles from multiple slices. Pick the slice whose extruded
    # volume is closest to the actual mesh volume — this handles tapered parts
    # better than just picking the largest cross-section.
    mesh_volume = abs(float(mesh.volume)) if mesh.is_watertight else 0
    extent = float(all_proj.max() - all_proj.min())

    candidates: list[tuple[float, list]] = []  # (vol_error, profiles)

    # Slice at the very top and bottom faces — just 0.1mm inside the surface.
    # This captures the true outer profile without any hole interference.
    surface_inset = 0.1  # mm inside the surface
    slice_positions = [
        float(all_proj.min()) + surface_inset,  # bottom face
        float(all_proj.max()) - surface_inset,  # top face
    ]
    for slice_pos in slice_positions:
        origin = [0.0, 0.0, 0.0]
        origin[primary_axis] = slice_pos

        try:
            section = mesh.section(plane_origin=origin, plane_normal=normal)
            if section is None:
                continue
            section_verts = np.array(section.vertices)
            s2d, to_planar_tform = section.to_planar()
            polygons = s2d.polygons_closed
            if len(polygons) == 0:
                continue

            total_area = sum(p.area for p in polygons)
            if total_area < 1.0:
                continue

            # Build profiles by projecting section vertices to (ax_u, ax_v).
            # s2d.vertices correspond 1:1 with section_verts, so use s2d
            # polygon topology but get coordinates from 3D projection.
            from scipy.spatial import cKDTree
            from shapely.geometry import Polygon as SPolyProj
            from shapely.ops import unary_union

            s2d_tree = cKDTree(s2d.vertices)

            projected_polys = []
            for poly in polygons:
                coords_2d = np.array(poly.exterior.coords)[:-1]
                _, s2d_indices = s2d_tree.query(coords_2d)
                pts_3d = section_verts[s2d_indices]
                proj_coords = [(float(p[ax_u]), float(p[ax_v])) for p in pts_3d]
                if len(proj_coords) >= 3:
                    try:
                        sp = SPolyProj(proj_coords)
                        if sp.is_valid and sp.area > 1.0:
                            projected_polys.append(sp)
                    except Exception:
                        pass

            # Build profiles: largest polygon = outer boundary.
            # Other large polygons that are INSIDE the outer = cutouts.
            # Small polygons near the outer boundary = split pieces to union.
            if projected_polys:
                projected_polys.sort(key=lambda p: p.area, reverse=True)
                outer = projected_polys[0]

                # Check remaining polygons:
                # - Large + inside outer = cutout (window/pocket)
                # - Small + inside outer = hole cross-section, DROP it
                # - Outside outer = split piece, union to extend boundary
                cutouts = []
                to_union = [outer]
                for sp in projected_polys[1:]:
                    centroid = sp.centroid
                    if outer.contains(centroid):
                        if sp.area > outer.area * 0.05:
                            # Large polygon inside → cutout
                            cutouts.append(sp)
                        # else: small polygon inside → hole cross-section, skip
                    else:
                        # Outside the outer → union to extend boundary
                        to_union.append(sp)

                # Union the boundary pieces
                merged = unary_union(to_union)
                if merged.is_empty:
                    continue
                if merged.geom_type == 'MultiPolygon':
                    merged = merged.buffer(0.5).buffer(-0.3)
                    if merged.geom_type == 'MultiPolygon':
                        merged = max(merged.geoms, key=lambda g: g.area)

                # Subtract cutouts from the merged boundary
                for cut in cutouts:
                    merged = merged.difference(cut)
                    if merged.is_empty:
                        break

                if merged.is_empty:
                    continue
                if merged.geom_type == 'MultiPolygon':
                    merged = max(merged.geoms, key=lambda g: g.area)

                outer_coords = list(merged.exterior.coords)[:-1]
                outer_coords = _simplify_profile(outer_coords, tolerance=0.1)
                slice_profiles = [outer_coords] if len(outer_coords) >= 3 else []
                # Add inner rings as separate profiles (cutouts)
                for ring in merged.interiors:
                    inner_coords = list(ring.coords)[:-1]
                    inner_coords = _simplify_profile(inner_coords, tolerance=0.1)
                    if len(inner_coords) >= 3:
                        slice_profiles.append(inner_coords)
            else:
                slice_profiles = []

            if not slice_profiles:
                continue

            # Score: prefer the slice with the largest profile area.
            # For chamfered parts, this picks the base (widest) profile
            # rather than the chamfered (narrower) side.
            profile_area = _polygon_area(slice_profiles[0])
            candidates.append((-profile_area, slice_profiles))
        except Exception:
            continue

    if not candidates:
        return None, extrude_start, [], [], []

    # Pick best candidate
    candidates.sort(key=lambda x: x[0])
    best_profiles = candidates[0][1]

    # If the best slice has multiple large outer polygons (frame split by
    # cutout), find a slice where the frame is one connected piece and
    # combine: outer boundary from connected slice + inner cutouts from
    # the best slice. This handles frames/bezels with through-cutouts.
    if len(best_profiles) > 1:
        largest_area = max(_polygon_area(p) for p in best_profiles)
        total_area = sum(_polygon_area(p) for p in best_profiles)
        if largest_area < total_area * 0.7:
            # Frame is split — find a nearby slice where the frame is one
            # connected piece but total area is similar (not a filled cutout).
            from shapely.geometry import Polygon as SPoly, MultiPolygon
            from shapely.ops import unary_union

            # Search from near-center outward for a connected slice
            outer_profile_found = None
            for edge_frac in [0.65, 0.60, 0.70, 0.55, 0.75, 0.40, 0.35]:
                edge_pos = float(all_proj.min()) + extent * edge_frac
                edge_origin = [0.0, 0.0, 0.0]
                edge_origin[primary_axis] = edge_pos
                try:
                    edge_section = mesh.section(
                        plane_origin=edge_origin, plane_normal=normal,
                    )
                    if edge_section is None:
                        continue
                    edge_2d, _ = edge_section.to_planar()
                    edge_polys = edge_2d.polygons_closed
                    if len(edge_polys) == 0:
                        continue
                    edge_areas = sorted([p.area for p in edge_polys], reverse=True)
                    edge_total = sum(edge_areas)
                    # Connected: one polygon dominates AND total area similar
                    if edge_areas[0] > edge_total * 0.85 and edge_total < total_area * 2.0:
                        outer_poly = max(edge_polys, key=lambda p: p.area)
                        from scipy.spatial import cKDTree as cKDTree3
                        edge_verts = np.array(edge_section.vertices)
                        edge_tree = cKDTree3(edge_2d.vertices)
                        coords_2d = np.array(outer_poly.exterior.coords)[:-1]
                        _, indices = edge_tree.query(coords_2d)
                        matched = edge_verts[indices]
                        outer_profile_found = [
                            (float(p[ax_u]), float(p[ax_v])) for p in matched
                        ]
                        outer_profile_found = _simplify_profile(
                            outer_profile_found, tolerance=0.1,
                        )
                        logger.info(
                            "Found connected slice at %.0f%% (%d pts)",
                            edge_frac * 100, len(outer_profile_found),
                        )
                        break
                except Exception:
                    continue

            # Bbox for fallback
            u_min = float(cleaned.bbox[0][ax_u])
            u_max = float(cleaned.bbox[1][ax_u])
            v_min = float(cleaned.bbox[0][ax_v])
            v_max = float(cleaned.bbox[1][ax_v])
            bbox_profile = [
                (u_min, v_min), (u_max, v_min),
                (u_max, v_max), (u_min, v_max),
            ]

            if outer_profile_found and len(outer_profile_found) >= 3:
                # Use connected outer boundary + mid-height cutouts
                outer_shape = SPoly(outer_profile_found)
                best_polys = [
                    SPoly(p).buffer(0) for p in best_profiles
                    if len(p) >= 3 and _polygon_area(p) > total_area * 0.05
                    and SPoly(p).is_valid
                ]
                if best_polys:
                    mid_solid = unary_union(best_polys)
                    diff = outer_shape.difference(mid_solid)
                    if not diff.is_empty:
                        diff_clean = diff.buffer(-2.0, join_style='mitre', mitre_limit=5.0)
                        diff_clean = diff_clean.buffer(2.0, join_style='mitre', mitre_limit=5.0)
                        if not diff_clean.is_empty:
                            diff = diff_clean
                else:
                    diff = SPoly()
                bbox_profile = outer_profile_found
                logger.info("Used connected slice for outer + mid-height cutout")
            else:
                # Fallback: bbox + morph-close
                bbox_poly = SPoly(bbox_profile)
                best_polys = [
                    SPoly(p).buffer(0) for p in best_profiles
                    if len(p) >= 3 and _polygon_area(p) > total_area * 0.05
                    and SPoly(p).is_valid
                ]
                if best_polys:
                    merged = unary_union(best_polys)
                    merged_clean = merged.buffer(4.0, join_style='mitre', mitre_limit=5.0)
                    merged_clean = merged_clean.buffer(-4.0, join_style='mitre', mitre_limit=5.0)
                    merged_clean = merged_clean.intersection(bbox_poly)
                    if merged_clean.is_empty:
                        merged_clean = merged
                    diff = bbox_poly.difference(merged_clean)
                else:
                    diff = SPoly()
                logger.info("Used bbox fallback for split frame")

            # Build profile list: outer boundary + inner cutouts
            combined = [bbox_profile]
            geoms = [diff] if diff.geom_type == 'Polygon' else (
                list(diff.geoms) if hasattr(diff, 'geoms') else []
            )
            for geom in geoms:
                if geom.geom_type != 'Polygon' or geom.area < 5.0:
                    continue
                coords = list(geom.exterior.coords)[:-1]
                simplified = _simplify_profile(coords, tolerance=0.1)
                if len(simplified) >= 3:
                    combined.append(simplified)
            combined.sort(key=lambda pts: -_polygon_area(pts))
            best_profiles = combined

    if not best_profiles:
        return None, extrude_start, []

    # Sort largest first (main body first, then smaller features)
    best_profiles.sort(key=lambda pts: -_polygon_area(pts))

    holes = [
        f for f in feature_tree.features
        if f.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)
    ]
    chamfers = [
        f for f in feature_tree.features
        if f.type == FeatureType.CHAMFER
    ]
    fillets = [
        f for f in feature_tree.features
        if f.type == FeatureType.FILLET
    ]

    # Note: profile smoothing removed — the polygon union step above
    # already handles merging split cross-sections. Smoothing via
    # convex hull was destroying concave shapes (L-brackets, etc.).

    # Filter out small inner profiles that are hole cross-sections.
    # These appear when the slice cuts through holes in the mesh.
    # Keep only: the outer profile (largest) and large inner cutouts
    # (like windows/pockets). Remove small circles that match hole sizes.
    if len(best_profiles) > 1 and holes:
        hole_radii = set()
        for h in holes:
            r = h.sketch_geometry.get("radius", 0) if h.sketch_geometry else 0
            if r > 0:
                hole_radii.add(round(r, 1))

        filtered = [best_profiles[0]]  # always keep outer profile
        for p in best_profiles[1:]:
            area = abs(_polygon_area(p))
            # Estimate radius if this were a circle: A = pi*r^2
            est_radius = (area / 3.14159) ** 0.5 if area > 0 else 0
            # Skip if this profile matches a hole radius (within 50%)
            is_hole = any(abs(est_radius - hr) < hr * 0.5 for hr in hole_radii)
            if is_hole and area < _polygon_area(best_profiles[0]) * 0.05:
                continue  # skip — this is a hole cross-section
            filtered.append(p)
        best_profiles = filtered

    return best_profiles, extrude_start, holes, chamfers, fillets


def _polygon_area(pts: list[tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _nudge_hole_into_solid(
    wp: object,
    center: np.ndarray,
    h_dir: np.ndarray,
    radius: float,
    is_through: bool,
    cut_length: float,
    h_depth: float,
    bbox_corners: np.ndarray,
    cq,
) -> object | None:
    """Nudge a hole center toward the solid until the cut removes material.

    When a hole's center is outside the extruded solid (e.g., on a
    flange not captured by the profile), we find the nearest point on
    the solid's B-rep surface and shift the hole center toward it.
    This avoids adding pad geometry (which inflates volume) while still
    producing a functional hole cut.

    Returns the workplane after cutting, or None on failure.
    """
    try:
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCP.gp import gp_Pnt

        # Find the nearest point on the solid surface to the hole center
        vertex = BRepBuilderAPI_MakeVertex(
            gp_Pnt(float(center[0]), float(center[1]), float(center[2]))
        ).Shape()
        dist_calc = BRepExtrema_DistShapeShape(
            vertex, wp.val().wrapped,
        )
        if not dist_calc.IsDone() or dist_calc.NbSolution() == 0:
            return None

        nearest_pt = dist_calc.PointOnShape2(1)
        nearest = np.array([nearest_pt.X(), nearest_pt.Y(), nearest_pt.Z()])

        # Compute the shift vector: from center toward nearest point,
        # but only in the plane perpendicular to the hole axis.
        # This preserves the hole direction while moving it sideways.
        shift = nearest - center
        along_axis = np.dot(shift, h_dir) * h_dir
        shift_perp = shift - along_axis  # component perpendicular to axis

        shift_dist = float(np.linalg.norm(shift_perp))
        if shift_dist < 0.01:
            # The center is already close to the solid along the
            # perpendicular; the issue might be along the axis.
            return None

        # Nudge the center past the nearest point and deep enough that
        # the full hole circle (center ± radius) is inside the solid.
        # shift_dist gets us TO the surface; we need radius more to be
        # fully embedded.
        nudge_extra = radius + 1.0
        if shift_dist > 0:
            nudge_dir = shift_perp / shift_dist
            nudged_center = center + nudge_dir * (shift_dist + nudge_extra)
        else:
            return None

        logger.info(
            "Nudged hole center by %.1fmm: [%.1f,%.1f,%.1f] -> [%.1f,%.1f,%.1f]",
            shift_dist, center[0], center[1], center[2],
            nudged_center[0], nudged_center[1], nudged_center[2],
        )

        # Recompute cut geometry with nudged center
        proj = bbox_corners @ h_dir
        solid_min = float(proj.min())
        solid_max = float(proj.max())
        nc_along = float(nudged_center @ h_dir)

        if is_through:
            new_cut_start = nudged_center + h_dir * (solid_min - nc_along - 1)
            new_cut_end = nudged_center + h_dir * (solid_max - nc_along + 1)
            new_cut_length = float(np.linalg.norm(new_cut_end - new_cut_start))
        else:
            new_cut_start = nudged_center - h_dir * (h_depth / 2) - h_dir * 1.0
            new_cut_length = h_depth + 2.0

        hole_plane = cq.Plane(
            origin=cq.Vector(*new_cut_start.tolist()),
            normal=cq.Vector(*h_dir.tolist()),
        )
        hole_solid = cq.Workplane(hole_plane).circle(radius).extrude(new_cut_length)
        return wp.cut(hole_solid)

    except Exception as e:
        logger.warning("Nudge failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# CadQuery solid construction
# ---------------------------------------------------------------------------


@dataclass
class BuildPlan:
    """LLM-generated construction plan for the CadQuery solid."""

    feature_order: list[int]  # feature indices in construction order
    notes: dict[int, str] = field(default_factory=dict)  # per-feature notes
    strategy: str = ""  # overall construction strategy description


def _llm_visual_verify(
    result: CadQueryResult,
    cleaned: CleanedMesh,
) -> dict | None:
    """Render the CAD result and original mesh, ask Claude to compare them.

    Renders both from 3 angles (top-iso, front, bottom), composites into
    a comparison image, and sends to Claude for visual QA.

    Returns dict with {"pass": bool, "issues": str, "score": int} or None.
    """
    from mesh2param.ai import is_available, call_ai_vision

    if not is_available():
        return None

    import tempfile
    import os

    # Render comparison images using matplotlib (no VTK needed)
    try:
        img_path = _render_comparison_images(result, cleaned)
    except Exception as e:
        logger.info("Could not render comparison images: %s", e)
        return None

    prompt = """Compare these two 3D models shown side by side from multiple angles.

LEFT = Original mesh (input)
RIGHT = Reconstructed CAD model (output)

Check for these issues:
1. Missing features (holes, chamfers, fillets not present in CAD)
2. Wrong proportions (CAD too thin/thick/wide/narrow)
3. Hollow surfaces (faces that should be solid appear as wireframe/transparent)
4. Extra geometry not in the original
5. Misplaced features (holes in wrong position)

Volume comparison is already done numerically — focus on VISUAL shape match.

Respond with ONLY this JSON:
{
  "pass": true/false,
  "score": 1-10 (10 = perfect match),
  "issues": "comma-separated list of issues, or 'none'"
}"""

    try:
        response = call_ai_vision(prompt, [img_path], timeout=90)
        # Parse JSON
        text = response.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])

        import json
        data = json.loads(text)
        logger.info(
            "Visual verification: pass=%s, score=%s, issues=%s",
            data.get("pass"), data.get("score"), data.get("issues"),
        )
        return data

    except Exception as e:
        logger.info("Visual verification LLM call failed: %s", e)
        return None
    finally:
        try:
            os.unlink(img_path)
        except Exception:
            pass


def _render_comparison_images(
    result: CadQueryResult,
    cleaned: CleanedMesh,
) -> str:
    """Render original mesh and CAD result side by side from 3 angles.

    Uses matplotlib's 3D projection (no VTK/PyVista needed).
    Returns path to the saved comparison image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import tempfile

    final = result.construction_steps[-1] if result.construction_steps else None
    if final is None or final.tess_vertices is None:
        raise ValueError("No tessellated CAD result")

    # Camera angles: (elevation, azimuth)
    angles = [
        (30, -60, "Top-Iso"),
        (0, -90, "Front"),
        (-30, -120, "Bottom-Iso"),
    ]

    fig, axes = plt.subplots(len(angles), 2, figsize=(14, 5 * len(angles)),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("#1A1A2E")

    def _plot_mesh(ax, verts, faces, title, elev, azim):
        ax.set_facecolor("#1A1A2E")
        tris = verts[faces]
        pc = Poly3DCollection(tris, alpha=0.9, edgecolor="#2D3436",
                              linewidth=0.1, facecolor="#8899AA")
        ax.add_collection3d(pc)

        # Set limits from data
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        center = (mins + maxs) / 2
        span = (maxs - mins).max() / 2 * 1.2
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, color="#E2E8F0", fontsize=10, pad=2)
        ax.set_axis_off()

    for row, (elev, azim, label) in enumerate(angles):
        # Original mesh (left)
        _plot_mesh(axes[row, 0], cleaned.vertices, cleaned.faces,
                   f"Original - {label}", elev, azim)
        # CAD result (right)
        _plot_mesh(axes[row, 1], final.tess_vertices, final.tess_faces,
                   f"CAD - {label}", elev, azim)

    plt.suptitle("Visual Verification: Original vs CAD",
                 color="#E2E8F0", fontsize=14, y=0.98)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def _llm_plan_build_order(
    feature_tree: FeatureTree,
    cleaned: CleanedMesh,
) -> BuildPlan | None:
    """Ask Claude for the optimal feature construction order.

    The LLM receives the feature list and part geometry, and returns
    a JSON build plan with ordered feature indices and per-feature notes.
    The actual CadQuery code generation stays deterministic.

    Returns a BuildPlan or None if the LLM is unavailable.
    """
    from mesh2param.ai import is_available, call_ai

    if not is_available():
        logger.info("Claude CLI not available — using default feature order")
        return None

    # Build feature descriptions
    feature_lines = []
    for i, feat in enumerate(feature_tree.features):
        line = f"F{i}: type={feat.type.value}, op={feat.operation}, depth={feat.depth:.2f}"
        line += f", dir=[{', '.join(f'{d:.2f}' for d in feat.direction)}]"
        line += f", faces={len(feat.face_indices)}, depends_on={feat.dependencies}"
        if feat.sketch_geometry:
            s = feat.sketch_geometry
            if s.get("type") == "circle":
                line += f", circle r={s['radius']}, center=[{', '.join(f'{c:.1f}' for c in s['center'])}]"
        feature_lines.append(line)

    prompt = f"""You are a CAD manufacturing expert. Given these recognized features from a prismatic
mechanical part, determine the optimal construction order for building it in CadQuery.

## Part
- Bounding box: {cleaned.bbox[0].tolist()} to {cleaned.bbox[1].tolist()}
- Total features: {feature_tree.num_features}

## Features
{chr(10).join(feature_lines)}

## Rules
- BASE_EXTRUSION must always be first (index 0)
- After the base: apply subtractive features (holes, pockets, slots) before additive (bosses)
- Group holes by direction — cut all Y-axis holes together, then Z-axis holes, etc.
- Larger holes before smaller ones (more stable boolean operations)
- Fillets and chamfers must come last (after the edges they reference exist)
- Respect dependency constraints

## Response format — ONLY valid JSON, nothing else:
{{
  "strategy": "one sentence describing the build approach",
  "order": [0, 3, 1, 2, ...],
  "notes": {{
    "0": "base extrusion along Y axis",
    "3": "cut largest through-hole first for cleaner booleans"
  }}
}}"""

    try:
        logger.info("Requesting build order from Claude CLI...")
        response = call_ai(prompt, model="claude-sonnet-4-6", timeout=60)

        # Parse JSON from response (strip markdown fences if present)
        text = response.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])

        import json
        data = json.loads(text)

        plan = BuildPlan(
            feature_order=data.get("order", list(range(feature_tree.num_features))),
            notes={int(k): v for k, v in data.get("notes", {}).items()},
            strategy=data.get("strategy", ""),
        )

        # Validate: must include all feature indices
        expected = set(range(feature_tree.num_features))
        got = set(plan.feature_order)
        if got != expected:
            logger.warning("LLM order missing features %s, adding them", expected - got)
            for idx in sorted(expected - got):
                plan.feature_order.append(idx)

        logger.info("LLM build plan: %s", plan.strategy)
        for idx in plan.feature_order:
            note = plan.notes.get(idx, "")
            feat = feature_tree.features[idx]
            logger.info("  Step: F%d %s %s", idx, feat.type.value, f"— {note}" if note else "")

        return plan

    except Exception as e:
        logger.warning("LLM planning failed: %s — using default order", e)
        return None



# ---------------------------------------------------------------------------
# Heuristic construction (fallback)
# ---------------------------------------------------------------------------


def _build_stepwise(
    profiles: list[list[tuple[float, float]]],
    depth: float,
    extrude_dir: np.ndarray,
    extrude_start: float,
    holes: list[Feature],
    chamfers: list[Feature] | None = None,
    fillets: list[Feature] | None = None,
    build_order: BuildPlan | None = None,
    cleaned: object = None,
) -> tuple[object, str, list[ConstructionStep]]:
    """Build a clean CAD solid from scratch, step by step.

    Steps:
      0. Sketch — 2D profile wire on the workplane
      1. Extrude — sweep the sketch into a solid
      2+. Cut holes — one per step
      N+. Apply chamfers (on all edges at once)
      N+1. Apply fillets (on all edges at once)

    Each step is tessellated for the visual slideshow. The resulting
    solid is clean B-rep geometry (no mesh triangles).

    Returns (final_solid, script, construction_steps).
    """
    import cadquery as cq

    logger.debug(
        "_build_stepwise: %d profiles, depth=%.2f, extrude_dir=%s, %d holes, "
        "%d chamfers, %d fillets, build_order=%s",
        len(profiles), depth, extrude_dir.tolist(), len(holes),
        len(chamfers) if chamfers else 0, len(fillets) if fillets else 0,
        build_order.strategy if build_order else "heuristic",
    )

    abs_dir = np.abs(extrude_dir)
    primary_axis = int(np.argmax(abs_dir))
    plane_map = {0: "YZ", 1: "XZ", 2: "XY"}
    plane_name = plane_map[primary_axis]
    origin_3d = [0.0, 0.0, 0.0]
    origin_3d[primary_axis] = extrude_start

    steps: list[ConstructionStep] = []
    script_lines = ["import cadquery as cq", ""]

    # --- Step 0: Sketch (2D profile wires — outer + inner contours) ---
    axes = [0, 1, 2]
    axes.remove(primary_axis)
    ax_u, ax_v = axes

    all_sketch_verts = []
    all_sketch_faces = []
    vert_offset = 0

    for profile in profiles:
        n_pts = len(profile)
        verts = np.zeros((n_pts, 3))
        for pi, (u, v_coord) in enumerate(profile):
            verts[pi, ax_u] = u
            verts[pi, ax_v] = v_coord
            verts[pi, primary_axis] = extrude_start
        # Edge connectivity: degenerate triangles (v0, v1, v0) for wireframe
        faces = np.array(
            [[(vert_offset + i, vert_offset + (i + 1) % n_pts, vert_offset + i)]
             for i in range(n_pts)],
            dtype=np.int64,
        ).reshape(-1, 3)
        all_sketch_verts.append(verts)
        all_sketch_faces.append(faces)
        vert_offset += n_pts

    sketch_verts = np.vstack(all_sketch_verts)
    sketch_faces = np.vstack(all_sketch_faces)

    steps.append(ConstructionStep(
        name="Sketch Profile", tess_vertices=sketch_verts, tess_faces=sketch_faces, volume=0.0,
    ))
    script_lines.append(f"# Step 1: Sketch profile on {plane_name} plane")
    script_lines.append(f'result = cq.Workplane("{plane_name}", origin={tuple(origin_3d)})')
    script_lines.append(f"result = result.polyline(profile).close()")

    # --- Step 1: Extrude base profile ---
    # Try both directions, keep the one that overlaps with the actual part bbox.
    part_min = float(cleaned.bbox[0][primary_axis])
    part_max = float(cleaned.bbox[1][primary_axis])

    wp_pos = cq.Workplane(plane_name, origin=tuple(origin_3d))
    wp_pos = wp_pos.polyline(profiles[0]).close().extrude(depth)
    bb_pos = wp_pos.val().BoundingBox()
    pos_vals = [(bb_pos.xmin, bb_pos.xmax), (bb_pos.ymin, bb_pos.ymax), (bb_pos.zmin, bb_pos.zmax)]
    pos_overlap = min(pos_vals[primary_axis][1], part_max) - max(pos_vals[primary_axis][0], part_min)

    wp_neg = cq.Workplane(plane_name, origin=tuple(origin_3d))
    wp_neg = wp_neg.polyline(profiles[0]).close().extrude(-depth)
    bb_neg = wp_neg.val().BoundingBox()
    neg_vals = [(bb_neg.xmin, bb_neg.xmax), (bb_neg.ymin, bb_neg.ymax), (bb_neg.zmin, bb_neg.zmax)]
    neg_overlap = min(neg_vals[primary_axis][1], part_max) - max(neg_vals[primary_axis][0], part_min)

    if neg_overlap > pos_overlap:
        wp = wp_neg
        extrude_sign = -depth
        logger.info("Using negative extrude (overlap %.1f > %.1f)", neg_overlap, pos_overlap)
    else:
        wp = wp_pos
        extrude_sign = depth
    # Additional profiles: union outer bodies, cut inner holes.
    # A profile is "inner" if its centroid falls inside the main profile polygon.
    from shapely.geometry import Polygon, Point
    main_poly = Polygon(profiles[0])
    for profile in profiles[1:]:
        try:
            centroid = np.mean(profile, axis=0)
            is_inner = main_poly.contains(Point(centroid[0], centroid[1]))
            extra = cq.Workplane(plane_name, origin=tuple(origin_3d))
            extra = extra.polyline(profile).close().extrude(extrude_sign)
            if is_inner:
                wp = wp.cut(extra)
                logger.info("Cut inner profile (%d pts) from base", len(profile))
            else:
                wp = wp.union(extra)
                logger.info("Unioned outer profile (%d pts) with base", len(profile))
        except Exception:
            pass

    script_lines.append(f"# Step 2: Extrude {depth:.2f} mm")
    script_lines.append(f"result = result.extrude({depth})")

    v, f = _tessellate_solid(wp)
    vol = abs(float(wp.val().Volume())) if wp else 0
    steps.append(ConstructionStep(
        name="Extrude Base", tess_vertices=v, tess_faces=f, volume=vol,
    ))

    # --- Chamfers BEFORE holes ---
    # Detect chamfer faces directly from mesh geometry (diagonal normals),
    # match each group to CadQuery edges, and apply individually.
    # After each chamfer, validate the volume change is proportional to
    # the expected material removal. Revert if it looks wrong.
    mesh_chamfers = _detect_mesh_chamfers(cleaned, extrude_dir)
    for cham in mesh_chamfers:
        d1 = cham["size"]
        d2 = cham.get("size2", d1)
        sel = cham["selector"]
        label = cham["label"]
        edge_length_hint = cham.get("edge_length", 0.0)
        vol_before = abs(float(wp.val().Volume()))

        # Try asymmetric first, then progressively smaller
        attempts = [(d1, d2), (d1 * 0.8, d2 * 0.8), (d1 * 0.6, d2 * 0.6)]
        applied = False
        for a1, a2 in attempts:
            try:
                min_len = min(a1, a2) * 2.0
                sel_edges = wp.edges(sel).filter(lambda e: e.Length() > min_len)
                if abs(a1 - a2) / max(a1, a2) < 0.15:
                    wp_test = sel_edges.chamfer(min(a1, a2))
                else:
                    wp_test = sel_edges.chamfer(a1, a2)
                vol_after = abs(float(wp_test.val().Volume()))
                if vol_after <= 0:
                    logger.debug("Chamfer %s (%.2fx%.2f): zero volume, skipping attempt", label, a1, a2)
                    continue

                # --- Volume validation ---
                # Expected volume removal for a chamfer:
                #   V_expected ~= 0.5 * d1 * d2 * total_edge_length
                # Sum the lengths of the edges that were chamfered.
                try:
                    total_edge_len = sum(
                        e.Length() for e in wp.edges(sel).filter(lambda e: e.Length() > min_len).vals()
                    )
                except Exception:
                    total_edge_len = edge_length_hint if edge_length_hint > 0 else 0.0

                vol_removed = vol_before - vol_after
                if total_edge_len > 0:
                    expected_vol = 0.5 * a1 * a2 * total_edge_len
                    vol_ratio = vol_removed / max(expected_vol, 1e-10)
                    logger.debug(
                        "Chamfer %s (%.2fx%.2f): vol_before=%.2f vol_after=%.2f "
                        "removed=%.2f expected=%.2f ratio=%.2f edge_len=%.2f",
                        label, a1, a2, vol_before, vol_after,
                        vol_removed, expected_vol, vol_ratio, total_edge_len,
                    )
                    # Allow 0.1x to 10x of expected — reject gross mismatches
                    if vol_ratio < 0.1 or vol_ratio > 10.0:
                        logger.warning(
                            "Chamfer %s: volume change ratio %.2f outside "
                            "[0.1, 10.0] — reverting (removed=%.2f, expected=%.2f)",
                            label, vol_ratio, vol_removed, expected_vol,
                        )
                        continue
                else:
                    # No edge length info — just check volume didn't increase
                    if vol_after > vol_before * 1.01:
                        logger.warning("Chamfer %s: volume increased, reverting", label)
                        continue
                    logger.debug(
                        "Chamfer %s (%.2fx%.2f): vol_before=%.2f vol_after=%.2f "
                        "(no edge length for validation)",
                        label, a1, a2, vol_before, vol_after,
                    )

                wp = wp_test
                v, f = _tessellate_solid(wp)
                cham_label = f"{a1:.1f}x{a2:.1f}" if abs(a1-a2) > 0.5 else f"{min(a1,a2):.1f}"
                steps.append(ConstructionStep(
                    name=f"Chamfer {label} ({cham_label}mm)",
                    tess_vertices=v, tess_faces=f, volume=vol_after,
                ))
                script_lines.append(f'# Chamfer {label}')
                script_lines.append(f'result = result.edges("{sel}").chamfer({a1:.4f}, {a2:.4f})')
                logger.info(
                    "Applied chamfer %s: %.2fx%.2f mm, vol %.2f -> %.2f (removed %.2f)",
                    label, a1, a2, vol_before, vol_after, vol_removed,
                )
                applied = True
                break
            except Exception as exc:
                logger.debug("Chamfer %s (%.2fx%.2f) failed: %s", label, a1, a2, exc)
                continue
        if not applied:
            logger.debug("Chamfer %s: all attempts failed, skipping", label)

    # Reorder holes
    if build_order is not None and len(holes) > 0:
        ordered = []
        used = set()
        for fi in build_order.feature_order:
            hi = fi - 1
            if 0 <= hi < len(holes):
                ordered.append((holes[hi], build_order.notes.get(fi, "")))
                used.add(hi)
        for hi, h in enumerate(holes):
            if hi not in used:
                ordered.append((h, ""))
    else:
        ordered = [(h, "") for h in holes]

    # Compute the solid's bounding box corners for projecting hole spans
    bb = wp.val().BoundingBox()
    bbox_corners = np.array([
        [bb.xmin, bb.ymin, bb.zmin], [bb.xmax, bb.ymin, bb.zmin],
        [bb.xmin, bb.ymax, bb.zmin], [bb.xmax, bb.ymax, bb.zmin],
        [bb.xmin, bb.ymin, bb.zmax], [bb.xmax, bb.ymin, bb.zmax],
        [bb.xmin, bb.ymax, bb.zmax], [bb.xmax, bb.ymax, bb.zmax],
    ])

    # --- Steps 2+: Cut each hole ---
    for i, (hole, note) in enumerate(ordered):
        sketch = hole.sketch_geometry
        if sketch.get("type") != "circle":
            continue

        center = np.array(sketch["center"], dtype=float)
        radius = float(sketch["radius"])
        h_depth = float(hole.depth)
        h_dir = np.array(hole.direction, dtype=float)
        h_dir = h_dir / (np.linalg.norm(h_dir) + 1e-15)
        is_through = hole.type == FeatureType.THROUGH_HOLE

        # Skip non-axis-aligned holes — these are misdetected features
        # (e.g. inner faces of circular cutouts). Real holes in prismatic
        # parts are always along principal axes.
        if np.max(np.abs(h_dir)) < 0.9:
            logger.debug("Skipping non-axis-aligned hole: dir=%s r=%.2f", h_dir, radius)
            continue

        # Project all bbox corners onto the hole axis to find the
        # full extent of the solid in that direction
        proj = bbox_corners @ h_dir
        solid_min_along_axis = float(proj.min())
        solid_max_along_axis = float(proj.max())
        center_along_axis = float(center @ h_dir)

        if is_through:
            # Span the full solid extent plus margin
            cut_start = center + h_dir * (solid_min_along_axis - center_along_axis - 1)
            cut_end = center + h_dir * (solid_max_along_axis - center_along_axis + 1)
            cut_length = float(np.linalg.norm(cut_end - cut_start))
        else:
            # Blind hole: start from the entry surface and cut inward.
            # center (axis_point) is at the midpoint of the cylinder.
            # The entry surface is at center - h_dir * (depth/2),
            # the bottom is at center + h_dir * (depth/2).
            # Start the cut slightly outside the entry surface.
            cut_start = center - h_dir * (h_depth / 2) - h_dir * 1.0
            cut_length = h_depth + 2.0

        hole_name = f"{'Through' if is_through else 'Blind'} Hole (r={radius:.1f})"
        if note:
            hole_name += f" — {note}"

        try:
            hole_plane = cq.Plane(
                origin=cq.Vector(*cut_start.tolist()),
                normal=cq.Vector(*h_dir.tolist()),
            )
            hole_solid = cq.Workplane(hole_plane).circle(radius).extrude(cut_length)

            # Try the cut; if it removes little/no material, the hole
            # center is likely outside the solid (e.g., on a flange/tab
            # not captured by the profile cross-section).  In that case,
            # nudge the hole center toward the solid so it intersects.
            vol_before = abs(float(wp.val().Volume()))
            wp_test = wp.cut(hole_solid)
            vol_after = abs(float(wp_test.val().Volume()))

            # Expected hole volume: pi * r^2 * cut_length
            expected_removal = 3.14159 * radius * radius * cut_length
            actual_removal = vol_before - vol_after
            # Trigger nudge if less than 10% of expected volume was removed
            barely_cut = actual_removal < expected_removal * 0.10

            if barely_cut and cleaned is not None:
                # Find the nearest point on the solid to the hole center
                # by projecting the center onto the profile plane and
                # shifting it inward until the cut succeeds.
                wp_nudged = _nudge_hole_into_solid(
                    wp, center, h_dir, radius, is_through,
                    cut_length, h_depth, bbox_corners, cq,
                )
                if wp_nudged is not None:
                    wp = wp_nudged
                else:
                    wp = wp_test
            else:
                wp = wp_test

            v, f = _tessellate_solid(wp)
            vol = abs(float(wp.val().Volume()))
            steps.append(ConstructionStep(
                name=hole_name, tess_vertices=v, tess_faces=f, volume=vol,
            ))

            step_num = len(steps) + 1
            script_lines.append(f"# Step {step_num}: Cut {hole_name}")
            script_lines.append(
                f"h{i} = cq.Plane(origin=cq.Vector({cut_start[0]:.4f}, "
                f"{cut_start[1]:.4f}, {cut_start[2]:.4f}), "
                f"normal=cq.Vector({h_dir[0]:.4f}, {h_dir[1]:.4f}, {h_dir[2]:.4f}))"
            )
            script_lines.append(
                f"result = result.cut(cq.Workplane(h{i})"
                f".circle({radius:.4f}).extrude({cut_length:.4f}))"
            )
        except Exception as e:
            logger.warning(
                "Failed to cut hole %d (r=%.2f, center=%s, dir=%s): %s",
                i, radius, center.tolist(), h_dir.tolist(), e,
            )

    # (Small chamfers are now handled by _detect_mesh_chamfers above)

    # --- Fillets: detect from mesh geometry and apply individually ---
    # Same approach as chamfers: analyze the mesh to find fillet faces,
    # compute radius from arc geometry, build edge selectors, and apply
    # each fillet individually with volume validation and progressive
    # fallback.
    mesh_fillets = _detect_mesh_fillets(cleaned, extrude_dir)
    for fil in mesh_fillets:
        fil_r = fil["radius"]
        sel = fil["selector"]
        label = fil["label"]
        edge_length_hint = fil.get("edge_length", 0.0)
        vol_before = abs(float(wp.val().Volume()))

        # Progressive fallback: try detected radius, then 80%, then 60%
        attempts = [fil_r, fil_r * 0.8, fil_r * 0.6]
        applied = False
        for attempt_r in attempts:
            try:
                min_len = attempt_r * 2.0
                sel_edges = wp.edges(sel).filter(lambda e: e.Length() > min_len)
                wp_test = sel_edges.fillet(attempt_r)
                vol_after = abs(float(wp_test.val().Volume()))
                if vol_after <= 0:
                    logger.debug(
                        "Fillet %s (r=%.2f): zero volume, skipping attempt",
                        label, attempt_r,
                    )
                    continue

                # --- Volume validation ---
                # A fillet removes material proportional to:
                #   V_expected ~= (1 - pi/4) * R^2 * total_edge_length
                # (The removed material is roughly the corner minus the
                # quarter-circle, per unit edge length.)
                try:
                    total_edge_len = sum(
                        e.Length() for e in wp.edges(sel).filter(
                            lambda e: e.Length() > min_len
                        ).vals()
                    )
                except Exception:
                    total_edge_len = edge_length_hint if edge_length_hint > 0 else 0.0

                vol_removed = vol_before - vol_after
                if total_edge_len > 0:
                    expected_vol = (1.0 - 3.14159 / 4.0) * attempt_r * attempt_r * total_edge_len
                    vol_ratio = vol_removed / max(expected_vol, 1e-10)
                    logger.debug(
                        "Fillet %s (r=%.2f): vol_before=%.2f vol_after=%.2f "
                        "removed=%.2f expected=%.2f ratio=%.2f edge_len=%.2f",
                        label, attempt_r, vol_before, vol_after,
                        vol_removed, expected_vol, vol_ratio, total_edge_len,
                    )
                    # Allow 0.05x to 15x of expected — fillets can vary more
                    # than chamfers due to concave/convex edge differences
                    if vol_ratio < 0.05 or vol_ratio > 15.0:
                        logger.warning(
                            "Fillet %s: volume change ratio %.2f outside "
                            "[0.05, 15.0] — reverting (removed=%.2f, expected=%.2f)",
                            label, vol_ratio, vol_removed, expected_vol,
                        )
                        continue
                else:
                    # No edge length info — just check volume didn't increase
                    if vol_after > vol_before * 1.01:
                        logger.warning("Fillet %s: volume increased, reverting", label)
                        continue
                    logger.debug(
                        "Fillet %s (r=%.2f): vol_before=%.2f vol_after=%.2f "
                        "(no edge length for validation)",
                        label, attempt_r, vol_before, vol_after,
                    )

                wp = wp_test
                v, f = _tessellate_solid(wp)
                steps.append(ConstructionStep(
                    name=f"Fillet {label} (r={attempt_r:.2f}mm)",
                    tess_vertices=v, tess_faces=f, volume=vol_after,
                ))
                script_lines.append(f'# Fillet {label}')
                script_lines.append(f'result = result.edges("{sel}").fillet({attempt_r:.4f})')
                logger.info(
                    "Applied fillet %s: r=%.2f mm, vol %.2f -> %.2f (removed %.2f)",
                    label, attempt_r, vol_before, vol_after, vol_removed,
                )
                applied = True
                break
            except Exception as exc:
                logger.debug("Fillet %s (r=%.2f) failed: %s", label, attempt_r, exc)
                continue
        if not applied:
            logger.debug("Fillet %s: all attempts failed, skipping", label)

    # --- Fallback: if no mesh fillets detected but recognition stage found
    # fillets, try a conservative global application ---
    if not mesh_fillets and fillets:
        fillet_radii = []
        for fl in fillets:
            r = fl.sketch_geometry.get("radius", 0)
            if 0.1 < r < 50:
                fillet_radii.append(r)

        if fillet_radii:
            fillet_r = float(np.median(fillet_radii))
            fillet_r = max(0.2, min(fillet_r, 3.0))
            vol_before = abs(float(wp.val().Volume()))

            fallback_applied = False
            for attempt_r in [fillet_r, fillet_r * 0.8, fillet_r * 0.6]:
                # Try each edge-axis selector individually
                for sel_str in ["|X", "|Y", "|Z"]:
                    try:
                        min_len = attempt_r * 2.0
                        wp_test = wp.edges(sel_str).filter(
                            lambda e: e.Length() > min_len
                        ).fillet(attempt_r)
                        vol_after = abs(float(wp_test.val().Volume()))
                        if vol_after <= 0:
                            continue
                        vol_removed = vol_before - vol_after
                        # Sanity: fillet should remove material, not add
                        if vol_after > vol_before * 1.01:
                            continue
                        # Sanity: should not remove more than 10% of volume
                        if vol_removed > vol_before * 0.10:
                            logger.debug(
                                "Fallback fillet %s r=%.2f removed %.1f%% of volume — "
                                "too much, skipping",
                                sel_str, attempt_r,
                                vol_removed / vol_before * 100,
                            )
                            continue

                        wp = wp_test
                        v, f = _tessellate_solid(wp)
                        steps.append(ConstructionStep(
                            name=f"Fillet (r={attempt_r:.1f}mm, fallback {sel_str})",
                            tess_vertices=v, tess_faces=f, volume=vol_after,
                        ))
                        script_lines.append(
                            f'result = result.edges("{sel_str}").fillet({attempt_r:.4f})'
                        )
                        logger.info(
                            "Applied fallback fillet %s: r=%.2f, vol %.2f -> %.2f",
                            sel_str, attempt_r, vol_before, vol_after,
                        )
                        fallback_applied = True
                        break
                    except Exception:
                        continue
                if fallback_applied:
                    break
            if not fallback_applied:
                logger.warning(
                    "All fillet sizes/selectors failed (mesh + fallback) — "
                    "skipping fillet step"
                )

    script_lines.extend(["", '# cq.exporters.export(result, "output.step")'])
    return wp, "\n".join(script_lines), steps


def _build_from_mesh_and_cut_holes(
    cleaned: CleanedMesh,
    holes: list[Feature],
    build_order: BuildPlan | None = None,
) -> tuple[object, str]:
    """Build a CadQuery solid by importing the mesh body, then cutting
    clean parametric holes.

    This avoids the polyline profile approximation problem entirely.
    The mesh gives us the exact body shape; the recognized holes give
    us the parametric cut geometry (center, radius, depth, direction).
    """
    logger.debug("_build_from_mesh_and_cut_holes: %d holes, verts=%d", len(holes), len(cleaned.vertices))
    import cadquery as cq
    import tempfile, os

    # Export the cleaned mesh to a temp STL for CadQuery import
    mesh_tri = trimesh.Trimesh(
        vertices=cleaned.vertices, faces=cleaned.faces, process=False,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp_path = tmp.name
    tmp.close()
    mesh_tri.export(tmp_path)

    # Import STL → sew into a closed shell → make solid.
    # The mesh already contains the full part geometry including holes,
    # so no boolean cuts are needed — the solid is correct as-is.
    from OCP.StlAPI import StlAPI_Reader
    from OCP.TopoDS import TopoDS_Shape, TopoDS
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer

    reader = StlAPI_Reader()
    shape = TopoDS_Shape()
    reader.Read(shape, tmp_path)
    os.unlink(tmp_path)

    sewer = BRepBuilderAPI_Sewing(0.1)
    sewer.Add(shape)
    sewer.Perform()
    sewn = sewer.SewedShape()

    exp = TopExp_Explorer(sewn, TopAbs_SHELL)
    if not exp.More():
        logger.error("STL sewing produced no shell")
        return cq.Workplane("XY"), ""
    shell = TopoDS.Shell_s(exp.Current())
    occ_solid = BRepBuilderAPI_MakeSolid(shell).Solid()
    wp = cq.Workplane("XY").add(cq.Solid(occ_solid))

    # The mesh solid already contains all holes. Generate a parametric
    # script that shows how to reconstruct the part from scratch.
    script_lines = [
        "import cadquery as cq",
        "",
        "# Parametric reconstruction script",
        "# Base body: extrude profile, then cut holes",
    ]

    # Reorder holes for the script based on LLM build plan
    if build_order is not None and len(holes) > 0:
        ordered_holes = []
        used_indices = set()
        for fi in build_order.feature_order:
            hi = fi - 1
            if 0 <= hi < len(holes):
                ordered_holes.append((holes[hi], build_order.notes.get(fi, "")))
                used_indices.add(hi)
        for hi, h in enumerate(holes):
            if hi not in used_indices:
                ordered_holes.append((h, ""))
        if build_order.strategy:
            script_lines.append(f"# Strategy: {build_order.strategy}")
    else:
        ordered_holes = [(h, "") for h in holes]

    # Document hole cuts in the script
    for i, (hole, note) in enumerate(ordered_holes):
        sketch = hole.sketch_geometry
        if sketch.get("type") != "circle":
            continue

        center = np.array(sketch["center"], dtype=float)
        radius = float(sketch["radius"])
        h_depth = float(hole.depth)
        h_dir = np.array(hole.direction, dtype=float)
        h_dir = h_dir / (np.linalg.norm(h_dir) + 1e-15)
        is_through = hole.type == FeatureType.THROUGH_HOLE

        if is_through:
            cut_start = center - h_dir * (h_depth / 2 + h_depth * 0.5)
            cut_length = h_depth * 2
        else:
            cut_start = center - h_dir * (h_depth / 2)
            cut_length = h_depth

        note_suffix = f"  ({note})" if note else ""
        script_lines.append(
            f"# Hole {i}: {'through' if is_through else 'blind'}, "
            f"r={radius:.2f}, depth={h_depth:.1f}{note_suffix}"
        )
        script_lines.append(
            f"h{i} = cq.Plane(origin=cq.Vector({cut_start[0]:.4f}, "
            f"{cut_start[1]:.4f}, {cut_start[2]:.4f}), "
            f"normal=cq.Vector({h_dir[0]:.4f}, {h_dir[1]:.4f}, {h_dir[2]:.4f}))"
        )
        script_lines.append(
            f"result = result.cut(cq.Workplane(h{i})"
            f".circle({radius:.4f}).extrude({cut_length:.4f}))"
        )

    script_lines.extend(["", '# cq.exporters.export(result, "output.step")'])
    return wp, "\n".join(script_lines)


def _simplify_profile(
    points: list[tuple[float, float]],
    tolerance: float = 0.1,
) -> list[tuple[float, float]]:
    """Simplify a 2D profile to only its corner vertices.

    Removes points that are collinear with their neighbors (on straight
    edges), keeping only vertices where the direction changes significantly.
    Also snaps coordinates to clean values.
    """
    if len(points) < 3:
        return points

    # Step 1: Remove near-duplicate consecutive points
    deduped = [points[0]]
    for p in points[1:]:
        dx = p[0] - deduped[-1][0]
        dy = p[1] - deduped[-1][1]
        if (dx * dx + dy * dy) > tolerance * tolerance:
            deduped.append(p)
    # Check closing duplicate
    if len(deduped) > 2:
        dx = deduped[-1][0] - deduped[0][0]
        dy = deduped[-1][1] - deduped[0][1]
        if (dx * dx + dy * dy) < tolerance * tolerance:
            deduped = deduped[:-1]

    if len(deduped) < 3:
        return deduped

    # Step 2: Keep corners + curve points (direction change > 2°)
    # Low threshold preserves gentle arcs while removing straight-edge noise.
    pts = deduped
    corners = []
    n = len(pts)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        angle = abs(np.degrees(np.arctan2(cross, dot)))
        if angle > 2.0:
            corners.append(p1)

    if len(corners) < 3:
        return deduped

    # Step 3: Snap coordinates to clean values (nearest 0.1mm)
    snapped = []
    for x, y in corners:
        sx = round(x * 10) / 10
        sy = round(y * 10) / 10
        snapped.append((sx, sy))

    return snapped


def _build_solid(
    profiles: list[list[tuple[float, float]]],
    depth: float,
    extrude_dir: np.ndarray,
    extrude_start: float,
    holes: list[Feature],
    build_order: BuildPlan | None = None,
) -> tuple[object, str]:
    """Build the CadQuery solid from profiles and holes.

    Extrudes ALL cross-section profiles and unions them to capture the
    full part geometry (main body + leg extensions + bosses).
    If build_order is provided, holes are cut in the LLM-specified order.

    Returns (solid, script_text).
    """
    abs_dir = np.abs(extrude_dir)
    primary_axis = int(np.argmax(abs_dir))
    axes = [0, 1, 2]
    axes.remove(primary_axis)
    ax_u, ax_v = axes

    plane_map = {0: "YZ", 1: "XZ", 2: "XY"}
    plane_name = plane_map[primary_axis]

    origin_3d = [0.0, 0.0, 0.0]
    origin_3d[primary_axis] = extrude_start

    import cadquery as cq

    # Extrude the main profile (largest polygon)
    wp = cq.Workplane(plane_name, origin=tuple(origin_3d))
    wp = wp.polyline(profiles[0]).close().extrude(depth)

    # Union additional profiles (leg extensions, bosses, etc.)
    for i, profile in enumerate(profiles[1:], 1):
        try:
            extra = cq.Workplane(plane_name, origin=tuple(origin_3d))
            extra = extra.polyline(profile).close().extrude(depth)
            wp = wp.union(extra)
            logger.info("Unioned additional profile %d (%d points)", i, len(profile))
        except Exception as e:
            logger.warning("Failed to union profile %d: %s", i, e)

    # Reorder holes if LLM provided a build plan.
    # Holes are features[1:] in the tree, so feature index i maps to holes[i-1].
    if build_order is not None and len(holes) > 0:
        ordered_holes = []
        used_indices = set()
        for fi in build_order.feature_order:
            hi = fi - 1  # feature index 0 = base, feature 1 = holes[0], etc.
            if 0 <= hi < len(holes):
                ordered_holes.append((holes[hi], build_order.notes.get(fi, "")))
                used_indices.add(hi)
        # Append any holes not covered by the plan
        for hi, h in enumerate(holes):
            if hi not in used_indices:
                ordered_holes.append((h, ""))
    else:
        ordered_holes = [(h, "") for h in holes]

    # Generate script text
    strategy_comment = ""
    if build_order and build_order.strategy:
        strategy_comment = f"# Build strategy: {build_order.strategy}\n"

    script_lines = [
        "import cadquery as cq",
        "",
        f"{strategy_comment}# Main profile on {plane_name} plane ({len(profiles)} total profiles)",
        f"profile_0 = {profiles[0]!r}",
        "",
        f'result = cq.Workplane("{plane_name}", origin={tuple(origin_3d)})',
        "result = result.polyline(profile_0).close()",
        f"result = result.extrude({depth})",
    ]

    # Add script lines for additional profiles
    for pi, profile in enumerate(profiles[1:], 1):
        script_lines.append(f"# Additional profile {pi} ({len(profile)} points)")
        script_lines.append(f"profile_{pi} = {profile!r}")
        script_lines.append(
            f'extra_{pi} = cq.Workplane("{plane_name}", origin={tuple(origin_3d)})'
            f".polyline(profile_{pi}).close().extrude({depth})"
        )
        script_lines.append(f"result = result.union(extra_{pi})")

    # Add boss material around blind holes. Blind holes are typically in
    # rectangular boss sections that may extend beyond the arch profile.
    # Create a box around each blind hole position to ensure the solid
    # has material there before cutting the hole.
    for hole, _ in ordered_holes:
        if hole.type != FeatureType.BLIND_HOLE:
            continue
        sketch = hole.sketch_geometry
        if sketch.get("type") != "circle":
            continue
        center = np.array(sketch["center"], dtype=float)
        radius = float(sketch["radius"])
        h_depth = float(hole.depth)
        h_dir = np.array(hole.direction, dtype=float)
        h_dir = h_dir / (np.linalg.norm(h_dir) + 1e-15)

        # Boss box: just large enough to contain the hole
        box_half = radius + 0.5
        # Create a box centered on the hole, aligned to the hole axis
        end_a = center - h_dir * (h_depth / 2)
        end_b = center + h_dir * (h_depth / 2)
        box_center = (end_a + end_b) / 2

        try:
            boss_plane = cq.Plane(
                origin=cq.Vector(*end_a.tolist()),
                normal=cq.Vector(*h_dir.tolist()),
            )
            boss = (
                cq.Workplane(boss_plane)
                .rect(box_half * 2, box_half * 2)
                .extrude(h_depth)
            )
            wp = wp.union(boss)
            script_lines.append(
                f"# Boss for blind hole at ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
            )
        except Exception as e:
            logger.warning("Failed to add blind hole boss: %s", e)

    # Cut holes using direct cylinder subtraction at the actual 3D position
    # and direction. The feature center is the midpoint of the cylinder axis,
    # so we extend depth/2 in both directions to get the full span.
    # Through-holes get extra margin to ensure full penetration.
    for i, (hole, note) in enumerate(ordered_holes):
        sketch = hole.sketch_geometry
        if sketch.get("type") != "circle":
            continue

        center = np.array(sketch["center"], dtype=float)
        radius = float(sketch["radius"])
        h_depth = float(hole.depth)
        h_dir = np.array(hole.direction, dtype=float)
        h_dir = h_dir / (np.linalg.norm(h_dir) + 1e-15)

        is_through = hole.type == FeatureType.THROUGH_HOLE

        # Cylinder spans from center ± depth/2 along the axis.
        # For through-holes, add generous margin to ensure full penetration.
        if is_through:
            margin = h_depth * 0.5
            cut_start = center - h_dir * (h_depth / 2 + margin)
            cut_length = h_depth + 2 * margin
        else:
            cut_start = center - h_dir * (h_depth / 2)
            cut_length = h_depth

        # Direction from cut_start toward the other end
        cut_dir = h_dir.copy()

        try:
            hole_plane = cq.Plane(
                origin=cq.Vector(*cut_start.tolist()),
                normal=cq.Vector(*cut_dir.tolist()),
            )
            hole_solid = (
                cq.Workplane(hole_plane)
                .circle(radius)
                .extrude(cut_length)
            )
            wp = wp.cut(hole_solid)

            note_suffix = f"  ({note})" if note else ""
            script_lines.append(
                f"# Hole {i}: {'through' if is_through else 'blind'}, "
                f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                f"dir=({h_dir[0]:.2f}, {h_dir[1]:.2f}, {h_dir[2]:.2f}), "
                f"r={radius:.4f}, depth={h_depth:.4f}{note_suffix}"
            )
            script_lines.append(
                f"hole_plane_{i} = cq.Plane("
                f"origin=cq.Vector({cut_start[0]:.4f}, {cut_start[1]:.4f}, {cut_start[2]:.4f}), "
                f"normal=cq.Vector({cut_dir[0]:.4f}, {cut_dir[1]:.4f}, {cut_dir[2]:.4f}))"
            )
            script_lines.append(
                f"result = result.cut("
                f"cq.Workplane(hole_plane_{i}).circle({radius:.4f}).extrude({cut_length:.4f}))"
            )
        except Exception as e:
            logger.warning("Failed to cut hole %d: %s", i, e)
            script_lines.append(f"# Hole {i} failed: {e}")

    script_lines.extend([
        "",
        '# Export STEP',
        'cq.exporters.export(result, "output.step")',
    ])

    return wp, "\n".join(script_lines)


# ---------------------------------------------------------------------------
# Volume estimation
# ---------------------------------------------------------------------------


def _tessellate_solid(solid) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Tessellate a CadQuery solid for visualization.

    Uses per-face triangulation: extracts each B-rep face, gets its boundary
    wire points, and triangulates with Constrained Delaunay (CDT) for planar
    faces.  This guarantees interior triangles for large flat surfaces that
    OCCT's BRepMesh otherwise leaves hollow.
    Curved faces use standard BRepMesh triangulation.
    """
    logger.debug("Tessellating CadQuery solid")
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_REVERSED
        from OCP.TopoDS import TopoDS
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Plane
        from OCP.TopLoc import TopLoc_Location
        from OCP.BRepTools import BRepTools_WireExplorer
        from OCP.GCPnts import GCPnts_UniformDeflection
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from scipy.spatial import Delaunay
        import shapely.geometry as sg

        shape = solid.val().wrapped

        # Use OCCT's BRepMesh for all faces with tight tolerance
        # (custom Delaunay was leaving gaps on concave planar faces)
        BRepMesh_IncrementalMesh(shape, 0.05, False, 0.1, True)

        all_verts = []
        all_faces = []
        vert_offset = 0

        face_exp = TopExp_Explorer(shape, TopAbs_FACE)
        while face_exp.More():
            face = TopoDS.Face_s(face_exp.Current())
            face_exp.Next()

            # Use BRepMesh triangulation for all faces
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is None:
                continue
            trsf = loc.Transformation()
            n_nodes = tri.NbNodes()
            n_tris = tri.NbTriangles()
            verts = np.empty((n_nodes, 3), dtype=np.float64)
            for i in range(1, n_nodes + 1):
                pt = tri.Node(i).Transformed(trsf)
                verts[i - 1] = [pt.X(), pt.Y(), pt.Z()]
            faces = np.empty((n_tris, 3), dtype=np.int64)
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for i in range(1, n_tris + 1):
                t = tri.Triangle(i)
                i1, i2, i3 = t.Get()
                if reversed_face:
                    faces[i - 1] = [i1 - 1, i3 - 1, i2 - 1]
                else:
                    faces[i - 1] = [i1 - 1, i2 - 1, i3 - 1]
            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            vert_offset += n_nodes

        if not all_verts:
            return None, None

        return np.vstack(all_verts), np.vstack(all_faces)

    except Exception as e:
        logger.warning("Per-face tessellation failed: %s — falling back to STL", e)
        return _tessellate_solid_stl(solid)


def _triangulate_planar_face(face) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Triangulate a planar B-rep face using its boundary wire + Delaunay.

    Extracts outer wire + hole wires, projects to 2D (the face's plane),
    runs constrained Delaunay, filters out triangles outside the boundary,
    and maps back to 3D.
    """
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_WIRE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GCPnts import GCPnts_UniformDeflection
    from OCP.BRep import BRep_Tool
    from scipy.spatial import Delaunay
    try:
        import shapely.geometry as sg
    except ImportError:
        return None, None

    adaptor = BRepAdaptor_Surface(face)
    plane = adaptor.Plane()
    origin = plane.Location()
    ax = plane.Position()
    dx = ax.XDirection()
    dy = ax.YDirection()

    ox, oy, oz = origin.X(), origin.Y(), origin.Z()
    dxx, dxy, dxz = dx.X(), dx.Y(), dx.Z()
    dyx, dyy, dyz = dy.X(), dy.Y(), dy.Z()

    def to_2d(pt3d):
        rx, ry, rz = pt3d[0] - ox, pt3d[1] - oy, pt3d[2] - oz
        return (rx * dxx + ry * dxy + rz * dxz,
                rx * dyx + ry * dyy + rz * dyz)

    def to_3d(u, v):
        return np.array([ox + u * dxx + v * dyx,
                         oy + u * dxy + v * dyy,
                         oz + u * dxz + v * dyz])

    # Collect wires — careful about edge orientation within the wire
    from OCP.TopAbs import TopAbs_FORWARD
    wires_3d = []
    wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
    while wire_exp.More():
        wire = TopoDS.Wire_s(wire_exp.Current())
        wire_exp.Next()
        pts_3d = []
        edge_exp = BRepTools_WireExplorer(wire)
        while edge_exp.More():
            edge = TopoDS.Edge_s(edge_exp.Current())
            edge_exp.Next()
            curve = BRepAdaptor_Curve(edge)
            discretizer = GCPnts_UniformDeflection(curve, 0.1, True)
            if not discretizer.IsDone():
                continue
            edge_pts = []
            for j in range(1, discretizer.NbPoints() + 1):
                p = discretizer.Value(j)
                edge_pts.append([p.X(), p.Y(), p.Z()])
            # BRepAdaptor_Curve always parametrizes in the edge's natural direction.
            # If the edge is reversed in the wire, we need to flip the points.
            if edge.Orientation() != TopAbs_FORWARD:
                edge_pts = edge_pts[::-1]
            # Skip the last point to avoid duplicating the shared vertex
            # with the next edge
            pts_3d.extend(edge_pts[:-1] if len(edge_pts) > 1 else edge_pts)
        if len(pts_3d) >= 3:
            wires_3d.append(np.array(pts_3d))

    if not wires_3d:
        return None, None

    # Project all wires to 2D
    wires_2d = []
    for w in wires_3d:
        pts2 = np.array([to_2d(p) for p in w])
        wires_2d.append(pts2)

    # Identify outer wire (largest area polygon)
    polys = []
    for pts2 in wires_2d:
        ring = np.vstack([pts2, pts2[0:1]])  # close
        polys.append(sg.Polygon(ring))

    outer_idx = max(range(len(polys)), key=lambda i: abs(polys[i].area))
    outer_poly = polys[outer_idx]
    if not outer_poly.is_valid:
        outer_poly = outer_poly.buffer(0)

    hole_polys = [polys[i] for i in range(len(polys)) if i != outer_idx]

    # Build shapely polygon with holes for point-in-polygon tests
    try:
        holes_coords = [list(hp.exterior.coords) for hp in hole_polys if hp.is_valid]
        boundary = sg.Polygon(outer_poly.exterior.coords, holes=holes_coords)
        if not boundary.is_valid:
            boundary = boundary.buffer(0)
    except Exception:
        boundary = outer_poly

    # Collect all boundary points for Delaunay
    all_pts_2d = []
    for pts2 in wires_2d:
        all_pts_2d.extend(pts2.tolist())

    # Add interior grid points for better triangulation of large faces
    bounds = boundary.bounds  # (minx, miny, maxx, maxy)
    dx_span = bounds[2] - bounds[0]
    dy_span = bounds[3] - bounds[1]
    grid_step = max(dx_span, dy_span) / 20.0  # ~20 divisions
    if grid_step > 0.1:
        xs = np.arange(bounds[0] + grid_step, bounds[2], grid_step)
        ys = np.arange(bounds[1] + grid_step, bounds[3], grid_step)
        for x in xs:
            for y in ys:
                pt = sg.Point(x, y)
                if boundary.contains(pt):
                    all_pts_2d.append([x, y])

    pts_2d = np.array(all_pts_2d)
    if len(pts_2d) < 3:
        return None, None

    # Remove near-duplicate points
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_2d)
    pairs = tree.query_pairs(r=1e-6)
    remove = set()
    for i, j in pairs:
        remove.add(max(i, j))
    keep = sorted(set(range(len(pts_2d))) - remove)
    pts_2d = pts_2d[keep]

    if len(pts_2d) < 3:
        return None, None

    # Delaunay triangulation
    tri = Delaunay(pts_2d)
    faces = tri.simplices

    # Filter: keep only triangles whose centroid is inside or on the boundary.
    # Use a tiny buffer to avoid dropping triangles on thin/narrow faces
    # where centroids land exactly on the boundary edge.
    boundary_buffered = boundary.buffer(1e-4)
    centroids = pts_2d[faces].mean(axis=1)
    keep_mask = np.array([
        boundary_buffered.contains(sg.Point(c[0], c[1])) for c in centroids
    ])
    faces = faces[keep_mask]

    if len(faces) == 0:
        return None, None

    # Map back to 3D
    verts_3d = np.array([to_3d(u, v) for u, v in pts_2d])

    # Compute the face outward normal from the plane
    nz = ax.Direction()
    face_normal = np.array([nz.X(), nz.Y(), nz.Z()])
    reversed_face = face.Orientation() == TopAbs_REVERSED
    if reversed_face:
        face_normal = -face_normal

    # Ensure all triangles have normals consistent with the face normal.
    # Compute triangle normals and flip any that point the wrong way.
    v0 = verts_3d[faces[:, 0]]
    v1 = verts_3d[faces[:, 1]]
    v2 = verts_3d[faces[:, 2]]
    tri_normals = np.cross(v1 - v0, v2 - v0)
    dots = np.einsum('ij,j->i', tri_normals, face_normal)
    flip_mask = dots < 0
    faces[flip_mask] = faces[flip_mask][:, ::-1]

    return verts_3d, faces


def _tessellate_solid_stl(solid) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fallback: tessellate via STL export."""
    try:
        import cadquery as cq
        import tempfile
        import os

        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cq.exporters.export(solid, tmp_path, exportType="STL",
                                tolerance=0.01, angularTolerance=0.05)
            mesh = trimesh.load(tmp_path)
            if hasattr(mesh, 'geometry'):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            return np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.int64)
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        logger.warning("STL tessellation failed: %s", e)
        return None, None


def _detect_mesh_chamfers(
    cleaned: CleanedMesh, extrude_dir: np.ndarray,
) -> list[dict]:
    """Detect all chamfer edges from the mesh by analyzing diagonal-normal faces.

    General-purpose: works on any prismatic part. Clusters diagonal faces by
    normal direction, computes the chamfer size from each cluster's face width,
    and determines the correct CadQuery edge selector by analyzing which two
    principal faces each chamfer sits between.

    Uses scale-relative thresholds based on the part's bounding-box diagonal.
    Validates that each candidate cluster forms a narrow connected strip (not
    scattered faces or circular cutout geometry).

    Returns a list of dicts, one per individual chamfer:
        [{"size": mm, "size2": mm, "selector": "|Y and <Z", "label": "bottom Y",
          "face_count": N, "edge_length": mm}, ...]
    """
    from sklearn.cluster import DBSCAN

    bbox_diag = float(np.linalg.norm(cleaned.bbox[1] - cleaned.bbox[0]))
    bbox_size = cleaned.bbox[1] - cleaned.bbox[0]
    logger.debug(
        "Detecting mesh chamfers: %d faces, extrude_dir=%s, bbox_diag=%.2f",
        len(cleaned.faces), extrude_dir.tolist(), bbox_diag,
    )

    # --- Scale-relative limits ---
    size_min = bbox_diag * 0.001   # 0.1% of diagonal
    size_max = bbox_diag * 0.05    # 5% of diagonal
    # DBSCAN eps for normal clustering: normals live on the unit sphere so
    # eps=0.20 works well regardless of part scale.
    normal_eps = 0.20
    # Minimum faces in a cluster to be considered real geometry (not noise).
    # Scales with mesh density: at least 5 or 0.05% of total faces.
    min_cluster_faces = max(5, int(len(cleaned.faces) * 0.0005))

    normals = cleaned.normals
    ext = extrude_dir / (np.linalg.norm(extrude_dir) + 1e-15)

    # --- Identify diagonal faces ---
    # A chamfer face between two principal faces has a normal with significant
    # components in exactly 2 of 3 principal directions. The "diagonal
    # threshold" is how far off-axis a component must be to count as
    # significant. Using 0.25 (about 14.5 degrees from axis) is permissive
    # enough to catch 30/45/60 degree chamfers.
    diag_thresh = 0.25
    abs_normals = np.abs(normals)
    sorted_abs = np.sort(abs_normals, axis=1)  # ascending per row
    # Two largest components > diag_thresh AND smallest < diag_thresh
    diagonal_mask = (
        (sorted_abs[:, 1] > diag_thresh)
        & (sorted_abs[:, 2] > diag_thresh)
        & (sorted_abs[:, 0] < diag_thresh)
    )

    if diagonal_mask.sum() < min_cluster_faces:
        logger.debug("Only %d diagonal faces found (need %d), no chamfers",
                      int(diagonal_mask.sum()), min_cluster_faces)
        return []

    diag_indices = np.where(diagonal_mask)[0]
    diag_normals = normals[diag_indices]
    logger.debug("Found %d diagonal-normal candidate faces", len(diag_indices))

    # --- Cluster by normal direction ---
    clustering = DBSCAN(eps=normal_eps, min_samples=3).fit(diag_normals)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    logger.debug("DBSCAN produced %d clusters (%d noise faces)",
                  len(unique_labels), int((labels == -1).sum()))

    # --- Build face adjacency for connectivity validation ---
    # Two mesh faces are adjacent if they share an edge (2 vertices).
    from collections import defaultdict
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in diag_indices:
        tri = cleaned.faces[fi]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            key = (min(a, b), max(a, b))
            edge_to_faces[key].append(int(fi))

    def _faces_are_connected(face_set: set[int]) -> bool:
        """Check if faces form a single connected component via shared edges."""
        if len(face_set) <= 1:
            return True
        adj: dict[int, set[int]] = defaultdict(set)
        for faces_list in edge_to_faces.values():
            in_set = [f for f in faces_list if f in face_set]
            for i in range(len(in_set)):
                for j in range(i + 1, len(in_set)):
                    adj[in_set[i]].add(in_set[j])
                    adj[in_set[j]].add(in_set[i])
        # BFS
        start = next(iter(face_set))
        visited: set[int] = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj[node] - visited)
        return len(visited) == len(face_set)

    # --- Analyze each cluster ---
    results: list[dict] = []
    axis_names = ["X", "Y", "Z"]

    for label in sorted(unique_labels):
        cluster_idx = diag_indices[labels == label]
        if len(cluster_idx) < min_cluster_faces:
            logger.debug("Cluster %d: %d faces < min %d, skipping",
                          label, len(cluster_idx), min_cluster_faces)
            continue

        avg_normal = normals[cluster_idx].mean(axis=0)
        avg_normal /= (np.linalg.norm(avg_normal) + 1e-15)

        # --- Determine which two principal axes this chamfer sits between ---
        abs_avg = np.abs(avg_normal)
        # The edge is parallel to the axis with the SMALLEST normal component
        edge_axis_idx = int(np.argmin(abs_avg))
        edge_axis = axis_names[edge_axis_idx]
        # The two face-axes are the others, sorted by component magnitude
        face_axis_indices = sorted(
            [i for i in range(3) if i != edge_axis_idx],
            key=lambda i: abs_avg[i],
        )
        face_ax_a = face_axis_indices[0]  # smaller component
        face_ax_b = face_axis_indices[1]  # larger component

        # --- Strip width vs length check (narrow strip = chamfer) ---
        # Project cluster vertices onto the edge axis (length) and the
        # perpendicular plane (width).
        cluster_face_verts = cleaned.vertices[cleaned.faces[cluster_idx].flatten()]
        edge_dir = np.zeros(3)
        edge_dir[edge_axis_idx] = 1.0
        proj_length = cluster_face_verts @ edge_dir
        strip_length = float(proj_length.max() - proj_length.min())

        # Width: extent along the direction perpendicular to both edge and
        # the average normal (i.e. across the chamfer face).
        perp_dir = np.cross(edge_dir, avg_normal)
        perp_norm = np.linalg.norm(perp_dir)
        if perp_norm > 1e-10:
            perp_dir /= perp_norm
        else:
            # Fallback: use the face_ax_a direction
            perp_dir = np.zeros(3)
            perp_dir[face_ax_a] = 1.0
        proj_width = cluster_face_verts @ perp_dir
        strip_width = float(proj_width.max() - proj_width.min())

        if strip_length < 1e-10:
            logger.debug("Cluster %d: zero strip length, skipping", label)
            continue

        aspect_ratio = strip_length / max(strip_width, 1e-10)
        if aspect_ratio < 2.0:
            logger.debug(
                "Cluster %d: aspect ratio %.1f < 2.0 (length=%.2f, width=%.2f) "
                "— not a narrow strip, skipping (likely circular cutout)",
                label, aspect_ratio, strip_length, strip_width,
            )
            continue

        # --- Connectivity check ---
        cluster_set = set(int(x) for x in cluster_idx)
        if not _faces_are_connected(cluster_set):
            logger.debug("Cluster %d: %d faces not connected, skipping",
                          label, len(cluster_idx))
            continue

        # --- Edge coverage check ---
        # The chamfer strip should span >50% of the corresponding bbox
        # edge length (otherwise it is a scattered partial cluster).
        bbox_edge_length = float(bbox_size[edge_axis_idx])
        if bbox_edge_length > 1e-10:
            coverage = strip_length / bbox_edge_length
            if coverage < 0.5:
                logger.debug(
                    "Cluster %d: strip covers only %.0f%% of edge (%.2f / %.2f), skipping",
                    label, coverage * 100, strip_length, bbox_edge_length,
                )
                continue
        else:
            logger.debug("Cluster %d: zero bbox edge length on axis %s, skipping",
                          label, edge_axis)
            continue

        # --- Build CadQuery selector ---
        # Determine which side of each face-axis the chamfer is on.
        # We support chamfers between ANY two principal faces, not just
        # top/bottom.
        centroid = cluster_face_verts.mean(axis=0)
        bbox_center = (cleaned.bbox[0] + cleaned.bbox[1]) / 2.0

        # For the two face axes, determine <axis or >axis by comparing
        # the cluster centroid to the bbox center.
        sign_a = ">" if centroid[face_ax_a] > bbox_center[face_ax_a] else "<"
        sign_b = ">" if centroid[face_ax_b] > bbox_center[face_ax_b] else "<"

        # The selector uses "|edge_axis" to pick edges parallel to the
        # edge direction, then filters by the face the chamfer is
        # closest to. Use the face-axis with the LARGER normal component
        # (face_ax_b) as the primary filter for best specificity.
        face_filter = f"{sign_b}{axis_names[face_ax_b]}"
        selector = f"|{edge_axis} and {face_filter}"

        # Label for display
        side_label_b = "top" if sign_b == ">" else "bottom"
        side_label_a = "top" if sign_a == ">" else "bottom"
        label_str = f"{side_label_b}-{axis_names[face_ax_b]} {edge_axis}"

        # --- Compute chamfer dimensions ---
        # d2: inset along the face_ax_b direction (the face with larger
        # normal component). Measured from the bbox boundary.
        face_ext_vals = []
        for fi in cluster_idx:
            tri_v = cleaned.vertices[cleaned.faces[fi]]
            if sign_b == "<":
                face_ext_vals.append(float(tri_v[:, face_ax_b].max()))
            else:
                face_ext_vals.append(float(tri_v[:, face_ax_b].min()))
        face_ext_vals_arr = np.array(face_ext_vals)

        if sign_b == "<":
            ext_boundary = float(cleaned.bbox[0][face_ax_b])
            d2 = float(np.percentile(face_ext_vals_arr, 90)) - ext_boundary
        else:
            ext_boundary = float(cleaned.bbox[1][face_ax_b])
            d2 = ext_boundary - float(np.percentile(face_ext_vals_arr, 10))

        # d1: inset along face_ax_a. Computed from d2 and the chamfer angle
        # via the normal components: d1/d2 = |n_a| / |n_b|
        n_a = abs(avg_normal[face_ax_a])
        n_b = abs(avg_normal[face_ax_b])
        d1 = d2 * n_a / max(n_b, 0.01)

        d1 = abs(d1)
        d2 = abs(d2)

        # --- Scale-relative size filtering ---
        if d1 < size_min or d2 < size_min:
            logger.debug(
                "Cluster %d: chamfer too small (d1=%.3f, d2=%.3f, min=%.3f), skipping",
                label, d1, d2, size_min,
            )
            continue
        if d1 > size_max or d2 > size_max:
            logger.debug(
                "Cluster %d: chamfer too large (d1=%.3f, d2=%.3f, max=%.3f), skipping",
                label, d1, d2, size_max,
            )
            continue

        logger.info(
            "Chamfer cluster %d: %d faces, normal=[%.2f,%.2f,%.2f], "
            "d1=%.2f d2=%.2f, edge=|%s, selector='%s', "
            "strip L=%.2f W=%.2f aspect=%.1f coverage=%.0f%%",
            label, len(cluster_idx), *avg_normal, d1, d2,
            edge_axis, selector,
            strip_length, strip_width, aspect_ratio, coverage * 100,
        )
        results.append({
            "size": float(d1),
            "size2": float(d2),
            "selector": selector,
            "label": label_str,
            "face_count": len(cluster_idx),
            "edge_length": float(strip_length),
        })

    # --- Keep individual chamfers (do NOT merge by selector) ---
    # Different chamfers may share the same selector but have different
    # sizes. We keep them separate so CadQuery can apply each individually.
    # Sort largest first for stable ordering.
    results.sort(key=lambda x: -x["size"])

    logger.debug("Detected %d chamfer(s) total", len(results))
    return results


def _detect_mesh_fillets(
    cleaned: CleanedMesh, extrude_dir: np.ndarray,
) -> list[dict]:
    """Detect fillet edges from the mesh by analyzing smoothly-curving face strips.

    A fillet face is a narrow strip of faces whose normals sweep smoothly
    between two principal directions (forming an arc, unlike a chamfer which
    is a flat bevel with a single diagonal normal).

    Detection strategy:
    1. Identify faces whose normals are NOT axis-aligned (off-axis faces).
    2. Cluster these by spatial location (not by normal direction, since
       fillet normals rotate through an arc).
    3. For each cluster, verify:
       a) The normals sweep through an angular arc (not clustered at one angle
          like chamfers).
       b) The faces form a connected narrow strip (high aspect ratio).
       c) The strip width (fillet radius) is small relative to the part.
    4. Compute the fillet radius from the arc geometry.
    5. Build CadQuery edge selectors.

    Uses scale-relative thresholds based on the part's bounding-box diagonal.

    Returns a list of dicts, one per detected fillet:
        [{"radius": mm, "selector": "|Y and <Z", "label": "bottom Y",
          "face_count": N, "edge_length": mm}, ...]
    """
    from sklearn.cluster import DBSCAN

    bbox_diag = float(np.linalg.norm(cleaned.bbox[1] - cleaned.bbox[0]))
    bbox_size = cleaned.bbox[1] - cleaned.bbox[0]
    logger.debug(
        "Detecting mesh fillets: %d faces, extrude_dir=%s, bbox_diag=%.2f",
        len(cleaned.faces), extrude_dir.tolist(), bbox_diag,
    )

    # --- Scale-relative limits ---
    radius_min = bbox_diag * 0.001   # 0.1% of diagonal
    radius_max = bbox_diag * 0.05    # 5% of diagonal
    # Minimum faces in a cluster to be considered real geometry.
    min_cluster_faces = max(5, int(len(cleaned.faces) * 0.0005))

    normals = cleaned.normals
    ext = extrude_dir / (np.linalg.norm(extrude_dir) + 1e-15)

    # --- Identify off-axis faces (potential fillet faces) ---
    # Fillet faces have normals that are NOT aligned to any principal axis.
    # Unlike chamfer faces (which have normals diagonal between exactly 2
    # axes), fillet normals can point in any off-axis direction because
    # the arc sweeps through a range.
    #
    # We look for faces where the largest normal component is < 0.85
    # (more than ~32 degrees off any axis). This captures the curved
    # middle portion of fillets while excluding the principal faces and
    # the flat chamfer faces (which are caught by _detect_mesh_chamfers).
    abs_normals = np.abs(normals)
    max_component = abs_normals.max(axis=1)

    # Exclude nearly axis-aligned faces (principal faces)
    off_axis_mask = max_component < 0.85

    # Also exclude faces that look like chamfers (exactly 2 large + 1 small
    # component). Fillet faces typically have the normal rotating, so the
    # second-largest component varies more.
    sorted_abs = np.sort(abs_normals, axis=1)
    chamfer_like = (
        (sorted_abs[:, 1] > 0.25)
        & (sorted_abs[:, 2] > 0.25)
        & (sorted_abs[:, 0] < 0.25)
    )
    # Keep off-axis faces that are NOT chamfer-like
    fillet_candidate_mask = off_axis_mask & ~chamfer_like

    if fillet_candidate_mask.sum() < min_cluster_faces:
        logger.debug(
            "Only %d off-axis non-chamfer faces found (need %d), no fillets",
            int(fillet_candidate_mask.sum()), min_cluster_faces,
        )
        return []

    cand_indices = np.where(fillet_candidate_mask)[0]
    logger.debug("Found %d off-axis candidate fillet faces", len(cand_indices))

    # --- Cluster by spatial location ---
    # Unlike chamfers (clustered by normal direction), fillet faces along
    # the same edge have VARYING normals but are SPATIALLY close. So we
    # cluster by the face centroid position.
    face_centroids = np.zeros((len(cand_indices), 3))
    for i, fi in enumerate(cand_indices):
        tri = cleaned.vertices[cleaned.faces[fi]]
        face_centroids[i] = tri.mean(axis=0)

    # DBSCAN eps: scale-relative. Use ~2% of bbox diagonal as the
    # neighborhood radius — faces on the same fillet edge are very close.
    spatial_eps = bbox_diag * 0.02
    clustering = DBSCAN(eps=spatial_eps, min_samples=3).fit(face_centroids)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    logger.debug(
        "Spatial DBSCAN: %d clusters, %d noise faces (eps=%.2f)",
        len(unique_labels), int((labels == -1).sum()), spatial_eps,
    )

    # --- Build face adjacency for connectivity validation ---
    from collections import defaultdict
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in cand_indices:
        tri = cleaned.faces[fi]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            key = (min(a, b), max(a, b))
            edge_to_faces[key].append(int(fi))

    def _faces_are_connected(face_set: set[int]) -> bool:
        """Check if faces form a single connected component via shared edges."""
        if len(face_set) <= 1:
            return True
        adj: dict[int, set[int]] = defaultdict(set)
        for faces_list in edge_to_faces.values():
            in_set = [f for f in faces_list if f in face_set]
            for i in range(len(in_set)):
                for j in range(i + 1, len(in_set)):
                    adj[in_set[i]].add(in_set[j])
                    adj[in_set[j]].add(in_set[i])
        start = next(iter(face_set))
        visited: set[int] = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj[node] - visited)
        return len(visited) == len(face_set)

    # --- Analyze each cluster ---
    results: list[dict] = []
    axis_names = ["X", "Y", "Z"]

    for label in sorted(unique_labels):
        cluster_mask = labels == label
        cluster_idx = cand_indices[cluster_mask]
        if len(cluster_idx) < min_cluster_faces:
            logger.debug("Fillet cluster %d: %d faces < min %d, skipping",
                         label, len(cluster_idx), min_cluster_faces)
            continue

        cluster_normals = normals[cluster_idx]
        cluster_centroids = face_centroids[cluster_mask]

        # --- Normal arc check ---
        # Fillet faces should have normals that sweep through an angular
        # range (the arc of the fillet). Chamfer faces have normals all
        # pointing in roughly the same direction.
        # Compute the angular spread of normals in the cluster.
        mean_normal = cluster_normals.mean(axis=0)
        mean_normal /= (np.linalg.norm(mean_normal) + 1e-15)

        # Angle of each face normal relative to the mean
        dots = np.clip(cluster_normals @ mean_normal, -1, 1)
        angles_from_mean = np.degrees(np.arccos(dots))
        angular_spread = float(angles_from_mean.max() - angles_from_mean.min())

        # A fillet arc should have normals spanning at least 15 degrees
        # (otherwise it's nearly flat — more like a chamfer or noise).
        # It should span less than ~120 degrees (a 90-degree fillet
        # sweeps about 90 degrees of normal change).
        if angular_spread < 15:
            logger.debug(
                "Fillet cluster %d: angular spread %.1f° < 15° — "
                "too flat, skipping (likely chamfer or principal face)",
                label, angular_spread,
            )
            continue
        if angular_spread > 120:
            logger.debug(
                "Fillet cluster %d: angular spread %.1f° > 120° — "
                "too wide, skipping (likely complex geometry)",
                label, angular_spread,
            )
            continue

        # --- Determine the edge axis (direction along the fillet strip) ---
        # The fillet strip is elongated along the edge it rounds. Use PCA
        # on the cluster centroids to find the principal direction.
        centroid_mean = cluster_centroids.mean(axis=0)
        centered = cluster_centroids - centroid_mean

        # SVD to find principal direction
        try:
            _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
            principal_dir = vt[0]  # direction of maximum variance = strip direction
        except Exception:
            logger.debug("Fillet cluster %d: SVD failed, skipping", label)
            continue

        # Snap to the nearest principal axis
        abs_principal = np.abs(principal_dir)
        edge_axis_idx = int(np.argmax(abs_principal))
        edge_axis = axis_names[edge_axis_idx]

        # --- Strip dimensions ---
        cluster_face_verts = cleaned.vertices[cleaned.faces[cluster_idx].flatten()]
        edge_dir = np.zeros(3)
        edge_dir[edge_axis_idx] = 1.0
        proj_length = cluster_face_verts @ edge_dir
        strip_length = float(proj_length.max() - proj_length.min())

        # Width: extent perpendicular to both the edge direction and the
        # mean normal direction (across the fillet arc).
        perp_dir = np.cross(edge_dir, mean_normal)
        perp_norm = np.linalg.norm(perp_dir)
        if perp_norm > 1e-10:
            perp_dir /= perp_norm
        else:
            # Fallback: pick any axis perpendicular to the edge
            other_axes = [i for i in range(3) if i != edge_axis_idx]
            perp_dir = np.zeros(3)
            perp_dir[other_axes[0]] = 1.0

        proj_width = cluster_face_verts @ perp_dir
        strip_width = float(proj_width.max() - proj_width.min())

        if strip_length < 1e-10:
            logger.debug("Fillet cluster %d: zero strip length, skipping", label)
            continue

        # --- Aspect ratio check (narrow strip = fillet) ---
        aspect_ratio = strip_length / max(strip_width, 1e-10)
        if aspect_ratio < 2.0:
            logger.debug(
                "Fillet cluster %d: aspect ratio %.1f < 2.0 "
                "(L=%.2f, W=%.2f) — not a narrow strip, skipping",
                label, aspect_ratio, strip_length, strip_width,
            )
            continue

        # --- Connectivity check ---
        cluster_set = set(int(x) for x in cluster_idx)
        if not _faces_are_connected(cluster_set):
            logger.debug("Fillet cluster %d: %d faces not connected, skipping",
                         label, len(cluster_idx))
            continue

        # --- Edge coverage check ---
        # The fillet strip should span >40% of the corresponding bbox
        # edge length (lower threshold than chamfers since fillets may
        # not cover the full edge).
        bbox_edge_length = float(bbox_size[edge_axis_idx])
        if bbox_edge_length > 1e-10:
            coverage = strip_length / bbox_edge_length
            if coverage < 0.40:
                logger.debug(
                    "Fillet cluster %d: strip covers only %.0f%% of edge "
                    "(%.2f / %.2f), skipping",
                    label, coverage * 100, strip_length, bbox_edge_length,
                )
                continue
        else:
            logger.debug("Fillet cluster %d: zero bbox edge length on axis %s, skipping",
                         label, edge_axis)
            continue

        # --- Compute fillet radius from the arc geometry ---
        # Method: fit a circle to the cross-section of the fillet strip.
        # Project the cluster face vertices onto the plane perpendicular
        # to the edge direction, then find the radius of the arc they form.
        #
        # Simpler approach: the strip width is approximately
        # 2 * R * sin(sweep/2) and the strip "depth" (extent along the
        # mean normal) is R * (1 - cos(sweep/2)). For a 90-degree fillet,
        # width ~ R * sqrt(2) and depth ~ R * (1 - 1/sqrt(2)) ~ 0.29*R.
        #
        # Most robust: use the strip width and the angular spread to
        # estimate R. For a circular arc spanning angle theta:
        # chord = 2*R*sin(theta/2), so R = chord / (2*sin(theta/2))
        sweep_rad = np.radians(angular_spread)
        sin_half = np.sin(sweep_rad / 2)
        if sin_half > 0.1:
            # Use the strip width as the chord length
            fillet_radius = strip_width / (2.0 * sin_half)
        else:
            # Very small sweep — approximate
            fillet_radius = strip_width / 2.0

        # Cross-check: the strip width should also be consistent with
        # the radius estimate (not wildly off from the expected arc size)
        if fillet_radius < radius_min:
            logger.debug(
                "Fillet cluster %d: radius %.3f too small (min=%.3f), skipping",
                label, fillet_radius, radius_min,
            )
            continue
        if fillet_radius > radius_max:
            logger.debug(
                "Fillet cluster %d: radius %.3f too large (max=%.3f), skipping",
                label, fillet_radius, radius_max,
            )
            continue

        # --- Build CadQuery selector ---
        # Same pattern as chamfers: "|edge_axis and </>face_axis"
        # Determine which two face axes the fillet sits between.
        face_axis_indices = sorted(
            [i for i in range(3) if i != edge_axis_idx],
            key=lambda i: np.abs(mean_normal[i]),
        )
        face_ax_a = face_axis_indices[0]  # smaller normal component
        face_ax_b = face_axis_indices[1]  # larger normal component

        centroid = cluster_face_verts.mean(axis=0)
        bbox_center = (cleaned.bbox[0] + cleaned.bbox[1]) / 2.0

        sign_a = ">" if centroid[face_ax_a] > bbox_center[face_ax_a] else "<"
        sign_b = ">" if centroid[face_ax_b] > bbox_center[face_ax_b] else "<"

        face_filter = f"{sign_b}{axis_names[face_ax_b]}"
        selector = f"|{edge_axis} and {face_filter}"

        side_label_b = "top" if sign_b == ">" else "bottom"
        label_str = f"{side_label_b}-{axis_names[face_ax_b]} {edge_axis}"

        logger.info(
            "Fillet cluster %d: %d faces, mean_normal=[%.2f,%.2f,%.2f], "
            "radius=%.2f, sweep=%.0f°, edge=|%s, selector='%s', "
            "strip L=%.2f W=%.2f aspect=%.1f coverage=%.0f%%",
            label, len(cluster_idx), *mean_normal, fillet_radius, angular_spread,
            edge_axis, selector,
            strip_length, strip_width, aspect_ratio, coverage * 100,
        )
        results.append({
            "radius": float(fillet_radius),
            "selector": selector,
            "label": label_str,
            "face_count": len(cluster_idx),
            "edge_length": float(strip_length),
        })

    # Sort largest radius first for stable ordering.
    results.sort(key=lambda x: -x["radius"])

    logger.debug("Detected %d fillet(s) total", len(results))
    return results


def _estimate_mesh_volume(cleaned: CleanedMesh) -> float:
    """Estimate mesh volume using trimesh."""
    try:
        mesh = trimesh.Trimesh(
            vertices=cleaned.vertices, faces=cleaned.faces, process=False,
        )
        if mesh.is_watertight:
            vol = abs(float(mesh.volume))
            logger.debug("CQ mesh volume (watertight): %.2f", vol)
            return vol
        vol = abs(float(mesh.convex_hull.volume))
        logger.debug("CQ mesh volume (convex hull): %.2f", vol)
        return vol
    except Exception:
        extents = cleaned.bbox[1] - cleaned.bbox[0]
        vol = float(np.prod(extents))
        logger.debug("CQ mesh volume (bbox fallback): %.2f", vol)
        return vol
