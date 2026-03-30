"""Stage 6b: CadQuery Export.

Alternative offline export path that reconstructs the recognized part as a
CadQuery solid and exports a STEP file. No API calls required.

Strategy:
1. Slice the original mesh at the midplane to extract the 2D base profile.
2. Extrude the profile to form the base body.
3. Cut holes at detected positions and radii.
4. Return the solid for STEP export and visualization.
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
class CadQueryResult:
    """Output of the CadQuery export pipeline."""

    solid: object | None = None  # cq.Workplane, lazy-imported
    tess_vertices: np.ndarray | None = None  # (V, 3) tessellated solid verts
    tess_faces: np.ndarray | None = None  # (F, 3) tessellated solid faces
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
        }


def export_cadquery(
    feature_tree: FeatureTree,
    cleaned: CleanedMesh,
    output_path: str | Path | None = None,
) -> CadQueryResult:
    """Reconstruct the part as a CadQuery solid and optionally export STEP.

    Args:
        feature_tree: Output of Stage 5 (Feature Recognition).
        cleaned: Output of Stage 1 (for mesh geometry and volume).
        output_path: Path for the .step file. None skips file export.

    Returns:
        CadQueryResult with the solid, generated script, and validation.
    """
    mesh_volume = _estimate_mesh_volume(cleaned)

    # Extract base profile from mesh cross-section
    base_feature = feature_tree.features[0]
    extrude_dir = np.array(base_feature.direction)
    depth = base_feature.depth

    profile_2d, extrude_start, hole_features = _extract_profile_and_holes(
        feature_tree, cleaned, extrude_dir,
    )

    if profile_2d is None or len(profile_2d) < 3:
        logger.error("Failed to extract base profile from mesh")
        return CadQueryResult(mesh_volume=mesh_volume, status="FAIL")

    # Build the CadQuery solid
    solid, script = _build_solid(
        profile_2d, depth, extrude_dir, extrude_start, hole_features,
    )

    # Compute CAD volume
    try:
        import cadquery as cq
        cad_volume = abs(float(solid.val().Volume()))
    except Exception:
        cad_volume = 0.0

    volume_error = abs(cad_volume - mesh_volume) / (mesh_volume + 1e-15) * 100
    status = "PASS" if volume_error < 10.0 else "FAIL"

    # Tessellate for visualization (done now, before pyvista might conflict)
    tess_verts, tess_faces = _tessellate_solid(solid)

    # Export STEP
    step_path = ""
    if output_path is not None:
        import cadquery as cq
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(solid, str(output_path))
        step_path = str(output_path)
        logger.info("Exported STEP to %s", step_path)

    logger.info(
        "CadQuery export: mesh_vol=%.1f, cad_vol=%.1f, error=%.2f%%, status=%s",
        mesh_volume, cad_volume, volume_error, status,
    )

    return CadQueryResult(
        solid=solid,
        tess_vertices=tess_verts,
        tess_faces=tess_faces,
        script=script,
        step_path=step_path,
        mesh_volume=mesh_volume,
        cad_volume=cad_volume,
        volume_error_pct=volume_error,
        status=status,
    )


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------


def _extract_profile_and_holes(
    feature_tree: FeatureTree,
    cleaned: CleanedMesh,
    extrude_dir: np.ndarray,
) -> tuple[list[tuple[float, float]] | None, float, list[Feature]]:
    """Extract the 2D base profile and collect hole features.

    Slices the mesh at the midplane perpendicular to the extrusion direction.
    Returns (profile_2d, extrude_start, holes) where profile_2d is a list
    of (u, v) coordinates in the profile plane.
    """
    mesh = trimesh.Trimesh(
        vertices=cleaned.vertices, faces=cleaned.faces, process=False,
    )

    # Determine the two axes forming the profile plane
    abs_dir = np.abs(extrude_dir)
    primary_axis = int(np.argmax(abs_dir))
    axes = [0, 1, 2]
    axes.remove(primary_axis)
    ax_u, ax_v = axes

    # Extrusion start position (min along extrude axis)
    all_proj = cleaned.vertices[:, primary_axis]
    extrude_start = float(all_proj.min())

    # Slice at midpoint
    mid = (all_proj.min() + all_proj.max()) / 2.0
    origin = [0.0, 0.0, 0.0]
    origin[primary_axis] = mid
    normal = [0.0, 0.0, 0.0]
    normal[primary_axis] = 1.0

    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return None, extrude_start, []

        # Get 3D vertices of the cross-section directly
        # and project onto the profile axes
        section_verts = np.array(section.vertices)
        s2d, _ = section.to_planar()
        polygons = s2d.polygons_closed
    except Exception as e:
        logger.error("Mesh section failed: %s", e)
        return None, extrude_start, []

    if len(polygons) == 0:
        return None, extrude_start, []

    # Take largest polygon and map its 2D coords to mesh axes
    # The to_planar transform maps (ax_u, ax_v) -> 2D, but the mapping
    # isn't always identity. Instead, use the section's 3D vertices directly.
    # Find which 3D vertices correspond to the largest polygon's outline.
    largest = max(polygons, key=lambda p: p.area)
    poly_coords_2d = np.array(largest.exterior.coords)[:-1]

    # Match 2D polygon vertices to 3D section vertices via nearest-neighbor
    from scipy.spatial import cKDTree
    all_2d = s2d.vertices
    tree_2d = cKDTree(all_2d)
    _, indices = tree_2d.query(poly_coords_2d)

    # Get the corresponding 3D coordinates and project onto profile axes
    matched_3d = section_verts[indices]
    profile_2d = [(float(p[ax_u]), float(p[ax_v])) for p in matched_3d]
    profile_2d = _simplify_profile(profile_2d, tolerance=0.1)

    holes = [
        f for f in feature_tree.features
        if f.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)
    ]

    return profile_2d, extrude_start, holes


# ---------------------------------------------------------------------------
# CadQuery solid construction
# ---------------------------------------------------------------------------


def _simplify_profile(
    points: list[tuple[float, float]],
    tolerance: float = 0.1,
) -> list[tuple[float, float]]:
    """Remove near-duplicate consecutive points from a 2D profile."""
    if len(points) < 3:
        return points
    result = [points[0]]
    for p in points[1:]:
        dx = p[0] - result[-1][0]
        dy = p[1] - result[-1][1]
        if (dx * dx + dy * dy) > tolerance * tolerance:
            result.append(p)
    # Also check closing distance (last point vs first)
    if len(result) > 2:
        dx = result[-1][0] - result[0][0]
        dy = result[-1][1] - result[0][1]
        if (dx * dx + dy * dy) < tolerance * tolerance:
            result = result[:-1]
    return result


def _build_solid(
    profile_2d: list[tuple[float, float]],
    depth: float,
    extrude_dir: np.ndarray,
    extrude_start: float,
    holes: list[Feature],
) -> tuple[cq.Workplane, str]:
    """Build the CadQuery solid from profile and holes.

    Returns (solid, script_text).
    """
    abs_dir = np.abs(extrude_dir)
    primary_axis = int(np.argmax(abs_dir))
    axes = [0, 1, 2]
    axes.remove(primary_axis)
    ax_u, ax_v = axes

    # Determine workplane string
    plane_map = {0: "YZ", 1: "XZ", 2: "XY"}
    plane_name = plane_map[primary_axis]

    # Origin at the start face
    origin_3d = [0.0, 0.0, 0.0]
    origin_3d[primary_axis] = extrude_start

    import cadquery as cq

    # Build the solid
    wp = cq.Workplane(plane_name, origin=tuple(origin_3d))
    wp = wp.polyline(profile_2d).close().extrude(depth)

    # Generate script text
    script_lines = [
        "import cadquery as cq",
        "",
        f"# Base profile on {plane_name} plane",
        f"profile = {profile_2d!r}",
        "",
        f'result = cq.Workplane("{plane_name}", origin={tuple(origin_3d)})',
        "result = result.polyline(profile).close()",
        f"result = result.extrude({depth})",
    ]

    # Cut holes using direct cylinder subtraction at the actual 3D position
    # and direction. The feature center is the midpoint of the cylinder axis,
    # so we extend depth/2 in both directions to get the full span.
    # Through-holes get extra margin to ensure full penetration.
    for i, hole in enumerate(holes):
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

            script_lines.append(
                f"# Hole {i}: {'through' if is_through else 'blind'}, "
                f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                f"dir=({h_dir[0]:.2f}, {h_dir[1]:.2f}, {h_dir[2]:.2f}), "
                f"r={radius:.4f}, depth={h_depth:.4f}"
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
    """Tessellate a CadQuery solid into vertices and faces for visualization."""
    try:
        from OCP.StlAPI import StlAPI_Writer
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        import tempfile
        import os

        shape = solid.val().wrapped
        BRepMesh_IncrementalMesh(shape, 0.1, False, 0.1, True)

        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            writer = StlAPI_Writer()
            writer.Write(shape, tmp_path)
            mesh = trimesh.load(tmp_path)
            return np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.int64)
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        logger.warning("Tessellation failed: %s", e)
        return None, None


def _estimate_mesh_volume(cleaned: CleanedMesh) -> float:
    """Estimate mesh volume using trimesh."""
    try:
        mesh = trimesh.Trimesh(
            vertices=cleaned.vertices, faces=cleaned.faces, process=False,
        )
        if mesh.is_watertight:
            return abs(float(mesh.volume))
        return abs(float(mesh.convex_hull.volume))
    except Exception:
        extents = cleaned.bbox[1] - cleaned.bbox[0]
        return float(np.prod(extents))
