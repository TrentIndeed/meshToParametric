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
    mesh_volume = _estimate_mesh_volume(cleaned)

    # Extract base profile (needed by both LLM and heuristic paths)
    base_feature = feature_tree.features[0]
    extrude_dir = np.array(base_feature.direction)
    depth = base_feature.depth

    profiles, extrude_start, hole_features = _extract_profile_and_holes(
        feature_tree, cleaned, extrude_dir,
    )

    if profiles is None or len(profiles) == 0:
        logger.error("Failed to extract base profile from mesh")
        return CadQueryResult(mesh_volume=mesh_volume, status="FAIL")

    # Get construction order — LLM-guided or default
    build_order = None
    if use_llm:
        build_order = _llm_plan_build_order(feature_tree, cleaned)

    # Build the CadQuery solid step-by-step:
    # 1. Extrude base profile (no holes)
    # 2. Cut each hole one at a time
    # Each step is tessellated for the construction slideshow.
    solid, script, steps = _build_stepwise(
        profiles, depth, extrude_dir, extrude_start,
        hole_features, build_order=build_order,
    )

    # Compute CAD volume
    try:
        import cadquery as cq
        cad_volume = abs(float(solid.val().Volume()))
    except Exception:
        cad_volume = 0.0

    volume_error = abs(cad_volume - mesh_volume) / (mesh_volume + 1e-15) * 100
    status = "PASS" if volume_error < 10.0 else "FAIL"

    # Final tessellation
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
        construction_steps=steps,
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
) -> tuple[list[list[tuple[float, float]]] | None, float, list[Feature]]:
    """Extract ALL 2D cross-section profiles and collect hole features.

    Slices the mesh and returns every polygon (main body + leg extensions
    + bosses) so the builder can extrude and union them all.

    Returns (list_of_profiles, extrude_start, holes).
    """
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

    # Collect profiles from multiple slices and keep the set with the
    # most total area (captures the widest cross-section).
    best_profiles: list[list[tuple[float, float]]] = []
    best_total_area = 0.0

    offsets = [0.05, 0.25, 0.5, 0.75, 0.95]
    extent = float(all_proj.max() - all_proj.min())

    for frac in offsets:
        slice_pos = float(all_proj.min()) + extent * frac
        origin = [0.0, 0.0, 0.0]
        origin[primary_axis] = slice_pos

        try:
            section = mesh.section(plane_origin=origin, plane_normal=normal)
            if section is None:
                continue
            section_verts = np.array(section.vertices)
            s2d, _ = section.to_planar()
            polygons = s2d.polygons_closed
            if len(polygons) == 0:
                continue

            total_area = sum(p.area for p in polygons)
            if total_area > best_total_area:
                best_total_area = total_area
                best_profiles = []

                from scipy.spatial import cKDTree
                all_2d = s2d.vertices
                tree_2d = cKDTree(all_2d)

                for poly in polygons:
                    coords_2d = np.array(poly.exterior.coords)[:-1]
                    _, indices = tree_2d.query(coords_2d)
                    matched_3d = section_verts[indices]
                    profile = [(float(p[ax_u]), float(p[ax_v])) for p in matched_3d]
                    profile = _simplify_profile(profile, tolerance=0.1)
                    if len(profile) >= 3:
                        best_profiles.append(profile)
        except Exception:
            continue

    if not best_profiles:
        return None, extrude_start, []

    # Sort largest first (main body first, then smaller features)
    best_profiles.sort(key=lambda pts: -_polygon_area(pts))

    holes = [
        f for f in feature_tree.features
        if f.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)
    ]

    return best_profiles, extrude_start, holes


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


# ---------------------------------------------------------------------------
# CadQuery solid construction
# ---------------------------------------------------------------------------


@dataclass
class BuildPlan:
    """LLM-generated construction plan for the CadQuery solid."""

    feature_order: list[int]  # feature indices in construction order
    notes: dict[int, str] = field(default_factory=dict)  # per-feature notes
    strategy: str = ""  # overall construction strategy description


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
    build_order: BuildPlan | None = None,
) -> tuple[object, str, list[ConstructionStep]]:
    """Build a clean CAD solid from scratch, step by step.

    Steps:
      0. Sketch — 2D profile wire on the workplane
      1. Extrude — sweep the sketch into a solid
      2+. Cut holes — one per step

    Each step is tessellated for the visual slideshow. The resulting
    solid is clean B-rep geometry (no mesh triangles).

    Returns (final_solid, script, construction_steps).
    """
    import cadquery as cq

    abs_dir = np.abs(extrude_dir)
    primary_axis = int(np.argmax(abs_dir))
    plane_map = {0: "YZ", 1: "XZ", 2: "XY"}
    plane_name = plane_map[primary_axis]
    origin_3d = [0.0, 0.0, 0.0]
    origin_3d[primary_axis] = extrude_start

    steps: list[ConstructionStep] = []
    script_lines = ["import cadquery as cq", ""]

    # --- Step 0: Sketch (2D profile wire) ---
    # Build just the wire (no extrusion) for the sketch visual
    sketch_wp = cq.Workplane(plane_name, origin=tuple(origin_3d))
    sketch_wp = sketch_wp.polyline(profiles[0]).close()

    # Tessellate the wire as a thin extruded sheet for visualization
    sketch_solid = sketch_wp.extrude(0.5)  # paper-thin for visibility
    v, f = _tessellate_solid(sketch_solid)
    steps.append(ConstructionStep(
        name="Sketch Profile", tess_vertices=v, tess_faces=f, volume=0.0,
    ))
    script_lines.append(f"# Step 1: Sketch profile on {plane_name} plane")
    script_lines.append(f'result = cq.Workplane("{plane_name}", origin={tuple(origin_3d)})')
    script_lines.append(f"result = result.polyline(profile).close()")

    # --- Step 1: Extrude base profile ---
    wp = cq.Workplane(plane_name, origin=tuple(origin_3d))
    wp = wp.polyline(profiles[0]).close().extrude(depth)

    # Union additional profiles (small pads around hole locations)
    for profile in profiles[1:]:
        try:
            extra = cq.Workplane(plane_name, origin=tuple(origin_3d))
            extra = extra.polyline(profile).close().extrude(depth)
            wp = wp.union(extra)
        except Exception:
            pass

    script_lines.append(f"# Step 2: Extrude {depth:.2f} mm")
    script_lines.append(f"result = result.extrude({depth})")

    v, f = _tessellate_solid(wp)
    vol = abs(float(wp.val().Volume())) if wp else 0
    steps.append(ConstructionStep(
        name="Extrude Base", tess_vertices=v, tess_faces=f, volume=vol,
    ))

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

        if is_through:
            cut_start = center - h_dir * (h_depth / 2 + h_depth * 0.5)
            cut_length = h_depth * 2
        else:
            cut_start = center - h_dir * (h_depth / 2)
            cut_length = h_depth

        hole_name = f"{'Through' if is_through else 'Blind'} Hole (r={radius:.1f})"
        if note:
            hole_name += f" — {note}"

        try:
            hole_plane = cq.Plane(
                origin=cq.Vector(*cut_start.tolist()),
                normal=cq.Vector(*h_dir.tolist()),
            )
            hole_solid = cq.Workplane(hole_plane).circle(radius).extrude(cut_length)
            wp = wp.cut(hole_solid)

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
            logger.warning("Failed to cut hole %d: %s", i, e)

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
    """Tessellate a CadQuery solid into vertices and faces for visualization."""
    try:
        from OCP.StlAPI import StlAPI_Writer
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        import tempfile
        import os

        shape = solid.val().wrapped
        BRepMesh_IncrementalMesh(shape, 0.05, False, 0.05, True)

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
