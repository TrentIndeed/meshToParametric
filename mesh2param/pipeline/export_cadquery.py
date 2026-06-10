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
    sketch_overlay: np.ndarray | None = None  # (N, 3) — pairs of 3D points for blue sketch lines


@dataclass
class BuildVariant:
    """One build variant — a specific construction order for the same part."""

    name: str  # e.g. "LLM Optimized", "Largest First", etc.
    strategy: str  # one-line description of the build approach
    construction_steps: list[ConstructionStep] = field(default_factory=list)
    script: str = ""
    cad_volume: float = 0.0


@dataclass
class CadQueryResult:
    """Output of the CadQuery export pipeline."""

    solid: object | None = None  # cq.Workplane, lazy-imported
    tess_vertices: np.ndarray | None = None  # (V, 3) tessellated solid verts
    tess_faces: np.ndarray | None = None  # (F, 3) tessellated solid faces
    # Snapshot of the solid right BEFORE fillets/chamfers were applied —
    # consumed by the "Fillet & Chamfer Intent" pipeline stage so the
    # viewer can show the un-blended body with cyan overlay markers
    # before the apply stage.
    pre_blend_tess_vertices: np.ndarray | None = None
    pre_blend_tess_faces: np.ndarray | None = None
    # Records the intent stage draws as cyan overlays.
    #   {"fillets": [{radius, axis_dir, axis_point, length}, ...],
    #    "chamfers": [{corner, along_a, along_b, size}, ...]}
    intent_records: dict = field(default_factory=dict)
    # The actual CAD edges (list of [[x,y,z],[x,y,z]] segments) that
    # received a fillet/chamfer — clean lines lying on the model, used by
    # the intent overlay instead of synthetic circles.
    blend_edge_overlay: list = field(default_factory=list)
    # Combined intent-stage mesh: the un-blended solid (gray) plus
    # analytic fillet/chamfer surface patches (cyan/orange), as one
    # per-face-colored mesh so the viewer shows what each blend looks
    # like. {"vertices": [...], "faces": [...], "colors": [...]} or {}.
    intent_mesh: dict = field(default_factory=dict)
    # Per-record match diagnostics from the fillet matcher (matched /
    # why-not), for debugging the blend-vs-apply gap. See
    # _apply_recognized_fillets_to_solid(out_diag=...).
    fillet_match_diag: list = field(default_factory=list)
    construction_steps: list[ConstructionStep] = field(default_factory=list)
    variants: list[BuildVariant] = field(default_factory=list)
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
    num_variants: int = 1,
    body_results: list | None = None,
    stitched_clusters: list[dict] | None = None,
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

    # Extract base profile (needed by both LLM and heuristic paths).
    # For multi-body parts, use the primary body's sub_cleaned mesh so the
    # profile/extrude_start reflect that body alone, not the full assembly.
    base_feature = feature_tree.features[0]
    extrude_dir = np.array(base_feature.direction)
    depth = base_feature.depth

    if body_results and len(body_results) > 0:
        primary_body, primary_sub, primary_tree = body_results[0]
        primary_base = primary_tree.features[0] if primary_tree.features else None
        if primary_base is not None:
            extrude_dir = np.array(primary_base.direction)
            depth = primary_base.depth
        else:
            extrude_dir = primary_body.extrude_direction
            depth = primary_body.depth
        profile_mesh = primary_sub
        profile_tree = primary_tree
    else:
        profile_mesh = cleaned
        profile_tree = feature_tree

    try:
        profiles, extrude_start, hole_features, chamfer_features, fillet_features = \
            _extract_profile_and_holes(profile_tree, profile_mesh, extrude_dir)
    except Exception:
        logger.exception(
            "Stage 6b failed extracting profile: extrude_dir=%s, depth=%.2f",
            extrude_dir.tolist(), depth,
        )
        raise

    if profiles is None or len(profiles) == 0:
        logger.error("Failed to extract base profile from mesh")
        return CadQueryResult(mesh_volume=mesh_volume, status="FAIL")

    # --- Multi-body construction ---
    # If body_results are provided, build each body and union them.
    # Each body gets its own profile extraction and extrusion.
    additional_bodies_data = []
    if body_results and len(body_results) > 1:
        logger.info("Multi-body export: %d bodies", len(body_results))
        for bi, (body, sub_cleaned, body_tree) in enumerate(body_results[1:], 1):
            try:
                body_dir = body.extrude_direction
                body_depth = body.depth
                body_base = body_tree.features[0] if body_tree.features else None
                if body_base:
                    body_dir = np.array(body_base.direction)
                    body_depth = body_base.depth
                body_profiles, body_start, body_holes, body_chamfers, body_fillets = \
                    _extract_profile_and_holes(body_tree, sub_cleaned, body_dir)
                if body_profiles:
                    additional_bodies_data.append({
                        'profiles': body_profiles,
                        'depth': body_depth,
                        'extrude_dir': body_dir,
                        'extrude_start': body_start,
                        'holes': body_holes,
                        'chamfers': body_chamfers,
                        'fillets': body_fillets,
                        'cleaned': sub_cleaned,
                        'body': body,
                    })
                    logger.info("Body %d: %d profiles, depth=%.1f, dir=%s",
                                bi, len(body_profiles), body_depth, body_dir.tolist())
            except Exception:
                logger.warning("Failed to extract profile for body %d, skipping", bi)

    # --- Build variants with different construction orders ---
    # For multi-body, pass the primary body's sub_cleaned as `cleaned` so
    # all downstream logic (chamfer detection, hole cutting bbox, etc.)
    # operates on that body's geometry, not the full assembly.
    build_cleaned = profile_mesh
    build_args = dict(
        profiles=profiles, depth=depth, extrude_dir=extrude_dir,
        extrude_start=extrude_start, holes=hole_features,
        chamfers=chamfer_features, fillets=fillet_features, cleaned=build_cleaned,
    )

    logger.debug(
        "Building CadQuery solid: %d profiles, depth=%.2f, %d holes, %d chamfers, %d fillets",
        len(profiles), depth, len(hole_features), len(chamfer_features), len(fillet_features),
    )

    variants: list[BuildVariant] = []

    # LLM-guided build order is currently disabled — the pipeline runs
    # heuristic-only construction until the heuristic baseline is solid
    # enough that an LLM reorder is a measurable improvement, not a
    # confound when debugging fillet/edge issues.
    llm_order = None
    is_multi_body = body_results is not None and len(body_results) > 1
    llm_future = None
    # if is_multi_body:
    #     llm_future = None
    # elif use_llm and num_features >= 10:
    #     from concurrent.futures import ThreadPoolExecutor, as_completed
    #     # Start LLM call in background while building the default variant
    #     llm_future = ThreadPoolExecutor(1).submit(_llm_plan_build_order, feature_tree, cleaned)
    # elif use_llm:
    #     llm_future = None
    #     # For simple parts, try a quick LLM call with short timeout
    #     try:
    #         llm_order = _llm_plan_build_order(feature_tree, cleaned)
    #     except Exception:
    #         pass
    # else:
    #     llm_future = None

    # Build default variant immediately (no waiting for LLM)
    result_pre_blend_tess: tuple[np.ndarray, np.ndarray] | None = None
    intent_records_out: dict = {"fillets": [], "chamfers": []}
    blend_edge_overlay_out: list = []
    blend_surface_patches_out: list = []
    fillet_match_diag_out: list = []
    try:
        # Multi-body: skip per-body fillet application so the union doesn't
        # freeze fillets at body boundaries. Fillets get applied once to the
        # combined solid below, after all bodies are unioned.
        # Always defer chamfers in the default variant so the
        # subprocess can hand the un-blended solid + intent records to
        # job_queue's two post-build pipeline stages (intent overlay,
        # then apply).
        deferred_chamfers_all: list[dict] = []
        solid, script, steps, body0_chamfers = _build_stepwise(
            **build_args, build_order=llm_order,
            skip_fillets=is_multi_body, skip_chamfers=True,
        )
        deferred_chamfers_all.extend(body0_chamfers)

        # Multi-body: build each additional body and union into the base.
        # Each body step shows the accumulated solid + this step's body geometry
        # so the viewer displays progressive construction with all prior bodies.
        if additional_bodies_data and solid is not None:
            for abd in additional_bodies_data:
                try:
                    body_solid, body_script, body_steps, body_chamfers = _build_stepwise(
                        profiles=abd['profiles'], depth=abd['depth'],
                        extrude_dir=abd['extrude_dir'], extrude_start=abd['extrude_start'],
                        holes=abd['holes'], chamfers=abd['chamfers'],
                        fillets=abd['fillets'], cleaned=abd['cleaned'],
                        skip_fillets=True, skip_chamfers=True,
                    )
                    deferred_chamfers_all.extend(body_chamfers)
                    if body_solid is None:
                        logger.warning("Body %d: _build_stepwise returned None",
                                       abd['body'].body_id)
                        continue

                    # Snapshot the accumulated base tessellation — every
                    # new body step gets merged with this so all prior
                    # bodies stay visible in frame.
                    base_v, base_f = _tessellate_solid(solid, fast=True)
                    base_v_arr = np.asarray(base_v, dtype=np.float64)
                    base_f_arr = np.asarray(base_f, dtype=np.int64)
                    base_vert_count = len(base_v_arr)

                    # Merge each body step's tessellation with the base
                    for bstep in body_steps:
                        is_sketch = "sketch" in bstep.name.lower()
                        # If the step already has a sketch_overlay set (e.g.
                        # "Sketch Cutout" from profile-based cuts), its tess
                        # is already the solid state — just offset it.
                        already_has_overlay = (
                            getattr(bstep, "sketch_overlay", None) is not None
                        )
                        if bstep.tess_vertices is None or bstep.tess_faces is None:
                            steps.append(bstep)
                            continue
                        step_v = np.asarray(bstep.tess_vertices, dtype=np.float64)
                        step_f = np.asarray(bstep.tess_faces, dtype=np.int64)
                        if len(step_v) == 0 or len(step_f) == 0:
                            steps.append(bstep)
                            continue

                        if already_has_overlay:
                            # Step's tess is a solid (base of the current body)
                            # and overlay is already set. Combine the body's
                            # solid with the accumulated base like a normal
                            # step, keep the overlay untouched.
                            if base_vert_count > 0:
                                combined_v = np.vstack([base_v_arr, step_v])
                                combined_f = np.vstack(
                                    [base_f_arr, step_f + base_vert_count]
                                )
                                bstep.tess_vertices = combined_v
                                bstep.tess_faces = combined_f
                            steps.append(bstep)
                            continue

                        if is_sketch and base_vert_count > 0:
                            # For sketch steps: show the base solid with the
                            # sketch edges drawn on top as a blue overlay.
                            # Convert sketch face triangles (degenerate = edges)
                            # into point pairs for sketch_overlay.
                            overlay_pts = []
                            for fi in range(len(step_f)):
                                a, b = int(step_f[fi, 0]), int(step_f[fi, 1])
                                overlay_pts.append(step_v[a])
                                overlay_pts.append(step_v[b])
                            bstep.tess_vertices = base_v_arr
                            bstep.tess_faces = base_f_arr
                            bstep.sketch_overlay = np.array(overlay_pts, dtype=np.float64) if overlay_pts else None
                            # Rename so frontend treats it as a solid+overlay, not pure sketch
                            # (the endpoint checks "sketch" in name.lower() for is_sketch)
                        else:
                            # Combine: stack vertices, offset body faces by base count
                            combined_v = np.vstack([base_v_arr, step_v]) if base_vert_count > 0 else step_v
                            combined_f = (
                                np.vstack([base_f_arr, step_f + base_vert_count])
                                if base_vert_count > 0 else step_f
                            )
                            bstep.tess_vertices = combined_v
                            bstep.tess_faces = combined_f
                        steps.append(bstep)

                    # Union into the accumulated solid for the next body
                    vol_before = abs(float(solid.val().Volume()))
                    solid = solid.union(body_solid)
                    # Fuse near-coplanar adjacent faces left by the
                    # irregular extruded polygons. Without this the
                    # solid has many small face fragments that OCC's
                    # BRepFilletAPI refuses to fillet ("BRep_API:
                    # command not done"). .clean() invokes
                    # ShapeUpgrade_UnifySameDomain internally.
                    try:
                        solid = solid.clean()
                    except Exception as _clean_exc:
                        logger.debug("Workplane.clean failed: %s", _clean_exc)
                    vol_after = abs(float(solid.val().Volume()))
                    v, f = _tessellate_solid(solid, fast=True)
                    steps.append(ConstructionStep(
                        name="Combine Bodies",
                        tess_vertices=v, tess_faces=f, volume=vol_after,
                    ))
                    script += f"\n# Body {abd['body'].body_id}\n"
                    script += body_script
                    logger.info("Combined body %d: vol %.0f -> %.0f, added %d steps",
                                abd['body'].body_id, vol_before, vol_after,
                                len(body_steps) + 1)
                except Exception as e:
                    logger.warning("Failed to build body %d: %s",
                                   abd['body'].body_id, e)

        # --- Post-union / single-body fillet pass ---
        # Prefer the pre-peeled fillet list (stage 1.75) — those were
        # detected on the full, un-decomposed mesh so each fillet record
        # has the TRUE edge length and radius, not a fragment chopped by
        # a body boundary. Fall back to per-body mesh cluster detection
        # only if peeled_fillets wasn't supplied (legacy path / testing).
        try:
            if solid is not None:
                clusters_in = list(stitched_clusters or [])

                # --- Snapshot the un-blended solid for the Stage 7
                # (Fillet/Chamfer Intent) pipeline-stage viz. The
                # subprocess returns this alongside the final solid so
                # job_queue can show the unfilleted body with cyan
                # overlay before showing the post-apply result.
                try:
                    pre_v, pre_f = _tessellate_solid(solid, fast=True)
                    result_pre_blend_tess = (
                        np.asarray(pre_v, dtype=np.float64),
                        np.asarray(pre_f, dtype=np.int64),
                    )
                except Exception as pre_exc:
                    logger.debug("Pre-blend tessellation failed: %s", pre_exc)
                    result_pre_blend_tess = None
                intent_records_out = {
                    "fillets": [
                        {
                            "group_id": int(c.get("group_id", -1)),
                            "n_records": int(c.get("n_records", 0)),
                            "n_vertices": int(
                                len(c.get("vertices", []))
                                if c.get("vertices") is not None else 0
                            ),
                        }
                        for c in clusters_in
                    ],
                    "chamfers": [
                        {
                            "corner": np.asarray(c.get("corner"), dtype=float).tolist(),
                            "along_a": np.asarray(c.get("along_a"), dtype=float).tolist(),
                            "along_b": np.asarray(c.get("along_b"), dtype=float).tolist(),
                            "size": float(c.get("size", 0.0)),
                        }
                        for c in deferred_chamfers_all
                        if c.get("corner") is not None
                    ],
                }

                # Accumulators for the intent overlay: the actual CAD
                # edges that receive a blend (clean lines) and analytic
                # surface patches (quarter-round fillets / bevel chamfers)
                # so the viz shows what each blend looks like.
                blend_edge_overlay = blend_edge_overlay_out
                blend_surface_patches = blend_surface_patches_out

                # --- Apply deferred chamfers ---
                if deferred_chamfers_all:
                    try:
                        solid, cham_steps, cham_script = _apply_deferred_chamfers(
                            solid, deferred_chamfers_all,
                            out_edges=blend_edge_overlay,
                            out_surfaces=blend_surface_patches,
                        )
                        steps.extend(cham_steps)
                        if cham_script:
                            script += "\n# --- Deferred chamfers ---\n"
                            script += "\n".join(cham_script) + "\n"
                        logger.info(
                            "Deferred chamfer pass: %d/%d applied",
                            len(cham_steps), len(deferred_chamfers_all),
                        )
                    except Exception as cham_exc:
                        logger.warning(
                            "Deferred chamfer pass failed: %s", cham_exc,
                        )

                # --- Apply fillets ---
                # Vertex-proximity matcher: each stitched cluster's
                # face-vertex set picks out the CAD edges that belong
                # to that physical fillet, and the radius is measured
                # per-edge from the same vertex set.
                if clusters_in:
                    solid, fillet_steps, fillet_script = (
                        _apply_recognized_fillets_to_solid(
                            solid,
                            stitched_clusters=clusters_in,
                            out_edges=blend_edge_overlay,
                            out_surfaces=blend_surface_patches,
                            out_diag=fillet_match_diag_out,
                        )
                    )
                    for fs in fillet_steps:
                        steps.append(fs)
                    if fillet_script:
                        script += "\n# --- Stitched fillets ---\n"
                        script += "\n".join(fillet_script) + "\n"
                    logger.info(
                        "Fillet pass: %d fillet(s) applied from %d stitched cluster(s)",
                        len(fillet_steps), len(clusters_in),
                    )
        except Exception as fil_exc:
            logger.warning("Stitched fillet apply pass failed: %s", fil_exc)
        logger.info("CHECKPOINT: post-fillet-apply, continuing to variants/tess")

        v1_name = "AI Optimized" if llm_order else "Default"
        v1_strategy = llm_order.strategy if llm_order else "Default heuristic ordering"
        try:
            v1_vol = abs(float(solid.val().Volume()))
        except Exception:
            v1_vol = 0.0
        variants.append(BuildVariant(
            name=v1_name, strategy=v1_strategy,
            construction_steps=steps, script=script, cad_volume=v1_vol,
        ))
    except Exception:
        logger.exception("Variant 1 (LLM/default) failed")
        solid, script, steps = None, "", []
        deferred_chamfers_all = []

    # If LLM was running in background, collect result and add as a variant
    if llm_future is not None:
        try:
            llm_result = llm_future.result(timeout=30)
            if llm_result:
                llm_solid, llm_script, llm_steps, _ = _build_stepwise(
                    **build_args, build_order=llm_result,
                )
                try:
                    llm_vol = abs(float(llm_solid.val().Volume()))
                except Exception:
                    llm_vol = 0.0
                variants.append(BuildVariant(
                    name="AI Optimized", strategy=llm_result.strategy,
                    construction_steps=llm_steps, script=llm_script, cad_volume=llm_vol,
                ))
                # Promote AI variant to first position
                if len(variants) > 1:
                    variants[0], variants[-1] = variants[-1], variants[0]
                    solid = llm_solid  # use AI solid as primary
        except Exception:
            logger.debug("LLM build order timed out or failed, using default")

    # Variants 2-N: Heuristic orderings (only if num_variants > 1).
    # Skip for multi-body — heuristic variants only rebuild body 0.
    heuristic_orders = (
        _make_heuristic_orders(feature_tree.features)
        if num_variants > 1 and not is_multi_body
        else []
    )
    for h_name, h_strategy, h_plan in heuristic_orders[:num_variants - 1]:
        try:
            h_solid, h_script, h_steps, _ = _build_stepwise(
                **build_args, build_order=h_plan,
            )
            try:
                h_vol = abs(float(h_solid.val().Volume()))
            except Exception:
                h_vol = 0.0
            variants.append(BuildVariant(
                name=h_name, strategy=h_strategy,
                construction_steps=h_steps, script=h_script, cad_volume=h_vol,
            ))
        except Exception:
            logger.warning("Variant '%s' failed, skipping", h_name)

    logger.info("Built %d variants: %s", len(variants), [v.name for v in variants])

    # Primary result uses the first (best) variant
    cad_volume = variants[0].cad_volume if variants else 0.0
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

    pre_blend_v = result_pre_blend_tess[0] if result_pre_blend_tess else None
    pre_blend_f = result_pre_blend_tess[1] if result_pre_blend_tess else None

    # Assemble the intent-stage mesh: the un-blended solid in gray, plus
    # each analytic fillet (cyan) / chamfer (orange) surface patch, as one
    # per-face-colored mesh. The patch verts are nudged very slightly
    # outward along their face normals to avoid z-fighting with the solid.
    intent_mesh_out: dict = {}
    if pre_blend_v is not None and pre_blend_f is not None:
        try:
            GRAY = [0.62, 0.64, 0.67]
            CYAN = [0.0, 0.85, 0.95]
            ORANGE = [1.0, 0.55, 0.0]
            base_v = np.asarray(pre_blend_v, dtype=float).reshape(-1, 3)
            base_f = np.asarray(pre_blend_f, dtype=np.int64).reshape(-1, 3)
            all_v = [base_v]
            all_f = [base_f]
            face_colors = [np.tile(GRAY, (len(base_f), 1))]
            voff = len(base_v)
            for patch in blend_surface_patches_out:
                pv = np.asarray(patch["verts"], dtype=float).reshape(-1, 3)
                pf = np.asarray(patch["faces"], dtype=np.int64).reshape(-1, 3)
                if len(pv) == 0 or len(pf) == 0:
                    continue
                col = CYAN if patch.get("kind") == "fillet" else ORANGE
                all_v.append(pv)
                all_f.append(pf + voff)
                face_colors.append(np.tile(col, (len(pf), 1)))
                voff += len(pv)
            merged_v = np.vstack(all_v)
            merged_f = np.vstack(all_f)
            merged_c = np.vstack(face_colors)
            intent_mesh_out = {
                "vertices": merged_v.flatten().tolist(),
                "faces": merged_f.flatten().tolist(),
                "colors": merged_c.flatten().tolist(),
            }
        except Exception as _im_exc:
            logger.debug("Intent mesh assembly failed: %s", _im_exc)
            intent_mesh_out = {}

    result = CadQueryResult(
        solid=solid,
        tess_vertices=tess_verts,
        tess_faces=tess_faces,
        pre_blend_tess_vertices=pre_blend_v,
        pre_blend_tess_faces=pre_blend_f,
        intent_records=intent_records_out,
        blend_edge_overlay=blend_edge_overlay_out,
        intent_mesh=intent_mesh_out,
        fillet_match_diag=fillet_match_diag_out,
        construction_steps=steps,
        variants=variants,
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

    # Collect profiles from multiple slices across the full depth.
    # Pick the slice with the largest outer profile area — this finds the
    # widest cross-section, avoiding positions where holes/pockets reduce it.
    extent = float(all_proj.max() - all_proj.min())

    candidates: list[tuple[float, list]] = []  # (-area, profiles)

    # Sample slices across the full depth: top, bottom, and intermediate
    # positions.  Intermediate slices avoid blind holes near the faces and
    # find the true widest outer boundary.
    surface_inset = 0.1  # mm inside the surface
    z_min = float(all_proj.min())
    z_max = float(all_proj.max())
    slice_positions = [
        z_min + surface_inset,                   # bottom face
        z_max - surface_inset,                   # top face
    ]
    # Add intermediate slices (10%, 25%, 50%, 75%, 90% of depth)
    for frac in [0.10, 0.25, 0.50, 0.75, 0.90]:
        pos = z_min + extent * frac
        # Avoid duplicates near the surface inset positions
        if pos > z_min + surface_inset + 0.2 and pos < z_max - surface_inset - 0.2:
            slice_positions.append(pos)
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
                # - Large + inside outer = cutout (window/pocket/large hole)
                # - Small circular + inside outer = bolt hole cross-section, DROP
                # - Outside outer = split piece, union to extend boundary
                cutouts = []
                to_union = [outer]
                for sp in projected_polys[1:]:
                    centroid = sp.centroid
                    if outer.contains(centroid):
                        circ = 4.0 * 3.14159 * sp.area / (sp.length ** 2) if sp.length > 0 else 0
                        is_circular = circ > 0.70
                        area_ratio = sp.area / outer.area
                        # Large features (>20% of outer) are always cutouts,
                        # even if circular — they're structural (windows, large bores).
                        # Small circular features (<10%) are bolt holes — drop them.
                        if area_ratio > 0.20:
                            cutouts.append(sp)
                        elif area_ratio > 0.10 and not is_circular:
                            cutouts.append(sp)
                        # else: small hole cross-section, skip
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

                # DON'T subtract cutouts from the profile — use the solid
                # outer boundary. Cutouts (windows, pockets) become separate
                # cut features applied after the base extrusion.
                outer_boundary_area = merged.area

                outer_coords = list(merged.exterior.coords)[:-1]
                outer_coords = _simplify_profile(outer_coords, tolerance=0.1)
                slice_profiles = [outer_coords] if len(outer_coords) >= 3 else []
            else:
                slice_profiles = []

            if not slice_profiles:
                continue

            # Prefer the slice with the largest outer boundary.
            # Using the boundary area (before hole subtraction) ensures
            # slices through blind holes aren't penalized.
            candidates.append((-outer_boundary_area, slice_profiles))
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
        # Check if this is truly a frame split (outer boundary broken into pieces)
        # vs a cutout (one large polygon contains another).
        # If the largest profile contains the centroid of the 2nd largest,
        # it's a cutout, not a split.
        from shapely.geometry import Polygon as _SP, Point as _PT
        sorted_by_area = sorted(best_profiles, key=lambda p: -abs(_polygon_area(p)))
        outer_poly = _SP(sorted_by_area[0])
        second_poly = _SP(sorted_by_area[1]) if len(sorted_by_area) > 1 else None
        is_cutout = second_poly and outer_poly.contains(_PT(second_poly.centroid))
        if largest_area < total_area * 0.7 and not is_cutout:
            # Frame is split — find a nearby slice where the frame is one
            # connected piece but total area is similar (not a filled cutout).
            from shapely.geometry import Polygon as SPoly
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

    # Validate profile area against expected area from bbox.
    # If the slice-based profile is too small, fall back to projecting
    # all mesh faces onto the profile plane (works for thin plates where
    # slicing produces open contours instead of closed polygons).
    if best_profiles:
        bbox_size = cleaned.bbox[1] - cleaned.bbox[0]
        sorted_dims = sorted(bbox_size.tolist(), reverse=True)
        expected_area = sorted_dims[0] * sorted_dims[1]
        profile_area = abs(_polygon_area(best_profiles[0]))

        if profile_area < expected_area * 0.3:
            logger.warning(
                "Profile area %.0f is <30%% of expected %.0f, "
                "falling back to face projection",
                profile_area, expected_area,
            )
            from shapely.geometry import Polygon as SProjPoly
            from shapely.ops import unary_union as proj_union

            # Only project faces that are roughly perpendicular to the
            # extrude direction — these are the top/bottom faces of the
            # extruded body. Faces parallel to the extrude direction are
            # the side walls or cut cap and should not contribute to the
            # profile outline.
            ext = extrude_dir / (np.linalg.norm(extrude_dir) + 1e-15)
            face_normals = cleaned.normals
            dot = np.abs(face_normals @ ext)
            top_bottom_mask = dot > 0.9  # faces whose normal is within ~25° of extrude dir

            face_polys = []
            for fi in np.where(top_bottom_mask)[0]:
                tri = cleaned.faces[fi]
                tri_pts = [
                    (float(cleaned.vertices[v][ax_u]), float(cleaned.vertices[v][ax_v]))
                    for v in tri
                ]
                try:
                    p = SProjPoly(tri_pts)
                    if p.is_valid and p.area > 0.001:
                        face_polys.append(p)
                except Exception:
                    pass

            if face_polys:
                merged = proj_union(face_polys)
                if merged.geom_type == 'MultiPolygon':
                    merged = max(merged.geoms, key=lambda g: g.area)
                if merged.area > profile_area:
                    outer_coords = list(merged.exterior.coords)[:-1]
                    outer_coords = _simplify_profile(outer_coords, tolerance=0.5)
                    if len(outer_coords) >= 3:
                        best_profiles = [outer_coords]
                        logger.info("Face projection profile: area=%.0f (vs slice %.0f)",
                                    merged.area, profile_area)

    # ── 2D-based hole detection ──
    # The canonical way to detect holes in a plate: project all top/bottom
    # faces (those perpendicular to the extrude direction) onto the 2D
    # profile plane, union them into a polygon, and any INTERIOR RINGS in
    # that polygon are physical holes through the plate. This is simpler
    # and more reliable than 3D cylinder fitting because holes always
    # appear as inner rings of the cross-section regardless of mesh noise.
    #
    # This REPLACES the recognition-based holes with 2D-detected ones
    # whenever the 2D detection produces results — that way we always get
    # holes that are perpendicular to the plate face (not random Y/Z axes
    # from misidentified cylindrical features).
    try:
        from shapely.geometry import Polygon as _HolePoly
        from shapely.ops import unary_union as _HoleUnion
        from mesh2param.models.feature import Feature, FeatureType as _FType

        ext_norm = extrude_dir / (np.linalg.norm(extrude_dir) + 1e-15)
        face_dots = np.abs(cleaned.normals @ ext_norm)
        top_bot_mask = face_dots > 0.9
        top_bot_faces = np.where(top_bot_mask)[0]

        face_polys = []
        for fi in top_bot_faces:
            tri = cleaned.faces[fi]
            tri_pts = [
                (float(cleaned.vertices[v][ax_u]), float(cleaned.vertices[v][ax_v]))
                for v in tri
            ]
            try:
                p = _HolePoly(tri_pts)
                if p.is_valid and p.area > 0.01:
                    face_polys.append(p)
            except Exception:
                pass

        detected_holes = []
        if face_polys:
            merged = _HoleUnion(face_polys)
            # If multiple components, pick the largest
            if merged.geom_type == 'MultiPolygon':
                merged = max(merged.geoms, key=lambda g: g.area)

            # Replace the outer profile with face-projection's clean exterior
            # whenever we detect interior rings. The slice-based profile
            # sometimes traces cutouts as slits connected to the outer
            # boundary, which baked the cutout into the initial sketch.
            if merged.geom_type == 'Polygon' and merged.interiors:
                clean_outer = list(merged.exterior.coords)[:-1]
                clean_outer = _simplify_profile(clean_outer, tolerance=0.5)
                if len(clean_outer) >= 3:
                    # Keep any extra profiles (they'll be replaced/re-added below)
                    extras = best_profiles[1:] if len(best_profiles) > 1 else []
                    best_profiles = [clean_outer] + extras
                    logger.info(
                        "Replaced outer profile with clean exterior: %d pts",
                        len(clean_outer),
                    )

            if merged.geom_type == 'Polygon' and merged.interiors:
                for ring in merged.interiors:
                    inner = _HolePoly(ring)
                    if not inner.is_valid or inner.area < 1.0:
                        continue
                    area = float(inner.area)
                    perimeter = float(ring.length)
                    if perimeter < 1e-6:
                        continue
                    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                    est_radius = float((area / np.pi) ** 0.5)

                    # Non-circular OR large interior rings are polygonal
                    # cutouts (windows, pockets). Append them as additional
                    # profiles so _build_stepwise cuts them from the base
                    # with their own Sketch Profile + Extrude.
                    if circularity < 0.5 or est_radius > 30.0:
                        if area > 10.0:  # ignore tiny noise
                            cutout_coords = list(ring.coords)[:-1]
                            cutout_coords = _simplify_profile(cutout_coords, tolerance=0.5)
                            if len(cutout_coords) >= 3:
                                best_profiles.append(cutout_coords)
                                logger.info(
                                    "Polygon cutout profile: %d pts, area=%.0f",
                                    len(cutout_coords), area,
                                )
                        continue
                    if est_radius < 1.5:
                        continue

                    # Compute hole center in 3D
                    c_2d = inner.centroid
                    hole_center_3d = np.zeros(3)
                    hole_center_3d[ax_u] = c_2d.x
                    hole_center_3d[ax_v] = c_2d.y
                    # Use the body's midpoint along the extrude axis
                    hole_center_3d[primary_axis] = (
                        float(cleaned.bbox[0][primary_axis]) +
                        float(cleaned.bbox[1][primary_axis])
                    ) / 2

                    hole_depth = extent  # full thickness (through hole)
                    detected_holes.append(Feature(
                        type=_FType.THROUGH_HOLE,
                        face_indices=np.array([], dtype=np.int64),
                        sketch_geometry={
                            "type": "circle",
                            "radius": est_radius,
                            "center": hole_center_3d.tolist(),
                        },
                        direction=ext_norm,
                        depth=hole_depth,
                    ))
                    logger.info(
                        "2D-profile hole: r=%.2f at %s, circularity=%.2f",
                        est_radius, np.round(hole_center_3d, 1).tolist(), circularity,
                    )

        # If 2D detection found holes, REPLACE the recognition-based ones.
        # 2D detection guarantees holes go perpendicular to the plate face,
        # whereas recognition often picks up sideways cylindrical features.
        if detected_holes:
            logger.info("2D detection replaced %d recognition holes with %d 2D holes",
                        len(holes), len(detected_holes))
            holes = detected_holes
    except Exception as e:
        logger.debug("2D profile hole detection failed: %s", e)

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


def _make_heuristic_orders(
    features: list,
) -> list[tuple[str, str, BuildPlan]]:
    """Generate 3 heuristic build orders (name, strategy, BuildPlan).

    These complement the LLM order to give the user 4 choices total.
    Feature 0 (base extrusion) is always first.
    """
    from mesh2param.models.feature import FeatureType

    n = len(features)
    if n <= 1:
        return []

    holes = [(i, f) for i, f in enumerate(features) if f.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE)]
    chamfers = [(i, f) for i, f in enumerate(features) if f.type == FeatureType.CHAMFER]
    fillets = [(i, f) for i, f in enumerate(features) if f.type == FeatureType.FILLET]
    other = [(i, f) for i, f in enumerate(features) if i > 0 and (i, f) not in holes + chamfers + fillets]

    results = []

    # --- Variant 2: Largest holes first ---
    sorted_holes = sorted(holes, key=lambda x: -(x[1].sketch_geometry.get("radius", 0) if x[1].sketch_geometry else 0))
    order = [0] + [i for i, _ in sorted_holes] + [i for i, _ in other] + [i for i, _ in chamfers] + [i for i, _ in fillets]
    results.append((
        "Largest First",
        "Cut largest holes first for stable booleans, chamfers and fillets last",
        BuildPlan(feature_order=order, strategy="Largest holes first"),
    ))

    # --- Variant 3: Smallest holes first ---
    sorted_holes_rev = sorted(holes, key=lambda x: (x[1].sketch_geometry.get("radius", 0) if x[1].sketch_geometry else 0))
    order = [0] + [i for i, _ in sorted_holes_rev] + [i for i, _ in other] + [i for i, _ in fillets] + [i for i, _ in chamfers]
    results.append((
        "Smallest First",
        "Cut smallest holes first, finish with fillets then chamfers",
        BuildPlan(feature_order=order, strategy="Smallest holes first"),
    ))

    # --- Variant 4: Grouped by type ---
    through = [(i, f) for i, f in holes if f.type == FeatureType.THROUGH_HOLE]
    blind = [(i, f) for i, f in holes if f.type == FeatureType.BLIND_HOLE]
    order = [0] + [i for i, _ in through] + [i for i, _ in blind] + [i for i, _ in other] + [i for i, _ in chamfers] + [i for i, _ in fillets]
    results.append((
        "By Type",
        "All through-holes, then blind holes, then chamfers, then fillets",
        BuildPlan(feature_order=order, strategy="Grouped by feature type"),
    ))

    return results


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
    skip_fillets: bool = False,
    skip_chamfers: bool = False,
) -> tuple[object, str, list[ConstructionStep], list[dict]]:
    """Build a clean CAD solid from scratch, step by step.

    Steps:
      0. Sketch — 2D profile wire on the workplane
      1. Extrude — sweep the sketch into a solid
      2+. Cut holes — one per step
      N+. Apply chamfers (on all edges at once)
      N+1. Apply fillets (on all edges at once)

    Each step is tessellated for the visual slideshow. The resulting
    solid is clean B-rep geometry (no mesh triangles).

    Returns (final_solid, script, construction_steps, deferred_chamfers).
    deferred_chamfers is the list of chamfer records detected but not
    applied when skip_chamfers=True; empty otherwise. The caller is
    expected to apply them post-union via _apply_deferred_chamfers.
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

    # --- Step 0: Sketch (outer profile only — cutouts get their own steps) ---
    axes = [0, 1, 2]
    axes.remove(primary_axis)
    ax_u, ax_v = axes

    all_sketch_verts = []
    all_sketch_faces = []
    vert_offset = 0

    # Only the first profile (outer boundary) goes in the initial sketch.
    # Any additional profiles (cutouts) will emit their own Sketch + Cut
    # Extrude steps later in the function so the feature tree shows them
    # as separate editable features.
    for profile in profiles[:1]:
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
    script_lines.append("result = result.polyline(profile).close()")

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

    script_lines.append(f"# Step 2: Extrude {depth:.2f} mm")
    script_lines.append(f"result = result.extrude({depth})")

    v, f = _tessellate_solid(wp, fast=True)
    vol = abs(float(wp.val().Volume())) if wp else 0
    steps.append(ConstructionStep(
        name="Extrude Base", tess_vertices=v, tess_faces=f, volume=vol,
    ))

    # Additional profiles: each becomes its own Sketch + Extrude step.
    # Inner profiles (centroid inside the main outer) are cut as pockets,
    # outer profiles are unioned as additional material.
    from shapely.geometry import Polygon, Point
    main_poly = Polygon(profiles[0])
    for pi, profile in enumerate(profiles[1:], 1):
        try:
            centroid = np.mean(profile, axis=0)
            is_inner = main_poly.contains(Point(centroid[0], centroid[1]))

            # Build the sketch wireframe points for overlay display
            n_pts = len(profile)
            overlay_pts = []
            for pj in range(n_pts):
                u1, v1 = profile[pj]
                u2, v2 = profile[(pj + 1) % n_pts]
                p1 = [0.0, 0.0, 0.0]
                p2 = [0.0, 0.0, 0.0]
                p1[ax_u] = u1
                p1[ax_v] = v1
                p1[primary_axis] = extrude_start
                p2[ax_u] = u2
                p2[ax_v] = v2
                p2[primary_axis] = extrude_start
                overlay_pts.append(p1)
                overlay_pts.append(p2)
            overlay_arr = np.array(overlay_pts, dtype=np.float64)

            # For the sketch step, show the CURRENT solid with the sketch
            # drawn as a blue overlay on top (not the sketch alone).
            base_v_now, base_f_now = _tessellate_solid(wp, fast=True)
            sketch_name = "Sketch Cutout" if is_inner else "Sketch Extra"
            steps.append(ConstructionStep(
                name=sketch_name,
                tess_vertices=np.asarray(base_v_now, dtype=np.float64),
                tess_faces=np.asarray(base_f_now, dtype=np.int64),
                volume=abs(float(wp.val().Volume())),
                sketch_overlay=overlay_arr,
            ))

            # Apply the cut or union and emit the step
            extra = cq.Workplane(plane_name, origin=tuple(origin_3d))
            extra = extra.polyline(profile).close().extrude(extrude_sign)
            vol_before = abs(float(wp.val().Volume()))
            if is_inner:
                wp = wp.cut(extra)
                op_name = "Cut Extrude"
            else:
                wp = wp.union(extra)
                op_name = "Union Extrude"
            vol_after = abs(float(wp.val().Volume()))
            v2, f2 = _tessellate_solid(wp, fast=True)
            steps.append(ConstructionStep(
                name=op_name, tess_vertices=v2, tess_faces=f2, volume=vol_after,
            ))
            logger.info(
                "%s profile %d (%d pts): vol %.0f -> %.0f",
                "Cut" if is_inner else "Union", pi, len(profile),
                vol_before, vol_after,
            )
        except Exception as e:
            logger.warning("Failed to apply profile %d: %s", pi, e)

    # --- Chamfers BEFORE holes ---
    # Detect chamfer faces from mesh geometry. For each chamfer:
    #   - Full-edge chamfers (bevel spans >=80% of face): use .chamfer()
    #     so it maps to an editable Onshape Chamfer feature.
    #   - Partial/stepped chamfers: use a side-profile cut (triangle
    #     extruded along the edge) since .chamfer() can't model these.
    mesh_chamfers = _detect_mesh_chamfers(cleaned, extrude_dir)
    deferred_chamfers: list[dict] = []
    if skip_chamfers:
        # Defer to a post-union pass so the "Fillet & Chamfer Intent"
        # overlay can show all chamfers on the un-chamfered solid.
        # Carry along the per-body context the application needs.
        bbox_diag_local = float(np.linalg.norm(cleaned.bbox[1] - cleaned.bbox[0]))
        for cham in mesh_chamfers:
            rec = dict(cham)
            rec["bbox_diag"] = bbox_diag_local
            deferred_chamfers.append(rec)
        mesh_chamfers = []
    for cham in mesh_chamfers:
        label = cham["label"]
        d1 = cham["size"]
        d2 = cham["size2"]
        sel = cham["selector"]
        edge_length_hint = cham.get("edge_length", 0.0)
        vol_before = abs(float(wp.val().Volume()))

        if cham["is_full_edge"]:
            # --- Full-edge chamfer: use .chamfer() API ---
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
                        continue

                    vol_removed = vol_before - vol_after
                    try:
                        total_edge_len = sum(
                            e.Length() for e in wp.edges(sel).filter(
                                lambda e: e.Length() > min_len
                            ).vals()
                        )
                    except Exception:
                        total_edge_len = edge_length_hint

                    if total_edge_len > 0:
                        expected_vol = 0.5 * a1 * a2 * total_edge_len
                        vol_ratio = vol_removed / max(expected_vol, 1e-10)
                        if vol_ratio < 0.1 or vol_ratio > 10.0:
                            logger.debug(
                                "Chamfer %s (%.2fx%.2f): ratio %.2f out of range, trying smaller",
                                label, a1, a2, vol_ratio,
                            )
                            continue
                    elif vol_after > vol_before * 1.01:
                        continue

                    wp = wp_test
                    v, f = _tessellate_solid(wp, fast=True)
                    cham_label = f"{a1:.1f}x{a2:.1f}" if abs(a1 - a2) > 0.5 else f"{min(a1, a2):.1f}"
                    steps.append(ConstructionStep(
                        name=f"Chamfer {label} ({cham_label}mm)",
                        tess_vertices=v, tess_faces=f, volume=vol_after,
                    ))
                    script_lines.append(f'# Chamfer {label}')
                    script_lines.append(f'result = result.edges("{sel}").chamfer({a1:.4f}, {a2:.4f})')
                    logger.info(
                        "Applied chamfer %s: %.2fx%.2f mm (.chamfer), vol %.2f -> %.2f",
                        label, a1, a2, vol_before, vol_after,
                    )
                    applied = True
                    break
                except Exception as exc:
                    logger.debug("Chamfer %s (%.2fx%.2f) failed: %s", label, a1, a2, exc)
                    continue
            if applied:
                continue  # success — move to next chamfer

            # .chamfer() failed — fall through to profile cut
            logger.debug("Chamfer %s: .chamfer() failed, trying profile cut", label)

        # --- Partial/stepped chamfer OR .chamfer() fallback: side-profile cut ---
        corner = cham["corner"]
        along_b = cham["along_b"]
        along_a = cham["along_a"]
        eidx = cham["edge_axis_idx"]
        e_min = cham["edge_min"]
        e_max = cham["edge_max"]
        try:
            ax_u, ax_v = [i for i in range(3) if i != eidx]
            # "Through All" — extrude far beyond the part in both directions
            bbox_diag_local = float(np.linalg.norm(cleaned.bbox[1] - cleaned.bbox[0]))
            through_all = bbox_diag_local * 5
            cut_origin = [0.0, 0.0, 0.0]
            cut_origin[eidx] = e_min - through_all

            plane_normal = [0.0, 0.0, 0.0]
            plane_normal[eidx] = 1.0
            # Choose xDir so that the CQ plane's yDir aligns with +ax_v.
            # For normal along Y with xDir=X, yDir=cross(Y,X)=-Z (flipped).
            # Using xDir along ax_v instead gives yDir along ax_u — then
            # moveTo(v, u) maps correctly. Pick whichever xDir avoids a flip.
            x_dir = [0.0, 0.0, 0.0]
            x_dir[ax_u] = 1.0
            y_dir_check = np.cross(plane_normal, x_dir)
            if y_dir_check[ax_v] < 0:
                # Flip: use ax_v as xDir instead, then u→ax_v, v→ax_u
                x_dir = [0.0, 0.0, 0.0]
                x_dir[ax_v] = 1.0
                swap_uv = True
            else:
                swap_uv = False

            cut_plane = cq.Plane(
                origin=cq.Vector(*cut_origin),
                normal=cq.Vector(*plane_normal),
                xDir=cq.Vector(*x_dir),
            )

            def _pt(p):
                return (p[ax_v], p[ax_u]) if swap_uv else (p[ax_u], p[ax_v])

            tri_pts = [along_b, corner, along_a]
            p0 = _pt(tri_pts[0])
            cut_wp = cq.Workplane(cut_plane).moveTo(*p0)
            for pt in tri_pts[1:]:
                cut_wp = cut_wp.lineTo(*_pt(pt))
            cut_wp = (cut_wp.close()
                    .extrude(2 * through_all + e_max - e_min))

            wp_test = wp.cut(cut_wp)
            vol_after = abs(float(wp_test.val().Volume()))
            if vol_after <= 0:
                logger.debug("Chamfer %s: profile cut zero volume, skipping", label)
                continue

            vol_removed = vol_before - vol_after
            expected_vol = 0.5 * d1 * d2 * (e_max - e_min)
            vol_ratio = vol_removed / max(expected_vol, 1e-10)

            if vol_ratio < 0.05 or vol_ratio > 10.0:
                logger.warning(
                    "Chamfer %s: profile cut ratio %.2f out of range — skipping",
                    label, vol_ratio,
                )
                continue

            # --- Sketch step: solid body + blue triangle sketch overlay ---
            body_v, body_f = _tessellate_solid(wp, fast=True)

            # Build sketch line pairs for the polygon edges
            poly_3d = []
            for pt in tri_pts:
                p = np.zeros(3)
                p[ax_u] = pt[ax_u]
                p[ax_v] = pt[ax_v]
                p[eidx] = e_min
                poly_3d.append(p)
            # N edges: pt0→pt1, pt1→pt2, ..., ptN→pt0
            sketch_pairs = []
            for i in range(len(poly_3d)):
                sketch_pairs.append(poly_3d[i])
                sketch_pairs.append(poly_3d[(i + 1) % len(poly_3d)])
            sketch_lines = np.array(sketch_pairs)

            cham_label = f"{d1:.1f}x{d2:.1f}" if abs(d1 - d2) > 0.5 else f"{min(d1, d2):.1f}"
            steps.append(ConstructionStep(
                name=f"Side Profile {label}",
                tess_vertices=body_v, tess_faces=body_f, volume=vol_before,
                sketch_overlay=sketch_lines,
            ))

            # --- Cut step: the actual extrude-cut ---
            wp = wp_test
            v, f = _tessellate_solid(wp, fast=True)
            steps.append(ConstructionStep(
                name=f"Side Cut {label} ({cham_label}mm)",
                tess_vertices=v, tess_faces=f, volume=vol_after,
            ))
            script_lines.append(f'# Chamfer {label} — profile cut')
            script_lines.append(
                f'cut = cq.Workplane(cq.Plane(origin=cq.Vector{tuple(cut_origin)}, '
                f'normal=cq.Vector{tuple(plane_normal)}, xDir=cq.Vector{tuple(x_dir)}))'
                f'.moveTo({corner[ax_u]:.4f}, {corner[ax_v]:.4f})'
                f'.lineTo({along_b[ax_u]:.4f}, {along_b[ax_v]:.4f})'
                f'.lineTo({along_a[ax_u]:.4f}, {along_a[ax_v]:.4f})'
                f'.close().extrude({e_max - e_min:.4f})'
            )
            script_lines.append('result = result.cut(cut)')
            logger.info(
                "Applied chamfer %s: %.2fx%.2f mm profile cut, "
                "vol %.2f -> %.2f (ratio %.2f)",
                label, d1, d2, vol_before, vol_after, vol_ratio,
            )
        except Exception as exc:
            logger.debug("Chamfer %s profile cut failed: %s", label, exc)

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

        # Symmetric-hole groups carry a `centers` list — one sketch step
        # with multiple circles, then one cut. A single-hole feature has
        # only `center`; treat that as a 1-element group.
        centers_list = sketch.get("centers")
        if centers_list:
            centers_all = [np.array(c, dtype=float) for c in centers_list]
        else:
            centers_all = [np.array(sketch["center"], dtype=float)]
        center = centers_all[0]  # primary center for legacy callers
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

        n_holes = len(centers_all)
        if n_holes > 1:
            hole_name = (
                f"{'Through' if is_through else 'Blind'} Hole Pattern "
                f"({n_holes}× r={radius:.1f})"
            )
        else:
            hole_name = f"{'Through' if is_through else 'Blind'} Hole (r={radius:.1f})"
        if note:
            hole_name += f" — {note}"

        try:
            # Compose the hole-cut solid as the union of cylinders at all
            # group centers. cut_start was computed for the primary center;
            # for each additional center we shift the start parallel along
            # the projected XY plane so the cylinder sits where the center
            # is, while keeping the same axis depth.
            hole_plane = cq.Plane(
                origin=cq.Vector(*cut_start.tolist()),
                normal=cq.Vector(*h_dir.tolist()),
            )
            hole_solid = cq.Workplane(hole_plane).circle(radius).extrude(cut_length)
            for c in centers_all[1:]:
                shift = c - center
                shift_plane = cq.Plane(
                    origin=cq.Vector(*(cut_start + shift).tolist()),
                    normal=cq.Vector(*h_dir.tolist()),
                )
                extra = cq.Workplane(shift_plane).circle(radius).extrude(cut_length)
                hole_solid = hole_solid.union(extra)

            # Emit a "Sketch Hole" step showing the circle outline(s) on
            # the pre-cut solid. For a multi-center group, this is one
            # sketch with N circles (matching how a CAD operator would
            # actually draw a hole pattern in one sketch).
            try:
                tmp_axis = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(h_dir, tmp_axis)) > 0.95:
                    tmp_axis = np.array([1.0, 0.0, 0.0])
                u_perp = np.cross(h_dir, tmp_axis)
                u_perp = u_perp / (np.linalg.norm(u_perp) + 1e-15)
                v_perp = np.cross(h_dir, u_perp)
                v_perp = v_perp / (np.linalg.norm(v_perp) + 1e-15)
                N_SAMPLES = 48
                overlay_pts = []
                for c_each in centers_all:
                    shift = c_each - center
                    sketch_plane_origin = np.asarray(cut_start + shift, dtype=float)
                    prev_pt = None
                    for k in range(N_SAMPLES + 1):
                        theta = 2 * np.pi * k / N_SAMPLES
                        pt = (
                            sketch_plane_origin
                            + radius * (np.cos(theta) * u_perp + np.sin(theta) * v_perp)
                        )
                        if prev_pt is not None:
                            overlay_pts.append(prev_pt)
                            overlay_pts.append(pt)
                        prev_pt = pt
                circle_overlay = np.asarray(overlay_pts, dtype=np.float64)
                v_pre, f_pre = _tessellate_solid(wp, fast=True)
                steps.append(ConstructionStep(
                    name=f"Sketch {hole_name}",
                    tess_vertices=np.asarray(v_pre, dtype=np.float64),
                    tess_faces=np.asarray(f_pre, dtype=np.int64),
                    volume=abs(float(wp.val().Volume())),
                    sketch_overlay=circle_overlay,
                ))
            except Exception as _sk_exc:
                logger.debug("Could not build hole sketch overlay: %s", _sk_exc)

            # Try the cut; if it removes little/no material, the hole
            # center is likely outside the solid (e.g., on a flange/tab
            # not captured by the profile cross-section).  In that case,
            # nudge the hole center toward the solid so it intersects.
            vol_before = abs(float(wp.val().Volume()))
            wp_test = wp.cut(hole_solid)
            vol_after = abs(float(wp_test.val().Volume()))

            # Expected hole volume: pi * r^2 * cut_length × N holes
            expected_removal = 3.14159 * radius * radius * cut_length * n_holes
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

            v, f = _tessellate_solid(wp, fast=True)
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

    # Fillet application is handled by the caller via
    # _apply_recognized_fillets_to_solid, which runs AFTER all bodies have
    # been unioned (or once for single-body parts). _build_stepwise no
    # longer emits fillet steps itself.

    script_lines.extend(["", '# cq.exporters.export(result, "output.step")'])
    return wp, "\n".join(script_lines), steps, deferred_chamfers


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
    import tempfile
    import os

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
    angle_thresh_deg: float = 8.0,
) -> list[tuple[float, float]]:
    """Simplify a 2D profile to only its corner vertices.

    Removes points that are collinear with their neighbors (on straight
    edges), keeping only vertices where the direction changes by more
    than ``angle_thresh_deg``. Also snaps coordinates to a coarse grid.

    The default 8° threshold is intentionally more aggressive than the
    older 2° (which kept too many near-collinear runs from the peeled
    mesh's boundary noise — those become individual short edges on the
    extruded solid that OCC's fillet kernel can't handle). 8° still
    preserves chamfers (45°), corners (90°), and gentle arcs sampled
    coarser than 8°.
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
        if angle > angle_thresh_deg:
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


def _tessellate_solid(solid, fast=False) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Tessellate a CadQuery solid for visualization.

    Uses per-face triangulation: extracts each B-rep face, gets its boundary
    wire points, and triangulates with Constrained Delaunay (CDT) for planar
    faces.  This guarantees interior triangles for large flat surfaces that
    OCCT's BRepMesh otherwise leaves hollow.
    Curved faces use standard BRepMesh triangulation.

    fast=True uses coarser tolerance for intermediate construction steps.
    """
    logger.debug("Tessellating CadQuery solid (fast=%s)", fast)
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopoDS import TopoDS
        from OCP.TopLoc import TopLoc_Location

        shape = solid.val().wrapped

        # Clear any cached triangulation from previous operations
        # (BRepMesh reuses stale mesh data after boolean cuts otherwise)
        from OCP.BRepTools import BRepTools
        BRepTools.Clean_s(shape)

        # Mesh tolerance: coarse for intermediate steps, fine for final
        mesh_tol = 0.15 if fast else 0.02
        BRepMesh_IncrementalMesh(shape, mesh_tol, False, 0.3 if fast else 0.05, True)

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
    size_max = bbox_diag * 0.10    # 10% of diagonal
    # DBSCAN eps for normal clustering: normals live on the unit sphere so
    # eps=0.20 works well regardless of part scale.
    normal_eps = 0.20
    # Minimum faces in a cluster to be considered real geometry (not noise).
    # Scales with mesh density: at least 5 or 0.05% of total faces.
    min_cluster_faces = max(5, int(len(cleaned.faces) * 0.0005))

    normals = cleaned.normals

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
    # Use ALL mesh faces (not just diagonal) so chamfer strips that share
    # edges with non-diagonal faces still appear connected.
    from collections import defaultdict
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(len(cleaned.faces)):
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

    # --- Diagonal-face adjacency, for the flat-vs-curved test below ---
    # A chamfer's bevel is a single FLAT face: all its diagonal faces share
    # one normal. A fillet's rolled surface is CURVED: its diagonal faces
    # fan across an arc. DBSCAN (eps=0.20) only groups a thin sub-band of
    # that arc, so a fillet sub-band can masquerade as a small chamfer.
    # To tell them apart we grow the full connected band of diagonal faces
    # around each cluster and measure how far the normals fan out.
    diag_set = set(int(x) for x in diag_indices)
    diag_adj: dict[int, set[int]] = defaultdict(set)
    for faces_list in edge_to_faces.values():
        dl = [f for f in faces_list if f in diag_set]
        for i in range(len(dl)):
            for j in range(i + 1, len(dl)):
                diag_adj[dl[i]].add(dl[j])
                diag_adj[dl[j]].add(dl[i])

    def _diagonal_band_spread_deg(seed_faces) -> float:
        """Grow the connected component of diagonal faces around `seed_faces`
        and return the angular spread (deg) of their normals from the mean.
        Flat chamfer → ~0°; curved fillet band → tens of degrees."""
        comp: set[int] = set()
        stack = [int(f) for f in seed_faces]
        while stack:
            fc = stack.pop()
            if fc in comp:
                continue
            comp.add(fc)
            stack.extend(diag_adj[fc] - comp)
        if len(comp) < 2:
            return 0.0
        comp_n = normals[list(comp)]
        mean_n = comp_n.mean(axis=0)
        mn = np.linalg.norm(mean_n)
        if mn < 1e-9:
            return 180.0
        mean_n /= mn
        dots = np.clip(comp_n @ mean_n, -1.0, 1.0)
        return float(np.degrees(np.arccos(dots.min())))

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

        # --- Flat-vs-curved gate: reject fillet bands ---
        # If the full connected band of diagonal faces around this cluster
        # fans across a wide arc, it's the curved surface of a fillet, not
        # a flat chamfer. A genuine chamfer's diagonal faces stay within a
        # few degrees of one normal.
        band_spread = _diagonal_band_spread_deg(cluster_idx)
        if band_spread > 18.0:
            logger.debug(
                "Cluster %d: diagonal band fans %.0f° (>18°) — curved "
                "fillet surface, not a chamfer, skipping",
                label, band_spread,
            )
            continue

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
        # Determine which bbox corner the chamfer removes material from.
        # The chamfer face normal points AWAY from the solid. Its components
        # indicate which direction the face points: the corner to cut is
        # on the side the normal points toward (the outside of the removed wedge).
        sign_a = ">" if avg_normal[face_ax_a] > 0 else "<"
        sign_b = ">" if avg_normal[face_ax_b] > 0 else "<"

        # The selector uses "|edge_axis" to pick edges parallel to the
        # edge direction, then filters by BOTH face axes to distinguish
        # e.g. left-side vs right-side chamfers that share the same Z.
        selector = f"|{edge_axis} and {sign_b}{axis_names[face_ax_b]} and {sign_a}{axis_names[face_ax_a]}"

        # Label for display
        side_label_b = "top" if sign_b == ">" else "bottom"
        label_str = f"{side_label_b}-{axis_names[face_ax_b]} {edge_axis}"

        # --- Compute chamfer triangle ---
        # d2: measured from cluster vertices near the side face boundary.
        boundary_tol = max(1.0, bbox_diag * 0.005)

        corner_a = float(cleaned.bbox[1][face_ax_a]) if sign_a == ">" else float(cleaned.bbox[0][face_ax_a])
        corner_b = float(cleaned.bbox[1][face_ax_b]) if sign_b == ">" else float(cleaned.bbox[0][face_ax_b])

        if sign_a == ">":
            near_a_mask = cluster_face_verts[:, face_ax_a] > corner_a - boundary_tol
        else:
            near_a_mask = cluster_face_verts[:, face_ax_a] < corner_a + boundary_tol
        if near_a_mask.sum() > 0:
            near_a_verts = cluster_face_verts[near_a_mask]
            if sign_b == "<":
                inner_b = float(near_a_verts[:, face_ax_b].max())
            else:
                inner_b = float(near_a_verts[:, face_ax_b].min())
        else:
            inner_b = corner_b

        d2 = abs(inner_b - corner_b)

        # d1: derive from d2 and the chamfer angle ratio.
        n_a_comp = abs(avg_normal[face_ax_a])
        n_b_comp = abs(avg_normal[face_ax_b])
        d1 = d2 * n_a_comp / max(n_b_comp, 0.01)

        # Subtract margin so the triangle extends PAST the mesh edge,
        # eliminating slivers from tessellation gaps.  Smaller d1/d2 →
        # along_a/along_b move further from the corner → bigger cut.
        d1 -= 0.5
        d2 -= 0.5
        d1 = max(d1, 0.5)
        d2 = max(d2, 0.5)

        # Build triangle with corner pushed 1mm outside the bbox.
        # The two legs that sit on the model's faces extend past the
        # boundary, eliminating slivers.  The hypotenuse shifts slightly
        # (~1mm on a 13mm chamfer = <8%) which is within mesh reconstruction
        # tolerance.  This approach is simple and robust for all part sizes.
        overshoot = 1.0  # mm past bbox on face-aligned legs

        corner = [0.0, 0.0, 0.0]
        corner[face_ax_a] = corner_a + (overshoot if sign_a == ">" else -overshoot)
        corner[face_ax_b] = corner_b + (overshoot if sign_b == ">" else -overshoot)

        along_b = list(corner)
        along_b[face_ax_b] = corner_b + d2 * (-1.0 if sign_b == ">" else 1.0)

        along_a = list(corner)
        along_a[face_ax_a] = corner_a + d1 * (-1.0 if sign_a == ">" else 1.0)


        # --- Size filtering ---
        if d1 < size_min or d2 < size_min:
            logger.debug(
                "Cluster %d: chamfer too small (d1=%.3f, d2=%.3f, min=%.3f), skipping",
                label, d1, d2, size_min,
            )
            continue
        if d1 > size_max or d2 > size_max:
            logger.debug(
                "Cluster %d: chamfer too large (d1=%.2f, d2=%.2f, max=%.2f), "
                "likely a window edge, skipping",
                label, d1, d2, size_max,
            )
            continue

        # Extrude range along edge axis
        edge_min = float(cleaned.bbox[0][edge_axis_idx])
        edge_max = float(cleaned.bbox[1][edge_axis_idx])

        # Determine if this is a simple full-edge chamfer (can use .chamfer())
        # or a partial/stepped chamfer that needs a profile cut.
        # A full-edge chamfer's d2 spans >=80% of the face extent.
        face_extent = float(bbox_size[face_ax_b])
        face_coverage = d2 / face_extent if face_extent > 1e-10 else 0
        is_full_edge = face_coverage >= 0.80

        logger.info(
            "Chamfer cluster %d: %d faces, normal=[%.2f,%.2f,%.2f], "
            "d1=%.2f d2=%.2f, edge=|%s, selector='%s', "
            "strip L=%.2f W=%.2f aspect=%.1f coverage=%.0f%% "
            "face_coverage=%.0f%% -> %s",
            label, len(cluster_idx), *avg_normal, d1, d2,
            edge_axis, selector,
            strip_length, strip_width, aspect_ratio, coverage * 100,
            face_coverage * 100,
            "chamfer" if is_full_edge else "profile cut",
        )
        results.append({
            "size": float(d1),
            "size2": float(d2),
            "selector": selector,
            "label": label_str,
            "face_count": len(cluster_idx),
            "edge_length": float(strip_length),
            "is_full_edge": is_full_edge,
            "edge_axis_idx": edge_axis_idx,
            "face_ax_a": face_ax_a,
            "face_ax_b": face_ax_b,
            "corner": corner,
            "along_b": along_b,
            "along_a": along_a,
            "edge_min": edge_min,
            "edge_max": edge_max,
        })

    # --- Keep individual chamfers (do NOT merge by selector) ---
    # Different chamfers may share the same selector but have different
    # sizes. We keep them separate so CadQuery can apply each individually.
    # Sort largest first for stable ordering.
    results.sort(key=lambda x: -x["size"])

    logger.debug("Detected %d chamfer(s) total", len(results))
    return results


def _apply_deferred_chamfers(
    wp: object,
    chamfer_records: list[dict],
    out_edges: list | None = None,
    out_surfaces: list | None = None,
) -> tuple[object, list[ConstructionStep], list[str]]:
    """Apply chamfers that were detected per-body but deferred to a
    post-union pass. Mirrors the inline chamfer loop in `_build_stepwise`
    but operates on a single solid and returns fresh step/script lists.

    Each record must carry its source body's `bbox_diag` so the
    profile-cut path can compute a "through-all" extrusion length without
    a CleanedMesh reference.

    If out_edges is provided, each successfully chamfered edge's
    [start, end] endpoints (world coords) are appended to it — used by
    the intent-overlay viz.
    """
    import cadquery as cq

    applied_steps: list[ConstructionStep] = []
    script_lines: list[str] = []
    if not chamfer_records:
        return wp, applied_steps, script_lines

    for cham in chamfer_records:
        label = cham["label"]
        d1 = cham["size"]
        d2 = cham["size2"]
        sel = cham["selector"]
        edge_length_hint = cham.get("edge_length", 0.0)
        bbox_diag_local = float(cham.get("bbox_diag", 0.0))
        try:
            vol_before = abs(float(wp.val().Volume()))
        except Exception:
            continue

        applied = False
        if cham.get("is_full_edge"):
            attempts = [(d1, d2), (d1 * 0.8, d2 * 0.8), (d1 * 0.6, d2 * 0.6)]
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
                        continue
                    vol_removed = vol_before - vol_after
                    try:
                        total_edge_len = sum(
                            e.Length() for e in wp.edges(sel).filter(
                                lambda e: e.Length() > min_len
                            ).vals()
                        )
                    except Exception:
                        total_edge_len = edge_length_hint
                    if total_edge_len > 0:
                        expected_vol = 0.5 * a1 * a2 * total_edge_len
                        vol_ratio = vol_removed / max(expected_vol, 1e-10)
                        if vol_ratio < 0.1 or vol_ratio > 10.0:
                            continue
                    elif vol_after > vol_before * 1.01:
                        continue
                    # Record the chamfered edges (from the pre-chamfer
                    # solid) for the intent overlay before reassigning wp.
                    if out_edges is not None:
                        try:
                            for _e in wp.edges(sel).filter(
                                lambda e: e.Length() > min_len
                            ).vals():
                                _sp = _e.startPoint()
                                _ep = _e.endPoint()
                                out_edges.append(
                                    [[_sp.x, _sp.y, _sp.z], [_ep.x, _ep.y, _ep.z]]
                                )
                        except Exception:
                            pass
                    wp = wp_test
                    v, f = _tessellate_solid(wp, fast=True)
                    cham_label = (
                        f"{a1:.1f}x{a2:.1f}" if abs(a1 - a2) > 0.5
                        else f"{min(a1, a2):.1f}"
                    )
                    applied_steps.append(ConstructionStep(
                        name=f"Chamfer {label} ({cham_label}mm)",
                        tess_vertices=v, tess_faces=f, volume=vol_after,
                    ))
                    script_lines.append(f'# Chamfer {label}')
                    script_lines.append(
                        f'result = result.edges("{sel}").chamfer({a1:.4f}, {a2:.4f})'
                    )
                    logger.info(
                        "Deferred chamfer %s: %.2fx%.2f mm, vol %.2f -> %.2f",
                        label, a1, a2, vol_before, vol_after,
                    )
                    if out_surfaces is not None:
                        patch = _build_chamfer_surface_patch(cham)
                        if patch is not None:
                            pv, pf = patch
                            out_surfaces.append({
                                "verts": pv.tolist(), "faces": pf.tolist(),
                                "kind": "chamfer",
                            })
                    applied = True
                    break
                except Exception as exc:
                    logger.debug(
                        "Deferred chamfer %s (%.2fx%.2f) failed: %s",
                        label, a1, a2, exc,
                    )
                    continue
            if applied:
                continue

        # --- Side-profile cut fallback (partial chamfers OR .chamfer() failed) ---
        try:
            corner = cham["corner"]
            along_b = cham["along_b"]
            along_a = cham["along_a"]
            eidx = cham["edge_axis_idx"]
            e_min = cham["edge_min"]
            e_max = cham["edge_max"]
            ax_u, ax_v = [i for i in range(3) if i != eidx]
            through_all = bbox_diag_local * 5
            cut_origin = [0.0, 0.0, 0.0]
            cut_origin[eidx] = e_min - through_all
            plane_normal = [0.0, 0.0, 0.0]
            plane_normal[eidx] = 1.0
            x_dir = [0.0, 0.0, 0.0]
            x_dir[ax_u] = 1.0
            y_dir_check = np.cross(plane_normal, x_dir)
            if y_dir_check[ax_v] < 0:
                x_dir = [0.0, 0.0, 0.0]
                x_dir[ax_v] = 1.0
                swap_uv = True
            else:
                swap_uv = False
            cut_plane = cq.Plane(
                origin=cq.Vector(*cut_origin),
                normal=cq.Vector(*plane_normal),
                xDir=cq.Vector(*x_dir),
            )

            def _pt(p):
                return (p[ax_v], p[ax_u]) if swap_uv else (p[ax_u], p[ax_v])

            tri_pts = [along_b, corner, along_a]
            cut_wp = cq.Workplane(cut_plane).moveTo(*_pt(tri_pts[0]))
            for pt in tri_pts[1:]:
                cut_wp = cut_wp.lineTo(*_pt(pt))
            cut_wp = cut_wp.close().extrude(2 * through_all + e_max - e_min)

            wp_test = wp.cut(cut_wp)
            vol_after = abs(float(wp_test.val().Volume()))
            if vol_after <= 0:
                continue
            vol_removed = vol_before - vol_after
            expected_vol = 0.5 * d1 * d2 * (e_max - e_min)
            vol_ratio = vol_removed / max(expected_vol, 1e-10)
            if vol_ratio < 0.05 or vol_ratio > 10.0:
                continue

            wp = wp_test
            v, f = _tessellate_solid(wp, fast=True)
            cham_label = (
                f"{d1:.1f}x{d2:.1f}" if abs(d1 - d2) > 0.5
                else f"{min(d1, d2):.1f}"
            )
            applied_steps.append(ConstructionStep(
                name=f"Side Cut {label} ({cham_label}mm)",
                tess_vertices=v, tess_faces=f, volume=vol_after,
            ))
            if out_edges is not None:
                # The chamfered edge runs along the edge axis at the
                # corner, from e_min to e_max.
                p_lo = list(map(float, corner))
                p_lo[eidx] = float(e_min)
                p_hi = list(map(float, corner))
                p_hi[eidx] = float(e_max)
                out_edges.append([p_lo, p_hi])
            if out_surfaces is not None:
                patch = _build_chamfer_surface_patch(cham)
                if patch is not None:
                    pv, pf = patch
                    out_surfaces.append({
                        "verts": pv.tolist(), "faces": pf.tolist(),
                        "kind": "chamfer",
                    })
            script_lines.append(f'# Chamfer {label} — profile cut')
            logger.info(
                "Deferred chamfer %s (profile cut): %.2fx%.2f mm, vol %.2f -> %.2f",
                label, d1, d2, vol_before, vol_after,
            )
        except Exception as exc:
            logger.debug("Deferred chamfer %s side-cut failed: %s", label, exc)

    return wp, applied_steps, script_lines


def _build_fillet_surface_patch(
    e0, e1, axis_point, axis_dir, radius, n_along: int = 3, n_arc: int = 10,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a quarter-round surface patch (the rolling-ball fillet
    surface) along a matched edge, so the intent viz can show what the
    fillet actually looks like rather than a bare line.

    The fillet is a portion of a cylinder of `radius` whose axis is the
    line (axis_point, axis_dir) — the rolling-ball centre line. The patch
    sweeps a 90° arc centred on the direction from the axis toward the
    edge (the corner bisector), tangent to the two adjacent faces.
    Returns (verts (N,3), faces (M,3)) or None.
    """
    e0 = np.asarray(e0, dtype=float)
    e1 = np.asarray(e1, dtype=float)
    D = np.asarray(axis_dir, dtype=float)
    nd = np.linalg.norm(D)
    if nd < 1e-12 or radius <= 0:
        return None
    D = D / nd
    P = np.asarray(axis_point, dtype=float)
    # Project edge endpoints onto the axis line → cylinder-axis segment.
    A0 = P + np.dot(e0 - P, D) * D
    A1 = P + np.dot(e1 - P, D) * D
    rad_vec = e0 - A0
    rn = np.linalg.norm(rad_vec)
    if rn < 1e-9:
        return None
    rad = rad_vec / rn               # axis → edge (corner bisector)
    tang = np.cross(D, rad)
    tn = np.linalg.norm(tang)
    if tn < 1e-9:
        return None
    tang = tang / tn
    sweep = np.pi / 2.0              # 90° fillet
    verts = []
    for i in range(n_along + 1):
        t = i / n_along
        A = A0 * (1.0 - t) + A1 * t
        for j in range(n_arc + 1):
            th = -sweep / 2.0 + sweep * (j / n_arc)
            pt = A + radius * (np.cos(th) * rad + np.sin(th) * tang)
            verts.append(pt)
    stride = n_arc + 1
    faces = []
    for i in range(n_along):
        for j in range(n_arc):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def _build_chamfer_surface_patch(cham: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the flat bevel face of a chamfer as a quad (2 triangles),
    spanning the edge from edge_min to edge_max between the two contact
    lines along_a / along_b."""
    try:
        along_a = np.asarray(cham["along_a"], dtype=float)
        along_b = np.asarray(cham["along_b"], dtype=float)
        eidx = int(cham["edge_axis_idx"])
        e_min = float(cham["edge_min"])
        e_max = float(cham["edge_max"])
    except (KeyError, TypeError, ValueError):
        return None
    pa0 = along_a.copy()
    pa0[eidx] = e_min
    pa1 = along_a.copy()
    pa1[eidx] = e_max
    pb0 = along_b.copy()
    pb0[eidx] = e_min
    pb1 = along_b.copy()
    pb1[eidx] = e_max
    verts = np.array([pa0, pb0, pb1, pa1], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return verts, faces


def build_blend_intent_edges(
    fillet_records: list[dict],
    chamfer_records: list[dict],
) -> np.ndarray | None:
    """Return an (N, 2, 3) array of line-segment endpoints visualizing
    where each fillet/chamfer record from the blend stage attaches.

    Used by the Stage 7 "Fillet & Chamfer Intent" pipeline-stage viz
    (called from job_queue.py after the CAD subprocess returns). The
    output is the same edge-pair format `_set_mesh_viz(edge_pts=...)`
    consumes, so it lays cyan lines on top of the pre-blend solid.

    Fillet markers: an axis line of length `length` (or >=1mm) centered
    at `axis_point`, plus two end-circles at the cylinder radius
    perpendicular to the axis. Chamfer markers: the triangle outline at
    corner / along_a / along_b world coordinates.
    """
    overlay_pairs: list[np.ndarray] = []
    N_CIRC = 24

    for rec in fillet_records or []:
        try:
            r = float(rec.get("radius", 0.0))
            ad = np.asarray(rec.get("axis_dir"), dtype=float)
            ap = np.asarray(
                rec.get("axis_point", rec.get("centroid", [0, 0, 0])), dtype=float,
            )
            length = float(rec.get("length", 0.0))
            n = float(np.linalg.norm(ad))
            if r <= 0 or n < 1e-12:
                continue
            ad = ad / n
            half = max(length, 1.0) * 0.5
            p0 = ap - ad * half
            p1 = ap + ad * half
            overlay_pairs.append(p0)
            overlay_pairs.append(p1)

            tmp = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(ad, tmp))) > 0.95:
                tmp = np.array([1.0, 0.0, 0.0])
            u_perp = np.cross(ad, tmp)
            u_perp = u_perp / (np.linalg.norm(u_perp) + 1e-15)
            v_perp = np.cross(ad, u_perp)
            v_perp = v_perp / (np.linalg.norm(v_perp) + 1e-15)
            for end in (p0, p1):
                prev = None
                for k in range(N_CIRC + 1):
                    th = 2 * np.pi * k / N_CIRC
                    pt = end + r * (np.cos(th) * u_perp + np.sin(th) * v_perp)
                    if prev is not None:
                        overlay_pairs.append(prev)
                        overlay_pairs.append(pt)
                    prev = pt
        except Exception as exc:
            logger.debug("Intent overlay: skipped fillet record (%s)", exc)
            continue

    for cham in chamfer_records or []:
        try:
            corner = np.asarray(cham.get("corner"), dtype=float)
            along_a = np.asarray(cham.get("along_a"), dtype=float)
            along_b = np.asarray(cham.get("along_b"), dtype=float)
            if corner is None or along_a is None or along_b is None:
                continue
            tri = [along_b, corner, along_a]
            for i in range(len(tri)):
                overlay_pairs.append(tri[i])
                overlay_pairs.append(tri[(i + 1) % len(tri)])
        except Exception as exc:
            logger.debug("Intent overlay: skipped chamfer record (%s)", exc)
            continue

    if not overlay_pairs:
        return None
    return np.asarray(overlay_pairs, dtype=np.float64).reshape(-1, 2, 3)


def _select_crease_edges(cand, snap, band_normal, min_spine_len):
    """Select the connected crease edges for a curve/straight fillet.

    ``cand`` is a list of ``(edge_idx, frac_on, mean_dist, edge)``
    candidates already within snap of the cluster's vertex cloud. The
    crease is the LONG outline edges plus the SHORT corner arcs that
    bridge them (corner to corner). Excluded:
      - rim-thickness spanners — run along the band normal (across the
        plate), not in the outline plane;
      - junction connectors at an OPEN end of the U — short edges that
        dangle off a single spine edge instead of bridging two.
    """
    import collections

    inplane = []
    for x in cand:
        e = x[3]
        d = e["end"] - e["start"]
        n = float(np.linalg.norm(d))
        if n < 1e-9 or e["len"] < 1.0 or x[2] > snap:
            continue
        if abs(float((d / n) @ band_normal)) < 0.6:
            inplane.append(x)
    if not inplane:
        return [x for x in cand if x[3]["len"] >= min_spine_len]
    max_len = max(x[3]["len"] for x in inplane)
    spine = [x for x in inplane if x[3]["len"] >= 0.25 * max_len]
    shorts = [x for x in inplane if x[3]["len"] < 0.25 * max_len]
    if not shorts or len(spine) < 2:
        return spine

    def key(p):
        return tuple(np.round(p, 1).tolist())

    ep2short = collections.defaultdict(list)
    for i, x in enumerate(shorts):
        e = x[3]
        ep2short[key(e["start"])].append(i)
        ep2short[key(e["end"])].append(i)
    ep2spine = collections.defaultdict(set)
    for x in spine:
        e = x[3]
        ep2spine[key(e["start"])].add(x[0])
        ep2spine[key(e["end"])].add(x[0])

    # Connected components of short edges; keep a component only if it
    # touches >= 2 distinct spine edges (a corner bridge), not a
    # dangling junction connector.
    seen: set = set()
    keep: list = []
    for i in range(len(shorts)):
        if i in seen:
            continue
        comp = []
        stack = [i]
        seen.add(i)
        touched: set = set()
        while stack:
            j = stack.pop()
            comp.append(j)
            e = shorts[j][3]
            for ep in (key(e["start"]), key(e["end"])):
                touched |= ep2spine.get(ep, set())
                for k in ep2short.get(ep, []):
                    if k not in seen:
                        seen.add(k)
                        stack.append(k)
        if len(touched) >= 2:
            keep.extend(comp)
    return spine + [shorts[i] for i in keep]


def _apply_recognized_fillets_to_solid(
    wp: object,
    stitched_clusters: list[dict] | None = None,
    out_edges: list | None = None,
    out_surfaces: list | None = None,
    out_diag: list | None = None,
) -> tuple[object, list[object], list[str]]:
    """Apply stitched fillets to the rebuilt solid by category-aware
    vertex-proximity snapping.

    Each entry in ``stitched_clusters`` is one physical fillet group
    carrying ``{group_id, vertices: (N,3), radius, category, ...}``.
    The matcher (validated end-to-end against the rackmount in
    ``scripts/match_fillets_to_edges.py``) works like this, per
    cluster:

      1. **Robust radius.** Topology's ``radius`` field is unreliable
         (it stores the sphere-fit / hole radius, not the fillet tube
         radius). Use ``min(topo_radius, smallest_PCA_extent)`` — the
         cluster's thinnest principal extent bounds the tube radius —
         so a garbage 20 mm collapses to ~1.5 mm. The snap tolerance
         scales off this, not the raw field.

      2. **Snap.** Sample every rebuilt edge; an edge is a candidate
         if a good fraction of its samples fall within the snap
         tolerance of the cluster's fillet-surface vertex cloud. The
         fillet surface sits ~0.4·r from the sharp rebuilt edge, so a
         few-mm snap is correct.

      3. **Category gate.**
         - ``hole_edge``  → the single nearest CIRCLE/ELLIPSE edge
           (the hole opening on this cluster's own face).
         - everything else → the long crease edges only: keep edges
           whose length is ≥ 25 % of the longest matched edge, so a
           perimeter wrap keeps its outline sides corner-to-corner and
           a tab keeps its few short sides, dropping the perpendicular
           connector / rim-thickness fragments.

    If two clusters claim the same edge, the cluster with the smaller
    vertex-cloud bbox wins ("more specific" feature). OCC's
    ``BRepFilletAPI`` tangent-chains co-incident edges automatically,
    so a perimeter U gets handled as one rolling operation.
    """
    import time as _time
    from scipy.spatial import cKDTree as _cKDTree

    pass_start = _time.time()
    applied_steps: list = []
    applied_script: list = []

    clusters = list(stitched_clusters or [])
    if not clusters:
        logger.info("Fillet pass: no stitched clusters supplied")
        return wp, [], []

    # --- Enumerate ALL edges (line / circle / ellipse) with sampled
    # points. We deliberately do NOT restrict to concave LINE edges:
    # hole openings are convex ELLIPSE edges, and perimeter-wrap
    # creases are convex outer edges — the old concave-line-only
    # enumeration structurally could not match either. Selection is
    # done by proximity + category instead. ---
    N_SAMP = 15
    solid_edges: list[dict] = []
    try:
        for edge in wp.edges().vals():
            try:
                gtype = edge.geomType()
                sp = edge.startPoint()
                ep = edge.endPoint()
            except Exception:
                continue
            start = np.array([sp.x, sp.y, sp.z], dtype=float)
            end = np.array([ep.x, ep.y, ep.z], dtype=float)
            try:
                length = float(edge.Length())
            except Exception:
                length = float(np.linalg.norm(end - start))
            try:
                samples = np.array([
                    [(_p := edge.positionAt(t)).x, _p.y, _p.z]
                    for t in np.linspace(0.0, 1.0, N_SAMP)
                ], dtype=float)
            except Exception:
                ts = np.linspace(0.0, 1.0, N_SAMP)
                samples = start[None, :] * (1 - ts)[:, None] + end[None, :] * ts[:, None]
            solid_edges.append({
                "obj": edge, "type": gtype,
                "start": start, "end": end, "len": length,
                "samples": samples, "midpoint": samples[len(samples) // 2],
            })
    except Exception as exc:
        logger.debug("Edge enumeration failed: %s", exc)
        return wp, [], []

    logger.info("Edge enumeration: %d edges on rebuilt solid", len(solid_edges))
    if not solid_edges:
        logger.info("Fillet pass: no edges on solid")
        return wp, [], []

    # --- Category-aware vertex-proximity matching ---
    SNAP_TOL_BASE = 3.5
    MIN_SPINE_LEN = 3.0
    MIN_EDGE_FRAC = 0.5
    matched_by_cluster: dict[int, list[int]] = {}
    cluster_radius: dict[int, float] = {}
    for ci, c in enumerate(clusters):
        verts = np.asarray(c.get("vertices"), dtype=float)
        if verts.ndim != 2 or len(verts) == 0:
            continue
        category = c.get("category", "curve_edge")
        topo_r = float(c.get("radius", 0.0) or 0.0)
        # Robust radius from geometry (smallest PCA extent).
        cv = verts - verts.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(cv, full_matrices=False)
            smallest_ext = float((cv @ vh[2]).max() - (cv @ vh[2]).min())
        except Exception:
            smallest_ext = topo_r
        robust_r = min(topo_r, smallest_ext) if topo_r > 0 else smallest_ext
        robust_r = max(0.3, robust_r)
        snap = max(SNAP_TOL_BASE, 0.6 * robust_r + 2.0)

        ctree = _cKDTree(verts)
        cand: list[tuple] = []  # (edge_idx, frac_on, mean_dist, edge)
        for ei, e in enumerate(solid_edges):
            d, _ = ctree.query(e["samples"])
            frac = float(np.mean(d < snap))
            if frac >= MIN_EDGE_FRAC:
                cand.append((ei, frac, float(d.mean()), e))

        if category == "hole_edge":
            circ = [x for x in cand if x[3]["type"] in ("CIRCLE", "ELLIPSE")]
            circ.sort(key=lambda x: x[2])
            chosen = circ[:1]
        else:
            # Crease = long spine edges + the SHORT corner-arc edges
            # that bridge two spine edges (so the fillet wraps the U's
            # 90° corners instead of leaving them sharp). Rim-thickness
            # spanners (run along the band normal) and junction
            # connectors (dangle off a single spine end) are excluded.
            chosen = _select_crease_edges(
                cand, snap, vh[2], MIN_SPINE_LEN,
            )
        if not chosen:
            continue
        matched_by_cluster[ci] = [x[0] for x in chosen]
        cluster_radius[ci] = robust_r
        logger.info(
            "  cluster %d (%s): r=%.1f->%.1f snap=%.1f -> %d edge(s)",
            int(c.get("group_id", ci)), category, topo_r, robust_r,
            snap, len(chosen),
        )

    n_total_edges_matched = sum(len(v) for v in matched_by_cluster.values())
    logger.info(
        "Matching: %d/%d clusters claimed %d edges (of %d) in %.1fs",
        len(matched_by_cluster), len(clusters),
        n_total_edges_matched, len(solid_edges), _time.time() - pass_start,
    )
    if not matched_by_cluster:
        return wp, [], []

    # Resolve edge-ownership conflicts: smallest vertex-cloud bbox wins.
    cluster_bbox_vol: dict[int, float] = {}
    for ci in matched_by_cluster:
        v = np.asarray(clusters[ci]["vertices"], dtype=float)
        dims = v.max(axis=0) - v.min(axis=0)
        cluster_bbox_vol[ci] = float(np.prod(np.maximum(dims, 1e-3)))
    edge_to_cluster: dict[int, int] = {}
    for ci, edges in matched_by_cluster.items():
        for ei in edges:
            prev = edge_to_cluster.get(ei)
            if prev is None or cluster_bbox_vol[ci] < cluster_bbox_vol[prev]:
                edge_to_cluster[ei] = ci
    matched_by_cluster = {}
    for ei, ci in edge_to_cluster.items():
        matched_by_cluster.setdefault(ci, []).append(ei)

    if out_diag is not None:
        for ci, c in enumerate(clusters):
            n_e = len(matched_by_cluster.get(ci, []))
            out_diag.append({
                "group_id": int(c.get("group_id", ci)),
                "n_records": int(c.get("n_records", 0)),
                "category": c.get("category", "?"),
                "matched": n_e > 0,
                "n_edges": n_e,
                "reason": ("matched" if n_e > 0 else
                           "no edge within snap of cluster"),
            })

    if not matched_by_cluster:
        return wp, [], []

    # Populate the INTENT overlay BEFORE running OCC fillet. The
    # intent stage is meant to show every edge the matcher
    # decided was a fillet location — whether or not OCC's
    # BRepFilletAPI succeeds at applying it. Putting these
    # outputs here means the user can see what the matcher is
    # claiming, even on edges where the rebuild's topology
    # defeats OCC. Applied surfaces are STILL appended again in
    # the success branch; duplicates are fine because the viz
    # just paints them once.
    def _append_intent(ci: int, ei: int, r_use: float) -> None:
        se = solid_edges[ei]
        sp = se["start"]
        ep = se["end"]
        samples = se["samples"]
        is_curve = se.get("type") in ("CIRCLE", "ELLIPSE") or \
            float(np.linalg.norm(ep - sp)) < 1e-6
        if out_edges is not None:
            if is_curve:
                # Emit the curve as a polyline of consecutive sample
                # segments so the hole-opening ring renders in the
                # intent overlay (start==end for a closed ellipse).
                for k in range(len(samples) - 1):
                    out_edges.append([
                        list(map(float, samples[k])),
                        list(map(float, samples[k + 1])),
                    ])
            else:
                out_edges.append(
                    [list(map(float, sp)), list(map(float, ep))]
                )
        if out_surfaces is not None:
            if is_curve:
                # Surface patch for a swept ring isn't built here; the
                # polyline overlay above already marks the location.
                return
            edge_dir = ep - sp
            edge_len = float(np.linalg.norm(edge_dir))
            if edge_len < 1e-6:
                return
            edge_dir = edge_dir / edge_len
            arb = np.array([1.0, 0.0, 0.0])
            if abs(float(arb @ edge_dir)) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            perp_dir = arb - float(arb @ edge_dir) * edge_dir
            perp_dir = perp_dir / (np.linalg.norm(perp_dir) + 1e-12)
            mid = 0.5 * (sp + ep)
            patch = _build_fillet_surface_patch(
                sp, ep, mid + r_use * perp_dir, edge_dir, r_use,
            )
            if patch is not None:
                pv, pf = patch
                out_surfaces.append({
                    "verts": pv.tolist(), "faces": pf.tolist(),
                    "kind": "fillet",
                })

    for ci, edges in matched_by_cluster.items():
        r_pre = cluster_radius[ci]
        for ei in edges:
            shortest = solid_edges[ei]["len"]
            r_clamped = max(0.2, min(r_pre, shortest / 3.0))
            _append_intent(ci, ei, r_clamped)
    logger.info(
        "Intent overlay: %d edges queued from %d clusters "
        "(populated before OCC apply so the viz reflects "
        "matcher decisions, not just OCC successes)",
        sum(len(es) for es in matched_by_cluster.values()),
        len(matched_by_cluster),
    )

    buckets: list[dict] = []
    for ci, edges in matched_by_cluster.items():
        buckets.append({
            "cluster_idx": ci,
            "members": [
                {"edge_idx": ei, "radius": cluster_radius[ci]}
                for ei in edges
            ],
        })

    for bi, bucket in enumerate(buckets):
        members = bucket["members"]
        # --- Applied radius: whole millimetre, floor 2 mm. ---
        # The user's parts are designed with whole-mm fillets, never
        # below 2 mm. The geometry-derived robust radius only sets the
        # ballpark; round it to a whole mm and clamp to >= 2.
        r_robust = float(np.median([m["radius"] for m in members]))
        applied_r = max(2.0, float(round(r_robust)))
        # --- Radius cap from SPINE edges only. ---
        # A U-crease bundles long spine edges with tiny corner-arc
        # segments. OCC tangent-chains them, so the effective edge is
        # the whole chain — the 1.6 mm corner segments must NOT cap the
        # radius (that produced sub-mm phantom fillets). Cap by the
        # shortest LONG edge instead.
        member_lens = sorted(
            solid_edges[m["edge_idx"]]["len"] for m in members
        )
        max_len = member_lens[-1] if member_lens else 1.0
        spine_lens = [L for L in member_lens if L >= 0.25 * max_len]
        cap_len = min(spine_lens) if spine_lens else max_len
        r_cap = cap_len / 3.0
        if applied_r > r_cap:
            applied_r = max(2.0, float(int(r_cap)))

        edges_obj = [solid_edges[m["edge_idx"]]["obj"] for m in members]
        n_edges = len(edges_obj)
        t_bucket = _time.time()
        vol_before = abs(float(wp.val().Volume()))

        # --- Apply the WHOLE crease in one rolling operation. ---
        # Never fall back to per-edge: filleting the sides separately
        # leaves the corners sharp (the exact failure the user saw).
        # If OCC can't roll the chain at applied_r, retry the whole
        # chain at successively smaller whole-mm radii down to 2 mm;
        # if even 2 mm fails, skip the cluster (no broken partial).
        r_candidates = [applied_r]
        for rr in range(int(applied_r) - 1, 1, -1):
            if rr >= 2:
                r_candidates.append(float(rr))
        applied_this_bucket = False
        for r in r_candidates:
            try:
                wp_test = wp.newObject(edges_obj).fillet(r)
                vol_after = abs(float(wp_test.val().Volume()))
                if vol_after <= 0 or vol_after > vol_before * 1.05:
                    continue
                wp = wp_test
                v, f = _tessellate_solid(wp, fast=True)
                applied_steps.append(ConstructionStep(
                    name=f"Fillet (r={r:.0f}mm) x{n_edges}",
                    tess_vertices=v, tess_faces=f, volume=vol_after,
                ))
                applied_script.append(
                    f"# Fillet r={r:.0f} on {n_edges} crease edge(s)"
                )
                logger.info(
                    "Bucket %d (cluster %d): r=%.0fmm on %d edges, "
                    "vol %.0f -> %.0f (dt=%.1fs)",
                    bi, bucket["cluster_idx"], r, n_edges,
                    vol_before, vol_after, _time.time() - t_bucket,
                )
                applied_this_bucket = True
                break
            except Exception as exc:
                logger.debug(
                    "Bucket %d r=%.0f on %d edges failed: %s",
                    bi, r, n_edges, exc,
                )
                continue
        if not applied_this_bucket:
            logger.warning(
                "Bucket %d (cluster %d): could not roll %d-edge crease at "
                "any radius >= 2mm; skipped (no partial per-edge apply so "
                "corners aren't left sharp)",
                bi, bucket["cluster_idx"], n_edges,
            )

    logger.info(
        "Fillet pass done: %d clusters, %d edges in buckets, %d buckets, "
        "%d steps applied, %.1fs",
        len(clusters), n_total_edges_matched, len(buckets),
        len(applied_steps), _time.time() - pass_start,
    )
    return wp, applied_steps, applied_script


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
