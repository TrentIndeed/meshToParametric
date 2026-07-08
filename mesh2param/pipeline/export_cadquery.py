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
    sketch_overlay: np.ndarray | None = None  # (N, 3) — pairs of 3D points for blue sketch LINES
    # True-arc overlay segments so the viewer draws curves, not chords. Each:
    # {ax, pos, cu, cv, r, a0, a1} = a circular arc in the sketch plane (the
    # out-of-plane axis `ax` sits at `pos`; centre (cu,cv), radius r, angles).
    overlay_arcs: list | None = None


# --- Live construction-step streaming -------------------------------------
# The web UI's dual-pane "live build" view wants each construction step to
# appear on the right pane the moment it's built, not all at once when the
# CadQuery SUBPROCESS finishes and the result pickle is read back. Because the
# build runs in a separate process (VTK/Three.js isolation), we stream over the
# filesystem: when `stream_dir` is set, every ConstructionStep appended to the
# primary build list is written atomically to `{stream_dir}/NNNN.pkl`. The
# parent (viz_server/job_queue) scans that dir in its existing poll loop and
# appends new steps to `job.steps` live. Same-process callers leave it None.
#
# Cloud builds (Modal) can't share a filesystem with the web server, so the same
# per-step payload can ALSO be handed to a `sink` callback — `sink(payload, n)` —
# which the Modal function points at a modal.Queue. Either, both, or neither may
# be armed; the payload is built once and dispatched to whatever is set.
_stream_ctx: dict = {"dir": None, "n": 0, "sink": None,
                     "phase_name": None, "phase_t": None}


def _stream_phase(name: str) -> None:
    """Mark the current build SUB-PHASE (e.g. 'Applying fillets…', 'Refining…',
    'Validating…') and LOG how long the previous phase took.

    The post-build passes (fillet apply / residual refine / IoU validation)
    stream NO construction steps, so without this the UI sits on a silent
    "Building model" for minutes. Writing `_phase.txt` into the step stream dir
    lets the server surface the sub-phase as a stage; the timing log tells us
    where the minutes actually go. Best-effort — never breaks the build."""
    import os
    import time as _t
    now = _t.time()
    prev, pt = _stream_ctx.get("phase_name"), _stream_ctx.get("phase_t")
    if prev and pt:
        logger.info("[build-phase] '%s' took %.1fs", prev, now - pt)
    _stream_ctx["phase_name"] = name
    _stream_ctx["phase_t"] = now
    logger.info("[build-phase] -> %s", name)
    d = _stream_ctx.get("dir")
    if d:
        try:
            tmp = os.path.join(d, "._phase.tmp")
            final = os.path.join(d, "_phase.txt")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(name)
            os.replace(tmp, final)
        except Exception:
            logger.debug("phase marker write failed", exc_info=True)


def _set_step_stream_dir(stream_dir: str | None) -> None:
    """Arm (or disarm) live step streaming for the current build."""
    _stream_ctx["dir"] = stream_dir
    _stream_ctx["n"] = 0
    _stream_ctx["phase_name"] = None
    _stream_ctx["phase_t"] = None
    if stream_dir:
        try:
            import os
            os.makedirs(stream_dir, exist_ok=True)
        except Exception:
            logger.debug("Could not create step stream dir %s", stream_dir,
                         exc_info=True)


def _set_step_stream_sink(sink) -> None:
    """Arm (or disarm) a callback sink for live steps: `sink(payload: dict, n)`.

    Resets the shared step counter, so call this (or _set_step_stream_dir)
    exactly once at the start of a build before any steps are emitted."""
    _stream_ctx["sink"] = sink
    _stream_ctx["n"] = 0


def _stream_step(step: "ConstructionStep") -> None:
    """Mirror one step to whichever live sinks are armed (dir and/or callback).

    Best-effort: any failure here must never break the build."""
    d = _stream_ctx["dir"]
    sink = _stream_ctx["sink"]
    if not d and sink is None:
        return
    n = _stream_ctx["n"]
    payload = {
        "name": step.name,
        "tess_vertices": step.tess_vertices,
        "tess_faces": step.tess_faces,
        "volume": step.volume,
        "sketch_overlay": step.sketch_overlay,
        "overlay_arcs": step.overlay_arcs,
    }
    if d:
        try:
            import os
            import pickle
            tmp = os.path.join(d, f".{n:04d}.tmp")
            final = os.path.join(d, f"{n:04d}.pkl")
            with open(tmp, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp, final)
        except Exception:
            logger.debug("Step file-streaming failed for %r",
                         getattr(step, "name", "?"), exc_info=True)
    if sink is not None:
        try:
            sink(payload, n)
        except Exception:
            logger.debug("Step sink-streaming failed for %r",
                         getattr(step, "name", "?"), exc_info=True)
    _stream_ctx["n"] = n + 1


def _await_stitched_clusters(path: str, timeout: float = 180.0):
    """Block until the parent drops the stitched fillet clusters at `path`,
    then load and return them.

    Two-phase build: the prismatic model (base + bosses + holes) builds and
    streams FIRST — the moment the feature tree is ready — then waits HERE for
    the fillet clusters, which the parent computes CONCURRENTLY (overlapping all
    the analysis stages) and drops atomically at `path`. Returns [] on timeout
    so the model still exports (without fillets) if the clusters never arrive.
    """
    import os
    import pickle
    import time as _t
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                logger.info("Two-phase build: received %d fillet cluster(s)",
                            len(data) if data else 0)
                return data or []
            except Exception:
                logger.exception("Failed to load stitched clusters from %s", path)
                return []
        _t.sleep(0.2)
    logger.warning("Stitched clusters not delivered within %.0fs; exporting "
                   "without fillets", timeout)
    return []


class _StreamingStepList(list):
    """A list that mirrors each appended ConstructionStep to the stream dir.

    Used for the ONE primary build list so deferred chamfers / stitched
    fillets — which `.append`/`.extend` onto that same list after the base
    build — also stream in order, with no second list to double-count."""

    def append(self, step) -> None:
        super().append(step)
        if isinstance(step, ConstructionStep):
            _stream_step(step)

    def extend(self, it) -> None:
        for s in it:
            self.append(s)


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
    # AS-BUILT record: what the rebuild actually cut/blended after all
    # recovery passes (2D ring recall, duplicate-drill skip, greedy
    # fillet subsets). The Onshape live export consumes this instead of
    # the raw recognition tree — exporting the tree directly dropped 3
    # of 4 tab holes because hole recovery lives in this module.
    #   {"holes": [{centers, radius, axis_dir, through, depth}],
    #    "blends": [{kind, radius, edge_midpoints}]}
    as_built: dict = field(default_factory=dict)
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
    stream_dir: str | None = None,
    stream_sink=None,
    stitched_clusters_path: str | None = None,
    residual_refine: bool = True,
    residual_refine_min_error_pct: float = 5.0,
    fillet_refine: bool = False,
    contour_fallback: bool = True,
    contour_fallback_min_error_pct: float = 15.0,
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
    # Arm live step streaming (parent ingests files as they appear, or a sink
    # callback forwards them — e.g. to a modal.Queue for cloud builds). Only the
    # primary/default variant streams — see _build_stepwise's `stream` flag.
    _set_step_stream_dir(stream_dir)
    _set_step_stream_sink(stream_sink)
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

    # Ensure the base extrudes INTO the material, not away from it. recognize()
    # fixes the base-plane normal axis but its SIGN can point outward — then the
    # solid is built on the wrong side of the start face: right volume, wrong
    # position (0 exact IoU while volume "passes"; AI10). If the extruded span's
    # midpoint lands outside the mesh extent on the extrude axis, flip the dir
    # and re-extract so the profile/holes stay consistent with the new basis.
    _ax = int(np.argmax(np.abs(extrude_dir)))
    _lo, _hi = float(profile_mesh.bbox[0][_ax]), float(profile_mesh.bbox[1][_ax])
    _mid = extrude_start + float(extrude_dir[_ax]) * depth / 2.0
    if not (_lo - 0.1 * depth <= _mid <= _hi + 0.1 * depth):
        logger.info("Base extrude dir points outward; flipping into material (axis %d)", _ax)
        extrude_dir = -extrude_dir
        try:
            profiles, extrude_start, hole_features, chamfer_features, fillet_features = \
                _extract_profile_and_holes(profile_tree, profile_mesh, extrude_dir)
        except Exception:
            logger.exception("Stage 6b re-extract after dir flip failed")
            raise

    if profiles is None or len(profiles) == 0:
        logger.error("Failed to extract base profile from mesh")
        if contour_fallback:
            fb = _contour_fallback_result(cleaned, mesh_volume)
            if fb is not None:
                return fb
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
    # JOIN extrudes from the extrusion engine's plan (one part, several
    # extrusions — RackMount's side plates). Decomposed multibody pieces
    # never carry these (the engine declines open shells), so the two
    # paths don't overlap.
    join_features = [
        f for f in profile_tree.features
        if f.type == FeatureType.BOSS and f.operation == "JOIN"
        and (f.sketch_geometry or {}).get("type") == "polygon"
    ]
    _base_sg = (profile_tree.features[0].sketch_geometry or {}) \
        if profile_tree.features else {}
    # REVOLVE as the BASE feature (turned parts, scan meshes): when the
    # body is a recognized body of revolution and the revolve solid fits,
    # it becomes the timeline's first feature ("Revolve") and every hole/
    # recess cutter runs ON it — instead of the old end-of-build solid
    # swap, which surfaced as an opaque boolean and ERASED all the drilled
    # holes (the motor mount plate).
    _revolve_base = None
    try:
        _rev_sg0 = (feature_tree.metadata or {}).get("revolve_candidate")
        if (_rev_sg0 is not None and mesh_volume > 0
                and (getattr(cleaned, "metadata", None) or {}).get(
                    "scan_mode")):
            from mesh2param.models.feature import Feature as _F0
            _rev_feat0 = _F0(type=FeatureType.REVOLVE,
                             face_indices=np.array([0]),
                             sketch_geometry=_rev_sg0)
            _rev0 = _build_revolve_result(_rev_feat0, cleaned, mesh_volume,
                                          refine=False)
            if _rev0 is not None and _rev0.solid is not None:
                # The bare revolve carries no holes yet, so expect it a bit
                # OVER the mesh volume; accept a generous window.
                _rerr = (_rev0.cad_volume - mesh_volume) / mesh_volume
                if -0.05 <= _rerr <= 0.30:
                    _revolve_base = {"solid": _rev0.solid, "sg": _rev_sg0}
                    logger.info(
                        "Revolve adopted as BASE feature (vol %+.1f%% "
                        "before holes)", 100 * _rerr)
    except Exception:
        logger.debug("revolve-base adoption failed", exc_info=True)

    build_args = dict(
        profiles=profiles, depth=depth, extrude_dir=extrude_dir,
        extrude_start=extrude_start, holes=hole_features,
        chamfers=chamfer_features, fillets=fillet_features, cleaned=build_cleaned,
        joins=join_features,
        # Carried parametric arc data for the BASE outer profile -> exact OCC
        # tangent fillets matching the Onshape sketch.
        base_corners=_base_sg.get("corners"),
        base_fillets=_base_sg.get("fillets"),
        revolve_base=_revolve_base,
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
    as_built_record: dict = {"holes": [], "blends": []}
    # Record the AS-BUILT base profile (outer boundary + interior cutouts)
    # so the Onshape export replays the exact frame CadQuery extruded. The
    # recognizer's engine profile carries only the OUTER outline; large
    # through-openings (Mesh3's frame windows) are found by the slice path
    # here as profiles[1:]. Without this the Onshape base extrudes a solid
    # plate (~2x volume). Same philosophy as the holes/blends records.
    try:
        _bax = int(np.argmax(np.abs(extrude_dir)))
        as_built_record["base"] = {
            "axis_idx": _bax,
            "span": [float(extrude_start), float(extrude_start + depth)],
            "outer": [[float(p[0]), float(p[1])] for p in profiles[0]],
            "inner": [[[float(p[0]), float(p[1])] for p in ring]
                      for ring in profiles[1:] if ring is not None and len(ring) >= 3],
        }
    except Exception:
        logger.debug("Could not record as-built base profile", exc_info=True)
    # Multi-body: record each ADDITIONAL body's base profile too, so the
    # Onshape export extrudes every body (not just body 0). Without this the
    # side plates (RackMount bodies 1-2) never get built — their volume is
    # missing (~72% short) and the fillets/holes that ride them ERROR with no
    # geometry to select. Same coordinate frame as body 0, so they share the
    # single origin offset and the assembly stays intact.
    as_built_record["extra_bodies"] = []
    for abd in additional_bodies_data:
        try:
            _ap = abd.get("profiles") or []
            if not _ap:
                continue
            _bax2 = int(np.argmax(np.abs(np.asarray(abd["extrude_dir"]))))
            _bs = float(abd["extrude_start"])
            as_built_record["extra_bodies"].append({
                "axis_idx": _bax2,
                "span": [_bs, _bs + float(abd["depth"])],
                "outer": [[float(p[0]), float(p[1])] for p in _ap[0]],
                "inner": [[[float(p[0]), float(p[1])] for p in ring]
                          for ring in _ap[1:] if ring is not None and len(ring) >= 3],
            })
        except Exception:
            logger.debug("Could not record as-built extra body", exc_info=True)
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
            as_built_out=as_built_record, stream=True,
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
                        as_built_out=as_built_record,
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
                # Two-phase build: the prismatic solid above already built and
                # streamed. If the fillet clusters weren't passed inline, wait
                # for the parent to drop them (it computes them concurrently),
                # then apply blends as phase two.
                if not stitched_clusters and stitched_clusters_path:
                    logger.info("CHECKPOINT: base solid built — awaiting fillet "
                                "clusters for phase two")
                    stitched_clusters = _await_stitched_clusters(
                        stitched_clusters_path)
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

                _stream_phase("Applying fillets…")

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
                            as_built_out=as_built_record,
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

    # CADFit-style residual-refinement post-pass: patch the primary solid
    # toward the target mesh, keeping the result only if the volume error
    # strictly improves. Pure safety net — a no-op when recognition already
    # matched the mesh (clean CAD), so it cannot regress those parts.
    # Skip entirely when the build already matches well (avoids the
    # tessellation+voxelization cost on the common clean-CAD case, where it
    # only ever no-ops anyway).
    # When a thin-member restructure explained the part, DON'T bolt refine
    # boxes on top: they carved visible slits into the side faces and read as
    # random subtractions in the tree ("three unnecessary cuts") — the
    # residual there is soft blend material, not missing features.
    if (residual_refine and solid is not None and mesh_volume > 0
            and not (as_built_record or {}).get("thin_member")
            and not (as_built_record or {}).get("revolve_base")):
        _pre_vol = variants[0].cad_volume if variants else _solid_volume(solid)
        _pre_err = abs(_pre_vol - mesh_volume) / (mesh_volume + 1e-15) * 100
        if _pre_err >= residual_refine_min_error_pct:
            _stream_phase("Refining model…")
            try:
                solid, _refined_vol = _residual_refine_solid(
                    solid, cleaned, mesh_volume,
                    steps_out=(variants[0].construction_steps
                               if variants else None))
                if variants:
                    variants[0].cad_volume = _refined_vol
            except Exception:
                logger.debug("residual_refine post-pass errored, ignoring",
                             exc_info=True)

    # Opt-in fillet detection by IoU sweep (alternative to the blend-graph
    # path). Off by default — see _fillet_refine_solid.
    if fillet_refine and solid is not None:
        try:
            solid = _fillet_refine_solid(solid, cleaned)
            if variants:
                variants[0].cad_volume = _solid_volume(solid)
        except Exception:
            logger.debug("fillet_refine errored, ignoring", exc_info=True)

    # Best-of-both router (Phase 5 outcome, docs/CADFIT_PORTING_PLAN.md): when
    # recognition STILL matches the mesh poorly, also build the
    # segmentation-free contour path and keep whichever solid matches the
    # target better. Cheap insurance — only poor builds pay for the second
    # reconstruction — and it rescues recognition mis-fires (validated:
    # AI11 49.7% -> 21%) with zero risk to good builds (skipped below the
    # threshold). Lazy import avoids a circular dependency (contour_recon
    # imports helpers from this module).
    # REVOLVE keep-better (Phase 4): if recognition flagged the body as a
    # revolution, build the revolve and keep it only if it matches the target
    # BETTER than the prismatic solid. Never forced — a revolve silhouette can
    # be badly wrong on a part prismatic models well (so this guards against
    # regression), and it wins for genuine turned parts.
    _rev_sg = (feature_tree.metadata or {}).get("revolve_candidate")
    if (_rev_sg is not None and mesh_volume > 0
            and not (as_built_record or {}).get("revolve_base")):
        try:
            from mesh2param.models.feature import Feature as _F
            rev_feat = _F(type=FeatureType.REVOLVE, face_indices=np.array([0]),
                          sketch_geometry=_rev_sg)
            rev_res = _build_revolve_result(rev_feat, cleaned, mesh_volume)
            cur_vol = (variants[0].cad_volume if variants
                       else _solid_volume(solid) if solid else 0.0)
            cur_err = (abs(cur_vol - mesh_volume) / (mesh_volume + 1e-15) * 100
                       if solid is not None else 1e9)
            # Clean CAD: STRICT — only swap when the revolve is genuinely better
            # (protects the volume-accuracy guardrail). Dirty/AI: PREFER the
            # revolve when it's close and not bad — for a turned part it's the
            # correct, smooth representation (the mesh-derived prismatic profile
            # is bumpy), which is what the user wants even at equal volume.
            scan_mode = bool((getattr(cleaned, "metadata", None) or {})
                             .get("scan_mode"))
            use_rev = False
            if rev_res is not None and rev_res.solid is not None:
                re = rev_res.volume_error_pct
                if re < cur_err - 1.0:
                    use_rev = True
                elif scan_mode and re <= cur_err + 3.0 and re < 25.0:
                    use_rev = True
            if use_rev:
                logger.info("Revolve chosen: prismatic %.1f%% -> revolve %.1f%%"
                            " (scan_mode=%s)", cur_err,
                            rev_res.volume_error_pct, scan_mode)
                solid = rev_res.solid
                if variants:
                    variants[0].cad_volume = rev_res.cad_volume
        except Exception:
            logger.debug("revolve keep-better failed", exc_info=True)

    # Build collapsed entirely (no solid from any variant): take the contour
    # path outright rather than returning nothing.
    if contour_fallback and solid is None:
        fb = _contour_fallback_result(cleaned, mesh_volume)
        if fb is not None:
            return fb

    if contour_fallback and solid is not None and mesh_volume > 0:
        _cur_vol = variants[0].cad_volume if variants else _solid_volume(solid)
        _cur_err = abs(_cur_vol - mesh_volume) / (mesh_volume + 1e-15) * 100
        _eff_thr = contour_fallback_min_error_pct
        # A symmetric thin-member restructure means the part IS explained by
        # real extrudes; its residual error is soft blends the arbiter can
        # only "fix" with an opaque solid swap ("Refined Model") — the exact
        # presentation the user rejected. The L-bracket hovers at 12-16%
        # run-to-run, flapping across the default 15% trigger: hold the
        # honest tree unless recognition is truly bad.
        if ((as_built_record or {}).get("thin_member")
                or (as_built_record or {}).get("revolve_base")):
            _eff_thr = max(_eff_thr, contour_fallback_min_error_pct + 6.0)
        if _cur_err >= _eff_thr:
            try:
                from mesh2param.pipeline.contour_recon import (
                    reconstruct_segmentation_free,
                )
                b = reconstruct_segmentation_free(cleaned)
                if b.solid is not None and b.volume_error_pct < _cur_err - 1.0:
                    logger.info(
                        "Contour fallback wins: recognition %.1f%% -> "
                        "contour %.1f%% volume error", _cur_err,
                        b.volume_error_pct)
                    fb_solid, fb_vol = b.solid, b.cad_volume
                    # The plain fallback is FEATURELESS — adopting it wholesale
                    # drops every recognized cut. Build repair CANDIDATES and
                    # let exact IoU pick the best (never worse than plain):
                    #  * HYBRID (fallback ∩ recognition + bosses): right when
                    #    recognition was OVER-solid (AI4 — the intersection
                    #    keeps the union of both cut-sets).
                    #  * TRANSPLANT (mesh-derived cuts applied to the
                    #    fallback): right when recognition was UNDER-solid
                    #    (the carriage block — intersecting there shrinks the
                    #    part further, but its holes are real and re-cuttable
                    #    straight from the mesh).
                    candidates = [("fallback", fb_solid, fb_vol)]
                    import cadquery as _cq2   # cq is shadowed by later local imports
                    try:
                        hy = fb_solid.intersect(solid)
                        try:
                            for _sb in _detect_standing_bosses(cleaned,
                                                               extrude_dir):
                                _o = [0.0, 0.0, 0.0]
                                _o[_sb["axis"]] = _sb["face_t"]
                                _sgn = _sb["sign"]
                                pads = []
                                if _sb.get("keep_outline"):
                                    pads.append(
                                        _cq2.Workplane(_sb["plane_name"],
                                                       origin=tuple(_o))
                                        .polyline(_sb["keep_outline"]).close()
                                        .extrude(_sgn * _sb["height"]))
                                for _k in _sb["keeps"]:
                                    pads.append(
                                        _cq2.Workplane(_sb["plane_name"],
                                                       origin=tuple(_o))
                                        .center(_k["cu"], _k["cv"])
                                        .circle(_k["r"])
                                        .extrude(_sgn * _sb["height"]))
                                for pad in pads:
                                    hy = hy.union(pad.intersect(solid))
                        except Exception:
                            logger.debug("boss re-union failed", exc_info=True)
                        hy_vol = _solid_volume(hy)
                        if hy_vol > 0.25 * mesh_volume:
                            candidates.append(("hybrid", hy, hy_vol))
                    except Exception:
                        logger.info("fallback-hybrid intersect failed",
                                     exc_info=True)
                    try:
                        # Transplant onto the PRE-REFINE fallback and refine
                        # LAST: refining first carved rectangular boxes around
                        # the jagged baked windows, and those over-cuts
                        # survived the transplant's clean slot cuts (the
                        # carriage block's "bad rectangular cutout"). With the
                        # slots already clean, the refine's residual there is
                        # ~zero and its boxes go where they're really needed.
                        try:
                            b2 = reconstruct_segmentation_free(
                                cleaned, residual=False)
                            tp_base = (b2.solid if b2.solid is not None
                                       else fb_solid)
                        except Exception:
                            tp_base = fb_solid
                        tp = _transplant_mesh_features(
                            tp_base, cleaned, feature_tree, extrude_dir, _cq2)
                        tp_steps: list = []
                        try:
                            tp, _ = _residual_refine_solid(
                                tp, cleaned, mesh_volume, steps_out=tp_steps)
                        except Exception:
                            logger.debug("post-transplant refine failed",
                                         exc_info=True)
                        tp_vol = _solid_volume(tp)
                        if tp_vol > 0.25 * mesh_volume:
                            candidates.append(("transplant", tp, tp_vol))
                    except Exception:
                        logger.info("fallback transplant failed",
                                     exc_info=True)
                    best_name = "fallback"
                    if len(candidates) > 1:
                        try:
                            from mesh2param.pipeline.iou_refine import (
                                exact_iou_available, mesh_iou_exact)
                            if exact_iou_available():
                                target = trimesh.Trimesh(
                                    vertices=cleaned.vertices,
                                    faces=cleaned.faces, process=False)

                                def _iou_of(wp2):
                                    tv2, tf2 = _tessellate_solid(wp2)
                                    m2 = trimesh.Trimesh(
                                        vertices=np.asarray(tv2, float).reshape(-1, 3),
                                        faces=np.asarray(tf2, np.int64).reshape(-1, 3),
                                        process=False)
                                    return mesh_iou_exact(target, m2)

                                scored = [(nm, s, v, _iou_of(s))
                                          for nm, s, v in candidates]
                                # plain fallback wins ties (simplest model)
                                best_name, fb_solid, fb_vol, _best_iou = max(
                                    scored, key=lambda c: c[3] - 1e-4 * (c[0] != "fallback"))
                                logger.info(
                                    "Fallback repair arbiter: %s wins (%s)",
                                    best_name,
                                    ", ".join(f"{nm}={io:.3f}"
                                              for nm, _, _, io in scored))
                            else:
                                best_name, fb_solid, fb_vol = min(
                                    candidates,
                                    key=lambda c: abs(c[2] - mesh_volume))
                        except Exception:
                            logger.info("fallback arbiter failed",
                                         exc_info=True)
                    _recog_solid = solid
                    solid = fb_solid
                    # Design-intent guarantee: recognized CIRCULAR boss pads
                    # (the lid's feet) survive the solid swap as perfect
                    # cylinders. The fallback bakes them as jagged blobs and
                    # the intersect mixes blob and cylinder — the user sees
                    # deformed feet. Carve a slightly dilated zone around
                    # each pad and re-union the clean pad clipped to the
                    # recognition solid.
                    try:
                        for _sb in _detect_standing_bosses(cleaned,
                                                           extrude_dir):
                            _o = [0.0, 0.0, 0.0]
                            _o[_sb["axis"]] = _sb["face_t"]
                            _sgn = _sb["sign"]
                            for _k in _sb["keeps"]:
                                _zone = (_cq2.Workplane(
                                            _sb["plane_name"],
                                            origin=tuple(_o))
                                         .center(_k["cu"], _k["cv"])
                                         .circle(_k["r"] + 0.6)
                                         .extrude(_sgn * _sb["height"]))
                                _pad = (_cq2.Workplane(
                                            _sb["plane_name"],
                                            origin=tuple(_o))
                                        .center(_k["cu"], _k["cv"])
                                        .circle(_k["r"])
                                        .extrude(_sgn * _sb["height"]))
                                solid = solid.cut(_zone).union(
                                    _pad.intersect(_recog_solid))
                        fb_vol = _solid_volume(solid)
                    except Exception:
                        logger.debug("boss preservation failed",
                                     exc_info=True)
                    # NEVER an opaque step: express the swap as itemized
                    # cut/add features applied to the recognition solid.
                    # Only if itemization fails badly does the plain swap
                    # remain (and surfaces as the catch-all boolean).
                    # When the TRANSPLANT won and already carries itemized
                    # refine steps, re-itemizing the swap is pure duplicate
                    # work (minutes of box-carving the pillow block then
                    # discarded by the quality floor) — the tp_steps below
                    # give the tree its real features.
                    _swap_itemized = False
                    try:
                        _skip_itemize = (best_name == "transplant"
                                         and bool(tp_steps))
                    except Exception:
                        _skip_itemize = False
                    if _skip_itemize:
                        logger.info(
                            "Transplant won with %d itemized refine "
                            "step(s) — skipping swap itemization",
                            len(tp_steps))
                    try:
                        _it_solid, _it_steps = ((None, None) if _skip_itemize
                                                else _itemize_solid_swap(
                            _recog_solid, solid, cleaned, mesh_volume))
                        if _it_solid is not None:
                            solid = _it_solid
                            fb_vol = _solid_volume(solid)
                            _swap_itemized = True
                            if variants:
                                variants[0].construction_steps.extend(
                                    _it_steps)
                    except Exception:
                        logger.debug("swap itemization errored",
                                     exc_info=True)
                    if variants:
                        variants[0].cad_volume = fb_vol
                        # Surface the transplant's refine ops as tree steps
                        # when the transplant won the arbiter (unless the
                        # itemized ops replaced the swap wholesale).
                        try:
                            if (best_name == "transplant" and tp_steps
                                    and not _swap_itemized):
                                variants[0].construction_steps.extend(tp_steps)
                        except Exception:
                            pass
            except Exception:
                logger.debug("contour fallback errored, ignoring",
                             exc_info=True)

    # Primary result uses the first (best) variant
    cad_volume = variants[0].cad_volume if variants else 0.0
    volume_error = abs(cad_volume - mesh_volume) / (mesh_volume + 1e-15) * 100
    status = "PASS" if volume_error < 10.0 else "FAIL"

    # Final tessellation
    _stream_phase("Validating…")
    tess_verts, tess_faces = _tessellate_solid(solid)

    # CATCH-ALL final step: the post-build passes (residual refine, contour
    # fallback, revolve keep-better) can REPLACE the solid without emitting a
    # construction step — the refined geometry then exists only in this final
    # tessellation/STEP while the UI's "final model" loads the LAST STEP, so
    # the user saw the pre-refine solid (AI4's underside cavity missing). If
    # the final solid's volume differs materially from the last step's, append
    # it as the closing step (reusing the tessellation just computed; the list
    # is the live-streaming one, so it also reaches the viewer).
    try:
        _steps_out = variants[0].construction_steps if variants else None
        if (_steps_out and tess_verts is not None and len(tess_verts)
                and mesh_volume > 0):
            _last_v = float(_steps_out[-1].volume or 0.0)
            if abs(cad_volume - _last_v) > 0.005 * mesh_volume:
                _steps_out.append(ConstructionStep(
                    name="Refined Model",
                    tess_vertices=np.asarray(tess_verts),
                    tess_faces=np.asarray(tess_faces),
                    volume=float(cad_volume),
                ))
                logger.info("Appended Refined Model step: %.0f -> %.0f mm^3",
                            _last_v, cad_volume)
    except Exception:
        logger.debug("refined-model catch-all step failed", exc_info=True)

    # Onshape-vocabulary step names for scan/AI parts: the user reads the
    # tree as an Onshape timeline, so every step is "<FeatureType> N (…)" —
    # Sketch/Extrude/Hole/Fillet/Chamfer/Boolean — never internal jargon
    # like "Refine Cut". Clean-CAD parts keep the legacy names (tests pin
    # them, and no user-facing scan tree is involved).
    if (getattr(cleaned, "metadata", None) or {}).get("scan_mode"):
        for _v in variants:
            try:
                _merge_coplanar_hole_steps(_v.construction_steps)
                _onshape_step_names(_v.construction_steps)
            except Exception:
                logger.debug("step rename failed", exc_info=True)

    # Export STEP. If ingest normalized the part's scale (unit_scale in
    # CleanedMesh.metadata), scale the solid back so the STEP file is in
    # the user's original units; everything in-pipeline stays normalized.
    step_path = ""
    if output_path is not None:
        try:
            import cadquery as cq
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_solid = solid
            unit_scale = float(
                (getattr(cleaned, "metadata", None) or {}).get("unit_scale", 1.0) or 1.0
            )
            if unit_scale != 1.0 and solid is not None:
                inv = 1.0 / unit_scale
                mat = cq.Matrix([
                    [inv, 0, 0, 0],
                    [0, inv, 0, 0],
                    [0, 0, inv, 0],
                ])
                export_solid = solid.newObject(
                    [o.transformGeometry(mat) for o in solid.objects]
                )
                logger.info(
                    "Scaling STEP back to original units (x%.6g)", inv,
                )
            cq.exporters.export(export_solid, str(output_path))
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
        as_built=as_built_record,
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

    # Multi-extrude part: the base covers only its engine segment's span
    # (RackMount's base plate is 9mm of an 85mm Z extent — joins fill the
    # rest). Slice within the span and start the extrude at its floor so
    # the base doesn't sweep through the side plates' territory.
    base_span = None
    base_engine_profile = None
    base_engine_inner: list = []
    if feature_tree.features:
        _bsg = feature_tree.features[0].sketch_geometry or {}
        base_span = _bsg.get("span")
        if _bsg.get("points"):
            base_engine_profile = _clean_engine_polygon(_bsg["points"])
            if len(base_engine_profile) < 3:
                base_engine_profile = None
            for ring in _bsg.get("inner") or []:
                rp = _clean_engine_polygon(ring, min_area=0.1)  # keep tiny bores
                if len(rp) >= 3:
                    base_engine_inner.append(rp)
    if base_span is not None:
        extrude_start = float(base_span[0])

    # Tapered/drafted plate: recognize chose to extrude along a NON-thin
    # axis (the side view). A single cross-section slice there is a flat
    # rectangle — it misses the taper. The SILHOUETTE (outline of all
    # vertices projected onto the extrude plane) is the trapezoid. Compute
    # it here and use it as the outer profile below. Only for this
    # tapered-side case; flat parts keep the slice path.
    _bbsize = cleaned.bbox[1] - cleaned.bbox[0]
    _thin_axis = int(np.argmin(_bbsize))
    _has_joins = any(
        f.type == FeatureType.BOSS and f.operation == "JOIN"
        for f in feature_tree.features
    )
    silhouette_profile = None
    # The silhouette is for single-extrude TAPERED plates (the trapezoid
    # side view). On a multi-extrude part the base's own cross-section is
    # already correct, and the silhouette would fill the hollow between
    # joins (RackMount's U-channel got stuffed solid: volume looked right,
    # every section was wrong).
    if primary_axis != _thin_axis and not _has_joins and base_span is None:
        try:
            from shapely import concave_hull, simplify
            from shapely.geometry import MultiPoint
            pts2d = np.column_stack([
                cleaned.vertices[:, ax_u], cleaned.vertices[:, ax_v],
            ])
            hull = concave_hull(MultiPoint(pts2d), ratio=0.2)
            hull = simplify(hull, 0.5)
            if hull.geom_type == "Polygon" and hull.area > 1.0:
                coords = list(hull.exterior.coords)[:-1]
                coords = _simplify_profile(coords, tolerance=0.5)
                if len(coords) >= 3:
                    silhouette_profile = coords
                    logger.info(
                        "Tapered side-extrude: using %d-pt silhouette profile "
                        "(area=%.0f) along axis %d",
                        len(coords), hull.area, primary_axis,
                    )
        except Exception as exc:
            logger.debug("Silhouette profile failed (%s) — using slices", exc)

    normal = [0.0, 0.0, 0.0]
    normal[primary_axis] = 1.0

    # Collect profiles from multiple slices across the full depth, then
    # pick the one that best reproduces the part with a SINGLE extrude —
    # the lowest-step base. The right profile is the cross-section whose
    # area matches the part's AVERAGE cross-section (volume / depth), not
    # the largest slice: for a tapered or corner-cut plate (BottomMount
    # grows 27k->32k mm^2 bottom-to-top) the largest slice over-fills,
    # while the average-area slice is the trapezoid that minimizes the
    # residual the downstream cut/add features must fix. This also
    # rejects fillet-inset slices (below average) without a special case.
    extent = float(all_proj.max() - all_proj.min())
    target_area = 0.0
    try:
        if mesh.is_watertight and extent > 1e-6:
            target_area = abs(float(mesh.volume)) / extent
    except Exception:
        target_area = 0.0

    candidates: list[tuple[float, float, list]] = []  # (rank_key, -area, profiles)

    # Sample slices across the full depth: top, bottom, and intermediate
    # positions.  Intermediate slices avoid blind holes near the faces and
    # find the true widest outer boundary.
    surface_inset = 0.1  # mm inside the surface
    z_min = float(all_proj.min())
    z_max = float(all_proj.max())
    if base_span is not None:
        z_min = max(z_min, float(base_span[0]))
        z_max = min(z_max, float(base_span[1]))
        extent = max(z_max - z_min, 1e-6)
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

            # Rank key: closeness of this slice's outer area to the
            # part's average cross-section (volume/depth) — the slice
            # whose single extrude best matches the part. Falls back to
            # largest-area (negative) when the mesh volume is unavailable
            # (non-watertight), preserving the old behavior there.
            if target_area > 0:
                rank_key = abs(outer_boundary_area - target_area)
            else:
                rank_key = -outer_boundary_area
            candidates.append(
                (rank_key, -outer_boundary_area, slice_profiles)
            )
        except Exception:
            continue

    if not candidates and base_engine_profile is None:
        return None, extrude_start, [], [], []

    # Pick best candidate
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        best_profiles = candidates[0][2]
    else:
        best_profiles = []

    # Multi-extrude base: the engine already measured this segment's
    # cross-section (outer + interior windows). The slice-ranking
    # heuristics target the FULL part's average area, which is wrong
    # when joins carry part of the volume — RackMount's U-profile base
    # was replaced by a volume-matched smaller slice, right volume,
    # wrong everywhere.
    if base_engine_profile is not None and len(base_engine_profile) >= 3:
        best_profiles = [base_engine_profile] + base_engine_inner

    # Tapered side-extrude: the silhouette IS the outer profile (the
    # trapezoid). It supersedes the slice-derived outer boundary, which is
    # only a flat rectangle for this axis. Inner cutouts from the slice
    # path (if any) are kept after it.
    if silhouette_profile is not None and base_engine_profile is None:
        best_profiles = [silhouette_profile] + best_profiles[1:]

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
    if best_profiles and base_engine_profile is None:
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
            if (
                merged.geom_type == 'Polygon' and merged.interiors
                and base_engine_profile is None
            ):
                clean_outer = list(merged.exterior.coords)[:-1]
                clean_outer = _simplify_profile(clean_outer, tolerance=0.5)
                # Only adopt the face-projection exterior to FIX hole-slit
                # artifacts in the slice outline — not to enlarge it. The
                # projection is the union of all top/bottom faces, i.e. the
                # full silhouette; on a tapered or corner-cut plate that is
                # bigger than the volume-matched slice and would over-fill
                # the base (the +9.8% on BottomMount). Slit removal barely
                # changes area; a taper adds ~10%. Keep the selected slice
                # when the clean exterior would grow it materially.
                cur_area = _polygon_area(best_profiles[0]) if best_profiles else 0.0
                clean_area = _polygon_area(clean_outer) if len(clean_outer) >= 3 else 0.0
                grows = cur_area > 0 and clean_area > cur_area * 1.03
                if len(clean_outer) >= 3 and not grows:
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
                            # Design-intent regularization on dirty meshes:
                            # this ring traces the slot fillet's jagged outer
                            # rim — snap it to a clean circle/capsule/smooth
                            # outline instead of baking the jags.
                            if (getattr(cleaned, "metadata", None)
                                    or {}).get("scan_mode"):
                                _reg = _void_intent_outline(cutout_coords)
                                if _reg and len(_reg) >= 3:
                                    cutout_coords = _reg
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
                    # The hole's LOCAL axial midpoint (its sketch draws
                    # here) — the global bbox midpoint put tab-hole
                    # sketches 40mm below the tabs they pierce.
                    _cb_d = np.linalg.norm(
                        np.column_stack([
                            cleaned.vertices[:, ax_u] - c_2d.x,
                            cleaned.vertices[:, ax_v] - c_2d.y,
                        ]), axis=1,
                    )
                    _cb_nb = _cb_d < est_radius * 3.0
                    if _cb_nb.any():
                        _cb_ax = cleaned.vertices[_cb_nb, primary_axis]
                        hole_center_3d[primary_axis] = (
                            float(_cb_ax.min()) + float(_cb_ax.max())
                        ) / 2.0
                    else:
                        hole_center_3d[primary_axis] = (
                            float(cleaned.bbox[0][primary_axis])
                            + float(cleaned.bbox[1][primary_axis])
                        ) / 2.0

                    # PIERCE CHECK: a 2D ring is only a through-hole if
                    # the bore is VOID just inside BOTH faces of the
                    # local material (parity sections, so bores read as
                    # voids). A partial-depth cavity has material over
                    # one end and is a pocket, not a hole. A stepped
                    # bore (RackMount's tab holes drop into their relief
                    # counterbores) is void at both probes and passes —
                    # the earlier column-span test wrongly rejected it.
                    _col_d = np.linalg.norm(
                        np.column_stack([
                            cleaned.vertices[:, ax_u] - c_2d.x,
                            cleaned.vertices[:, ax_v] - c_2d.y,
                        ]), axis=1,
                    )
                    _nb = _col_d < est_radius * 3.0
                    _ax_all = cleaned.vertices[:, primary_axis]
                    if not _nb.any():
                        continue
                    _lo, _hi = float(_ax_all[_nb].min()), float(_ax_all[_nb].max())
                    from mesh2param.pipeline.recognize import _section_material
                    from shapely.geometry import Point as _PiercePt
                    import trimesh as _pierce_tm
                    _pmesh = _pierce_tm.Trimesh(
                        vertices=cleaned.vertices, faces=cleaned.faces,
                        process=False,
                    )
                    _is_through_ring = True
                    for _probe in (_lo + 0.8, _hi - 0.8):
                        _geom = _section_material(_pmesh, primary_axis, _probe)
                        if _geom is not None and _geom.buffer(-0.2).contains(
                            _PiercePt(c_2d.x, c_2d.y)
                        ):
                            _is_through_ring = False
                            break
                    if not _is_through_ring:
                        logger.info(
                            "2D ring at (%.1f, %.1f) r=%.2f rejected: bore "
                            "covered at a face (partial-depth cavity)",
                            c_2d.x, c_2d.y, est_radius,
                        )
                        continue

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

        # Merge 2D-detected holes with recognition-based ones. This used
        # to wholesale-replace recognition holes whenever 2D found any
        # ring, but that threw away every hole 2D can't see: side holes
        # along non-extrude axes (Body1Mesh's two blind Z edge holes) and
        # fused tangent pairs whose ring isn't a clean circle. Policy:
        #   • recognition center MATCHED by a 2D ring → the ring wins
        #     (its geometry comes straight from the mesh faces and keeps
        #     hole edges where the fillet matcher expects them)
        #   • 2D ring with no recognition match → added (recall)
        #   • recognition hole 2D can't see → kept only if BLIND (a real
        #     side hole has a bottom cap) or on the extrude axis; a
        #     sideways THROUGH cylinder with no cap is fitter junk
        #     (RackMount's base grew two 25mm horizontal tunnels that
        #     chewed through real hole edges)
        if detected_holes:
            ext_axis = ext_norm

            from matplotlib.path import Path as _JoinPath
            _join_lookup = []
            for jf in feature_tree.features:
                if jf.type != FeatureType.BOSS or jf.operation != "JOIN":
                    continue
                jsg = jf.sketch_geometry or {}
                if not jsg.get("points") or not jsg.get("span"):
                    continue
                jax = int(jsg.get("axis_idx", 0))
                # Include EVERY loop of a pattern boss (AI1's four-screw sketch
                # carries the other three in extra_loops); otherwise only the
                # first screw of each pattern is known and the rest get drilled.
                for loop in [jsg["points"], *(jsg.get("extra_loops") or [])]:
                    if loop and len(loop) >= 3:
                        _join_lookup.append((
                            jax,
                            [a for a in range(3) if a != jax],
                            _JoinPath(np.asarray(loop, dtype=float)),
                            jsg["span"],
                        ))

            # A boss the planner already built (a screw/standoff) must NOT then
            # be drilled: the flat plate face reads a RING where the standoff
            # pokes through, and the 2D pass would cut the whole standoff out.
            # Drop detected holes whose centre sits inside a coaxial boss.
            def _on_join_boss(h) -> bool:
                hd = np.asarray(h.direction, dtype=float)
                hd = hd / (np.linalg.norm(hd) + 1e-15)
                hsg = h.sketch_geometry or {}
                centers = hsg.get("centers") or [hsg.get("center")]
                for cc in centers:
                    if cc is None:
                        continue
                    c3 = np.asarray(cc, dtype=float)
                    for jax, (ju, jv), jpath, jspan in _join_lookup:
                        if abs(hd[jax]) < 0.9:
                            continue
                        if jpath.contains_point((c3[ju], c3[jv]), radius=0.5):
                            return True
                return False

            _kept = [h for h in detected_holes if not _on_join_boss(h)]
            if len(_kept) != len(detected_holes):
                logger.info("Dropped %d detected hole(s) coaxial with a boss",
                            len(detected_holes) - len(_kept))
            detected_holes = _kept

            def _trustworthy(rh) -> bool:
                rdir = np.asarray(rh.direction, dtype=float)
                rdir = rdir / (np.linalg.norm(rdir) + 1e-15)
                if abs(float(rdir @ ext_axis)) >= 0.9:
                    return True
                # A sideways hole is real only if it pierces a JOIN body: its
                # centre must lie inside that join's 3D volume (its profile in
                # the join plane AND within the join's span). The hole may run
                # ALONG the join's extrude axis (a coaxial standoff bore) OR
                # PERPENDICULAR to it (a mounting hole drilled tangentially
                # through a wall the engine extrudes along the base axis) —
                # either is a real hole, not a tab-relief artifact floating in
                # the base air. (Don't require the hole to parallel the join
                # axis: a wall built as a base-axis box has its through-holes
                # running across the wall, perpendicular to the extrude.)
                sg_rh = rh.sketch_geometry or {}
                for cc in (sg_rh.get("centers") or [sg_rh.get("center")]):
                    if cc is None:
                        continue
                    c3 = np.asarray(cc, dtype=float)
                    for jax, (ju, jv), jpath, jspan in _join_lookup:
                        in_span = (
                            jspan[0] - 1.0 <= c3[jax] <= jspan[1] + 1.0
                        )
                        pt = np.array([[c3[ju], c3[jv]]])
                        if in_span and (
                            jpath.contains_points(pt, radius=0.5)[0]
                            or jpath.contains_points(pt, radius=-0.5)[0]
                        ):
                            return True
                return rh.type == FeatureType.BLIND_HOLE

            dropped = [rh for rh in holes if not _trustworthy(rh)]
            if dropped:
                logger.info(
                    "Dropped %d sideways through-hole feature(s) without "
                    "bottom caps (likely fitter artifacts)", len(dropped),
                )
                holes = [rh for rh in holes if _trustworthy(rh)]

            ring_inplane = [
                np.array([
                    float(h.sketch_geometry["center"][ax_u]),
                    float(h.sketch_geometry["center"][ax_v]),
                ])
                for h in detected_holes
            ]
            ring_radii = [float(h.sketch_geometry["radius"]) for h in detected_holes]

            # Strip recognition centers that a 2D ring supersedes.
            surviving: list = []
            for rh in holes:
                sg = rh.sketch_geometry or {}
                rdir = np.asarray(rh.direction, dtype=float)
                rdir = rdir / (np.linalg.norm(rdir) + 1e-15)
                if abs(float(rdir @ ext_axis)) < 0.9:
                    surviving.append(rh)  # different axis — rings can't see it
                    continue
                centers = sg.get("centers") or [sg.get("center")]
                kept_centers = []
                for cc in centers:
                    if cc is None:
                        continue
                    c2 = np.array([float(cc[ax_u]), float(cc[ax_v])])
                    matched = any(
                        float(np.linalg.norm(c2 - rc)) < (rr + float(sg.get("radius", 0.0)))
                        for rc, rr in zip(ring_inplane, ring_radii)
                    )
                    if not matched:
                        kept_centers.append(list(cc))
                if not kept_centers:
                    continue  # every center superseded by a ring
                if len(kept_centers) != len(centers):
                    sg = dict(sg)
                    sg["centers"] = kept_centers
                    sg["count"] = len(kept_centers)
                    sg["center"] = kept_centers[0]
                    rh = Feature(
                        type=rh.type,
                        face_indices=rh.face_indices,
                        sketch_geometry=sg,
                        direction=rh.direction,
                        depth=rh.depth,
                        operation=rh.operation,
                    )
                surviving.append(rh)
            holes = surviving

        if detected_holes:
            # Group identical holes (same radius + axis) into ONE feature
            # with a `centers` list, so the build emits a single sketch
            # with N circles + one cut — matching how a CAD operator draws
            # a hole pattern. Without this the 2D detector hands back one
            # feature per hole and the build makes N separate Sketch+Cut
            # steps. Grouping by rounded radius (0.1mm) and axis tolerates
            # fit noise across symmetric holes.
            grouped: dict[tuple, Feature] = {}
            for h in detected_holes:
                sg = h.sketch_geometry
                axis = np.asarray(h.direction, dtype=float)
                axis = axis / (np.linalg.norm(axis) + 1e-15)
                key = (
                    round(float(sg["radius"]), 1),
                    h.type,
                    round(float(axis[0]), 2),
                    round(float(axis[1]), 2),
                    round(float(axis[2]), 2),
                )
                if key not in grouped:
                    sg0 = dict(sg)
                    sg0["centers"] = [list(sg["center"])]
                    grouped[key] = Feature(
                        type=h.type,
                        face_indices=h.face_indices,
                        sketch_geometry=sg0,
                        direction=h.direction,
                        depth=h.depth,
                    )
                else:
                    grouped[key].sketch_geometry["centers"].append(list(sg["center"]))
            grouped_holes = list(grouped.values())
            for g in grouped_holes:
                g.sketch_geometry["count"] = len(g.sketch_geometry["centers"])
            logger.info(
                "2D detection added %d holes (%d pattern group(s)) alongside "
                "%d recognition holes",
                len(detected_holes), len(grouped_holes), len(holes),
            )
            holes = holes + grouped_holes
    except Exception as e:
        logger.debug("2D profile hole detection failed: %s", e)

    # Dedupe near-identical profiles: the engine's interior rings and the
    # 2D face-projection cutout detection can surface the SAME window,
    # and even-odd drawing of two coincident holes flips the region back
    # to solid (or corrupts the extrude outright).
    if best_profiles and len(best_profiles) > 1:
        from shapely.geometry import Polygon as _DedupPoly
        kept_profiles = [best_profiles[0]]
        for p in best_profiles[1:]:
            try:
                pp = _DedupPoly(p)
                if not pp.is_valid:
                    pp = pp.buffer(0)
                dup = False
                for q in kept_profiles:
                    qq = _DedupPoly(q)
                    if not qq.is_valid:
                        qq = qq.buffer(0)
                    union = pp.union(qq).area
                    # True duplicates only — concentric circles of
                    # different radius (a tube's bore is ~90% of its
                    # outer circle) share most of their area but are
                    # DISTINCT rings; deduping them turned the tube into
                    # a rod. Double-detections are near-exact.
                    if union > 1e-9 and pp.intersection(qq).area / union > 0.95:
                        dup = True
                        break
                if not dup:
                    kept_profiles.append(p)
            except Exception:
                kept_profiles.append(p)
        if len(kept_profiles) != len(best_profiles):
            logger.info(
                "Deduped %d near-identical profile(s)",
                len(best_profiles) - len(kept_profiles),
            )
        best_profiles = kept_profiles

    # Scan/AI: a slot whose rim is FILLETED double-traces — one ring at the
    # WALL, one at the fillet's outer rim ("the AI mesh's bumpy fillet makes
    # it think there's two levels"). Keep the WALL (smaller) contour; the rim
    # band is blend geometry, not a second cut. Only INNER profiles compete
    # (the outer profile at index 0 never participates, so a tube's bore
    # cannot dedupe against its outer ring).
    if len(best_profiles) > 2 and cleaned.metadata.get("scan_mode"):
        try:
            from shapely.geometry import Polygon as _DPoly
            inner = best_profiles[1:]
            polys = []
            for p in inner:
                try:
                    polys.append(_DPoly(p).buffer(0))
                except Exception:
                    polys.append(None)
            drop = set()
            for i in range(len(inner)):
                for j in range(len(inner)):
                    if i == j or polys[i] is None or polys[j] is None:
                        continue
                    pi, pj = polys[i], polys[j]
                    if pi.area <= pj.area:
                        continue          # only the LARGER twin is dropped
                    u = pi.union(pj).area
                    if (u > 1e-9
                            and pi.intersection(pj).area / u > 0.55
                            and pi.area < 2.5 * pj.area):
                        drop.add(i)
            if drop:
                logger.info("Dropped %d fillet-rim double-trace ring(s)",
                            len(drop))
                best_profiles = [best_profiles[0]] + [
                    p for k, p in enumerate(inner) if k not in drop]
        except Exception:
            logger.debug("rim double-trace dedupe failed", exc_info=True)

    # Stage F (scan/AI only): regularize the OUTER profile — snap its straight
    # sides to axis-aligned and its corners to exact 90 deg with tangent arcs.
    # A scan/AI base profile comes out wavy with non-90 corners (user's "bumpy
    # sketch"); clean CAD profiles are already exact, so this is gated off for
    # them. Inner cutouts are left as-is.
    if (best_profiles and cleaned.metadata.get("scan_mode")
            and len(best_profiles[0]) >= 6):
        try:
            from mesh2param.pipeline.regularize import regularize_profile_2d
            reg = regularize_profile_2d(best_profiles[0])
            if reg and len(reg) >= 3:
                logger.info("Regularized base profile: %d -> %d pts "
                            "(axis-snapped straights + tangent corners)",
                            len(best_profiles[0]), len(reg))
                best_profiles = [reg] + best_profiles[1:]
        except Exception:
            logger.debug("Profile regularization failed", exc_info=True)

    return best_profiles, extrude_start, holes, chamfers, fillets


def _clean_engine_polygon(
    pts: list[tuple[float, float]], tolerance: float = 0.5,
    min_area: float = 1.0,
) -> list[tuple[float, float]]:
    """Simplify + repair an engine-derived profile for CadQuery.

    Mesh-traced outlines can self-intersect after corner simplification
    (thin prongs at 0.5mm tolerance), and CadQuery's polyline().close()
    extrudes a self-intersecting wire into an EMPTY shape with no error.
    Round-trip through shapely: simplify, buffer(0)-repair, keep the
    largest part, return its exterior.

    `min_area` is the floor below which the polygon is treated as noise and
    dropped. Pass a small value (e.g. 0.1) for tiny mounting bores — an AI mesh
    makes them ~0.4 mm radius (~0.5 mm²), well under the default 1 mm² floor.
    """
    from shapely.geometry import Polygon as _CP

    if len(pts) < 3:
        return []
    try:
        p = _CP([(float(a), float(b)) for a, b in pts])
        if not p.is_valid:
            p = p.buffer(0)
        # Repair + light dedupe ONLY. Aggressive simplification flattened
        # the mesh-traced corner ARCS into sharp corners (297 -> 11 pts),
        # so _draw_profile_with_arcs had nothing to rebuild and the
        # stitched outline-fillet clusters found no edge within snap. And on a
        # tiny bore (a few mm² or less) simplify(0.05) collapses the circle to a
        # triangle or to nothing — skip it there and keep the sampled ring.
        if p.area >= 4.0:
            p = p.simplify(0.05, preserve_topology=True)
            if not p.is_valid:
                p = p.buffer(0)
        if p.is_empty:
            return []
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
        if p.area < min_area:
            return []
        return [(float(x), float(y)) for x, y in p.exterior.coords[:-1]]
    except Exception:
        return [(float(a), float(b)) for a, b in pts]


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
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.TopAbs import TopAbs_IN

        # If the hole center is VOID (not inside the solid), the hole
        # already exists — typically baked into a join/base profile as
        # an interior ring. A no-op cut is SUCCESS; nudging it carved a
        # phantom bore one diameter sideways into RackMount's side plate.
        try:
            classifier = BRepClass3d_SolidClassifier(
                wp.val().wrapped,
                gp_Pnt(float(center[0]), float(center[1]), float(center[2])),
                1e-6,
            )
            if classifier.State() != TopAbs_IN:
                logger.info(
                    "Hole at %s already void (baked into a profile); "
                    "skipping nudge", np.round(center, 1).tolist(),
                )
                return None
        except Exception:
            pass

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
        if shift_dist > radius:
            # The center is more than a radius away from any material.
            # That isn't a sliver-miss from profile rounding — the hole
            # simply has no solid where the mesh says it is (region cut
            # away, or a misdetected feature). Relocating it (Mesh3's big
            # bore was dragged 19.5mm) carves material the mesh never had
            # removed. Leave the no-op cut in place instead.
            logger.debug(
                "Not nudging hole r=%.2f: nearest solid is %.1fmm away",
                radius, shift_dist,
            )
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


def _detect_face_pockets(cleaned, extrude_dir):
    """Detect BLIND recessed pockets on the extrude-axis faces (the cam gear's
    circular face recess). A region SOLID at mid-thickness but VOID just inside a
    face is a face recess. Uses material(mid) - material(face), which INHERENTLY
    excludes through features (they're void at mid) — so it can't mis-fire on
    through slots/holes (the carriage-block lesson).

    Each recess interior is CLASSIFIED: solid at mid-thickness = a raised HUB
    (preserve, don't cut), void at mid = a THROUGH-HOLE (cut through, so it
    doesn't survive as a raised post). Depth is the MEDIAN floor over several
    sample points (robust to a noisy single point). Returns [{plane_name, axis,
    face_t, sign, depth, exterior_uv, hubs_uv, holes_uv, a_lo, a_hi}].
    """
    try:
        import trimesh
        from shapely.geometry import MultiPolygon, Point, Polygon as SPoly
        from mesh2param.pipeline.recognize import _section_material
    except Exception:
        return []
    try:
        axis = int(np.argmax(np.abs(extrude_dir)))
        a_lo = float(cleaned.bbox[0][axis])
        a_hi = float(cleaned.bbox[1][axis])
        span = a_hi - a_lo
        if span <= 1e-6:
            return []
        mesh = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces,
                               process=False)
        m_mid = _section_material(mesh, axis, 0.5 * (a_lo + a_hi))
        if m_mid is None or m_mid.is_empty:
            return []
        out = []
        for face_t, sign in ((a_hi, -1.0), (a_lo, 1.0)):
            try:
                m_face = _section_material(mesh, axis, face_t + sign * max(0.5, span * 0.05))
                if m_face is None:
                    continue
                recess = m_mid.difference(m_face.buffer(0))
                if recess.is_empty:
                    continue
                # difference may yield Polygon / MultiPolygon / GeometryCollection.
                geoms = list(getattr(recess, "geoms", [recess]))
                polys = [g for g in geoms if isinstance(g, SPoly) and not g.is_empty]
            except Exception:
                logger.debug("face-pocket: section/diff failed", exc_info=True)
                continue
            for p in polys:
                try:
                    # <1cm² shallow face recesses on scan/AI meshes are DENTS,
                    # not intent (the L-bracket's 92mm² blob — flat-floored and
                    # sharp-walled, so no downstream gate can tell it apart).
                    # The smallest real recess in the fleet is 149mm² (cam
                    # gear).
                    if p.area < max(100.0, 0.01 * m_mid.area):
                        continue
                    # SLIVER guard: on noisy AI meshes the face-vs-mid diff
                    # traces thin bands hugging the outer rim (jitter between
                    # the two sections), which then cut bogus edge "pockets".
                    # A real pocket is COMPACT: eroding ~1mm keeps most of its
                    # area. A rim band collapses to crumbs (a plain is_empty
                    # test misses the crumbs, so require the area fraction).
                    _ero = p.buffer(-max(0.8, 0.008 * span))
                    if _ero.is_empty or _ero.area < 0.25 * p.area:
                        continue
                    ps = p.simplify(max(0.2, 0.003 * span))
                    if not isinstance(ps, SPoly) or ps.is_empty:
                        continue
                    ext = [(float(x), float(y)) for x, y in zip(*ps.exterior.xy)]
                    if len(ext) < 4:
                        continue

                    # Floor depth over a GRID of sample points in the recess,
                    # resolved with ONE shared section sweep. The floor SPREAD
                    # is the phantom discriminator: a designed pocket has a
                    # FLAT floor (AI1's panel recess: every sample 2.1mm); a
                    # perimeter EDGE-FILLET band has a SLOPED floor (rounded
                    # edges make the near-face section grow gradually — the
                    # L-bracket's floors ran ~0.9→3.9mm across the band and
                    # cut 4 phantom pockets).
                    samples = [ps.representative_point()]
                    mnx, mny, mxx, mxy = ps.bounds
                    for fx in (0.1, 0.3, 0.5, 0.7, 0.9):
                        for fy in (0.1, 0.3, 0.5, 0.7, 0.9):
                            q = Point(mnx + (mxx - mnx) * fx,
                                      mny + (mxy - mny) * fy)
                            if ps.contains(q):
                                samples.append(q)
                    pending = dict(enumerate(samples))
                    depths = []
                    for dd in np.linspace(max(0.5, span * 0.05), span * 0.85, 24):
                        if not pending:
                            break
                        g = _section_material(mesh, axis, face_t + sign * dd)
                        if g is None:
                            continue
                        gb = g.buffer(-0.3)
                        hit = [i for i, q in pending.items() if gb.contains(q)]
                        for i in hit:
                            depths.append(float(dd))
                            del pending[i]
                    # <4 resolved floors = a sliver the grid barely lands in;
                    # compact real pockets get many hits from their own-bounds
                    # grid. Not enough evidence to cut.
                    if len(depths) < 4:
                        continue
                    depth = float(np.median(depths))
                    d_lo, d_hi = np.percentile(depths, [15.0, 85.0])
                    if (d_hi - d_lo) > max(0.8, 0.45 * depth):
                        continue                # sloped floor = fillet band
                    if depth < max(0.8, 0.03 * span):
                        continue                          # too shallow / through
                    # PERSISTENCE: a designed pocket has SHARP walls — its
                    # footprint at 80% of its depth still matches the near-face
                    # footprint (AI1's panel recess: 0.9). A wide ROUNDED
                    # transition shrinks steadily as the reference deepens
                    # (the L-bracket's arm-to-base blend: 0.66 — it cut 4
                    # phantom pockets). One extra section, per candidate.
                    try:
                        ref2 = max(max(0.5, span * 0.05), 0.8 * depth)
                        g2 = _section_material(mesh, axis,
                                               face_t + sign * ref2)
                        if g2 is not None:
                            p2 = m_mid.difference(g2.buffer(0))
                            keep = (float(ps.intersection(p2).area)
                                    / max(float(ps.area), 1e-9))
                            if keep < 0.8:
                                continue
                    except Exception:
                        logger.debug("face-pocket: persistence test failed",
                                     exc_info=True)

                    # CUT SHAPE = recess MINUS material near the floor. Any region
                    # still SOLID near the floor (a raised hub/boss — including one
                    # present on a single face) is excluded from the cut and thus
                    # PRESERVED. This is general (no per-feature guessing) and
                    # handles a variable-depth recess. Through-holes (void at mid)
                    # are additionally cut all the way through so they don't
                    # survive as raised posts.
                    mfloor = _section_material(mesh, axis, face_t + sign * 0.85 * depth)
                    cut_shape = (ps.difference(mfloor.buffer(0))
                                 if mfloor is not None else ps)
                    for cs in (g2 for g2 in getattr(cut_shape, "geoms", [cut_shape])
                               if isinstance(g2, SPoly) and not g2.is_empty):
                        if cs.area < max(20.0, 0.01 * m_mid.area):
                            continue
                        cext = [(float(x), float(y)) for x, y in zip(*cs.exterior.xy)]
                        if len(cext) < 4:
                            continue
                        interiors, throughs = [], []
                        for ring in cs.interiors:
                            iv = [(float(x), float(y)) for x, y in zip(*ring.xy)]
                            rp = SPoly(ring)
                            if len(iv) < 4 or abs(rp.area) <= 2.0:
                                continue
                            interiors.append(iv)        # preserved in the pocket cut
                            try:
                                frac = m_mid.intersection(rp).area / (rp.area + 1e-9)
                            except Exception:
                                frac = 0.0
                            if frac < 0.5:
                                throughs.append(iv)     # void at mid -> cut through
                        out.append({
                            "plane_name": {0: "YZ", 1: "XZ", 2: "XY"}[axis],
                            "axis": axis, "face_t": face_t, "sign": sign, "depth": depth,
                            "exterior_uv": cext, "interiors_uv": interiors,
                            "holes_uv": throughs, "a_lo": a_lo, "a_hi": a_hi,
                        })
                except Exception:
                    logger.debug("face-pocket: polygon failed", exc_info=True)
                    continue
        return out
    except Exception:
        logger.debug("face-pocket detection failed", exc_info=True)
        return []


def _detect_symmetric_thinning(cleaned, extrude_dir):
    """Detect a profile region that is THINNER than the base along the extrude
    axis, set back symmetrically from BOTH faces — the L-bracket's diagonal
    arm (17.2mm base, arm set back ~2.1mm per side). Design intent is a
    second, thinner extrude of that region, NOT face pockets and NOT a
    "Refined Model" chop after the fact.

    Evidence = the same recess footprint on BOTH opposite faces (mirror IoU
    ≥ 0.65) with similar measured setbacks. That mirror match replaces the
    sharp-wall persistence gate `_detect_face_pockets` uses — a rounded
    arm-to-base transition fails persistence by design but is exactly what
    this detector is for.

    Returns {axis, region_uv, d_lo, d_hi} or None. region_uv is the convex
    hull of the matched footprints (the seam it cuts lies inside material).
    """
    try:
        import trimesh
        from shapely.geometry import Polygon as SPoly, Point
        from mesh2param.pipeline.recognize import _section_material
    except Exception:
        return None
    try:
        axis = int(np.argmax(np.abs(extrude_dir)))
        a_lo = float(cleaned.bbox[0][axis])
        a_hi = float(cleaned.bbox[1][axis])
        span = a_hi - a_lo
        if span <= 1e-6:
            return None
        mesh = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces,
                               process=False)
        m_mid = _section_material(mesh, axis, 0.5 * (a_lo + a_hi))
        if m_mid is None or m_mid.is_empty:
            return None
        found = {}
        for face_t, sign in ((a_hi, -1.0), (a_lo, 1.0)):
            m_face = _section_material(mesh, axis,
                                       face_t + sign * max(0.5, span * 0.05))
            if m_face is None:
                continue
            recess = m_mid.difference(m_face.buffer(0))
            polys = [g for g in getattr(recess, "geoms", [recess])
                     if isinstance(g, SPoly) and not g.is_empty]
            if not polys:
                continue
            p = max(polys, key=lambda g: g.area)
            # A thinned MEMBER is a substantial partial region: not a sliver,
            # not the whole face (a ~full-face mirror pair is a panel recess
            # on both sides — AI1 — which stays a pocket pair).
            if not (0.15 * m_mid.area <= p.area <= 0.85 * m_mid.area):
                continue
            _ero = p.buffer(-max(0.8, 0.008 * span))
            if _ero.is_empty or _ero.area < 0.25 * p.area:
                continue
            # Setback = floor of the region's INTERIOR. Points near the
            # rounded arm-to-base blend resolve early and bias the setback
            # shallow (2.1 measured vs ~2.9 true on the L-bracket, leaving
            # +13% volume) — so sample only well inside the region, and take
            # a high percentile: the member's face is where the LAST interior
            # points resolve, not the middle of the blend.
            ps = p.simplify(max(0.2, 0.003 * span))
            if not isinstance(ps, SPoly) or ps.is_empty:
                continue
            ps_in = ps.buffer(-max(1.2, 0.03 * float(np.sqrt(ps.area))))
            if ps_in.is_empty:
                ps_in = ps
            if ps_in.geom_type != "Polygon":
                ps_in = max(ps_in.geoms, key=lambda g: g.area)
            samples = [ps_in.representative_point()]
            mnx, mny, mxx, mxy = ps_in.bounds
            for fx in np.linspace(0.08, 0.92, 7):
                for fy in np.linspace(0.08, 0.92, 7):
                    q = Point(mnx + (mxx - mnx) * fx, mny + (mxy - mny) * fy)
                    if ps_in.contains(q):
                        samples.append(q)
            pending = dict(enumerate(samples))
            depths = []
            for dd in np.linspace(max(0.5, span * 0.05), 0.45 * span, 16):
                if not pending:
                    break
                g = _section_material(mesh, axis, face_t + sign * dd)
                if g is None:
                    continue
                gb = g.buffer(-0.3)
                for i in [i for i, q in pending.items() if gb.contains(q)]:
                    depths.append(float(dd))
                    del pending[i]
            if len(depths) < 4:
                continue
            d = float(np.percentile(depths, 75.0))
            if not (max(0.8, 0.03 * span) <= d <= 0.25 * span):
                continue
            found[face_t] = (p, d)
        if len(found) != 2:
            return None
        (p_hi, d_hi) = found[a_hi]
        (p_lo, d_lo) = found[a_lo]
        iou = (p_hi.intersection(p_lo).area
               / max(p_hi.union(p_lo).area, 1e-9))
        if iou < 0.65:
            return None
        if abs(d_hi - d_lo) > max(1.0, 0.5 * min(d_hi, d_lo)):
            return None
        # The ACTUAL matched footprint, smoothed — NOT its convex hull: the
        # hull of the L-bracket's diagonal band overhangs into the
        # full-thickness bar and the carve bit a visible notch out of it.
        hull = p_hi.intersection(p_lo)
        hull = hull.buffer(1.5, join_style=1).buffer(-1.5, join_style=1)
        if hull.geom_type == "MultiPolygon":
            hull = max(hull.geoms, key=lambda g: g.area)
        hull = hull.simplify(0.6)
        if hull.is_empty or hull.geom_type != "Polygon":
            return None
        region = [(float(x), float(y)) for x, y in hull.exterior.coords]
        if len(region) < 4:
            return None
        logger.info(
            "Symmetric thinning: region area %.0f (%.0f%% of mid), setbacks "
            "%.2f/%.2f over span %.1f (mirror IoU %.2f)",
            hull.area, 100.0 * hull.area / m_mid.area, d_lo, d_hi, span, iou)
        return {"axis": axis, "region_uv": region,
                "d_lo": float(d_lo), "d_hi": float(d_hi)}
    except Exception:
        logger.debug("symmetric-thinning detection failed", exc_info=True)
        return None


def _void_intent_outline(coords):
    """Regularize a void outline (window/slot cutout) to its DESIGN-INTENT
    shape: circle, capsule (stadium), or a smoothed outline. Used for the
    cutout rings baked into the base sketch — on an AI mesh those trace the
    slot fillet's jagged outer rim (the carriage block's 'jagged, slightly
    wider sketch' around each smooth oval). Returns a point list, or None to
    keep the input as-is."""
    from shapely.geometry import Polygon as SPoly
    try:
        p = SPoly(coords)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.geom_type != "Polygon" or p.area < 4.0:
            return None
        filled = p.buffer(2.0, join_style=1).buffer(-2.0, join_style=1)
        if filled.is_empty or filled.geom_type != "Polygon":
            filled = p
        circ = 4.0 * np.pi * filled.area / (filled.length ** 2 + 1e-9)
        bx = filled.bounds
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        aspect = max(w, h) / max(min(w, h), 1e-9)
        if circ > 0.82 and aspect < 1.3:
            cen = filled.centroid
            r = float(np.sqrt(filled.area / np.pi))
            th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            return [(float(cen.x + r * np.cos(t)),
                     float(cen.y + r * np.sin(t))) for t in th]
        if aspect >= 1.4:
            bpts = np.asarray(filled.exterior.coords, dtype=float)[:-1]
            cen = np.asarray(filled.centroid.coords[0], dtype=float)
            d0 = bpts - cen
            _, _, vt = np.linalg.svd(d0, full_matrices=False)
            u_maj, u_min = vt[0], vt[1]
            # PARALLEL intent: a slot within ~12 deg of a part axis was
            # designed axis-aligned — snap it (also makes sibling slots
            # parallel to each other and to the outline).
            for ax_v in ((1.0, 0.0), (0.0, 1.0)):
                axv = np.asarray(ax_v)
                if abs(float(u_maj @ axv)) > np.cos(np.radians(12.0)):
                    u_maj = axv if (u_maj @ axv) > 0 else -axv
                    u_min = np.asarray([-u_maj[1], u_maj[0]])
                    break
            half_w = float(np.percentile(np.abs(d0 @ u_min), 95))
            half_l = float(np.percentile(np.abs(d0 @ u_maj), 98))
            r_cap = half_w
            straight = max(half_l - r_cap, 0.1)
            th = np.linspace(-np.pi / 2, np.pi / 2, 20)
            end1 = (cen + straight * u_maj
                    + np.outer(np.cos(th), r_cap * u_maj)
                    + np.outer(np.sin(th), r_cap * u_min))
            end2 = (cen - straight * u_maj
                    - np.outer(np.cos(th), r_cap * u_maj)
                    - np.outer(np.sin(th), r_cap * u_min))
            cap = SPoly([(float(x), float(y))
                         for x, y in np.vstack([end1, end2])])
            if cap.is_valid:
                iou = (cap.intersection(filled).area
                       / max(cap.union(filled).area, 1e-9))
                if iou >= 0.8:
                    return [(float(x), float(y))
                            for x, y in cap.exterior.coords[:-1]]
        sm = filled.simplify(0.4)
        if sm.is_empty or sm.geom_type != "Polygon":
            return None
        return [(float(x), float(y)) for x, y in sm.exterior.coords[:-1]]
    except Exception:
        return None


def _detect_face_ring_recesses(cleaned, extrude_dir,
                               max_frac: float = 0.15,
                               min_area: float = 4.0):
    """Detect BLIND recesses on the extrude-axis faces of a HOLLOW part — the
    enclosure lid's oval top recess and its small blind holes.

    `_detect_face_pockets` references material at the MID-plane, which is VOID
    inside a hollow enclosure — every top-face feature there reads as "through"
    and is skipped by design. Here a feature is instead an INTERIOR RING of the
    face section: a void fully ENCLOSED by material just inside the face. Sweep
    deeper to find its floor: material covering the ring = a BLIND recess (cut
    face->floor); no floor within the span = a through window (skip — handled
    elsewhere). Two insets are sampled so shallow recesses aren't missed.

    Guards: `min_area` admits small blind holes but not mesh noise (the floor
    requirement is the real filter); `max_frac` of the face section EXCLUDES
    the giant open cavity — the volumetric refine reconstructs that better than
    a prismatic carve (A/B'd). Downstream the cut is kept only if it actually
    removes volume, so parts whose recognition already cut these features are
    untouched. Returns [{plane_name, axis, face_t, sign, depth, exterior_uv}].
    """
    try:
        import trimesh
        from shapely.geometry import Polygon as SPoly
        from mesh2param.pipeline.recognize import _section_material
    except Exception:
        return []
    def _enclosing_ring_area(g, pt):
        """Area of the smallest interior-ring void of `g` containing pt, or
        None if pt is not enclosed by material."""
        best = None
        for poly in getattr(g, "geoms", [g]):
            for ring in getattr(poly, "interiors", []):
                try:
                    q = SPoly(ring)
                    if q.contains(pt) and (best is None or q.area < best):
                        best = float(q.area)
                except Exception:
                    continue
        return best

    try:
        mesh = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces,
                               process=False)
        out = []
        # ALL axes, not just the extrude axis: the enclosure's 3 vent slots
        # live on a SIDE face.
        axes_faces = []
        for axis in (0, 1, 2):
            a_lo = float(cleaned.bbox[0][axis])
            a_hi = float(cleaned.bbox[1][axis])
            if a_hi - a_lo > 1e-6:
                axes_faces.append((axis, a_lo, a_hi))
        for axis, a_lo, a_hi in axes_faces:
            span = a_hi - a_lo
            for face_t, sign in ((a_hi, -1.0), (a_lo, 1.0)):
                # PASS 1 — collect candidate rings from both insets (cheap
                # geometric gates only), dedupe, keep the LARGEST few. The
                # expensive floor sweep runs only on survivors: a gear's 46
                # tooth gaps or a dirty mesh's dozens of sub-mm noise voids
                # otherwise explode the build (the cam-gear 600s timeout).
                cands: list = []      # (rp, inset, sect_area)
                for inset in (max(0.5, 0.02 * span), max(0.9, 0.05 * span)):
                    g = _section_material(mesh, axis, face_t + sign * inset)
                    if g is None or g.is_empty:
                        continue
                    sect_area = float(g.area)
                    for poly in getattr(g, "geoms", [g]):
                        for ring in getattr(poly, "interiors", []):
                            rp = SPoly(ring)
                            if not rp.is_valid:
                                rp = rp.buffer(0)
                            if (not isinstance(rp, SPoly) or rp.is_empty
                                    or rp.area < min_area
                                    or rp.area > max_frac * (sect_area + rp.area)):
                                continue
                            if any(rp.intersection(c[0]).area
                                   > 0.5 * min(rp.area, c[0].area)
                                   for c in cands):
                                continue    # same feature seen at other inset
                            cands.append((rp, float(inset), sect_area))
                if len(cands) > 8:
                    logger.info(
                        "Ring recesses: %d candidates on a face, keeping the "
                        "8 largest (rest = noise/teeth)", len(cands))
                    cands = sorted(cands, key=lambda c: -c[0].area)[:8]
                for rp, inset, sect_area in cands:
                    if True:
                        # Sweep deeper: material covering the ring = BLIND
                        # recess floor. The void merging into a MUCH larger
                        # enclosure (the cavity) or opening out of the part =
                        # a THROUGH slot/window through this wall — cut just
                        # past the exit. Never resolving = skip.
                        pt = rp.representative_point()
                        depth = None
                        through = False
                        for dd in np.linspace(inset + 0.4, 0.9 * span, 30):
                            gg = _section_material(mesh, axis, face_t + sign * dd)
                            if gg is None or gg.is_empty:
                                depth = float(dd)
                                through = True
                                break
                            if gg.buffer(-0.2).contains(pt):
                                depth = float(dd)
                                break
                            enc = _enclosing_ring_area(gg, pt)
                            if enc is None or enc > 6.0 * rp.area:
                                depth = float(dd) + 1.0
                                through = True
                                break
                        if depth is None or depth < max(0.8, inset + 0.3):
                            continue      # unresolved / too shallow
                        # SKETCH-INTENT regularization. A round ring is a HOLE:
                        # carry a fitted circle (centroid + area-equivalent
                        # radius) so the cut is a true CAD circle, not a jagged
                        # mesh trace. A non-round ring (the oval recess) gets a
                        # morphological-closing smooth so its outline reads as a
                        # designed shape rather than mesh noise.
                        # SKETCH-INTENT regularization. Raised bumps inside a
                        # recess bridge to its rim, so the traced void is a
                        # CRESCENT rather than the designed outline — a strong
                        # morphological closing (buffer +3/-3) re-bridges it
                        # into the designed shape before classification.
                        filled = rp.buffer(3.0, join_style=1).buffer(
                            -3.0, join_style=1)
                        if filled.is_empty or filled.geom_type != "Polygon":
                            filled = rp
                        circ = (4.0 * np.pi * filled.area
                                / (filled.length ** 2 + 1e-9))
                        _bx = filled.bounds
                        _w, _h = _bx[2] - _bx[0], _bx[3] - _bx[1]
                        aspect = max(_w, _h) / max(min(_w, _h), 1e-9)
                        circle = None
                        ellipse = None
                        stadium = None
                        if circ > 0.82 and aspect < 1.3:
                            # A true circle: high isoperimetric ratio + ~square
                            # bbox.
                            cen = filled.centroid
                            circle = {"cu": float(cen.x), "cv": float(cen.y),
                                      "r": float(np.sqrt(filled.area / np.pi))}
                        elif aspect >= 1.4:
                            # SLOT intent: an elongated void is a capsule
                            # (stadium) — two semicircle ends + straight
                            # sides — the designed shape of the carriage
                            # block's long ovals (a raw mesh trace leaves
                            # jagged rims).
                            try:
                                bpts = np.asarray(
                                    filled.exterior.coords, dtype=float)[:-1]
                                cen = np.asarray(
                                    filled.centroid.coords[0], dtype=float)
                                d0 = bpts - cen
                                _, _, vt = np.linalg.svd(d0,
                                                         full_matrices=False)
                                u_maj = vt[0]
                                u_min = vt[1]
                                for _axv in ((1.0, 0.0), (0.0, 1.0)):
                                    _axv = np.asarray(_axv)
                                    if abs(float(u_maj @ _axv)) > np.cos(
                                            np.radians(12.0)):
                                        u_maj = (_axv if (u_maj @ _axv) > 0
                                                 else -_axv)
                                        u_min = np.asarray(
                                            [-u_maj[1], u_maj[0]])
                                        break
                                half_w = float(np.percentile(
                                    np.abs(d0 @ u_min), 95))
                                half_l = float(np.percentile(
                                    np.abs(d0 @ u_maj), 98))
                                r_cap = half_w
                                straight = max(half_l - r_cap, 0.1)
                                th = np.linspace(-np.pi / 2, np.pi / 2, 20)
                                end1 = (cen + straight * u_maj
                                        + np.outer(np.cos(th), r_cap * u_maj)
                                        + np.outer(np.sin(th), r_cap * u_min))
                                end2 = (cen - straight * u_maj
                                        - np.outer(np.cos(th), r_cap * u_maj)
                                        - np.outer(np.sin(th), r_cap * u_min))
                                cap_pts = np.vstack([end1, end2])
                                capsule = SPoly([(float(x), float(y))
                                                 for x, y in cap_pts])
                                if capsule.is_valid:
                                    iou = (capsule.intersection(filled).area
                                           / max(capsule.union(filled).area,
                                                 1e-9))
                                    if iou >= 0.82:
                                        stadium = capsule
                            except Exception:
                                stadium = None
                        if stadium is None and circle is None:
                            # OVAL intent: fit an ellipse (principal axes of
                            # the outline) and keep it if it explains the
                            # shape; otherwise fall back to the smoothed
                            # outline.
                            try:
                                bpts = np.asarray(
                                    filled.exterior.coords, dtype=float)[:-1]
                                cen = np.asarray(
                                    filled.centroid.coords[0], dtype=float)
                                d0 = bpts - cen
                                cov = d0.T @ d0 / max(len(d0), 1)
                                evals, evecs = np.linalg.eigh(cov)
                                ax1 = evecs[:, 1]     # major direction
                                proj1 = d0 @ evecs[:, 1]
                                proj0 = d0 @ evecs[:, 0]
                                ra = float(np.percentile(np.abs(proj1), 98))
                                rb = float(np.percentile(np.abs(proj0), 98))
                                ang = float(np.degrees(
                                    np.arctan2(ax1[1], ax1[0])))
                                th = np.linspace(0, 2 * np.pi, 64)
                                epts = (cen[None, :]
                                        + np.outer(np.cos(th), ra * evecs[:, 1])
                                        + np.outer(np.sin(th), rb * evecs[:, 0]))
                                ep = SPoly([(float(x), float(y))
                                            for x, y in epts])
                                iou = (ep.intersection(filled).area
                                       / max(ep.union(filled).area, 1e-9))
                                if iou >= 0.8:
                                    ellipse = {"cu": float(cen[0]),
                                               "cv": float(cen[1]),
                                               "ra": ra, "rb": rb, "ang": ang}
                            except Exception:
                                ellipse = None
                        if stadium is not None:
                            smooth = stadium
                        else:
                            smooth = filled.simplify(max(0.25, 0.003 * span))
                        if smooth.is_empty or smooth.geom_type != "Polygon":
                            smooth = rp.simplify(max(0.15, 0.002 * span))
                        # THROUGH windows: dilate slightly so the clean cut
                        # swallows the jagged baked window walls the fallback
                        # base carries (otherwise both rims show).
                        if through and circle is None:
                            _dl = smooth.buffer(0.25, join_style=1)
                            if (not _dl.is_empty
                                    and _dl.geom_type == "Polygon"):
                                smooth = _dl
                        ext = [(float(x), float(y)) for x, y in
                               smooth.exterior.coords]
                        if len(ext) < 4:
                            continue
                        # SCALE gates: features must be meaningful at the
                        # part's scale. The ⌀300 motor plate grew tiny
                        # "slot" cuts (~250mm², 0.35% of the face) and
                        # r~1.6 hole circles from AI-mesh dimples.
                        _bbext = (np.asarray(cleaned.bbox[1])
                                  - np.asarray(cleaned.bbox[0]))
                        _maxdim = float(np.max(_bbext))
                        if circle is not None:
                            if circle["r"] < 0.006 * _maxdim:
                                logger.info(
                                    "Ring circle r=%.2f skipped (< 0.6%% "
                                    "of the part)", circle["r"])
                                continue
                        elif float(filled.area) < 0.008 * sect_area:
                            logger.info(
                                "Ring recess area %.0f skipped (< 0.8%% "
                                "of the face section %.0f)",
                                filled.area, sect_area)
                            continue
                        elif through and (float(filled.area)
                                          < 0.018 * sect_area):
                            # A THROUGH window is a major feature (real
                            # vents/ovals measure >=2.6% of their face); a
                            # through "slot" under ~2% is an AI-mesh dent
                            # punched through (the motor plate's six
                            # ~250mm² slots on the boss face).
                            logger.info(
                                "Through ring recess area %.0f skipped "
                                "(< 1.8%% of the face section %.0f)",
                                filled.area, sect_area)
                            continue
                        out.append({
                            "plane_name": {0: "YZ", 1: "XZ", 2: "XY"}[axis],
                            "axis": axis, "face_t": face_t, "sign": sign,
                            "depth": depth, "exterior_uv": ext,
                            "area": float(filled.area), "circle": circle,
                            "ellipse": ellipse, "through": through,
                        })
        # HOLE-PATTERN intent: holes on the same face that are the same feature
        # repeated (a bolt pattern) get ONE shared diameter and depth — snap
        # every circle within 25% of the group median to the median.
        circles = [r for r in out if r.get("circle")]
        if len(circles) >= 2:
            med_r = float(np.median([r["circle"]["r"] for r in circles]))
            med_d = float(np.median([r["depth"] for r in circles]))
            for r in circles:
                if abs(r["circle"]["r"] - med_r) <= 0.25 * med_r:
                    r["circle"]["r"] = med_r
                    if abs(r["depth"] - med_d) <= 0.25 * med_d:
                        r["depth"] = med_d
        return out
    except Exception:
        logger.debug("face-ring recess detection failed", exc_info=True)
        return []


def _detect_standing_bosses(cleaned, extrude_dir,
                            max_height: float = 4.0,
                            max_frac: float = 0.25):
    """Detect SHORT STANDING BOSSES on an extrude-axis face — the enclosure
    lid's four feet: small pads standing proud of the main face, so the base
    slab (which spans the full bbox) needs the layer between the pads CUT away.

    Signature: the section right AT the face contains only the pads (small
    total area), and a few mm deeper the full face appears (area jumps).
    Returns [{plane_name, axis, face_t, sign, height, keeps: [{cu,cv,r}],
    silhouette_uv}] — the cut is silhouette MINUS the kept pad circles, from
    the face to `height`. Round pads snap to one shared radius (pattern
    intent). Downstream keeps the cut only if it removes volume, so parts
    whose base already stops at the true face are untouched.
    """
    try:
        import trimesh
        from shapely.geometry import Polygon as SPoly
        from shapely.ops import unary_union
        from mesh2param.pipeline.recognize import _section_material
    except Exception:
        return []
    try:
        axis = int(np.argmax(np.abs(extrude_dir)))
        a_lo = float(cleaned.bbox[0][axis])
        a_hi = float(cleaned.bbox[1][axis])
        span = a_hi - a_lo
        if span <= 1e-6:
            return []
        mesh = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces,
                               process=False)
        out = []
        for face_t, sign in ((a_lo, 1.0), (a_hi, -1.0)):
            # Scan levels from the face inward, classifying each as a
            # PADS-level (1..8 small components) or the MAIN face (big area).
            # Feet = a run of pads-levels ending where the main face appears.
            # Pads are MEASURED at the deepest pads-only level: right at the
            # face the discs are jagged mesh nubs (roundness ~0.4), a little
            # deeper they read as clean circles.
            levels = np.arange(0.25, max_height, 0.35)
            pads_at: list = []          # (offset, [pad polys])
            height = None
            main = None
            first_total = None
            for off in levels:
                g = _section_material(mesh, axis, face_t + sign * float(off))
                if g is None or g.is_empty:
                    continue
                comps = [p for p in getattr(g, "geoms", [g])
                         if isinstance(p, SPoly) and p.area > 1.5]
                total = sum(p.area for p in comps)
                if first_total is None:
                    first_total = total
                # Main-face jump: ABRUPT growth (a step, not a fillet/draft
                # ramp) that lands well above the face level. 4x catches
                # small feet; 1.8x-of-first + 1.5x-abrupt catches a large
                # raised PANEL (~40% of the face) without firing on a
                # drafted face's gradual growth.
                _abrupt = (pads_at and total > 1.35 * pads_at[-1][2])
                if (pads_at and len(comps) <= 2
                        and (total > max(4.0 * pads_at[-1][2], 100.0)
                             or (_abrupt and total > 1.8 * first_total
                                 and total > 100.0))):
                    height = float(off) - 0.15      # main face appeared
                    main = g
                    break
                if 1 <= len(comps) <= 8:
                    pads_at.append((float(off), comps, total))
                else:
                    break
            if height is None or main is None or not pads_at:
                continue
            _, pads, pad_area = pads_at[-1]      # deepest pads-only level
            # PROMINENCE gates: real designed pads carry real area AND real
            # height (AI4's feet: ~7% of the face, 1.15mm). The bezel frame's
            # distorted corner nubs (2.1%, 0.8mm) are mesh noise — modelling
            # them as bosses adds junk the designer never drew.
            if height < 1.0 or pad_area < 0.03 * float(main.area):
                continue
            keeps = []
            keep_outline = None
            if pad_area > max_frac * float(main.area):
                # ONE LARGE RAISED PANEL (the lid's rounded-rect top plate,
                # ~40% of the face): keep its REGULARIZED outline instead of
                # circles. Scan-mode only, and the pad must be a compact
                # designed shape (high solidity), so a drafted clean-CAD face
                # can't trip it.
                scan = bool((getattr(cleaned, "metadata", None) or {})
                            .get("scan_mode"))
                if not (scan and len(pads) == 1):
                    continue
                p = pads[0]
                # Smooth FIRST (morphological closing bridges the bumps biting
                # into the outline), then judge solidity on the designed shape.
                sm = p.buffer(2.0, join_style=1).buffer(-2.0, join_style=1)
                if sm.is_empty or sm.geom_type != "Polygon":
                    sm = p
                sold = sm.area / max(sm.convex_hull.area, 1e-9)
                if not (0.08 * main.area <= sm.area <= 0.75 * main.area
                        and sold >= 0.75):
                    continue
                # DESIGN-INTENT snap: the panel is a designed shape, not a
                # wavy mesh trace. Try ROUNDED rectangles across a radius
                # sweep — the lid's panel has big corner rounds (~8mm), so
                # its sharp-box IoU is only ~0.82 and a sharp-rect gate
                # never fires; the rounded version fits.
                try:
                    from shapely.geometry import box as _PBox
                    _pb = _PBox(*sm.bounds)
                    _bx = _pb.bounds
                    _mins = min(_bx[2] - _bx[0], _bx[3] - _bx[1])
                    _best_iou = (sm.intersection(_pb).area
                                 / max(sm.union(_pb).area, 1e-9))
                    _best_shape = _pb
                    for _rp in (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0):
                        if _rp > 0.45 * _mins:
                            break
                        _cand = _pb.buffer(-_rp, join_style=1).buffer(
                            _rp, join_style=1)
                        if _cand.is_empty or _cand.geom_type != "Polygon":
                            continue
                        _iou = (sm.intersection(_cand).area
                                / max(sm.union(_cand).area, 1e-9))
                        if _iou > _best_iou:
                            _best_iou, _best_shape = _iou, _cand
                    if _best_iou >= 0.90:
                        sm = _best_shape
                    else:
                        # Not a rounded rect (the lid's panel: 0.87 at
                        # best) — smooth the real trace instead (Chaikin
                        # corner-cutting) so the sketch reads as a
                        # designed smooth curve, not a bumpy polyline.
                        _cpts = np.asarray(sm.exterior.coords[:-1], float)
                        for _ in range(3):
                            _nxt = []
                            _n2 = len(_cpts)
                            for _i2 in range(_n2):
                                _a2 = _cpts[_i2]
                                _b2 = _cpts[(_i2 + 1) % _n2]
                                _nxt.append(0.75 * _a2 + 0.25 * _b2)
                                _nxt.append(0.25 * _a2 + 0.75 * _b2)
                            _cpts = np.asarray(_nxt)
                        from shapely.geometry import Polygon as _ChPoly
                        _smc = _ChPoly(_cpts).buffer(0)
                        if (not _smc.is_empty
                                and _smc.geom_type == "Polygon"):
                            sm = _smc
                except Exception:
                    pass
                sm = sm.simplify(0.4)
                keep_outline = [(float(x), float(y))
                                for x, y in sm.exterior.coords]
            else:
                # Kept pads as circles — the feet pattern is specifically
                # ROUND pads. A non-round pad (e.g. a thin plate edge landing
                # on the bbox face, RackMount) can't be covered by a circle
                # and cutting around a wrong circle would eat real material:
                # skip the face.
                ok = True
                for p in pads:
                    circ = 4.0 * np.pi * p.area / (p.length ** 2 + 1e-9)
                    if circ < 0.6:
                        ok = False
                        break
                    cen = p.centroid
                    r = float(np.sqrt(p.area / np.pi))
                    keeps.append({"cu": float(cen.x), "cv": float(cen.y),
                                  "r": r})
                if not ok:
                    continue
                if len(keeps) >= 2:
                    med = float(np.median([k["r"] for k in keeps]))
                    for k in keeps:
                        if abs(k["r"] - med) <= 0.3 * med:
                            k["r"] = med
            # silhouette of the full face (outer boundary, holes filled)
            outers = [SPoly(g.exterior) for g in getattr(main, "geoms", [main])
                      if isinstance(g, SPoly)]
            sil = unary_union(outers)
            if sil.geom_type != "Polygon":
                sil = max(getattr(sil, "geoms", [sil]), key=lambda g: g.area)
            sil = sil.buffer(1.0).simplify(0.3)   # margin so the rim clears
            silhouette = [(float(x), float(y)) for x, y in sil.exterior.coords]
            out.append({
                "plane_name": {0: "YZ", 1: "XZ", 2: "XY"}[axis],
                "axis": axis, "face_t": face_t, "sign": sign,
                "height": height, "keeps": keeps,
                "keep_outline": keep_outline, "silhouette_uv": silhouette,
                # The cut is only meaningful when the between-pads layer is
                # actually FILLED by the (over-extruded) base slab — then it
                # removes ~this much. If the solid already models the pads as
                # separate bosses (AI1's screws), the cut removes almost
                # nothing but can still clip their domes — the cutter compares
                # against this to skip that case.
                "expected_cut": float((main.area - pad_area) * height),
            })
        return out
    except Exception:
        logger.debug("standing-boss detection failed", exc_info=True)
        return []


def _detect_tangent_holes(cleaned, extrude_dir):
    """Detect circular through-holes whose axis is PERPENDICULAR to the extrude
    axis (the carriage block's 'tangent' holes), straight from the MESH — robust
    to the broken fit that mis-locates them as short/blind cylinders. Each hole
    carries its radius LEVELS along the axis, so a STEPPED tangent hole (wide for
    part of its length, narrow for the rest) is cut correctly.

    Returns [{axis, cu, cv, levels: [(radius, t0, t1), ...]}] in world coords
    (outer ends overshoot the faces). Empty when there are no tangent holes.
    """
    try:
        import trimesh
        from shapely.geometry import MultiPolygon, Point, Polygon as SPoly
        from mesh2param.pipeline.recognize import _section_material
    except Exception:
        return []
    try:
        ext_axis = int(np.argmax(np.abs(extrude_dir)))
        mesh = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces,
                               process=False)
        out = []
        for axis in (0, 1, 2):
            if axis == ext_axis:
                continue                          # extrude-axis holes handled elsewhere
            a_lo = float(cleaned.bbox[0][axis])
            a_hi = float(cleaned.bbox[1][axis])
            span = a_hi - a_lo
            if span <= 1e-6:
                continue
            g = _section_material(mesh, axis, 0.5 * (a_lo + a_hi))
            if g is None:
                continue
            polys = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
            cands = []
            for p in polys:
                for ring in p.interiors:
                    ip = SPoly(ring)
                    per = ip.length
                    circ = 4.0 * np.pi * ip.area / (per * per + 1e-9)
                    if ip.area >= 3.0 and circ > 0.7:
                        rp = ip.representative_point()
                        cands.append((float(rp.x), float(rp.y)))
            for (cu, cv) in cands:
                n = int(np.clip(span / 2.0, 16, 60))
                ts = np.linspace(a_lo + span * 0.02, a_hi - span * 0.02, n)
                pt = Point(cu, cv)
                radii = np.zeros(n)
                for k, t in enumerate(ts):
                    gg = _section_material(mesh, axis, float(t))
                    if gg is not None and not gg.contains(pt):
                        radii[k] = float(gg.distance(pt))
                rmax = float(radii.max())
                if rmax < 1.0:
                    continue
                if np.mean(radii > max(0.5, 0.3 * rmax)) < 0.5:
                    continue                       # not through-ish
                tol = max(0.5, 0.25 * rmax)
                rmin = max(0.5, 0.3 * rmax)
                levels = []
                k = 0
                while k < n:
                    if radii[k] < rmin:
                        k += 1
                        continue
                    j = k
                    band = [radii[k]]
                    while (j + 1 < n and radii[j + 1] >= rmin
                           and abs(radii[j + 1] - float(np.median(band))) <= tol):
                        j += 1
                        band.append(radii[j])
                    levels.append({"radius": float(np.median(band)), "i0": k, "i1": j})
                    k = j + 1
                if not levels:
                    continue
                # Merge mesh-noise levels: absorb a level into the previous one
                # when it's short OR its radius is close (so a wide→narrow step
                # doesn't fragment into wide→narrow→bulge→narrow).
                min_len = max(2.0, 0.12 * span)
                merged = [levels[0]]
                for lv in levels[1:]:
                    prev = merged[-1]
                    short = (float(ts[lv["i1"]] - ts[lv["i0"]]) < min_len)
                    close = abs(lv["radius"] - prev["radius"]) <= tol
                    if short or close:
                        prev["i1"] = lv["i1"]      # keep prev (longer/earlier) radius
                    else:
                        merged.append(lv)
                levels = merged
                margin = max(1.0, 0.06 * span)
                lv_out = []
                for idx, lv in enumerate(levels):
                    start = (a_lo - margin) if idx == 0 else \
                        0.5 * (float(ts[levels[idx - 1]["i1"]]) + float(ts[lv["i0"]]))
                    end = (a_hi + margin) if idx == len(levels) - 1 else \
                        0.5 * (float(ts[lv["i1"]]) + float(ts[levels[idx + 1]["i0"]]))
                    lv_out.append((lv["radius"], float(start), float(end)))
                out.append({"axis": axis, "cu": cu, "cv": cv, "levels": lv_out,
                            "a_lo": a_lo, "a_hi": a_hi})
        return out
    except Exception:
        logger.debug("tangent hole detection failed", exc_info=True)
        return []


def _merge_coplanar_hole_steps(steps) -> None:
    """Merge consecutive (Sketch, op) pairs that share a PLANE and a
    FEATURE LENGTH into ONE sketch (all profiles) + ONE op — how a CAD
    user works: one sketch per face, then one cut/extrude at each depth.
    Applies to holes (through, or blind at the same depth), cut extrudes,
    and join extrudes; op classes never mix."""
    import re as _re

    _len_re = _re.compile(r"depth\s*([\d.]+)|\((?:[\d]+× )?([\d.]+)mm\)")

    def _plane_of(s):
        try:
            ov = s.sketch_overlay
            if ov is not None and len(ov):
                pts = np.asarray(ov, float).reshape(-1, 3)
                for ax in range(3):
                    if float(np.ptp(pts[:, ax])) < 0.5:
                        return (ax, round(float(np.mean(pts[:, ax])), 1))
            arcs = s.overlay_arcs
            if arcs:
                ax = arcs[0].get("ax")
                pos = arcs[0].get("pos")
                if (ax is not None and pos is not None
                        and all(a.get("ax") == ax
                                and abs(float(a.get("pos", 0)) - pos) < 0.5
                                for a in arcs)):
                    return (int(ax), round(float(pos), 1))
        except Exception:
            pass
        return None

    def _op_class(nm):
        if "Hole" in nm:
            return "hole"
        if nm.startswith(("Extrude Join", "Union Extrude")):
            return "join"
        if nm.startswith(("Cut Extrude", "Pocket", "Slot Cut", "Oval Cut",
                          "Recess Cut")):
            return "cut"
        return None

    def _feat_len(nm):
        if "through" in nm.lower():
            return "through"
        m = _len_re.search(nm)
        if m:
            return round(float(m.group(1) or m.group(2)), 1)
        return None

    def _pair_key(i):
        if i + 1 >= len(steps):
            return None
        a, b = steps[i], steps[i + 1]
        if not (a.name or "").startswith("Sketch"):
            return None
        cls = _op_class(b.name or "")
        if cls is None:
            return None
        ln = _feat_len(b.name or "")
        if ln is None:
            return None
        pl = _plane_of(a)
        if pl is None:
            return None
        return (cls, ln, pl)

    def _same_key(a, b):
        if a is None or b is None or a[0] != b[0]:
            return False
        la, lb = a[1], b[1]
        if isinstance(la, str) != isinstance(lb, str):
            return False
        if isinstance(la, str):
            if la != lb:
                return False
        elif abs(la - lb) > 0.3:
            return False
        return a[2][0] == b[2][0] and abs(a[2][1] - b[2][1]) < 2.0

    out = []
    i = 0
    while i < len(steps):
        key = _pair_key(i)
        if key is None:
            out.append(steps[i])
            i += 1
            continue
        run = [i]
        j = i + 2
        while _same_key(_pair_key(j), key):
            run.append(j)
            j += 2
        if len(run) == 1:
            out.append(steps[i])
            out.append(steps[i + 1])
            i += 2
            continue
        sk0 = steps[run[0]]
        ovs = [np.asarray(steps[k].sketch_overlay, float).reshape(-1, 3)
               for k in run
               if steps[k].sketch_overlay is not None
               and len(steps[k].sketch_overlay)]
        arcs = []
        for k in run:
            arcs.extend(steps[k].overlay_arcs or [])
        if ovs:
            sk0.sketch_overlay = np.vstack(ovs)
        sk0.overlay_arcs = arcs or None
        cls, ln, _pl = key
        _thr = ln == "through"
        _len_sfx = "" if _thr else f", depth {ln}mm"
        n = len(run)
        last = steps[run[-1] + 1]
        if cls == "hole":
            sk0.name = f"Sketch Holes ({n} on one plane)"
            last.name = (f"Through Holes ×{n}" if _thr
                         else f"Blind Holes ×{n} (depth {ln}mm)")
        elif cls == "join":
            sk0.name = f"Sketch Joins ({n} on one plane)"
            last.name = f"Extrude Join ×{n} ({ln}mm)"
        else:
            sk0.name = f"Sketch Cuts ({n} on one plane)"
            last.name = (f"Cut Extrude ×{n} (through)" if _thr
                         else f"Cut Extrude ×{n} (depth {ln}mm)")
        out.append(sk0)
        out.append(last)
        i = run[-1] + 2
    steps[:] = out


def _onshape_step_names(steps) -> None:
    """Rename construction steps IN PLACE to Onshape feature vocabulary:
    "Sketch N (…)", "Extrude N (Base|Join|Cut, …)", "Hole N (…)",
    "Fillet N", "Chamfer N", "Boolean N". The parenthetical keeps the
    descriptive detail (diameters, depths); the prefix is always a real
    Onshape feature type — never internal jargon like "Refine Cut"."""
    import re as _re

    counters: dict = {}

    def _n(kind):
        counters[kind] = counters.get(kind, 0) + 1
        return counters[kind]

    def _detail(nm):
        m = _re.search(r"\(([^)]*)\)", nm)
        return m.group(1).strip() if m else ""

    for s in steps:
        nm = s.name or ""
        if nm.startswith("Sketch"):
            rest = _re.sub(r"^Sketch\s*", "", nm)
            rest = rest.replace("(", "").replace(")", "").strip()
            s.name = f"Sketch {_n('Sketch')}" + (f" ({rest})" if rest else "")
        elif "Hole" in nm:
            # "Through Hole Pattern ⌀9.0mm ×2" -> "Hole 1 (⌀9.0mm ×2, through)"
            mode = ("through" if "Through" in nm
                    else "blind" if "Blind" in nm else "")
            rest = _re.sub(r"(Through|Blind|Tangent)?\s*Holes?(\s+Pattern)?",
                           "", nm)
            rest = rest.replace("(", "").replace(")", "").strip()
            # drop a bare leading hole index ("2 level 1") but never a
            # pattern count ("3× r=3.2")
            rest = _re.sub(r"^\d+\s+(?![×x(])", "", rest)
            parts = ", ".join(x for x in (rest, mode) if x)
            s.name = f"Hole {_n('Hole')}" + (f" ({parts})" if parts else "")
        elif nm.startswith("Fillet"):
            d = _detail(nm)
            s.name = f"Fillet {_n('Fillet')}" + (f" ({d})" if d else "")
        elif nm.startswith("Chamfer"):
            d = _detail(nm)
            s.name = f"Chamfer {_n('Chamfer')}" + (f" ({d})" if d else "")
        elif nm == "Extrude Base":
            s.name = f"Extrude {_n('Extrude')} (Base)"
        elif nm.startswith("Extrude Member"):
            d = _detail(nm)
            s.name = (f"Extrude {_n('Extrude')} (Join"
                      + (f", {d}" if d else "") + ")")
        elif nm.startswith(("Extrude Join", "Extrude ")):
            d = _detail(nm)
            s.name = (f"Extrude {_n('Extrude')} (Join"
                      + (f", {d}" if d else "") + ")")
        elif nm.startswith(("Boss", "Panel Boss", "Refine Add")):
            rest = _re.sub(r"^(Panel Boss|Boss Pattern|Boss|Refine Add)\s*",
                           "", nm)
            rest = rest.replace("(", "").replace(")", "").strip()
            s.name = (f"Extrude {_n('Extrude')} (Join"
                      + (f", {rest}" if rest else "") + ")")
        elif nm.startswith("Side Profile"):
            s.name = f"Sketch {_n('Sketch')} (side profile)"
        elif nm.startswith(("Refine Cut", "Cut Extrude", "Pocket",
                            "Oval Cut", "Slot Cut", "Recess Cut",
                            "Ring Recess", "Vent", "Cut ", "Side Cut")):
            m2 = _re.match(r"^(Oval|Slot|Recess)\s+Cut\s*(.*)$", nm)
            if m2:
                rest = f"{m2.group(1)} {m2.group(2)}"
            else:
                rest = _re.sub(r"^(Refine Cut|Cut Extrude|Side Cut|Cut)\s*",
                               "", nm)
            rest = rest.replace("(", "").replace(")", "").strip()
            s.name = (f"Extrude {_n('Extrude')} (Cut"
                      + (f", {rest}" if rest else "") + ")")
        elif nm == "Refined Model":
            s.name = f"Boolean {_n('Boolean')} (refine)"
        elif nm == "Combine Bodies":
            s.name = f"Boolean {_n('Boolean')} (union)"
        # anything else keeps its name


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
    joins: list[Feature] | None = None,
    as_built_out: dict | None = None,
    stream: bool = False,
    base_corners: list | None = None,
    base_fillets: list | None = None,
    revolve_base: object = None,
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

    # The primary variant streams each step to disk as it's built (live UI);
    # the deferred-chamfer / stitched-fillet passes in export_cadquery append
    # onto this same returned list, so they stream too. Other variants/bodies
    # use a plain list to avoid double-counting the stream.
    steps: list[ConstructionStep] = (
        _StreamingStepList()
        if (stream and (_stream_ctx["dir"] or _stream_ctx["sink"]))
        else []
    )
    script_lines = ["import cadquery as cq", ""]

    # --- Step 0: Sketch (outer profile only — cutouts get their own steps) ---
    axes = [0, 1, 2]
    axes.remove(primary_axis)
    ax_u, ax_v = axes

    all_sketch_verts = []
    all_sketch_faces = []
    vert_offset = 0

    # Partition the extra profiles. Only LARGE or NON-circular inner
    # cutouts belong to the base cross-section and get baked into the base
    # sketch as holes — a tube's bore or a plate's window — so the extrude
    # is the ring/windowed shape in one step. Small CIRCULAR inner cutouts
    # are bolt holes: they must NOT go in the base profile (that put a
    # spurious hole in the sketch on Body1Mesh) — they fall through to the
    # loop and get drilled as separate hole features. Outer extras
    # (legs/bosses) are unioned in the loop.
    from shapely.geometry import Polygon as _BasePoly, Point as _BasePoint
    _main_poly = _BasePoly(profiles[0])
    _main_area = abs(_main_poly.area)
    # Standoff/boss footprints in the base's (u,v) plane. A small mounting hole
    # that sits UNDER a standoff must be cut PLATE-ONLY (baked into the base
    # extrude so the solid standoff caps it from above), not drilled straight
    # through the standoff. A free bolt hole (no standoff over it) still cuts
    # through. Only same-axis bosses share the base's u,v plane.
    _boss_foot = []
    for _jf in (joins or []):
        _jsg = _jf.sketch_geometry or {}
        if _jsg.get('axis_idx') != primary_axis:
            continue
        for _loop in [_jsg.get('points')] + list(_jsg.get('extra_loops') or []):
            if _loop and len(_loop) >= 3:
                try:
                    _bp = _BasePoly(_loop)
                    if not _bp.is_valid:
                        _bp = _bp.buffer(0)
                    if not _bp.is_empty:
                        _boss_foot.append(_bp)
                except Exception:
                    pass

    base_inner_cutouts: list = []
    extra_profiles: list = []
    for _p in profiles[1:]:
        _poly = _BasePoly(_p)
        _c = _poly.centroid
        if not _main_poly.contains(_BasePoint(_c.x, _c.y)):
            extra_profiles.append(_p)            # outer leg/boss -> union
            continue
        # Size is the reliable discriminator: a bore/window is a large
        # fraction of the section (tube bore ~90%, frame window ~30-60%),
        # a bolt hole is < ~1%. Circularity is NOT usable here — a coarse
        # mesh traces a 2.5mm hole as a jagged 38-gon whose inflated
        # perimeter gives a misleadingly low circularity (0.44).
        _area = abs(_poly.area)
        _pt = _BasePoint(_c.x, _c.y)
        _under_boss = any(bf.contains(_pt) for bf in _boss_foot)
        if _area > 0.05 * _main_area:
            # DESIGN-INTENT snap (scan meshes): a round bore baked into the
            # base sketch is a true CIRCLE, not the jagged mesh trace (the
            # motor plate's ⌀91 center bore).
            if (cleaned is not None
                    and (getattr(cleaned, "metadata", None) or {}).get(
                        "scan_mode")):
                try:
                    _bcirc = (4.0 * np.pi * _area
                              / (_poly.length ** 2 + 1e-9))
                    if _bcirc >= 0.8:
                        _br = float(np.sqrt(_area / np.pi))
                        _bth = np.linspace(0.0, 2.0 * np.pi, 65)[:-1]
                        _p = [(float(_c.x + _br * np.cos(t)),
                               float(_c.y + _br * np.sin(t)))
                              for t in _bth]
                except Exception:
                    pass
            base_inner_cutouts.append(_p)        # window/bore -> bake into base
        elif _under_boss:
            # Mounting hole under a standoff: cut PLATE-ONLY (baked here), the
            # solid standoff above caps it.
            base_inner_cutouts.append(_p)
        else:
            extra_profiles.append(_p)            # free bolt hole -> through cut

    # Sketch wireframe overlay: outer boundary + each baked-in hole. The outer
    # boundary draws as STRAIGHT segments + true tangent ARCS (no chord polyline
    # on the rounded corners) when the parametric corners+fillets are carried —
    # the same source the solid and the Onshape sketch use. Holes chord their
    # own (already finely sampled) points.
    # REVOLVE base: the lathe solid replaces the extruded slab as the
    # timeline's first feature; every cutter below still runs on it.
    _revolved = revolve_base is not None
    if _revolved and as_built_out is not None:
        as_built_out["revolve_base"] = True

    _overlay_line_pts = []   # flat list of 3D points, consecutive PAIRS = a segment
    _overlay_arcs = []
    if _revolved:
        # A Revolve's sketch is its SIDE (lathe) profile in the half-plane
        # through the axis — not the base-face circles.
        try:
            _rsg = revolve_base["sg"]
            _rax_o = np.asarray(_rsg["axis_origin"], dtype=float)
            _rax_d = np.asarray(_rsg["axis_dir"], dtype=float)
            _rax_d = _rax_d / (np.linalg.norm(_rax_d) + 1e-12)
            _rprof = _regularize_revolve_profile(
                np.asarray(_rsg["profile"], dtype=float))
            _rtmp = (np.array([1.0, 0, 0]) if abs(_rax_d[0]) < 0.9
                     else np.array([0, 1.0, 0]))
            _rrad = np.cross(_rax_d, _rtmp)
            _rrad = _rrad / (np.linalg.norm(_rrad) + 1e-12)
            _rloop = ([(float(_rprof[0, 0]), 0.0)]
                      + [(float(h), float(r)) for h, r in _rprof]
                      + [(float(_rprof[-1, 0]), 0.0)])
            _rp3 = [_rax_o + _rax_d * h + _rrad * r for (h, r) in _rloop]
            for _k in range(len(_rp3) - 1):
                _overlay_line_pts.append([float(x) for x in _rp3[_k]])
                _overlay_line_pts.append([float(x) for x in _rp3[_k + 1]])
        except Exception:
            logger.debug("revolve profile overlay failed", exc_info=True)
    if not _overlay_line_pts:
        _ol, _oa = _overlay_for_ring(
            profiles[0], primary_axis, extrude_start,
            corners=base_corners, fillets=base_fillets)
        _overlay_line_pts.extend(_ol)
        _overlay_arcs.extend(_oa)
        for profile in base_inner_cutouts:
            _il, _ia = _overlay_for_ring(profile, primary_axis, extrude_start)
            _overlay_line_pts.extend(_il)
            _overlay_arcs.extend(_ia)

    # Build a degenerate-triangle wireframe from the straight segments so the
    # is_sketch render path (which walks faces as edges) still shows the lines;
    # the arcs ride alongside in overlay_arcs.
    if _overlay_line_pts:
        sketch_verts = np.asarray(_overlay_line_pts, dtype=np.float64)
        _m = len(sketch_verts) // 2
        sketch_faces = np.array(
            [[2 * i, 2 * i + 1, 2 * i] for i in range(_m)], dtype=np.int64)
    else:
        # Fully circular outer ring (no straights) — fall back to chords so the
        # step still carries geometry for the endpoint.
        sketch_verts = np.vstack(all_sketch_verts) if all_sketch_verts else \
            np.zeros((0, 3))
        sketch_faces = np.vstack(all_sketch_faces) if all_sketch_faces else \
            np.zeros((0, 3), dtype=np.int64)

    steps.append(ConstructionStep(
        name="Sketch Profile", tess_vertices=sketch_verts, tess_faces=sketch_faces,
        volume=0.0, overlay_arcs=_overlay_arcs or None,
    ))
    script_lines.append(f"# Step 1: Sketch profile on {plane_name} plane")
    script_lines.append(f'result = cq.Workplane("{plane_name}", origin={tuple(origin_3d)})')
    script_lines.append("result = result.polyline(outer_profile).close()")
    for _ci in range(len(base_inner_cutouts)):
        script_lines.append(f"result = result.polyline(hole_profile_{_ci}).close()")

    # --- Step 1: Extrude base profile ---
    # Try both directions, keep the one landing in the base's INTENDED SPAN
    # [extrude_start, extrude_start + extrude_dir*depth]. Anchoring on the part
    # bbox fails for a STACKED part (a base plate in the MIDDLE — AI1): the
    # part extends both ways from the base plane, so both directions overlap it
    # equally and the tie sends the base the wrong way (into the bottom plate's
    # region). The base span has a definite direction, so use it.
    _ed = 1.0 if float(extrude_dir[primary_axis]) >= 0 else -1.0
    tgt_lo = min(extrude_start, extrude_start + _ed * depth)
    tgt_hi = max(extrude_start, extrude_start + _ed * depth)

    wp_pos = cq.Workplane(plane_name, origin=tuple(origin_3d))
    wp_pos = _draw_profile_with_holes(
        wp_pos, profiles[0], base_inner_cutouts,
        corners=base_corners, fillets=base_fillets).extrude(depth)
    bb_pos = wp_pos.val().BoundingBox()
    pos_vals = [(bb_pos.xmin, bb_pos.xmax), (bb_pos.ymin, bb_pos.ymax), (bb_pos.zmin, bb_pos.zmax)]
    pos_overlap = min(pos_vals[primary_axis][1], tgt_hi) - max(pos_vals[primary_axis][0], tgt_lo)

    wp_neg = cq.Workplane(plane_name, origin=tuple(origin_3d))
    wp_neg = _draw_profile_with_holes(
        wp_neg, profiles[0], base_inner_cutouts,
        corners=base_corners, fillets=base_fillets).extrude(-depth)
    bb_neg = wp_neg.val().BoundingBox()
    neg_vals = [(bb_neg.xmin, bb_neg.xmax), (bb_neg.ymin, bb_neg.ymax), (bb_neg.zmin, bb_neg.zmax)]
    neg_overlap = min(neg_vals[primary_axis][1], tgt_hi) - max(neg_vals[primary_axis][0], tgt_lo)

    if neg_overlap > pos_overlap:
        wp = wp_neg
        extrude_sign = -depth
        logger.info("Using negative extrude (overlap %.1f > %.1f)", neg_overlap, pos_overlap)
    else:
        wp = wp_pos
        extrude_sign = depth

    if _revolved:
        wp = revolve_base["solid"]
        script_lines.append("# Step 2: Revolve (body of revolution)")
    else:
        script_lines.append(f"# Step 2: Extrude {depth:.2f} mm")
        script_lines.append(f"result = result.extrude({depth})")

    # --- Symmetric thinning: a profile region THINNER than the base span ---
    # The L-bracket: leg full 17.2mm, bar/arm ~12.7mm (set back from both
    # faces). Detected BEFORE the base step is recorded so the timeline
    # reads like Onshape (see the thin-first build below).
    _thin = None
    if (cleaned is not None and not _revolved
            and (getattr(cleaned, "metadata", None) or {}).get("scan_mode")):
        try:
            _thin = _detect_symmetric_thinning(cleaned, extrude_dir)
        except Exception:
            _thin = None
    # The setback is a soft TAPER (AI-mesh pillowing), not a designed step:
    # even the "thick" regions only touch the bbox span at a small bump.
    # So the honest model is ONE extrude — the whole profile at the
    # measured effective thickness — with the taper absorbed. No member
    # extrude, no pads sketch (two floating outlines was not a valid
    # Onshape sketch), no extra steps at all.
    if _thin is not None and _thin["axis"] == primary_axis:
        try:
            from shapely.geometry import Polygon as _ThPoly
            _prof = _ThPoly(profiles[0]).buffer(0)
            _A_lo = float(cleaned.bbox[0][primary_axis])
            _A_hi = float(cleaned.bbox[1][primary_axis])
            _t_lo = _A_lo + _thin["d_lo"]
            _t_hi = _A_hi - _thin["d_hi"]
            _thick = _t_hi - _t_lo
            if _thick > 1.0:
                _vol_before = abs(float(wp.val().Volume()))
                _us = [k for k in (0, 1, 2) if k != primary_axis]
                _c = [0.0, 0.0, 0.0]
                _c[primary_axis] = 0.5 * (_t_lo + _t_hi)
                _c[_us[0]] = 0.5 * (float(cleaned.bbox[0][_us[0]])
                                    + float(cleaned.bbox[1][_us[0]]))
                _c[_us[1]] = 0.5 * (float(cleaned.bbox[0][_us[1]])
                                    + float(cleaned.bbox[1][_us[1]]))
                _dims = [0.0, 0.0, 0.0]
                _dims[primary_axis] = _thick
                _dims[_us[0]] = (float(cleaned.bbox[1][_us[0]])
                                 - float(cleaned.bbox[0][_us[0]])) + 10.0
                _dims[_us[1]] = (float(cleaned.bbox[1][_us[1]])
                                 - float(cleaned.bbox[0][_us[1]])) + 10.0
                _slab = (cq.Workplane("XY", origin=tuple(_c))
                         .box(_dims[0], _dims[1], _dims[2]))
                # Clip the full solid to the thin span: keeps every baked
                # hole/corner of the real profile.
                _wp_thin = wp.intersect(_slab)
                _removed = (_vol_before
                            - abs(float(_wp_thin.val().Volume())))
                _exp = _prof.area * (_thin["d_lo"] + _thin["d_hi"])
                # Must remove ≈ the two setback layers over the profile.
                if 0.4 * _exp <= _removed <= 1.8 * _exp:
                    wp = _wp_thin
                    if as_built_out is not None:
                        as_built_out["thin_member"] = True
                    script_lines.append(
                        f"# Base extruded at effective thickness "
                        f"{_thick:.2f} mm (taper absorbed)")
                    logger.info(
                        "Base at effective thickness %.1fmm (bbox %.1fmm, "
                        "taper absorbed), %.0f mm^3 removed",
                        _thick, _A_hi - _A_lo, _removed)
                else:
                    logger.info(
                        "Effective-thickness base skipped: removed %.0f vs "
                        "expected %.0f mm^3", _removed, _exp)
        except Exception:
            logger.warning("Symmetric thinning failed", exc_info=True)

    v, f = _tessellate_solid(wp, fast=True)
    vol = abs(float(wp.val().Volume())) if wp else 0
    steps.append(ConstructionStep(
        name=("Revolve" if _revolved else "Extrude Base"),
        tess_vertices=v, tess_faces=f, volume=vol,
    ))

    # Baked windows/bores become real THROUGH cuts on the revolve (the
    # motor plate's ⌀91 center bore — the lathe profile carries no holes).
    if _revolved and base_inner_cutouts:
        _A_lo_rb = float(cleaned.bbox[0][primary_axis])
        _A_hi_rb = float(cleaned.bbox[1][primary_axis])
        _o_rb = [0.0, 0.0, 0.0]
        _o_rb[primary_axis] = 0.5 * (_A_lo_rb + _A_hi_rb)
        for _ci, _cl in enumerate(base_inner_cutouts, 1):
            try:
                _ol_rb, _oa_rb = _overlay_for_ring(
                    _cl, primary_axis, extrude_start)
                _sv_rb = (np.asarray(_ol_rb, np.float64)
                          if _ol_rb else np.zeros((0, 3)))
                # Sketch step = the CURRENT solid with the circle overlaid
                # (a wireframe-only step floats detached from the part).
                _bv_rb, _bf_rb = _tessellate_solid(wp, fast=True)
                steps.append(ConstructionStep(
                    name=f"Sketch Hole {_ci}",
                    tess_vertices=np.asarray(_bv_rb, np.float64),
                    tess_faces=np.asarray(_bf_rb, np.int64),
                    volume=abs(float(wp.val().Volume())),
                    sketch_overlay=_sv_rb, overlay_arcs=_oa_rb or None,
                ))
                _cut_rb = (cq.Workplane(plane_name, origin=tuple(_o_rb))
                           .polyline([(float(a), float(b))
                                      for a, b in _cl]).close()
                           .extrude(0.6 * (_A_hi_rb - _A_lo_rb) + 2.0,
                                    both=True))
                wp = wp.cut(_cut_rb)
                from shapely.geometry import Polygon as _RbPoly
                _r_rb = float(np.sqrt(abs(_RbPoly(_cl).area) / np.pi))
                v, f = _tessellate_solid(wp, fast=True)
                steps.append(ConstructionStep(
                    name=f"Through Hole (r={_r_rb:.1f})",
                    tess_vertices=v, tess_faces=f,
                    volume=abs(float(wp.val().Volume())),
                ))
            except Exception:
                logger.debug("revolve bore cut failed", exc_info=True)

    # --- Join extrusions: the engine plan's non-primary segments ---
    # One part, one timeline: each join sketches its own profile on its
    # own axis plane and extrudes over its span, unioned into the solid
    # BEFORE holes are drilled (a side plate must exist before its bores
    # are cut). RackMount: Extrude Base (bottom plate) -> Extrude Join x2
    # (side plates) replaces the decomposed "bodies" presentation.
    for ji, jf in enumerate(joins or [], 1):
        try:
            sg = jf.sketch_geometry or {}
            span = sg.get("span")
            raw_pts = sg.get("points")
            ax_j = int(sg.get("axis_idx", 0))
            if not span or not raw_pts:
                continue
            j_pts = _clean_engine_polygon(raw_pts)
            if len(j_pts) < 3:
                continue
            j_inner = []
            for ring in sg.get("inner") or []:
                ring_pts = _clean_engine_polygon(ring)
                if len(ring_pts) >= 3:
                    j_inner.append(ring_pts)
            # Pattern: a co-planar group (AI1's four corner screws) carries its
            # extra profile loops here, drawn on the SAME sketch plane so the
            # join is one sketch + one extrude with N loops.
            j_extra = []
            _ei = sg.get("extra_inner") or []
            for li, loop in enumerate(sg.get("extra_loops") or []):
                lp = _clean_engine_polygon(loop)
                if len(lp) < 3:
                    continue
                lin = []
                for ring in (_ei[li] if li < len(_ei) else []) or []:
                    rp = _clean_engine_polygon(ring)
                    if len(rp) >= 3:
                        lin.append(rp)
                j_extra.append((lp, lin))
            j_len = float(span[1]) - float(span[0])
            plane_j = plane_map[ax_j]
            jx_u, jx_v = [a for a in range(3) if a != ax_j]
            # Put the sketch on the span face ADJACENT to the existing solid and
            # extrude AWAY from it — otherwise a boss BELOW the base sketches at
            # the far end (span[0]) and extrudes back INTO the plate, so the
            # profile visibly floats off the part. Pick the span endpoint nearer
            # the current solid's centre along this axis.
            try:
                _bb = wp.val().BoundingBox()
                _ctr = [(_bb.xmin + _bb.xmax) / 2, (_bb.ymin + _bb.ymax) / 2,
                        (_bb.zmin + _bb.zmax) / 2][ax_j]
            except Exception:
                _ctr = (float(span[0]) + float(span[1])) / 2
            sketch_at = (float(span[0])
                         if abs(float(span[0]) - _ctr) <= abs(float(span[1]) - _ctr)
                         else float(span[1]))
            origin_j = [0.0, 0.0, 0.0]
            origin_j[ax_j] = sketch_at

            # Sketch overlay wireframe in world coords — include EVERY loop of a
            # pattern (AI1's four corner screws are ONE sketch with four loops),
            # so the sketch shows all four profiles, not just the first. The main
            # loop (and any rounded extra loop) draws as STRAIGHT segments + true
            # tangent ARCS — no chord polyline on curves — from the carried
            # corners+fillets; circular holes chord their own points.
            j_overlay = []
            j_arcs = []
            _ml, _ma = _overlay_for_ring(
                j_pts, ax_j, sketch_at,
                corners=sg.get("corners"), fillets=sg.get("fillets"))
            j_overlay.extend(_ml)
            j_arcs.extend(_ma)
            for ring_pts in j_inner:
                _rl, _ra = _overlay_for_ring(ring_pts, ax_j, sketch_at)
                j_overlay.extend(_rl)
                j_arcs.extend(_ra)
            for _li, (_lp, _lin) in enumerate(j_extra):
                _ec = (sg.get("extra_corners") or [None] * (_li + 1))[_li] \
                    if sg.get("extra_corners") else None
                _ef = (sg.get("extra_fillets") or [None] * (_li + 1))[_li] \
                    if sg.get("extra_fillets") else None
                _ll, _la = _overlay_for_ring(
                    _lp, ax_j, sketch_at, corners=_ec, fillets=_ef)
                j_overlay.extend(_ll)
                j_arcs.extend(_la)
                for ring_pts in _lin:
                    _rl, _ra = _overlay_for_ring(ring_pts, ax_j, sketch_at)
                    j_overlay.extend(_rl)
                    j_arcs.extend(_ra)
            v_pre, f_pre = _tessellate_solid(wp, fast=True)
            steps.append(ConstructionStep(
                name=f"Sketch Join {ji}",
                tess_vertices=np.asarray(v_pre, dtype=np.float64),
                tess_faces=np.asarray(f_pre, dtype=np.int64),
                volume=abs(float(wp.val().Volume())),
                sketch_overlay=np.asarray(j_overlay, dtype=np.float64),
                overlay_arcs=j_arcs or None,
            ))

            # Extrude with the sign whose result overlaps the span
            # (named CadQuery planes have fixed normals; XZ extrudes -Y).
            best_wpj, best_overlap = None, -1e30
            for sign in (1.0, -1.0):
                try:
                    wpj = cq.Workplane(plane_j, origin=tuple(origin_j))
                    wpj = _draw_profile_with_holes(
                        wpj, j_pts, j_inner,
                        corners=sg.get("corners"), fillets=sg.get("fillets"))
                    for _li, (lp, lin) in enumerate(j_extra):
                        _ec = (sg.get("extra_corners") or [None] * (_li + 1))[_li] \
                            if sg.get("extra_corners") else None
                        _ef = (sg.get("extra_fillets") or [None] * (_li + 1))[_li] \
                            if sg.get("extra_fillets") else None
                        wpj = _draw_profile_with_holes(wpj, lp, lin, _ec, _ef)
                    wpj = wpj.extrude(sign * j_len)
                    bbj = wpj.val().BoundingBox()
                    j_vals = [
                        (bbj.xmin, bbj.xmax), (bbj.ymin, bbj.ymax),
                        (bbj.zmin, bbj.zmax),
                    ]
                    ov = (
                        min(j_vals[ax_j][1], float(span[1]))
                        - max(j_vals[ax_j][0], float(span[0]))
                    )
                    if ov > best_overlap:
                        best_overlap, best_wpj = ov, wpj
                except Exception:
                    continue
            if best_wpj is None:
                logger.warning("Join %d failed to extrude; skipping", ji)
                continue
            _vol_pre_join = abs(float(wp.val().Volume()))
            _wp_joined_j = wp.union(best_wpj)
            _dv_j = abs(float(_wp_joined_j.val().Volume())) - _vol_pre_join
            # On scan/AI meshes a join that adds ~nothing is a deformed
            # corner NUB (the bezel frame's four sub-2mm "screws", 4-12mm³
            # each on an 11,400mm³ part) — mesh noise, not a feature. Drop
            # it and its sketch. Real joins add real volume (the L-bracket's
            # feet: ~260mm³ each).
            if (cleaned is not None
                    and (getattr(cleaned, "metadata", None) or {}).get(
                        "scan_mode")
                    and _dv_j < max(15.0, 0.002 * _vol_pre_join)):
                steps.pop()          # the Sketch Join step just appended
                logger.info("Join %d dropped: adds only %.1f mm^3 "
                            "(corner-nub noise)", ji, _dv_j)
                continue
            wp = _wp_joined_j
            v_j, f_j = _tessellate_solid(wp, fast=True)
            steps.append(ConstructionStep(
                name=f"Extrude Join {ji} ({j_len:.1f}mm)",
                tess_vertices=v_j, tess_faces=f_j,
                volume=abs(float(wp.val().Volume())),
            ))
            script_lines.append(f"# Join extrusion {ji}")
            script_lines.append(
                f'join{ji} = cq.Workplane("{plane_j}", origin={tuple(origin_j)})'
                f".polyline(join_profile_{ji}).close().extrude({j_len:.4f})"
            )
            script_lines.append(f"result = result.union(join{ji})")
            logger.info(
                "Join %d: axis %d span %.1f..%.1f, %d pts, vol now %.0f",
                ji, ax_j, span[0], span[1], len(j_pts),
                abs(float(wp.val().Volume())),
            )
        except Exception:
            logger.exception("Join extrusion %d failed; continuing", ji)

    # Remaining profiles: outer legs/bosses (union) and small circular
    # bolt-hole cutouts (cut). Large window/bore cutouts were baked into
    # the base sketch above.
    _mid_material = None
    if (cleaned is not None and extra_profiles
            and (getattr(cleaned, "metadata", None) or {}).get("scan_mode")):
        try:
            import trimesh as _MidTm
            from mesh2param.pipeline.recognize import (
                _section_material as _MidSect)
            _mm = _MidTm.Trimesh(vertices=cleaned.vertices,
                                 faces=cleaned.faces, process=False)
            _mid_t = 0.5 * (float(cleaned.bbox[0][primary_axis])
                            + float(cleaned.bbox[1][primary_axis]))
            _mid_material = _MidSect(_mm, primary_axis, _mid_t)
        except Exception:
            _mid_material = None
    for pi, profile in enumerate(extra_profiles, 1):
        try:
            _snapped_hole_r = None
            _pp = _BasePoly(profile)
            _cc = _pp.centroid
            is_inner = _main_poly.contains(_BasePoint(_cc.x, _cc.y))

            # NOISE guard (dirty meshes): a real small cutout is a designed
            # shape — round or rectangular, i.e. it FILLS its convex hull. A
            # mesh-noise blob (the enclosure lid's corner "V" squiggle) is
            # tiny AND ragged (low solidity). Skip those instead of cutting a
            # random divot into the part.
            if (getattr(cleaned, "metadata", None) or {}).get("scan_mode"):
                try:
                    _circ = (4.0 * np.pi * _pp.area
                             / (_pp.length ** 2 + 1e-9))
                    _rel = _pp.area / max(_main_poly.area, 1e-9)
                    if is_inner and _rel < 0.015:
                        # THROUGH-ness is the decisive test for a tiny
                        # inner loop: a real hole is VOID at mid-depth; a
                        # face artifact (the lid's foot shadow) is SOLID
                        # there. Shape gates alone couldn't separate them —
                        # a jagged real bolt circle scores circ 0.80, the
                        # foot shadow 0.75 (and the 0.85 gate ate the motor
                        # plate's real outer holes).
                        if _mid_material is not None:
                            if _mid_material.buffer(0).contains(
                                    _BasePoint(_cc.x, _cc.y)):
                                logger.info(
                                    "Skipping face-artifact profile %d "
                                    "(area %.0f: solid at mid-depth)",
                                    pi, _pp.area)
                                continue
                        elif _circ < 0.75:
                            logger.info(
                                "Skipping noise profile %d (area %.0f, "
                                "circ %.2f)", pi, _pp.area, _circ)
                            continue
                    # A kept round-ish inner loop is a HOLE: cut it as a
                    # true circle, not a jagged polyline trace.
                    if is_inner and _circ >= 0.6:
                        _r_eq = float(np.sqrt(_pp.area / np.pi))
                        _th = np.linspace(0.0, 2.0 * np.pi, 49)[:-1]
                        profile = [
                            (float(_cc.x + _r_eq * np.cos(t)),
                             float(_cc.y + _r_eq * np.sin(t)))
                            for t in _th]
                        _snapped_hole_r = _r_eq
                except Exception:
                    pass

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
            if _snapped_hole_r is not None:
                sketch_name = f"Sketch Hole (r={_snapped_hole_r:.1f})"
            else:
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
            extra = _draw_profile_with_arcs(extra, profile).extrude(extrude_sign)
            vol_before = abs(float(wp.val().Volume()))
            if is_inner:
                wp = wp.cut(extra)
                op_name = (f"Through Hole (r={_snapped_hole_r:.1f})"
                           if _snapped_hole_r is not None else "Cut Extrude")
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
    # Registry of drilled bores (3D center + radius). The mesh-derived ring
    # detector later re-finds the SAME bores (measured deeper, so slightly
    # larger) — without dedupe every hole was cut small first, then a bigger
    # twin on top (the user's "made them small first then another bigger
    # version"). Concentric ring circles get skipped against this registry.
    _drilled_bores: list = []
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
        # NOISE gate (scan meshes): a hole whose radius is a fraction of a
        # percent of the part is a distortion dimple, not a feature — the
        # motor plate grew "7× r=1.6" and "3× r=1.5" hole patterns from
        # AI-mesh pockmarks on a ⌀300 disc.
        if (cleaned is not None
                and (getattr(cleaned, "metadata", None) or {}).get(
                    "scan_mode")):
            _ext_max = float(np.max(np.asarray(cleaned.bbox[1])
                                    - np.asarray(cleaned.bbox[0])))
            if radius < 0.006 * _ext_max:
                logger.info(
                    "Skipping noise hole r=%.2f (< 0.6%% of the %.0fmm "
                    "part)", radius, _ext_max)
                continue
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

        # Drop centers whose bore ALREADY exists in the current solid
        # (baked into a join/base profile as an interior ring) — drilling
        # them again added duplicate no-op sketch+cut steps to the
        # timeline after the join already showed the hole.
        try:
            from OCP.BRepClass3d import BRepClass3d_SolidClassifier as _HoleCls
            from OCP.gp import gp_Pnt as _HolePnt
            from OCP.TopAbs import TopAbs_IN as _HoleIn
            still_solid = []
            solid_shape = wp.val().wrapped
            for c_each in centers_all:
                c_arr = np.asarray(c_each, dtype=float)
                half = max(float(h_depth) / 2.0 - 1.0, 0.5)
                # Probe SEVERAL stations along the bore: a stepped tab
                # hole's midpoint sits in its relief counterbore (already
                # void) while the bore above is still solid — a single
                # mid probe skipped 3 of the 4 tab holes.
                any_solid = False
                # Tight near-center offsets too: grouped 2D holes carry
                # depth = full part extent, so half-depth probes overshoot
                # a 10mm tab entirely and land in air on both sides.
                for off in (
                    -half, -half / 2, -2.0, -1.0, 0.0, 1.0, 2.0,
                    half / 2, half,
                ):
                    pt = c_arr + (off) * h_dir
                    cls = _HoleCls(
                        solid_shape,
                        _HolePnt(float(pt[0]), float(pt[1]), float(pt[2])),
                        1e-6,
                    )
                    if cls.State() == _HoleIn:
                        any_solid = True
                        break
                if any_solid:
                    still_solid.append(c_each)
            if not still_solid:
                logger.info(
                    "Skipping hole feature r=%.2f: all %d bore(s) already "
                    "exist (baked into profiles)", radius, len(centers_all),
                )
                continue
            if len(still_solid) < len(centers_all):
                logger.info(
                    "Hole r=%.2f: %d of %d bores already baked; drilling "
                    "the rest", radius, len(centers_all) - len(still_solid),
                    len(centers_all),
                )
                centers_all = still_solid
                center = centers_all[0]
        except Exception:
            pass

        for _cc in centers_all:
            _drilled_bores.append((np.asarray(_cc, dtype=float), radius))
        n_holes = len(centers_all)
        _d_sfx = "" if is_through else f", depth {h_depth:.1f}mm"
        if n_holes > 1:
            hole_name = (
                f"{'Through' if is_through else 'Blind'} Hole Pattern "
                f"({n_holes}× r={radius:.1f}{_d_sfx})"
            )
        else:
            hole_name = (f"{'Through' if is_through else 'Blind'} Hole "
                         f"(r={radius:.1f}{_d_sfx})")
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
                    # Draw each sketch circle AT its hole (the wall's
                    # axial midpoint), not at the cut start — a through
                    # cut starts at the solid's far bbox face, which put
                    # tab-hole sketches 85mm away from the tab.
                    sketch_plane_origin = (
                        sketch_plane_origin
                        + (float(np.asarray(c_each) @ h_dir)
                           - float(sketch_plane_origin @ h_dir)) * h_dir
                    )
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
            # AS-BUILT: this hole group was actually drilled (after all
            # recovery + duplicate-skip logic) — record it for exports
            # that rebuild elsewhere (Onshape). The raw recognition tree
            # is NOT equivalent: 2D-ring-recalled holes only exist here.
            if as_built_out is not None:
                as_built_out.setdefault("holes", []).append({
                    "centers": [
                        np.asarray(c, dtype=float).tolist()
                        for c in centers_all
                    ],
                    "radius": float(radius),
                    "axis_dir": h_dir.tolist(),
                    "through": bool(is_through),
                    "depth": float(h_depth),
                })

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

    # --- Tangent holes (axis PERPENDICULAR to the extrude axis) ---
    # The carriage block's side holes, which the broken fit mis-locates. Detected
    # straight from the mesh and cut as STEPPED bores (a wide section + a narrow
    # section). Each gets a SKETCH step before the cut. A redundant cut (hole
    # already there) removes nothing and is skipped, so this is inert when the
    # normal hole path already handled them and never over-cuts.
    if cleaned is not None:
        try:
            tangent_holes = _detect_tangent_holes(cleaned, extrude_dir)
        except Exception:
            tangent_holes = []
        for ti, th in enumerate(tangent_holes):
            try:
                axis = th["axis"]
                us = [k for k in (0, 1, 2) if k != axis]
                e = [0.0, 0.0, 0.0]
                e[axis] = 1.0
                nlv = len(th["levels"])
                # One SKETCH + one CUT per diameter level, so a stepped bore
                # reads as separate sketch->extrude pairs (each diameter drawn
                # then cut) instead of one sketch cutting both diameters.
                cut_any = False
                for li, (r, t0, t1) in enumerate(th["levels"]):
                    if t1 - t0 <= 0:
                        continue
                    o = [0.0, 0.0, 0.0]
                    o[us[0]] = th["cu"]
                    o[us[1]] = th["cv"]
                    o[axis] = t0
                    cutter = (cq.Workplane(cq.Plane(
                        origin=cq.Vector(*o), normal=cq.Vector(*e)))
                        .circle(r).extrude(t1 - t0))
                    vb = abs(float(wp.val().Volume()))
                    wt = wp.cut(cutter)
                    if vb - abs(float(wt.val().Volume())) < 0.5:
                        continue                   # this level removed nothing
                    suffix = (f" Ø{2 * r:.1f} (step {li + 1}/{nlv})"
                              if nlv > 1 else f" Ø{2 * r:.1f}")
                    # SKETCH step: THIS level's circle at its own entry plane.
                    try:
                        seg = []
                        prev = None
                        for kk in range(41):
                            ang = 2.0 * np.pi * kk / 40.0
                            p3 = [0.0, 0.0, 0.0]
                            p3[us[0]] = th["cu"] + r * np.cos(ang)
                            p3[us[1]] = th["cv"] + r * np.sin(ang)
                            p3[axis] = t0
                            if prev is not None:
                                seg.append(prev)
                                seg.append(list(p3))
                            prev = list(p3)
                        vsk, fsk = _tessellate_solid(wp, fast=True)
                        steps.append(ConstructionStep(
                            name=f"Sketch Tangent Hole {ti + 1}{suffix}",
                            tess_vertices=np.asarray(vsk, np.float64),
                            tess_faces=np.asarray(fsk, np.int64),
                            volume=abs(float(wp.val().Volume())),
                            sketch_overlay=np.asarray(seg, np.float64) if seg else None,
                        ))
                    except Exception:
                        logger.debug("tangent-hole sketch overlay failed", exc_info=True)
                    # CUT step: extrude just this diameter.
                    wp = wt
                    cut_any = True
                    v, f = _tessellate_solid(wp, fast=True)
                    steps.append(ConstructionStep(
                        name=f"Tangent Hole {ti + 1}{suffix}",
                        tess_vertices=v, tess_faces=f,
                        volume=abs(float(wp.val().Volume())),
                    ))
                    script_lines.append(
                        f"# Tangent hole {ti + 1} step {li + 1}: dia {2 * r:.2f} "
                        f"on {'XYZ'[axis]} axis")
                if cut_any:
                    logger.info("Cut tangent hole %d axis %s, %d level(s) r=%s",
                                ti + 1, "XYZ"[axis], nlv,
                                [round(r, 1) for r, _, _ in th["levels"]])
            except Exception as _th_exc:
                logger.warning("Failed to cut tangent hole: %s", _th_exc)

    # --- Face pockets: BLIND recesses on the extrude-axis faces ---
    # The cam gear's circular face recess. Detected via material(mid)-material(
    # face) (through-safe: through features are void at mid, so excluded). The
    # recess profile keeps its interior holes (through-holes + raised hubs), so
    # the cut preserves them — no boss. Each gets a SKETCH step then a CUT step.
    if cleaned is not None:
        try:
            face_pockets = _detect_face_pockets(cleaned, extrude_dir)
        except Exception:
            face_pockets = []
        for pi, pk in enumerate(face_pockets):
            try:
                o = [0.0, 0.0, 0.0]
                o[pk["axis"]] = pk["face_t"]

                def _pocket_cutter(sgn):
                    # Recess = exterior to depth, MINUS its interiors (hubs/bosses
                    # solid near the floor + hole regions), so anything solid stays
                    # solid. Through-holes are cut all the way through below.
                    cut = (cq.Workplane(pk["plane_name"], origin=tuple(o))
                           .polyline(pk["exterior_uv"]).close()
                           .extrude(sgn * pk["depth"]))
                    for iv in pk["interiors_uv"]:
                        try:
                            keep = (cq.Workplane(pk["plane_name"], origin=tuple(o))
                                    .polyline(iv).close()
                                    .extrude(sgn * pk["depth"]))
                            cut = cut.cut(keep)         # preserve hub/hole region
                        except Exception:
                            continue
                    return cut

                vb = abs(float(wp.val().Volume()))
                wp_done = None
                used_sgn = pk["sign"]
                for sgn in (pk["sign"], -pk["sign"]):
                    try:
                        wt = wp.cut(_pocket_cutter(sgn))
                        if vb - abs(float(wt.val().Volume())) > 0.5:
                            wp_done = wt
                            used_sgn = sgn
                            break
                    except Exception:
                        continue
                if wp_done is None:
                    continue
                # Cut the recess's THROUGH-HOLES all the way through (the scan's
                # hole detection misses them, so without this they'd remain as
                # raised posts in the recess floor).
                a_lo, a_hi = pk["a_lo"], pk["a_hi"]
                tspan = (a_hi - a_lo) + 2.0
                for iv in pk["holes_uv"]:
                    try:
                        oo = [0.0, 0.0, 0.0]
                        oo[pk["axis"]] = a_lo - 1.0
                        for s2 in (1.0, -1.0):
                            thru = (cq.Workplane(pk["plane_name"], origin=tuple(oo))
                                    .polyline(iv).close().extrude(s2 * tspan))
                            wt2 = wp_done.cut(thru)
                            if abs(float(wp_done.val().Volume())) - abs(float(wt2.val().Volume())) > 0.5:
                                wp_done = wt2
                                break
                    except Exception:
                        continue
                # SKETCH step: the recess outline on the face.
                try:
                    seg = []
                    prev = None
                    pts3 = []
                    us = [k for k in (0, 1, 2) if k != pk["axis"]]
                    for (u, v) in pk["exterior_uv"]:
                        p3 = [0.0, 0.0, 0.0]
                        p3[us[0]] = u
                        p3[us[1]] = v
                        p3[pk["axis"]] = pk["face_t"]
                        pts3.append(list(p3))
                    for q in pts3:
                        if prev is not None:
                            seg.append(prev)
                            seg.append(q)
                        prev = q
                    vsk, fsk = _tessellate_solid(wp, fast=True)
                    steps.append(ConstructionStep(
                        name=f"Sketch Pocket {pi + 1}",
                        tess_vertices=np.asarray(vsk, np.float64),
                        tess_faces=np.asarray(fsk, np.int64),
                        volume=abs(float(wp.val().Volume())),
                        sketch_overlay=np.asarray(seg, np.float64) if seg else None,
                    ))
                except Exception:
                    logger.debug("face-pocket sketch overlay failed", exc_info=True)
                wp = wp_done
                v, f = _tessellate_solid(wp, fast=True)
                steps.append(ConstructionStep(
                    name=f"Pocket {pi + 1} (depth {pk['depth']:.1f}mm)",
                    tess_vertices=v, tess_faces=f,
                    volume=abs(float(wp.val().Volume())),
                ))
                script_lines.append(
                    f"# Face pocket {pi + 1} on {pk['plane_name']}, depth {pk['depth']:.2f}")
                logger.info("Cut face pocket %d on %s depth %.1f "
                            "(%d regions kept, %d holes through)",
                            pi + 1, pk["plane_name"], pk["depth"],
                            len(pk["interiors_uv"]), len(pk["holes_uv"]))
            except Exception as _fp_exc:
                logger.warning("Failed to cut face pocket: %s", _fp_exc)

    # --- Face RING recesses: blind features of a HOLLOW part (the enclosure
    # lid's oval top recess + small blind holes), which the mid-plane pocket
    # detector above cannot see (void at mid). Kept only if the cut removes
    # volume, so a part whose recognition already cut them is untouched. ---
    if cleaned is not None:
        try:
            ring_recesses = _detect_face_ring_recesses(cleaned, extrude_dir)
        except Exception:
            ring_recesses = []
        # GROUP same-face circles with the same (snapped) diameter into ONE
        # pattern feature — one sketch of N circles, one cut — mirroring how a
        # CAD operator models a bolt pattern. Ellipse/outline recesses each get
        # their own sketch + cut pair.
        def _ring_sketch_step(name, arcs=None, lines_uv=None, ax=None, pos=None):
            try:
                vsk, fsk = _tessellate_solid(wp, fast=True)
                # ALWAYS build sketch_overlay line segments: the viewer renders
                # a step named "Sketch..." with NO overlay as a wireframe of
                # the whole tessellation (the "glitch"). Circles from arcs are
                # sampled into segments here so the solid+overlay path is taken.
                loops = list(lines_uv or [])
                if arcs:
                    ax = arcs[0]["ax"]
                    pos = arcs[0]["pos"]
                    for a in arcs:
                        th = np.linspace(a["a0"], a["a1"], 48)
                        loops.append([
                            (float(a["cu"] + a["r"] * np.cos(t)),
                             float(a["cv"] + a["r"] * np.sin(t))) for t in th])
                overlay = None
                if loops:
                    us = [k for k in (0, 1, 2) if k != ax]
                    seg3 = []
                    for loop in loops:
                        for k in range(len(loop)):
                            for pt in (loop[k], loop[(k + 1) % len(loop)]):
                                p3 = [0.0, 0.0, 0.0]
                                p3[us[0]], p3[us[1]] = pt
                                p3[ax] = pos
                                seg3.append(p3)
                    overlay = np.asarray(seg3, np.float64)
                steps.append(ConstructionStep(
                    name=name,
                    tess_vertices=np.asarray(vsk, np.float64),
                    tess_faces=np.asarray(fsk, np.int64),
                    volume=abs(float(wp.val().Volume())),
                    sketch_overlay=overlay, overlay_arcs=arcs,
                ))
            except Exception:
                logger.debug("ring sketch step failed", exc_info=True)

        def _is_drilled(rr, c):
            # Concentric with an already-drilled bore of similar radius?
            # (the ring re-finds the same hole slightly larger — cutting it
            # again produced the small-then-big duplicate steps).
            us2 = [k for k in (0, 1, 2) if k != rr["axis"]]
            for (dc, dr) in _drilled_bores:
                du = float(dc[us2[0]]) - c["cu"]
                dv = float(dc[us2[1]]) - c["cv"]
                if (np.hypot(du, dv) < 0.7 * max(c["r"], dr)
                        and abs(c["r"] - dr) <= 0.35 * max(c["r"], dr)):
                    return True
            return False

        circle_groups: dict = {}
        others = []
        for rr in ring_recesses:
            c = rr.get("circle")
            if c:
                if _is_drilled(rr, c):
                    logger.info("Ring circle r=%.2f already drilled — skip",
                                c["r"])
                    continue
                key = (rr["face_t"], round(c["r"], 2))
                circle_groups.setdefault(key, []).append(rr)
            else:
                others.append(rr)

        for (face_t_g, r_g), grp in circle_groups.items():
            try:
                rr0 = grp[0]
                o = [0.0, 0.0, 0.0]
                o[rr0["axis"]] = face_t_g
                _ring_sketch_step(
                    f"Sketch Hole Pattern ({len(grp)}× ⌀{2 * r_g:.1f}mm)",
                    arcs=[{"ax": rr0["axis"], "pos": face_t_g,
                           "cu": g["circle"]["cu"], "cv": g["circle"]["cv"],
                           "r": g["circle"]["r"], "a0": 0.0,
                           "a1": 2.0 * np.pi} for g in grp])
                vb = abs(float(wp.val().Volume()))
                wp_done = None
                for sgn in (rr0["sign"], -rr0["sign"]):
                    try:
                        wt = wp
                        _exp = 0.0
                        for g in grp:
                            cgeo = g["circle"]
                            cut = (cq.Workplane(rr0["plane_name"], origin=tuple(o))
                                   .center(cgeo["cu"], cgeo["cv"])
                                   .circle(cgeo["r"]).extrude(sgn * g["depth"]))
                            wt = wt.cut(cut)
                            _exp += np.pi * cgeo["r"] ** 2 * g["depth"]
                        # RELATIVE gate: removing only a thin shell of the
                        # expected volume = re-cutting an existing void; skip.
                        if (vb - abs(float(wt.val().Volume()))
                                > max(0.5, 0.1 * _exp)):
                            wp_done = wt
                            break
                    except Exception:
                        continue
                if wp_done is None:
                    continue
                wp = wp_done
                # A ring whose void exits the far side (or merges into the
                # cavity) is a THROUGH hole — say so; "Blind" on a through
                # hole reads as two pocket cuts in the feature tree.
                _is_thru_g = all(g.get("through") for g in grp)
                _kind = "Through Hole" if _is_thru_g else "Blind Hole"
                _dg_sfx = ("" if _is_thru_g
                           else f" (depth {rr0['depth']:.1f}mm)")
                _nm = (f"{_kind} Pattern ⌀{2 * r_g:.1f}mm ×{len(grp)}"
                       f"{_dg_sfx}"
                       if len(grp) > 1
                       else f"{_kind} ⌀{2 * r_g:.1f}mm{_dg_sfx}")
                v, f = _tessellate_solid(wp, fast=True)
                steps.append(ConstructionStep(
                    name=_nm, tess_vertices=v, tess_faces=f,
                    volume=abs(float(wp.val().Volume())),
                ))
                script_lines.append(
                    f"# Hole pattern {len(grp)}x r={r_g:.2f} on {rr0['plane_name']}")
                logger.info("Cut hole pattern %dx r=%.2f on %s",
                            len(grp), r_g, rr0["plane_name"])
            except Exception as _rr_exc:
                logger.warning("Failed hole-pattern cut: %s", _rr_exc)

        for ri, rr in enumerate(others):
            try:
                o = [0.0, 0.0, 0.0]
                o[rr["axis"]] = rr["face_t"]
                el = rr.get("ellipse")
                if el:
                    th = np.linspace(0.0, 2.0 * np.pi, 72)
                    _a = np.radians(el["ang"])
                    _u = (el["cu"] + el["ra"] * np.cos(th) * np.cos(_a)
                          - el["rb"] * np.sin(th) * np.sin(_a))
                    _v = (el["cv"] + el["ra"] * np.cos(th) * np.sin(_a)
                          + el["rb"] * np.sin(th) * np.cos(_a))
                    sketch_loop = list(zip(_u.tolist(), _v.tolist()))
                    _nm_sk = f"Sketch Oval ({2*el['ra']:.0f}×{2*el['rb']:.0f}mm)"
                    _nm = f"Oval Cut (depth {rr['depth']:.1f}mm)"
                elif rr.get("through"):
                    sketch_loop = rr["exterior_uv"]
                    _nm_sk = "Sketch Slot"
                    _nm = "Slot Cut (through)"
                else:
                    sketch_loop = rr["exterior_uv"]
                    _nm_sk = "Sketch Recess"
                    _nm = f"Recess Cut (depth {rr['depth']:.1f}mm)"
                _ring_sketch_step(_nm_sk, lines_uv=[sketch_loop],
                                  ax=rr["axis"], pos=rr["face_t"])
                vb = abs(float(wp.val().Volume()))
                wp_done = None
                for sgn in (rr["sign"], -rr["sign"]):
                    try:
                        base_wp = cq.Workplane(rr["plane_name"], origin=tuple(o))
                        if el:
                            cut = (base_wp.center(el["cu"], el["cv"])
                                   .ellipse(el["ra"], el["rb"],
                                            rotation_angle=el["ang"])
                                   .extrude(sgn * rr["depth"]))
                        else:
                            cut = (base_wp.polyline(rr["exterior_uv"]).close()
                                   .extrude(sgn * rr["depth"]))
                        wt = wp.cut(cut)
                        # RELATIVE gate: removing only a thin shell of the
                        # expected volume = re-cutting a void the base
                        # already carries (the baked slot window); skip
                        # instead of shaving a sliver + emitting a dup step.
                        _exp2 = float(rr.get("area", 0.0)) * rr["depth"]
                        if (vb - abs(float(wt.val().Volume()))
                                > max(0.5, 0.1 * _exp2)):
                            wp_done = wt
                            break
                    except Exception:
                        continue
                if wp_done is None:
                    continue
                wp = wp_done
                v, f = _tessellate_solid(wp, fast=True)
                steps.append(ConstructionStep(
                    name=_nm, tess_vertices=v, tess_faces=f,
                    volume=abs(float(wp.val().Volume())),
                ))
                script_lines.append(
                    f"# Ring recess {ri + 1} on {rr['plane_name']}, "
                    f"depth {rr['depth']:.2f}")
                logger.info("Cut ring recess %d on %s depth %.1f area %.0f",
                            ri + 1, rr["plane_name"], rr["depth"], rr["area"])
            except Exception as _rr_exc:
                logger.warning("Failed to cut ring recess: %s", _rr_exc)

    # --- Standing bosses (the enclosure lid's FEET): small pads standing
    # proud of a face. The base slab spans the full bbox, so the layer
    # BETWEEN the pads must be cut away: silhouette minus the kept pad
    # circles, from the face to the pad height. Volume-gated like the rest. ---
    if cleaned is not None:
        try:
            standing = _detect_standing_bosses(cleaned, extrude_dir)
        except Exception:
            standing = []
        for bi2, sb in enumerate(standing):
            try:
                o = [0.0, 0.0, 0.0]
                o[sb["axis"]] = sb["face_t"]
                _is_panel = bool(sb.get("keep_outline"))
                # Sketch step: pad circles (feet) or the panel outline.
                try:
                    vsk, fsk = _tessellate_solid(wp, fast=True)
                    if _is_panel:
                        us = [k for k in (0, 1, 2) if k != sb["axis"]]
                        seg3 = []
                        loop = sb["keep_outline"]
                        for k in range(len(loop)):
                            for pt in (loop[k], loop[(k + 1) % len(loop)]):
                                p3 = [0.0, 0.0, 0.0]
                                p3[us[0]], p3[us[1]] = pt
                                p3[sb["axis"]] = sb["face_t"]
                                seg3.append(p3)
                        _sk = ConstructionStep(
                            name=f"Sketch Panel (h={sb['height']:.1f}mm)",
                            tess_vertices=np.asarray(vsk, np.float64),
                            tess_faces=np.asarray(fsk, np.int64),
                            volume=abs(float(wp.val().Volume())),
                            sketch_overlay=np.asarray(seg3, np.float64),
                        )
                    else:
                        # sketch_overlay segments too — a "Sketch" step with
                        # no overlay renders as a wireframe of the whole solid.
                        us = [k for k in (0, 1, 2) if k != sb["axis"]]
                        seg3 = []
                        for k in sb["keeps"]:
                            th = np.linspace(0.0, 2.0 * np.pi, 48)
                            loop = [(k["cu"] + k["r"] * np.cos(t),
                                     k["cv"] + k["r"] * np.sin(t)) for t in th]
                            for j in range(len(loop)):
                                for pt in (loop[j], loop[(j + 1) % len(loop)]):
                                    p3 = [0.0, 0.0, 0.0]
                                    p3[us[0]], p3[us[1]] = pt
                                    p3[sb["axis"]] = sb["face_t"]
                                    seg3.append(p3)
                        _sk = ConstructionStep(
                            name=f"Sketch Boss Pattern ({len(sb['keeps'])}× "
                                 f"⌀{2 * sb['keeps'][0]['r']:.1f}mm)",
                            tess_vertices=np.asarray(vsk, np.float64),
                            tess_faces=np.asarray(fsk, np.int64),
                            volume=abs(float(wp.val().Volume())),
                            sketch_overlay=np.asarray(seg3, np.float64),
                            overlay_arcs=[{"ax": sb["axis"], "pos": sb["face_t"],
                                           "cu": k["cu"], "cv": k["cv"],
                                           "r": k["r"], "a0": 0.0,
                                           "a1": 2.0 * np.pi}
                                          for k in sb["keeps"]],
                        )
                    steps.append(_sk)
                except Exception:
                    logger.debug("boss sketch step failed", exc_info=True)
                vb = abs(float(wp.val().Volume()))
                # Cut region = the ENTIRE layer (oversized rectangle), not the
                # buffered face silhouette — the silhouette's margin left a
                # ledge artifact at the corners where it met the cavity. The
                # pads/panel are excluded, so they stand as positive bosses.
                _us = [k for k in (0, 1, 2) if k != sb["axis"]]
                _u0 = float(cleaned.bbox[0][_us[0]]) - 5.0
                _u1 = float(cleaned.bbox[1][_us[0]]) + 5.0
                _v0 = float(cleaned.bbox[0][_us[1]]) - 5.0
                _v1 = float(cleaned.bbox[1][_us[1]]) + 5.0
                _rect = [(_u0, _v0), (_u1, _v0), (_u1, _v1), (_u0, _v1)]
                wp_done = None
                for sgn in (sb["sign"], -sb["sign"]):
                    try:
                        cut = (cq.Workplane(sb["plane_name"], origin=tuple(o))
                               .polyline(_rect).close()
                               .extrude(sgn * sb["height"]))
                        if _is_panel:
                            keep = (cq.Workplane(sb["plane_name"],
                                                 origin=tuple(o))
                                    .polyline(sb["keep_outline"]).close()
                                    .extrude(sgn * sb["height"]))
                            cut = cut.cut(keep)
                        for k in sb["keeps"]:
                            keep = (cq.Workplane(sb["plane_name"], origin=tuple(o))
                                    .center(k["cu"], k["cv"]).circle(k["r"])
                                    .extrude(sgn * sb["height"]))
                            cut = cut.cut(keep)
                        wt = wp.cut(cut)
                        # Meaningful only if the layer was actually filled by
                        # the base slab (removes ~expected). A solid that
                        # already models the pads (AI1's screws) loses almost
                        # nothing here — skip rather than clip their domes.
                        _removed = vb - abs(float(wt.val().Volume()))
                        if _removed > max(0.5, 0.4 * sb.get("expected_cut", 0.0)):
                            wp_done = wt
                            break
                    except Exception:
                        continue
                if wp_done is None:
                    continue
                wp = wp_done
                v, f = _tessellate_solid(wp, fast=True)
                if _is_panel:
                    _bnm = f"Panel Boss (h={sb['height']:.1f}mm)"
                else:
                    _r0 = sb["keeps"][0]["r"] if sb["keeps"] else 0.0
                    _bnm = (f"Boss Pattern ⌀{2 * _r0:.1f}mm x{len(sb['keeps'])} "
                            f"(h={sb['height']:.1f}mm)")
                steps.append(ConstructionStep(
                    name=_bnm, tess_vertices=v, tess_faces=f,
                    volume=abs(float(wp.val().Volume())),
                ))
                script_lines.append(
                    f"# Standing bosses on {sb['plane_name']}: "
                    f"{len(sb['keeps'])} pads h={sb['height']:.2f}")
                logger.info("Cut around %d standing boss(es) on %s h=%.1f",
                            len(sb["keeps"]), sb["plane_name"], sb["height"])
            except Exception as _sb_exc:
                logger.warning("Failed standing-boss cut: %s", _sb_exc)

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
    wp = _draw_profile_with_arcs(wp, profiles[0]).extrude(depth)

    # Union additional profiles (leg extensions, bosses, etc.)
    for i, profile in enumerate(profiles[1:], 1):
        try:
            extra = cq.Workplane(plane_name, origin=tuple(origin_3d))
            extra = _draw_profile_with_arcs(extra, profile).extrude(depth)
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


def _draw_profile_with_holes(wp2d, outer: list, inners: list,
                             corners: list = None, fillets: list = None):
    """Draw the outer boundary plus inner cutout wires on one workplane.

    Extruding the result produces the cross-section WITH holes in a single
    operation (CadQuery's even-odd rule turns nested wires into a washer/
    windowed solid) — the lowest-step base for a tube (annular sketch) or
    any prismatic part whose through-cutouts belong to the profile, rather
    than extrude-then-cut.

    `corners`/`fillets` (if given) drive exact tangent fillets on the OUTER
    boundary from the carried parametric data (matches the Onshape sketch).
    """
    wp = _draw_profile_with_arcs(wp2d, outer, corners, fillets)
    for inner in inners:
        wp = _draw_profile_with_arcs(wp, inner)
    return wp


def _detect_profile_arc_runs(pts: list) -> list[tuple[int, int]]:
    """Find runs of polyline vertices that polygonize an outline arc.

    The profile extractor samples the mesh outline, so a designed
    corner radius arrives as N short segments turning a consistent
    ~10-20 degrees each. Returns [(start_vertex, end_vertex), ...]
    runs (inclusive, circular indexing) whose vertices lie on one arc:
    consistent turn sign, bounded per-vertex turn, short segments, and
    consistent implied radius.
    """
    P = np.asarray(pts, dtype=float)
    n = len(P)
    if n < 6:
        return []
    seg = np.roll(P, -1, axis=0) - P
    L = np.linalg.norm(seg, axis=1)
    if np.any(L < 1e-9):
        return []
    D = seg / L[:, None]
    Dp = np.roll(D, 1, axis=0)
    cosv = np.clip(np.einsum("ij,ij->i", Dp, D), -1.0, 1.0)
    crossv = Dp[:, 0] * D[:, 1] - Dp[:, 1] * D[:, 0]
    turn = np.arctan2(crossv, cosv)
    Lp = np.roll(L, 1)
    is_arc = (
        (np.abs(turn) > np.radians(4.0))
        & (np.abs(turn) < np.radians(40.0))
        & (L < 6.0) & (Lp < 6.0)
    )
    # implied radius at each arc vertex
    with np.errstate(divide="ignore", invalid="ignore"):
        r_v = ((L + Lp) / 2.0) / (2.0 * np.sin(np.abs(turn) / 2.0) + 1e-12)

    runs: list[tuple[int, int]] = []
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start] or not is_arc[start]:
            continue
        # grow forward (circular) while same turn sign + similar radius
        run = [start]
        visited[start] = True
        j = (start + 1) % n
        while (
            not visited[j] and is_arc[j]
            and np.sign(turn[j]) == np.sign(turn[start])
            and 0.5 < r_v[j] / (r_v[start] + 1e-12) < 2.0
            and len(run) < n
        ):
            run.append(j)
            visited[j] = True
            j = (j + 1) % n
        # grow backward
        j = (start - 1) % n
        while (
            not visited[j] and is_arc[j]
            and np.sign(turn[j]) == np.sign(turn[start])
            and 0.5 < r_v[j] / (r_v[start] + 1e-12) < 2.0
            and len(run) < n
        ):
            run.insert(0, j)
            visited[j] = True
            j = (j - 1) % n
        if len(run) >= 3:
            runs.append((run[0], run[-1]))
    return runs


def _sharp_polygon_with_fillets(pts: list):
    """Collapse polygonized corner arcs to SHARP corners + radii.

    Returns (sharp_pts, fillets) where fillets maps sharp-vertex index ->
    radius, or (None, None) when the outline has no usable corner-arc
    structure. Feeding these to OCC MakeFillet2d yields EXACTLY tangent
    corner arcs; mesh-fitted threePointArc reconstructions leave
    micro-kinks at arc/straight junctions that BRepFilletAPI refuses to
    roll over (RackMount's 13-edge side-plate rim creases).
    """
    runs = _detect_profile_arc_runs(pts)
    P = np.asarray(pts, dtype=float)
    n = len(P)
    if n < 6:
        return None, None
    in_run_member = set()
    run_info = []
    for (a, b) in runs:
        idxs = []
        j = a
        while True:
            idxs.append(j)
            if j == b:
                break
            j = (j + 1) % n
        runP = P[idxs]
        A = np.c_[2 * runP[:, 0], 2 * runP[:, 1], np.ones(len(runP))]
        try:
            sol, *_ = np.linalg.lstsq(A, (runP**2).sum(axis=1), rcond=None)
        except np.linalg.LinAlgError:
            return None, None
        cx, cy = float(sol[0]), float(sol[1])
        cr = float(np.sqrt(max(sol[2] + cx**2 + cy**2, 0.0)))
        if cr < 0.3 or cr > 60.0:
            return None, None
        prev_i = (idxs[0] - 1) % n
        next_i = (idxs[-1] + 1) % n
        p0, p1 = P[prev_i], P[idxs[0]]
        q0, q1 = P[idxs[-1]], P[next_i]
        d1 = p1 - p0
        d2 = q1 - q0
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            return None, None
        t = ((q0[0] - p0[0]) * d2[1] - (q0[1] - p0[1]) * d2[0]) / denom
        corner = p0 + t * d1
        if np.hypot(corner[0] - cx, corner[1] - cy) > 3.5 * cr:
            return None, None
        run_info.append((idxs, corner, cr))
        in_run_member.update(idxs)

    # SECOND PASS — tight corner chains. Small-radius corners (r~2 at
    # ~1.5mm sampling) turn >40 deg per vertex, past the arc detector's
    # cap, and survive as raw 2-3 segment fragments with kinked
    # junctions — exactly the sliver pairs that break the rim fillet.
    # Detect sign-consistent chains of sharp short turns and treat them
    # as fillet corners too.
    seg = np.roll(P, -1, axis=0) - P
    L = np.linalg.norm(seg, axis=1)
    D = seg / (L[:, None] + 1e-12)
    Dp = np.roll(D, 1, axis=0)
    cosv = np.clip(np.einsum("ij,ij->i", Dp, D), -1.0, 1.0)
    crossv = Dp[:, 0] * D[:, 1] - Dp[:, 1] * D[:, 0]
    turn = np.arctan2(crossv, cosv)
    tight = (
        (np.abs(turn) >= np.radians(25.0))
        & (np.abs(turn) <= np.radians(80.0))
        & (L < 4.0) & (np.roll(L, 1) < 4.0)
    )
    visited2 = set(in_run_member)
    for (a, b) in runs:
        visited2.add(a)
    k = 0
    while k < n:
        if k in visited2 or not tight[k]:
            k += 1
            continue
        chain = [k]
        j = (k + 1) % n
        while (
            j not in visited2 and tight[j]
            and np.sign(turn[j]) == np.sign(turn[k])
            and len(chain) < 4
        ):
            chain.append(j)
            j = (j + 1) % n
        total_turn = float(np.abs(turn[list(chain)]).sum())
        if total_turn < np.radians(35.0) or total_turn > np.radians(150.0):
            k += 1
            continue
        prev_i = (chain[0] - 1) % n
        next_i = (chain[-1] + 1) % n
        p0, p1 = P[prev_i], P[chain[0]]
        q0, q1 = P[chain[-1]], P[next_i]
        d1 = p1 - p0
        d2 = q1 - q0
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            k += 1
            continue
        t = ((q0[0] - p0[0]) * d2[1] - (q0[1] - p0[1]) * d2[0]) / denom
        corner = p0 + t * d1
        # Tangent-circle radius through the chain's first point: the
        # clipped vertex approximates the tangency point, which sits
        # r/tan(theta/2) before the sharp corner (theta = total turn).
        d_clip = float(np.linalg.norm(P[chain[0]] - corner))
        r_est = d_clip * np.tan(total_turn / 2.0)
        if not (0.3 <= r_est <= 30.0):
            k += 1
            continue
        run_info.append((list(chain), corner, float(r_est)))
        in_run_member.update(chain)
        for ci in chain:
            visited2.add(ci)
        k += 1

    if not run_info:
        return None, None

    sharp_pts = []
    fillets = {}
    handled = set()
    i = 0
    while i < n:
        if i in in_run_member:
            run = next(r for r in run_info if i in r[0])
            key = id(run)
            if key not in handled:
                handled.add(key)
                fillets[len(sharp_pts)] = float(run[2])
                sharp_pts.append((float(run[1][0]), float(run[1][1])))
            i += 1
            continue
        sharp_pts.append((float(P[i][0]), float(P[i][1])))
        i += 1
    if len(sharp_pts) < 3 or not fillets:
        return None, None
    return sharp_pts, fillets


def _draw_exact_corner_arcs(wp2d, sharp_pts, fillets):
    """Draw the profile via OCC MakeFillet2d, re-expressed as NATIVE
    CadQuery moveTo/lineTo/threePointArc commands.

    MakeFillet2d generates corner arcs EXACTLY tangent to their flanking
    straights. We then walk the resulting wire IN ORDER and re-draw it
    with stock workplane primitives, sampling each edge's exact curve
    points — so the workplane's pending-wire bookkeeping stays
    completely stock (a previous attempt that pushed the OCC wire onto
    the stack corrupted downstream step tessellation).
    """
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet2d
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.TopAbs import TopAbs_REVERSED, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    plane = wp2d.plane
    tmp = cq.Workplane(plane).polyline(sharp_pts).close()
    wire = tmp.ctx.pendingWires[-1]
    face_builder = BRepBuilderAPI_MakeFace(wire.wrapped, True)
    if not face_builder.IsDone():
        raise RuntimeError("face build failed")
    face = face_builder.Face()
    mk = BRepFilletAPI_MakeFillet2d(face)
    targets = []
    for vi, rad in fillets.items():
        loc3 = plane.toWorldCoords(sharp_pts[vi])
        targets.append((np.array([loc3.x, loc3.y, loc3.z]), rad))
    seen = set()
    n_added = 0
    exp = TopExp_Explorer(face, TopAbs_VERTEX)
    while exp.More():
        v = TopoDS.Vertex_s(exp.Current())
        pnt = BRep_Tool.Pnt_s(v)
        key = (round(pnt.X(), 4), round(pnt.Y(), 4), round(pnt.Z(), 4))
        if key not in seen:
            seen.add(key)
            wpt = np.array([pnt.X(), pnt.Y(), pnt.Z()])
            for tpos, rad in targets:
                if np.linalg.norm(wpt - tpos) < 0.2:
                    mk.AddFillet(v, rad)
                    n_added += 1
                    break
        exp.Next()
    if n_added == 0:
        raise RuntimeError("no fillet vertices matched")
    mk.Build()
    if not mk.IsDone():
        raise RuntimeError("MakeFillet2d failed")
    out_face = TopoDS.Face_s(mk.Shape())
    out_wire = BRepTools.OuterWire_s(out_face)

    # Walk the wire in order, sampling exact curve points
    def _local(p3):
        lp = plane.toLocalCoords(cq.Vector(p3.X(), p3.Y(), p3.Z()))
        return (float(lp.x), float(lp.y))

    from OCP.BRepAdaptor import BRepAdaptor_Curve

    segments = []  # ("line", end) | ("arc", mid, end)
    wexp = BRepTools_WireExplorer(out_wire)
    start_pt = None
    while wexp.More():
        edge = wexp.Current()
        crv = BRepAdaptor_Curve(edge)
        ufirst, ulast = crv.FirstParameter(), crv.LastParameter()
        if edge.Orientation() == TopAbs_REVERSED:
            ufirst, ulast = ulast, ufirst
        p_first = crv.Value(ufirst)
        p_mid = crv.Value(0.5 * (ufirst + ulast))
        p_last = crv.Value(ulast)
        if start_pt is None:
            start_pt = _local(p_first)
        chord = np.hypot(
            p_last.X() - p_first.X(), p_last.Y() - p_first.Y(),
        ) + abs(p_last.Z() - p_first.Z())
        mid_dev = np.linalg.norm(
            np.array([p_mid.X(), p_mid.Y(), p_mid.Z()])
            - 0.5 * (
                np.array([p_first.X(), p_first.Y(), p_first.Z()])
                + np.array([p_last.X(), p_last.Y(), p_last.Z()])
            )
        )
        if mid_dev < 1e-6 * max(chord, 1.0) + 1e-7:
            segments.append(("line", _local(p_last)))
        else:
            segments.append(("arc", _local(p_mid), _local(p_last)))
        wexp.Next()
    if start_pt is None or len(segments) < 3:
        raise RuntimeError("wire walk produced no segments")

    wp = wp2d.moveTo(*start_pt)
    for seg in segments[:-1]:
        if seg[0] == "line":
            wp = wp.lineTo(*seg[1])
        else:
            wp = wp.threePointArc(seg[1], seg[2])
    # Final segment returns to start: draw it then close.
    last = segments[-1]
    if last[0] == "arc":
        wp = wp.threePointArc(last[1], last[2])
    return wp.close()


def _sketch_overlay_arcs(corners, fillets, ax, plane_pos):
    """Sketch-overlay wireframe for a rounded outline as STRAIGHT segments +
    true ARC descriptors (no chord polyline on the curves).

    Mirrors onshape/feature_builder._analytic_tangent_arcs EXACTLY (same
    tangent-length / centre-on-bisector math) so the viewer overlay, the
    CadQuery solid, and the Onshape sketch are one geometry, not three
    independent re-detections.

    corners:   sharp-corner polygon, list of (u, v) in mm (sketch-plane coords).
    fillets:   per-corner radii (mm) aligned with corners; 0/None = sharp.
    ax:        out-of-plane axis index (0/1/2). The two in-plane axes are the
               remaining indices in ascending order -> (u_axis, v_axis).
    plane_pos: constant value along `ax`.

    Returns (line_pts, arcs):
      line_pts: list of 3D points, consecutive PAIRS = one straight segment
                (same flat format as ConstructionStep.sketch_overlay).
      arcs:     list of {"ax","pos","cu","cv","r","a0","a1"} — a circular arc
                in the sketch plane swept CCW from a0 to a1 about (cu, cv).
    """
    P = np.asarray(corners, dtype=float)
    n = len(P)
    if n < 3:
        return [], []
    fdict = {}
    if fillets:
        for i, r in enumerate(fillets):
            if r and float(r) > 0.05:
                fdict[i] = float(r)
    if not fdict:
        # nothing rounds — caller falls back to plain chords
        return [], []
    u_axis, v_axis = [a for a in range(3) if a != ax]

    def _to3(p2):
        q = [0.0, 0.0, 0.0]
        q[u_axis] = float(p2[0])
        q[v_axis] = float(p2[1])
        q[ax] = float(plane_pos)
        return q

    node_pts = []  # (entry_pt, exit_pt, arc or None)
    for i in range(n):
        V = P[i]
        r = fdict.get(i)
        if not r:
            node_pts.append((V, V, None))
            continue
        a = P[(i - 1) % n] - V
        b = P[(i + 1) % n] - V
        la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if la < 1e-9 or lb < 1e-9:
            node_pts.append((V, V, None))
            continue
        a, b = a / la, b / lb
        phi = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
        if phi < np.radians(10) or phi > np.radians(170):
            node_pts.append((V, V, None))
            continue
        t = r / np.tan(phi / 2.0)
        if t > 0.6 * min(la, lb):
            node_pts.append((V, V, None))
            continue
        T1 = V + a * t
        T2 = V + b * t
        C = V + (a + b) / np.linalg.norm(a + b) * (r / np.sin(phi / 2.0))
        ang1 = float(np.arctan2(T1[1] - C[1], T1[0] - C[0]))
        ang2 = float(np.arctan2(T2[1] - C[1], T2[0] - C[0]))
        sweep = (ang2 - ang1) % (2.0 * np.pi)
        if sweep > np.pi:
            ang1, ang2 = ang2, ang1
        if ang2 < ang1:
            ang2 += 2.0 * np.pi
        node_pts.append((T1, T2, (C, float(r), ang1, ang2)))

    line_pts = []
    arcs = []
    for i in range(n):
        _, exit_pt, _ = node_pts[i]
        entry_next, _, arc_next = node_pts[(i + 1) % n]
        d = entry_next - exit_pt
        seg_len = float(np.linalg.norm(d))
        if seg_len > 1e-9:
            line_pts.append(_to3(exit_pt))
            line_pts.append(_to3(entry_next))
        if arc_next is not None:
            C, r, a0, a1 = arc_next
            arcs.append({
                "ax": int(ax), "pos": float(plane_pos),
                "cu": float(C[0]), "cv": float(C[1]),
                "r": float(r), "a0": float(a0), "a1": float(a1),
            })
    return line_pts, arcs


def _overlay_for_ring(ring_pts, ax, plane_pos, corners=None, fillets=None):
    """(line_pts, arcs) for ONE closed sketch ring. With parametric
    corners+fillets it returns straights + true arcs; otherwise it chords the
    ring's own points (holes/circles, already finely sampled). Same flat
    point-pair format as ConstructionStep.sketch_overlay throughout."""
    if corners is not None and len(corners) >= 3 and fillets:
        try:
            lp, arcs = _sketch_overlay_arcs(corners, fillets, ax, plane_pos)
            if arcs:
                return lp, arcs
        except Exception:
            logger.debug("overlay arc build failed; chording ring", exc_info=True)
    u_axis, v_axis = [a for a in range(3) if a != ax]
    line_pts = []
    n = len(ring_pts)
    for i in range(n):
        for (uu, vv) in (ring_pts[i], ring_pts[(i + 1) % n]):
            q = [0.0, 0.0, 0.0]
            q[u_axis] = float(uu)
            q[v_axis] = float(vv)
            q[ax] = float(plane_pos)
            line_pts.append(q)
    return line_pts, []


def _draw_profile_with_arcs(wp2d, pts: list, corners: list = None,
                            fillets: list = None):
    """Draw a closed profile, rebuilding polygonized corner arcs as true
    arcs (threePointArc) so the solid gets ONE arc edge per outline
    corner instead of N sliver segments. Sliver chains made OCC's
    fillet of perimeter-wrap creases fail with "command not done";
    real arcs roll cleanly and match the original design intent.

    Falls back to a plain polyline on any irregularity.

    When the PARAMETRIC representation is supplied (`corners` = sharp polygon,
    `fillets` = per-corner radii aligned with corners), draw exact OCC tangent
    fillets from it — the SAME source the Onshape export uses, so the viewer's
    solid and the Onshape sketch are identical, not independently re-detected.
    """
    if corners and len(corners) >= 3 and fillets:
        _fdict = {i: float(r) for i, r in enumerate(fillets) if r and r > 0.05}
        if _fdict:
            try:
                return _draw_exact_corner_arcs(
                    wp2d, [(float(c[0]), float(c[1])) for c in corners], _fdict)
            except Exception:
                logger.debug("carried-arc draw failed; re-detecting", exc_info=True)
    try:
        # Preferred: EXACT tangent corner arcs (OCC MakeFillet2d on a
        # sharp polygon, re-drawn with native workplane commands).
        # Mesh-fitted arcs leave micro-kinks that block downstream rim
        # fillets (RackMount's 13-edge creases).
        try:
            sharp_pts, corner_fillets = _sharp_polygon_with_fillets(pts)
            if sharp_pts is not None:
                return _draw_exact_corner_arcs(
                    wp2d, sharp_pts, corner_fillets,
                )
        except Exception as exc:
            logger.debug("Exact corner arcs failed (%s); falling back", exc)

        runs = _detect_profile_arc_runs(pts)
        if not runs:
            # Finely sampled circles turn <4 degrees per vertex and form
            # no arc runs at all — check for a circle before punting.
            arr0 = np.asarray(pts, dtype=float)
            if len(arr0) >= 12:
                A0 = np.c_[2 * arr0[:, 0], 2 * arr0[:, 1], np.ones(len(arr0))]
                sol0, *_ = np.linalg.lstsq(
                    A0, (arr0**2).sum(axis=1), rcond=None,
                )
                cx0, cy0 = float(sol0[0]), float(sol0[1])
                cr0 = float(np.sqrt(max(sol0[2] + cx0**2 + cy0**2, 0.0)))
                resid0 = np.abs(
                    np.hypot(arr0[:, 0] - cx0, arr0[:, 1] - cy0) - cr0
                )
                if cr0 > 0.5 and float(resid0.max()) < max(0.15, 0.02 * cr0):
                    return wp2d.moveTo(cx0, cy0).circle(cr0)
            return wp2d.polyline(pts).close()
        P = [tuple(map(float, p)) for p in pts]
        n = len(P)
        in_run = {}
        for (a, b) in runs:
            # collect circular vertex list of this run
            idxs = []
            j = a
            while True:
                idxs.append(j)
                if j == b:
                    break
                j = (j + 1) % n
            in_run[a] = idxs

        run_member = set()
        for idxs in in_run.values():
            run_member.update(idxs[1:])  # all but the entry vertex

        # start at a vertex that is not inside any run
        start = None
        for k in range(n):
            if k not in run_member and k not in in_run:
                start = k
                break
        if start is None:
            # Every vertex is on an arc — circle-like profile. Fit the
            # circle and draw a TRUE circle edge: punting to polyline
            # made a 62-sliver bore ring that the fillet matcher could
            # not snap to and OCC could not roll.
            arr = np.asarray(P, dtype=float)
            A = np.c_[2 * arr[:, 0], 2 * arr[:, 1], np.ones(len(arr))]
            sol, *_ = np.linalg.lstsq(A, (arr**2).sum(axis=1), rcond=None)
            cx, cy = float(sol[0]), float(sol[1])
            cr = float(np.sqrt(max(sol[2] + cx**2 + cy**2, 0.0)))
            resid = np.abs(np.hypot(arr[:, 0] - cx, arr[:, 1] - cy) - cr)
            if cr > 0.5 and float(resid.max()) < max(0.15, 0.02 * cr):
                return wp2d.moveTo(cx, cy).circle(cr)
            return wp2d.polyline(pts).close()

        wp = wp2d.moveTo(*P[start])
        i = (start + 1) % n
        guard = 0
        while i != start and guard < 2 * n:
            guard += 1
            if i in in_run:
                idxs = in_run[i]
                a, b = idxs[0], idxs[-1]
                mid = idxs[len(idxs) // 2]
                wp = wp.lineTo(*P[a])
                wp = wp.threePointArc(P[mid], P[b])
                i = (b + 1) % n
            else:
                wp = wp.lineTo(*P[i])
                i = (i + 1) % n
        return wp.close()
    except Exception as exc:
        logger.debug("Arc-profile draw failed (%s) — plain polyline", exc)
        return wp2d.polyline(pts).close()


def _select_face_wire_edges(
    planar_faces, edge_key_to_idx, solid_edges, ctree, snap, band_normal, verts,
):
    """Face-driven crease selection for perimeter-wrap fillets.

    A wrap belongs to ONE face of the solid: the planar face whose
    normal is the band normal and whose plane coincides with one side
    of the band's vertex cloud. That face's outline wire IS the crease
    chain (spines + corner arcs, corner-to-corner). Selecting from the
    owning face's wire — instead of proximity-matching every edge on
    the solid — keeps the two parallel wraps of a plate (inner/outer
    face) on disjoint edge sets, so conflict resolution can't strip
    one of them down to fragments.

    Returns [(edge_idx, frac_on, mean_dist, edge), ...] like the
    proximity candidates, or [] when no owning face is found (caller
    falls back to proximity selection).
    """
    t = verts @ band_normal
    lo, hi = float(t.min()), float(t.max())

    best_edges: list = []
    best_score = 0.0
    for f in planar_faces:
        if abs(float(f["normal"] @ band_normal)) < 0.95:
            continue
        plane_t = float(f["center"] @ band_normal)
        if min(abs(plane_t - lo), abs(plane_t - hi)) > 1.0:
            continue
        edges_here: list = []
        total_len = 0.0
        for ek in f["wire_edge_keys"]:
            ei = edge_key_to_idx.get(ek)
            if ei is None:
                continue
            e = solid_edges[ei]
            # Sub-millimetre slivers (union-seam artifacts at chain
            # endpoints) poison BRepFilletAPI for the whole chain —
            # and rolling them adds nothing visible. Skip them.
            if e["len"] < 1.0:
                continue
            d, _ = ctree.query(e["samples"])
            frac = float(np.mean(d < snap))
            if frac < 0.5:
                continue
            if e["type"] == "LINE" and e["len"] > 4.0 * snap and \
                    _edge_angular_coverage(e, verts, snap * 1.5) < 45.0:
                continue
            edges_here.append((ei, frac, float(d.mean()), e))
            total_len += e["len"]
        if len(edges_here) >= 2 and total_len > best_score:
            best_score = total_len
            best_edges = edges_here
    return best_edges


def _edge_angular_coverage(e: dict, cluster_verts: np.ndarray, window: float) -> float:
    """Angular span (degrees) of nearby band vertices around an edge.

    A fillet band WRAPS the sharp edge it replaces — seen from the edge
    line, its vertices span ~the blend angle (90 deg for a plain
    fillet). A parallel bystander edge (e.g. a pocket's rim 2mm above
    the floor-fillet band) sees the band only from one side: span ~0.
    """
    d = e["end"] - e["start"]
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return 360.0  # closed curve (circle/ellipse): skip the test
    d = d / n
    rel = cluster_verts - e["start"]
    t = rel @ d
    on_span = (t > -window) & (t < n + window)
    if not np.any(on_span):
        return 0.0
    perp = rel[on_span] - np.outer(t[on_span], d)
    dist = np.linalg.norm(perp, axis=1)
    near = (dist > 1e-6) & (dist < window)
    if near.sum() < 4:
        return 0.0
    perp = perp[near]
    arb = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = arb - d * float(arb @ d)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(d, u)
    ang = np.sort(np.arctan2(perp @ v, perp @ u))
    gap = max(float(np.diff(ang).max()) if len(ang) > 1 else 0.0,
              2.0 * np.pi - (float(ang[-1]) - float(ang[0])))
    return float(np.degrees(2.0 * np.pi - gap))


def _resolve_edges_on_current_solid(wp, bucket_keys, cluster_verts, snap):
    """Find this bucket's edges on the CURRENT (possibly refilleted) solid.

    Exact geometric-identity matches first; edges that a neighboring
    fillet trimmed (midpoint/length changed) are recovered by proximity
    to the cluster's vertex cloud. Returns live Edge objects.
    """
    from scipy.spatial import cKDTree as _cKDTree

    try:
        current = wp.edges().vals()
    except Exception:
        return []
    key_to_edge = {}
    for e in current:
        k = _edge_identity_key(e)
        if k:
            key_to_edge[k] = e

    resolved = []
    missing = []
    for k in bucket_keys:
        e = key_to_edge.get(k)
        if e is not None:
            resolved.append(e)
        else:
            missing.append(k)
    if not missing or len(cluster_verts) == 0:
        return resolved

    # Proximity recovery for trimmed edges: a current edge qualifies if
    # most of its samples hug the cluster cloud and it isn't already in.
    ctree = _cKDTree(cluster_verts)
    have = {id(e) for e in resolved}
    for e in current:
        if id(e) in have:
            continue
        try:
            samples = np.array([
                [(_p := e.positionAt(t)).x, _p.y, _p.z]
                for t in np.linspace(0.0, 1.0, 9)
            ], dtype=float)
        except Exception:
            continue
        d, _ = ctree.query(samples)
        if float(np.mean(d < snap)) >= 0.5 and float(d.mean()) < snap:
            resolved.append(e)
            have.add(id(e))
    return resolved


def _edge_identity_key(edge) -> tuple:
    """Geometric identity key for an OCC edge: rounded midpoint + length."""
    try:
        mp = edge.positionAt(0.5)
        ln = float(edge.Length())
        return (round(mp.x, 2), round(mp.y, 2), round(mp.z, 2), round(ln, 2))
    except Exception:
        return ()


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
    as_built_out: dict | None = None,
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

    # Identity map from edge geometry to index, plus planar faces with
    # their outline-wire edges — drives the face-wire (feature-driven)
    # crease selection.
    edge_key_to_idx: dict[tuple, int] = {}
    for ei, e in enumerate(solid_edges):
        k = _edge_identity_key(e["obj"])
        if k:
            edge_key_to_idx[k] = ei
    planar_faces: list[dict] = []
    try:
        for f in wp.faces().vals():
            try:
                if f.geomType() != "PLANE":
                    continue
                n = f.normalAt()
                c = f.Center()
                wire_keys = []
                for we in f.outerWire().Edges():
                    k = _edge_identity_key(we)
                    if k:
                        wire_keys.append(k)
                planar_faces.append({
                    "normal": np.array([n.x, n.y, n.z], dtype=float),
                    "center": np.array([c.x, c.y, c.z], dtype=float),
                    "wire_edge_keys": wire_keys,
                })
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Face enumeration failed: %s", exc)
    logger.info("Face enumeration: %d planar faces", len(planar_faces))

    # --- Category-aware vertex-proximity matching ---
    SNAP_TOL_BASE = 3.5
    MIN_SPINE_LEN = 3.0
    MIN_EDGE_FRAC = 0.5
    matched_by_cluster: dict[int, list[int]] = {}
    cluster_radius: dict[int, float] = {}
    cluster_kind: dict[int, str] = {}
    for ci, c in enumerate(clusters):
        verts = np.asarray(c.get("vertices"), dtype=float)
        if verts.ndim != 2 or len(verts) == 0:
            continue
        category = c.get("category", "curve_edge")
        topo_r = float(c.get("radius", 0.0) or 0.0)
        # The cluster radius is now the median of per-record curvature
        # measurements (adjacent-face normal steps in
        # peel_via_topology._face_pair_curvature_radius) — trust it
        # when sane. The old min(topo_r, smallest_PCA_extent) clamp
        # measured the band's thickness extent, not its radius, and
        # systematically under-reported (1.4 for a true 2.0mm fillet).
        cv = verts - verts.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(cv, full_matrices=False)
            smallest_ext = float((cv @ vh[2]).max() - (cv @ vh[2]).min())
        except Exception:
            smallest_ext = topo_r
        if 0.2 < topo_r < 30.0:
            robust_r = topo_r
        else:
            robust_r = max(0.3, smallest_ext)
        snap = max(SNAP_TOL_BASE, 0.6 * robust_r + 2.0)

        ctree = _cKDTree(verts)
        cand: list[tuple] = []  # (edge_idx, frac_on, mean_dist, edge)
        for ei, e in enumerate(solid_edges):
            d, _ = ctree.query(e["samples"])
            frac = float(np.mean(d < snap))
            if frac < MIN_EDGE_FRAC:
                continue
            # The band must WRAP this edge, not just run past it — a
            # pocket's rim edge sits within snap of the floor-fillet
            # band but sees it from one side only.
            if e["type"] == "LINE" and _edge_angular_coverage(e, verts, snap * 1.5) < 45.0:
                continue
            cand.append((ei, frac, float(d.mean()), e))

        if category == "hole_edge":
            circ = [x for x in cand if x[3]["type"] in ("CIRCLE", "ELLIPSE")]
            circ.sort(key=lambda x: x[2])
            chosen = circ[:1]
        else:
            # Primary: face-wire (feature-driven) selection — the wrap's
            # owning face contributes its outline chain, corner-to-corner,
            # disjoint from the parallel wrap on the opposite plate face.
            chosen = _select_face_wire_edges(
                planar_faces, edge_key_to_idx, solid_edges,
                ctree, snap, vh[2], verts,
            )
            if chosen:
                logger.info(
                    "  cluster %d: face-wire selection -> %d edge(s)",
                    int(c.get("group_id", ci)), len(chosen),
                )
            else:
                # Fallback: proximity crease selection (long spines +
                # corner-arc bridges).
                chosen = _select_crease_edges(
                    cand, snap, vh[2], MIN_SPINE_LEN,
                )
        if not chosen:
            continue
        # OVER-CLAIM guard: a real fillet cluster owns a handful of edges. A
        # cluster claiming dozens-to-hundreds (AI5: one r=8.3 cluster matched
        # 205 of 7,696 edges on a complex solid) is a mis-match, and asking
        # OCC to roll a fillet across it SEGFAULTS the build. Skip it.
        if len(chosen) > 64:
            logger.warning(
                "  cluster %d claims %d edges — mis-match, skipping "
                "(OCC fillet on that many edges segfaults)",
                int(c.get("group_id", ci)), len(chosen))
            continue
        matched_by_cluster[ci] = [x[0] for x in chosen]
        cluster_radius[ci] = robust_r
        cluster_kind[ci] = str(c.get("kind", "fillet"))
        logger.info(
            "  cluster %d (%s %s): r=%.1f->%.1f snap=%.1f -> %d edge(s)",
            int(c.get("group_id", ci)), cluster_kind[ci], category,
            topo_r, robust_r, snap, len(chosen),
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

    # --- Apply order: holes first, then large edge wraps, then small
    # outlines. Grouping similar blends together (all hole rims, then
    # all plate-edge wraps, then tab outlines) keeps the construction
    # timeline readable and applies big structural blends before small
    # detail blends, so a tab fillet never reshapes an edge a plate
    # wrap still needs to resolve against. Within a tier, larger
    # features go first; ties break by cluster index for determinism.
    def _bucket_sort_key(bucket: dict) -> tuple:
        ci = bucket["cluster_idx"]
        cat = clusters[ci].get("category", "curve_edge")
        v = np.asarray(clusters[ci]["vertices"], dtype=float)
        max_ext = float((v.max(axis=0) - v.min(axis=0)).max()) if len(v) else 0.0
        tier = 0 if cat == "hole_edge" else 1
        return (tier, -max_ext, ci)

    buckets.sort(key=_bucket_sort_key)

    # Each cluster is applied as ITS OWN fillet, one at a time. The two
    # perimeter wraps of one plate (or the left/right plates) are
    # SEPARATE physical fillets and must not be merged into one
    # operation — the requirement is one continuous fillet per wrap, not
    # one operation spanning two wraps. Continuity within a wrap comes
    # from the arc-rebuilt profile + live edge re-resolution +
    # leave-one-out below; cross-wrap OCC conflicts are handled by
    # re-resolving each bucket's edges against the current solid.

    def _bucket_applied_radius(members: list) -> float:
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
        return applied_r

    def _live_bucket_edges(bucket: dict) -> list:
        # Re-resolve this bucket's edges against the CURRENT solid.
        # Earlier buckets rebuild the shape, which silently invalidates
        # the edge objects captured during enumeration. Match by
        # geometric identity first; for edges a prior fillet trimmed
        # (changed midpoint/length), fall back to proximity against
        # the cluster's vertex cloud.
        ci = bucket["cluster_idx"]
        members = bucket["members"]
        cluster_verts = np.asarray(clusters[ci].get("vertices"), dtype=float)
        bucket_keys = [
            _edge_identity_key(solid_edges[m["edge_idx"]]["obj"])
            for m in members
        ]
        edges_obj = _resolve_edges_on_current_solid(
            wp, bucket_keys, cluster_verts,
            max(SNAP_TOL_BASE, 0.6 * cluster_radius[ci] + 2.0),
        )
        if not edges_obj:
            edges_obj = [solid_edges[m["edge_idx"]]["obj"] for m in members]
        return edges_obj

    def _blend_edges(wp_solid, edges, size: float, kind: str):
        """Apply the blend op for this cluster kind to the selected edges.
        Chamfers and fillets share the entire matcher + robust-apply path;
        only the OCC operation differs (.chamfer(d) vs .fillet(r))."""
        sel = wp_solid.newObject(edges)
        return sel.chamfer(size) if kind == "chamfer" else sel.fillet(size)

    def _record_blend(edges_applied, size: float, kind: str) -> None:
        """AS-BUILT: midpoints of the PRE-blend edges that actually got
        rolled, so a geometrically identical rebuild elsewhere (Onshape)
        can re-select the same edges by closest-point queries. Must be
        called with the edge objects from BEFORE the blend op (they're
        invalid afterwards)."""
        if as_built_out is None:
            return
        mids = []
        for e in edges_applied:
            try:
                p = e.positionAt(0.5)
                mids.append([float(p.x), float(p.y), float(p.z)])
            except Exception:
                continue
        if mids:
            as_built_out.setdefault("blends", []).append({
                "kind": kind, "radius": float(size),
                "edge_midpoints": mids,
            })

    def _gate_blend_result(wp_test, vol_before: float):
        """Volume sanity + broken-shell gate for a candidate blend result.

        OCC can report SUCCESS while emitting a broken shell: rolling
        RackMount's outer-face ring returned a sane B-rep volume but a
        self-intersecting, non-watertight shape whose tessellation
        inverted (and poisoned every later step). Verify by SIGNED
        tessellation volume AND welded open-edge count (vertices must be
        deduplicated first — OCC tessellations duplicate vertices per
        face, which makes every edge look open otherwise).

        Returns (ok, v, f, vol_after, why) — v/f are the tessellation so
        the caller can reuse it for the ConstructionStep.
        """
        vol_after = abs(float(wp_test.val().Volume()))
        if vol_after <= 0 or vol_after > vol_before * 1.05:
            return False, None, None, vol_after, "volume out of range"
        v, f = _tessellate_solid(wp_test, fast=True)
        _va = np.asarray(v, dtype=float)
        _fa = np.asarray(f, dtype=np.int64)
        _tri = _va[_fa]
        signed_vol = float(np.einsum(
            "ij,ij->i", _tri[:, 0],
            np.cross(_tri[:, 1], _tri[:, 2]),
        ).sum() / 6.0)
        _rv = np.round(_va, 4)
        _uv, _inv = np.unique(_rv, axis=0, return_inverse=True)
        _fw = _inv[_fa]
        _ed = np.sort(
            _fw[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1,
        )
        _, _ec = np.unique(_ed, axis=0, return_counts=True)
        n_open_edges = int((_ec != 2).sum())
        if (
            signed_vol <= 0
            or not (0.7 < signed_vol / max(vol_after, 1e-9) < 1.3)
            or n_open_edges > 0
        ):
            why = (
                f"broken shell (signed tess vol {signed_vol:.0f} vs "
                f"B-rep {vol_after:.0f}, {n_open_edges} open edges)"
            )
            return False, v, f, vol_after, why
        return True, v, f, vol_after, "ok"

    def _apply_bucket(bi: int, bucket: dict) -> bool:
        nonlocal wp
        members = bucket["members"]
        kind = cluster_kind.get(bucket["cluster_idx"], "fillet")
        op_label = "Chamfer" if kind == "chamfer" else "Fillet"
        applied_r = _bucket_applied_radius(members)
        edges_obj = _live_bucket_edges(bucket)
        n_edges = len(edges_obj)
        t_bucket = _time.time()
        vol_before = abs(float(wp.val().Volume()))

        # --- Apply the WHOLE crease in one rolling operation. ---
        # Prefer the full chain: filleting the sides separately leaves
        # the corners sharp. If OCC can't roll the chain at applied_r,
        # retry the whole chain at successively smaller whole-mm radii
        # down to 2 mm, then leave-one-out, then a greedy maximal
        # subset — a cluster only fails if NO sub-chain passes the
        # shell gate.
        r_candidates = [applied_r]
        for rr in range(int(applied_r) - 1, 1, -1):
            if rr >= 2:
                r_candidates.append(float(rr))
        applied_this_bucket = False
        for r in r_candidates:
            try:
                wp_test = _blend_edges(wp, edges_obj, r, kind)
                ok, v, f, vol_after, why = _gate_blend_result(
                    wp_test, vol_before,
                )
                if not ok:
                    if v is not None:
                        logger.warning(
                            "Bucket %d r=%.0f on %d edges: OCC reported "
                            "success but %s; rejecting",
                            bi, r, n_edges, why,
                        )
                    continue
                wp = wp_test
                _record_blend(edges_obj, r, kind)
                applied_steps.append(ConstructionStep(
                    name=f"{op_label} (r={r:.0f}mm) x{n_edges}",
                    tess_vertices=v, tess_faces=f, volume=vol_after,
                ))
                applied_script.append(
                    f"# {op_label} r={r:.0f} on {n_edges} crease edge(s)"
                )
                logger.info(
                    "Bucket %d (cluster %d, %s): r=%.0fmm on %d edges, "
                    "vol %.0f -> %.0f (dt=%.1fs)",
                    bi, bucket["cluster_idx"], kind, r, n_edges,
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
        # Leave-one-out recovery: one degenerate edge (a sliver segment
        # or a seam OCC can't blend) poisons the whole chain with
        # "BRep_API: command not done". Find it by dropping one edge at
        # a time and applying the remaining chain in a single rolling
        # op — corners stay wrapped everywhere except the dropped
        # sliver. (RackMount: one of the two mirror-twin U-wraps failed
        # wholesale this way while its twin succeeded.)
        if not applied_this_bucket and n_edges >= 3:
            r_loo = r_candidates[-1]
            for skip_i in range(n_edges):
                subset = [e for k, e in enumerate(edges_obj) if k != skip_i]
                try:
                    wp_test = _blend_edges(wp, subset, r_loo, kind)
                    # Same broken-shell gate as the primary path: the
                    # unguarded leave-one-out acceptance was the
                    # corruption's second way in.
                    ok, v, f, vol_after, why = _gate_blend_result(
                        wp_test, vol_before,
                    )
                    if not ok:
                        if v is not None:
                            logger.warning(
                                "Bucket %d LOO subset %d: %s; rejecting",
                                bi, skip_i, why,
                            )
                        continue
                except Exception:
                    continue
                wp = wp_test
                _record_blend(subset, r_loo, kind)
                applied_steps.append(ConstructionStep(
                    name=f"{op_label} (r={r_loo:.0f}mm) x{len(subset)}",
                    tess_vertices=v, tess_faces=f, volume=vol_after,
                ))
                applied_script.append(
                    f"# {op_label} r={r_loo:.0f} on {len(subset)} crease "
                    f"edge(s) (1 degenerate edge dropped)"
                )
                dropped = edges_obj[skip_i]
                try:
                    dmp = dropped.positionAt(0.5)
                    where = f"({dmp.x:.0f},{dmp.y:.0f},{dmp.z:.0f})"
                except Exception:
                    where = "?"
                logger.info(
                    "Bucket %d (cluster %d): r=%.0fmm on %d/%d edges after "
                    "dropping degenerate edge at %s, vol %.0f -> %.0f",
                    bi, bucket["cluster_idx"], r_loo, len(subset), n_edges,
                    where, vol_before, vol_after,
                )
                applied_this_bucket = True
                break
        # Greedy maximal-subset recovery: when the whole chain AND every
        # leave-one-out subset fail, grow the largest sub-chain OCC can
        # roll instead of skipping the cluster outright. Long spine
        # edges are tried first (they carry the visible fillet); each
        # candidate is re-tested as one rolling op with the full shell
        # gate, so a poison edge is simply never admitted. A partial
        # fillet with a sharp corner near the degenerate edge beats the
        # user-visible alternative — the whole wrap staying sharp
        # (RackMount: 3 of 4 side-plate edge wraps were dropped this
        # way).
        if not applied_this_bucket and n_edges >= 2:
            r_g = r_candidates[-1]

            def _elen(e) -> float:
                try:
                    return float(e.Length())
                except Exception:
                    return 0.0

            order = sorted(
                range(n_edges), key=lambda k: -_elen(edges_obj[k]),
            )
            kept: list[int] = []
            best = None  # (wp_test, v, f, vol_after)
            for k in order:
                trial = [edges_obj[t] for t in (*kept, k)]
                try:
                    wp_test = _blend_edges(wp, trial, r_g, kind)
                    ok, v, f, vol_after, _why = _gate_blend_result(
                        wp_test, vol_before,
                    )
                except Exception:
                    continue
                if ok:
                    kept.append(k)
                    best = (wp_test, v, f, vol_after)
            if best is not None and kept:
                wp_test, v, f, vol_after = best
                wp = wp_test
                _record_blend([edges_obj[t] for t in kept], r_g, kind)
                applied_steps.append(ConstructionStep(
                    name=f"{op_label} (r={r_g:.0f}mm) x{len(kept)}",
                    tess_vertices=v, tess_faces=f, volume=vol_after,
                ))
                applied_script.append(
                    f"# {op_label} r={r_g:.0f} on {len(kept)}/{n_edges} "
                    f"crease edge(s) (greedy maximal subset)"
                )
                logger.info(
                    "Bucket %d (cluster %d): greedy subset rolled %d/%d "
                    "edges at r=%.0fmm, vol %.0f -> %.0f",
                    bi, bucket["cluster_idx"], len(kept), n_edges, r_g,
                    vol_before, vol_after,
                )
                applied_this_bucket = True

        if not applied_this_bucket:
            logger.warning(
                "Bucket %d (cluster %d): could not roll %d-edge crease at "
                "any radius >= 2mm, even with leave-one-out or greedy "
                "subsets; skipped",
                bi, bucket["cluster_idx"], n_edges,
            )
        return applied_this_bucket

    for bi, bucket in enumerate(buckets):
        _apply_bucket(bi, bucket)

    logger.info(
        "Fillet pass done: %d clusters, %d edges in buckets, %d buckets, "
        "%d steps applied, %.1fs",
        len(clusters), n_total_edges_matched, len(buckets),
        len(applied_steps), _time.time() - pass_start,
    )
    return wp, applied_steps, applied_script


def _regularize_revolve_profile(profile: np.ndarray) -> np.ndarray:
    """Snap a sampled (h, r) lathe profile to piecewise-CONSTANT radius
    steps where the samples plateau — a turned part is cylinders and
    shoulders, not 18 slightly-different radii. Sloped runs (cones) are
    kept as-is. Returns the cleaned profile."""
    try:
        hs = profile[:, 0].astype(float)
        rs = profile[:, 1].astype(float)
        segs = []          # (i0, i1, median_r, is_flat)
        i0 = 0
        for i in range(1, len(rs) + 1):
            if i == len(rs) or abs(rs[i] - np.median(rs[i0:i])) > 1.5:
                seg = rs[i0:i]
                flat = float(np.std(seg)) < 0.8 and len(seg) >= 2
                segs.append((i0, i, float(np.median(seg)), flat))
                i0 = i
        out = []
        for (a, b, med, flat) in segs:
            if flat:
                out.append((float(hs[a]), med))
                out.append((float(hs[b - 1]), med))
            else:
                out.extend((float(hs[k]), float(rs[k]))
                           for k in range(a, b))
        if len(out) >= 2:
            return np.asarray(out, dtype=float)
    except Exception:
        pass
    return profile


def _build_revolve_result(feature, cleaned: CleanedMesh, mesh_volume: float,
                          output_path=None, refine: bool = True):
    """Build a CadQueryResult for a REVOLVE feature (turned part).

    The feature's sketch_geometry carries axis_origin, axis_dir, and a
    profile = [(height_along_axis, radius), ...]. We lift the profile into the
    axis/radial half-plane, close it against the axis (r=0), and revolve 360
    about the axis. Returns None on any failure so the caller falls back to the
    prismatic path.
    """
    try:
        import cadquery as cq
    except Exception:
        return None
    sg = feature.sketch_geometry or {}
    try:
        origin = np.asarray(sg["axis_origin"], dtype=float)
        axis = np.asarray(sg["axis_dir"], dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        profile = np.asarray(sg["profile"], dtype=float)   # (M,2): (h, r)
        if profile.ndim != 2 or len(profile) < 3:
            return None
        # Turned parts are cylinders + shoulders: snap the sampled profile
        # to clean radius steps (a bumpy 18-sample lathe curve otherwise
        # revolves into a wavy barrel).
        profile = _regularize_revolve_profile(profile)
        tmp = (np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9
               else np.array([0, 1.0, 0]))
        radial = np.cross(axis, tmp)
        radial = radial / (np.linalg.norm(radial) + 1e-12)
        hs = profile[:, 0]
        rs = np.clip(profile[:, 1], 0.0, None)
        # Closed loop in the (h, r) half-plane: down the axis, out along the
        # profile, back to the axis.
        loop = ([(float(hs[0]), 0.0)]
                + list(zip(hs.tolist(), rs.tolist()))
                + [(float(hs[-1]), 0.0)])
        pts3 = [cq.Vector(*(origin + axis * h + radial * r)) for (h, r) in loop]
        wire = cq.Wire.makePolygon(pts3 + [pts3[0]])
        solid = cq.Solid.revolve(wire, [], 360.0, cq.Vector(*origin),
                                 cq.Vector(*(origin + axis)))
        wp = cq.Workplane(obj=solid)
    except Exception:
        logger.debug("Revolve solid build failed", exc_info=True)
        return None

    # Patch any residual mismatch (holes/steps the smooth profile missed).
    # Skipped when the revolve serves as the timeline's BASE feature — the
    # real cutters drill the holes there, and refine boxes on a bare lathe
    # solid would be noise.
    if refine:
        try:
            wp, _ = _residual_refine_solid(wp, cleaned, mesh_volume)
        except Exception:
            logger.debug("revolve residual refine failed", exc_info=True)

    cad_vol = _solid_volume(wp)
    if cad_vol <= 0:
        return None
    vol_err = (abs(cad_vol - mesh_volume) / (mesh_volume + 1e-15) * 100
               if mesh_volume > 0 else 100.0)
    tv, tf = _tessellate_solid(wp)

    step_path = ""
    if output_path is not None:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_solid = wp
            unit_scale = float((getattr(cleaned, "metadata", None) or {})
                               .get("unit_scale", 1.0) or 1.0)
            if unit_scale != 1.0:
                inv = 1.0 / unit_scale
                mat = cq.Matrix([[inv, 0, 0, 0], [0, inv, 0, 0], [0, 0, inv, 0]])
                export_solid = wp.newObject(
                    [o.transformGeometry(mat) for o in wp.objects])
            cq.exporters.export(export_solid, str(output_path))
            step_path = str(output_path)
        except Exception:
            logger.debug("revolve STEP export failed", exc_info=True)

    logger.info("Revolve export: cad_vol=%.1f, err=%.2f%%", cad_vol, vol_err)
    return CadQueryResult(
        solid=wp, tess_vertices=tv, tess_faces=tf,
        variants=[BuildVariant(name="revolve", strategy="revolve",
                               cad_volume=cad_vol)],
        script="# Revolve reconstruction\nimport cadquery as cq\n",
        step_path=step_path, mesh_volume=mesh_volume, cad_volume=cad_vol,
        volume_error_pct=vol_err,
        status="PASS" if vol_err < 10.0 else "FAIL")


def _solid_volume(solid) -> float:
    """abs volume of a CadQuery solid/Workplane, 0.0 on failure."""
    try:
        return abs(float(solid.val().Volume()))
    except Exception:
        return 0.0


def _transplant_mesh_features(solid, cleaned, feature_tree, extrude_dir, cq):
    """Cut the MESH-DERIVED features into an arbitrary solid (the contour
    fallback): ring recesses/slots, tangent holes, the tree's cylindrical
    holes; union detected standing bosses. Used when the fallback WINS but
    recognition was UNDER-solid — there the fallback∩recognition hybrid
    shrinks the part further (wrong), yet adopting the plain fallback DROPS
    every hole (the carriage block's "refined model removed all the holes").
    Every cut/union is volume-gated; the caller scores the result by exact
    IoU against the plain fallback, so this can never ship a regression."""
    out = solid

    def _cut(shape):
        nonlocal out
        try:
            before = _solid_volume(out)
            wt = out.cut(shape)
            if before - _solid_volume(wt) > 0.5:
                out = wt
        except Exception:
            pass

    plane_of = {0: "YZ", 1: "XZ", 2: "XY"}
    # ring recesses / through slots (mesh sections)
    try:
        for rr in _detect_face_ring_recesses(cleaned, extrude_dir):
            o = [0.0, 0.0, 0.0]
            o[rr["axis"]] = rr["face_t"]
            wp0 = cq.Workplane(rr["plane_name"], origin=tuple(o))
            el = rr.get("ellipse")
            c = rr.get("circle")
            try:
                if c:
                    sh = (wp0.center(c["cu"], c["cv"]).circle(c["r"])
                          .extrude(rr["depth"], both=True))
                elif el:
                    sh = (wp0.center(el["cu"], el["cv"])
                          .ellipse(el["ra"], el["rb"],
                                   rotation_angle=el["ang"])
                          .extrude(rr["depth"], both=True))
                else:
                    sh = (wp0.polyline(rr["exterior_uv"]).close()
                          .extrude(rr["depth"], both=True))
                _cut(sh)
            except Exception:
                continue
    except Exception:
        logger.info("transplant rings failed", exc_info=True)
    # tangent holes (mesh sections)
    try:
        for th in _detect_tangent_holes(cleaned, extrude_dir):
            for (r, t0, t1) in th["levels"]:
                o = [0.0, 0.0, 0.0]
                o[th["axis"]] = 0.5 * (t0 + t1)
                try:
                    sh = (cq.Workplane(plane_of[th["axis"]], origin=tuple(o))
                          .center(th["cu"], th["cv"]).circle(r)
                          .extrude(0.5 * (t1 - t0) + 0.5, both=True))
                    _cut(sh)
                except Exception:
                    continue
    except Exception:
        logger.info("transplant tangent holes failed", exc_info=True)
    # the tree's cylindrical holes (recognition features)
    try:
        from mesh2param.models.feature import FeatureType as _FT
        for f in getattr(feature_tree, "features", []) or []:
            if f.type not in (_FT.THROUGH_HOLE, _FT.BLIND_HOLE):
                continue
            sg = f.sketch_geometry or {}
            if sg.get("center") is None or not sg.get("radius"):
                continue
            cen = np.asarray(sg["center"], dtype=float)
            d = np.asarray(f.direction, dtype=float)
            ax = int(np.argmax(np.abs(d)))
            length = (float(f.depth) if f.type == _FT.BLIND_HOLE
                      and f.depth > 0 else 1e3)
            o = [0.0, 0.0, 0.0]
            o[ax] = float(cen[ax])
            us = [k for k in (0, 1, 2) if k != ax]
            try:
                sh = (cq.Workplane(plane_of[ax], origin=tuple(o))
                      .center(float(cen[us[0]]), float(cen[us[1]]))
                      .circle(float(sg["radius"]))
                      .extrude(length if f.type == _FT.BLIND_HOLE
                               else 1e3, both=True))
                _cut(sh)
            except Exception:
                continue
    except Exception:
        logger.info("transplant tree holes failed", exc_info=True)
    # standing bosses (union — they exist in the mesh)
    try:
        for sb in _detect_standing_bosses(cleaned, extrude_dir):
            o = [0.0, 0.0, 0.0]
            o[sb["axis"]] = sb["face_t"]
            pads = []
            if sb.get("keep_outline"):
                pads.append(cq.Workplane(sb["plane_name"], origin=tuple(o))
                            .polyline(sb["keep_outline"]).close()
                            .extrude(sb["sign"] * sb["height"]))
            for k in sb["keeps"]:
                pads.append(cq.Workplane(sb["plane_name"], origin=tuple(o))
                            .center(k["cu"], k["cv"]).circle(k["r"])
                            .extrude(sb["sign"] * sb["height"]))
            for pad in pads:
                try:
                    out = out.union(pad)
                except Exception:
                    continue
    except Exception:
        logger.info("transplant bosses failed", exc_info=True)
    return out


def _contour_fallback_result(cleaned: CleanedMesh, mesh_volume: float):
    """Build a CadQueryResult from the segmentation-free contour path, or None.

    Used when recognition produces NO usable solid (profile extraction failed,
    or the build collapsed) — the catastrophic case where the contour fallback
    is most valuable but the normal router (which guards on an existing solid)
    would skip it."""
    try:
        from mesh2param.pipeline.contour_recon import (
            reconstruct_segmentation_free)
        b = reconstruct_segmentation_free(cleaned)
        if b.solid is None:
            return None
        tv, tf = _tessellate_solid(b.solid)
        logger.info("Contour fallback produced a solid where recognition "
                    "could not (%.1f%% volume error)", b.volume_error_pct)
        return CadQueryResult(
            solid=b.solid, tess_vertices=tv, tess_faces=tf,
            mesh_volume=mesh_volume, cad_volume=b.cad_volume,
            volume_error_pct=b.volume_error_pct, status=b.status)
    except Exception:
        logger.debug("contour fallback result failed", exc_info=True)
        return None


def _refine_op_to_solid(op, cq):
    """Build the CadQuery solid for one iou_refine RefineOp (box / prism /
    cylinder), or None. Prisms/cylinders are built in world space along the
    op's axis so no plane-local bookkeeping is needed."""
    try:
        if op.kind == "box":
            lo = np.asarray(op.box_min, float)
            hi = np.asarray(op.box_max, float)
            dims = hi - lo
            if np.any(dims <= 1e-6):
                return None
            return cq.Solid.makeBox(
                float(dims[0]), float(dims[1]), float(dims[2]),
                cq.Vector(float(lo[0]), float(lo[1]), float(lo[2])))

        height = float(op.hi - op.lo)
        if height <= 1e-6:
            return None
        ax = int(op.axis)
        uv = tuple(i for i in range(3) if i != ax)
        axis_unit = np.zeros(3)
        axis_unit[ax] = 1.0

        if op.kind == "cylinder":
            if op.radius <= 1e-6:
                return None
            base = np.zeros(3)
            base[ax] = float(op.lo)
            base[uv[0]] = float(op.center[0])
            base[uv[1]] = float(op.center[1])
            return cq.Solid.makeCylinder(
                float(op.radius), height, cq.Vector(*base),
                cq.Vector(*axis_unit))

        if op.kind == "prism":
            poly = np.asarray(op.polygon, float)
            if poly.ndim != 2 or len(poly) < 3:
                return None
            # DESIGN-INTENT snap: the residual footprint is a voxel trace —
            # jagged, with notches where small artifacts perturbed it (the
            # bezel frame's pocket sketch had "two random notches"). A
            # near-rectangular footprint IS a rectangle; otherwise smooth
            # the outline. Runs inside the op builder so the scoring pass
            # and the final cut see the same solid.
            try:
                from shapely.geometry import Polygon as _RSP, box as _RSB
                _sp = _RSP(poly).buffer(0)
                if not _sp.is_empty and _sp.geom_type == "Polygon":
                    _rect = None
                    _ab = _RSB(*_sp.bounds)
                    if (_sp.intersection(_ab).area
                            / max(_sp.union(_ab).area, 1e-9)) >= 0.90:
                        _rect = _ab
                    else:
                        _mrr = _sp.minimum_rotated_rectangle
                        if (_sp.intersection(_mrr).area
                                / max(_sp.union(_mrr).area, 1e-9)) >= 0.90:
                            _rect = _mrr
                    if _rect is not None:
                        # Preserve ROUNDED corners: the area the true
                        # footprint misses at the rect corners gives the
                        # radius ((4-pi)r^2). The enclosure lid's pocket
                        # was snapped to a sharp-cornered rectangle.
                        _bx = _rect.bounds
                        _mins = min(_bx[2] - _bx[0], _bx[3] - _bx[1])
                        _r = float(np.sqrt(max(_rect.area - _sp.area, 0.0)
                                           / (4.0 - np.pi)))
                        if 0.5 <= _r <= 0.25 * _mins:
                            _rr = _rect.buffer(-_r, join_style=1).buffer(
                                _r, join_style=1)
                            if (not _rr.is_empty
                                    and _rr.geom_type == "Polygon"):
                                _rect = _rr
                        poly = np.asarray(_rect.exterior.coords[:-1], float)
                    else:
                        _sm = _void_intent_outline(
                            [(float(a), float(b)) for a, b in poly])
                        if _sm and len(_sm) >= 3:
                            poly = np.asarray(_sm, float)
            except Exception:
                pass
            pts = []
            for (u, v) in poly:
                p = np.zeros(3)
                p[ax] = float(op.lo)
                p[uv[0]] = float(u)
                p[uv[1]] = float(v)
                pts.append(cq.Vector(*p))
            wire = cq.Wire.makePolygon(pts + [pts[0]])
            face = cq.Face.makeFromWires(wire)
            return cq.Solid.extrudeLinear(face, cq.Vector(*(axis_unit * height)))
    except Exception:
        logger.debug("refine op -> solid failed (kind=%s)",
                     getattr(op, "kind", "?"), exc_info=True)
        return None
    return None


def _refine_op_sketch_segments(op, patch):
    """Wireframe outline of a refine patch's footprint, so each refine op
    gets a Sketch step like every other feature (they previously appeared
    as cuts with no sketch)."""
    try:
        if op.kind == "cylinder":
            ax = int(op.axis)
            us = [k for k in (0, 1, 2) if k != ax]
            th = np.linspace(0, 2 * np.pi, 49)
            ring = []
            for t in th:
                p3 = np.zeros(3)
                p3[ax] = float(op.lo)
                p3[us[0]] = float(op.center[0] + op.radius * np.cos(t))
                p3[us[1]] = float(op.center[1] + op.radius * np.sin(t))
                ring.append(p3)
        else:
            bb = patch.BoundingBox()
            lo = np.array([bb.xmin, bb.ymin, bb.zmin])
            hi = np.array([bb.xmax, bb.ymax, bb.zmax])
            dims = hi - lo
            ax = int(np.argmin(dims))   # sketch on the face plane
            us = [k for k in (0, 1, 2) if k != ax]
            ring = []
            for cu, cv in ((lo[us[0]], lo[us[1]]),
                           (hi[us[0]], lo[us[1]]),
                           (hi[us[0]], hi[us[1]]),
                           (lo[us[0]], hi[us[1]]),
                           (lo[us[0]], lo[us[1]])):
                p3 = np.zeros(3)
                p3[ax] = hi[ax]
                p3[us[0]], p3[us[1]] = cu, cv
                ring.append(p3)
        seg = []
        for k in range(len(ring) - 1):
            seg.append(ring[k])
            seg.append(ring[k + 1])
        return np.asarray(seg, np.float64)
    except Exception:
        return None


def _itemize_solid_swap(pre_solid, target_solid, cleaned,
                        mesh_volume: float):
    """Express an arbiter solid SWAP as itemized refine features.

    The contour-fallback arbiter used to REPLACE the whole solid with its
    winner ("Boolean 1 (refine)" — the exact opaque step the user rejected).
    Instead: run the box/prism carver with the WINNER as target and the
    pre-swap solid as start, so the difference becomes ordinary cut/add
    steps (each with a sketch). Returns (solid, steps) when the ops capture
    enough of the swap's improvement, else (None, None) — the caller then
    keeps the plain swap as a last resort.
    """
    import cadquery as cq
    import trimesh
    from mesh2param.pipeline.iou_refine import (refine_against_target,
                                                exact_iou_available,
                                                mesh_iou_exact)

    def _tm(wp):
        v, f = _tessellate_solid(wp)
        return trimesh.Trimesh(
            vertices=np.asarray(v, float).reshape(-1, 3),
            faces=np.asarray(f, np.int64).reshape(-1, 3), process=False)

    try:
        tgt = _tm(target_solid)
        pre = _tm(pre_solid)
        mesh = trimesh.Trimesh(vertices=cleaned.vertices,
                               faces=cleaned.faces, process=False)
        v_tgt = _solid_volume(target_solid)
        use_iou = exact_iou_available()

        def _apply_set(start_solid, ops):
            cur = start_solid
            steps = []
            prev_vol = _solid_volume(cur)
            _emitted = 0
            for op in ops:
                patch = _refine_op_to_solid(op, cq)
                if patch is None:
                    continue
                try:
                    nxt = (cur.union(patch) if op.op == "add"
                           else cur.cut(patch))
                except Exception:
                    continue
                vol = _solid_volume(nxt)
                dv = abs(vol - prev_vol)
                if dv >= 0.004 * mesh_volume and _emitted < 8:
                    _emitted += 1
                    try:
                        seg = _refine_op_sketch_segments(op, patch)
                        if seg is not None and len(seg):
                            v2, f2 = _tessellate_solid(cur, fast=True)
                            steps.append(ConstructionStep(
                                name=("Sketch Cut" if op.op != "add"
                                      else "Sketch Join"),
                                tess_vertices=v2, tess_faces=f2,
                                volume=prev_vol, sketch_overlay=seg,
                            ))
                        v3, f3 = _tessellate_solid(nxt, fast=True)
                        steps.append(ConstructionStep(
                            name=(f"Refine "
                                  f"{'Add' if op.op == 'add' else 'Cut'} "
                                  f"({op.kind}, {dv:,.0f}mm³)"),
                            tess_vertices=v3, tess_faces=f3, volume=vol,
                        ))
                    except Exception:
                        pass
                cur = nxt
                prev_vol = vol
            return cur, steps

        # MULTI-ROUND: the greedy box cover stalls on irregular shells;
        # re-lattice from the ACTUAL solid each round and go again until
        # the volume gap to the winner stops closing.
        def _mesh_score(wp2):
            return (mesh_iou_exact(mesh, _tm(wp2)) if use_iou
                    else -abs(_solid_volume(wp2) - mesh_volume))

        cur_solid = pre_solid
        cur_score = _mesh_score(cur_solid)
        _iou_swap_early = mesh_iou_exact(mesh, tgt) if use_iou else None
        # Pre-bail: when the pre-swap solid is ALREADY far below the swap,
        # ≤8 emitted boxes per round can never recover enough to clear the
        # 0.06 quality floor (pillow block: pre 0.32 vs swap 0.81 — round 1
        # alone cost minutes and its 600s timeout). Don't even start.
        if (_iou_swap_early is not None
                and cur_score < _iou_swap_early - 0.35):
            logger.info(
                "Swap itemization skipped: pre-swap fit %.3f too far "
                "below the swap %.3f for ops to recover", cur_score,
                _iou_swap_early)
            return None, None
        all_steps: list = []
        for _round in range(3):
            _cur_tm = _tm(cur_solid)
            op_sets = []
            for _res in (96, 128):
                try:
                    rr = refine_against_target(tgt, _cur_tm, res=_res,
                                               max_iters=12,
                                               min_residual_frac=0.003)
                    op_sets.extend(ops for ops in (rr.ops, rr.box_ops)
                                   if ops)
                except Exception:
                    continue
            if not op_sets:
                break
            # Ops are GUIDED by the arbiter's winner but each round is
            # ACCEPTED by fit to the actual mesh — chasing the winner's
            # volume alone traded mesh fit for volume.
            round_best = None
            for ops in op_sets:
                cand, steps = _apply_set(cur_solid, ops)
                if not steps:
                    continue
                sc = _mesh_score(cand)
                if round_best is None or sc > round_best[0]:
                    round_best = (sc, cand, steps)
            if round_best is None or round_best[0] < cur_score + 0.003:
                break
            cur_solid, cur_score = round_best[1], round_best[0]
            all_steps.extend(round_best[2])
            logger.info("Swap itemization round %d: vol %.0f (winner "
                        "%.0f), score %.3f, +%d step(s)", _round + 1,
                        _solid_volume(cur_solid), v_tgt, cur_score,
                        len(round_best[2]))
            # Hopeless bail: when round 1 lands FAR below the swap, later
            # rounds never close enough of the gap to clear the 0.06
            # quality floor (pillow block: 0.321 → 0.639 vs swap 0.809,
            # minutes wasted then discarded). Cut the losses now.
            if (_round == 0 and _iou_swap_early is not None
                    and cur_score < _iou_swap_early - 0.25):
                logger.info(
                    "Swap itemization round 1 far below the swap "
                    "(%.3f vs %.3f) — bailing early", cur_score,
                    _iou_swap_early)
                return None, None
        if not all_steps:
            return None, None
        # Adopt whenever the itemized result IMPROVES on the pre-swap
        # solid — real features always beat an opaque boolean. Only when
        # itemization fails to improve at all does the caller keep the
        # plain swap.
        if use_iou:
            iou_pre = mesh_iou_exact(mesh, pre)
            iou_swap = _iou_swap_early
            iou_ops = mesh_iou_exact(mesh, _tm(cur_solid))
            if iou_ops < iou_pre + 0.005:
                logger.info(
                    "Swap itemization did not improve: ops IoU %.3f vs "
                    "pre %.3f / swap %.3f", iou_ops, iou_pre, iou_swap)
                return None, None
            # Quality floor: itemized features may cost a LITTLE fit vs
            # the opaque swap (the lid traded 0.056) but not a lot — the
            # pillow block's boxes captured so little that adopting them
            # cost 0.18 IoU. Past the floor, correct geometry wins.
            if iou_ops < iou_swap - 0.06:
                logger.info(
                    "Swap itemization too lossy: ops IoU %.3f vs swap "
                    "%.3f — keeping the swap", iou_ops, iou_swap)
                return None, None
            logger.info(
                "Arbiter swap itemized: %d step(s), IoU %.3f (pre %.3f, "
                "swap %.3f)", len(all_steps), iou_ops, iou_pre, iou_swap)
        else:
            if (abs(_solid_volume(cur_solid) - mesh_volume)
                    > abs(_solid_volume(pre_solid) - mesh_volume)
                    - 0.005 * mesh_volume):
                return None, None
        return cur_solid, all_steps
    except Exception:
        logger.debug("swap itemization failed", exc_info=True)
        return None, None


def _residual_refine_solid(solid, cleaned: CleanedMesh, mesh_volume: float,
                           max_iters: int = 3, steps_out: list | None = None):
    """CADFit-style residual-refinement post-pass.

    The recognition pipeline is feed-forward — a mis-recognized or missed
    feature silently propagates. CADFit (github.com/ghadinehme/CADFit) instead
    BUILDS the result, diffs it against the target, and patches the residual
    (`under -> add`, `over -> cut`) until it vanishes. This is the keystone
    borrow: it gives the deterministic pipeline a global safety net.

    Here we tessellate the built solid, run the voxel-IoU residual analysis
    (pipeline/iou_refine.py) against the target mesh, and apply the proposed
    axis-aligned add/cut boxes to the CadQuery solid — keeping the result ONLY
    if it reduces the volume error. It is therefore a pure safety net: any
    failure or non-improvement returns the input solid unchanged, so it is safe
    to leave on by default. Returns (solid, cad_volume).

    Patches are PRISMS (extruded footprint), CYLINDERS (round holes/bosses —
    Phase 2), or boxes (fallback for chunky bits), chosen per residual by
    iou_refine. Cylinders are what let the loop cut round holes cleanly.
    """
    cad_vol = _solid_volume(solid)
    if solid is None or mesh_volume <= 0:
        return solid, cad_vol
    try:
        import cadquery as cq
        from mesh2param.pipeline.iou_refine import refine_against_target
    except Exception:
        return solid, cad_vol
    try:
        tv, tf = _tessellate_solid(solid)
        if tv is None or tf is None or len(tv) == 0:
            return solid, cad_vol
        recon = trimesh.Trimesh(
            vertices=np.asarray(tv, float).reshape(-1, 3),
            faces=np.asarray(tf, np.int64).reshape(-1, 3), process=False)
        target = trimesh.Trimesh(
            vertices=cleaned.vertices, faces=cleaned.faces, process=False)
        # ADAPTIVE resolution. The voxel box-carver at the default res=64 /
        # 3 iters under-resolves a part that is still BADLY over-solid — a deep
        # distributed cavity (AI4's hollow enclosure lid) reads as a few coarse
        # cells, so the carve stops far short (IoU 0.762, vol_err 22%). When the
        # current solid is still grossly off (>12%), the residual is large and
        # non-prismatic, so we ALSO refine at res=96 / 5 iters, which tracks the
        # cavity finely (AI4 -> IoU 0.845, vol_err 6.9%). Resolution is NOT
        # monotonic (res=128 scored below 96 on AI4), so we keep BOTH passes as
        # candidates and let the exact-IoU arbiter below pick the winner — the
        # res=64 candidate is always present, so escalation can only add upside,
        # never regress. Parts that already fit (<12%) run the cheap pass only.
        pre_err = abs(cad_vol - mesh_volume) / (mesh_volume + 1e-15)
        configs = [(64, max_iters)]
        if pre_err >= 0.12:
            configs.append((96, max(5, max_iters)))
        op_sets = []
        for _rr, _ri in configs:
            try:
                rr = refine_against_target(target, recon, res=_rr, max_iters=_ri)
            except Exception:
                logger.debug("refine at res=%d failed", _rr, exc_info=True)
                continue
            op_sets.extend(ops for ops in (rr.ops, rr.box_ops) if ops)
        if not op_sets:
            return solid, cad_vol

        def _apply(ops):
            refined = solid
            applied = 0
            for op in ops:
                patch = _refine_op_to_solid(op, cq)
                if patch is None:
                    continue
                # Skip TINY add-boxes: they buy a sliver of IoU but read as
                # random blocky extrusions stuck onto the model (user-visible
                # on the enclosure lid). Cuts stay unrestricted — removing
                # excess never adds phantom geometry.
                if op.op == "add":
                    try:
                        if _solid_volume(patch) < 0.004 * mesh_volume:
                            continue
                    except Exception:
                        pass
                try:
                    refined = (refined.union(patch) if op.op == "add"
                               else refined.cut(patch))
                    applied += 1
                except Exception:
                    logger.debug("residual %s %s failed, skipping", op.op,
                                 op.kind, exc_info=True)
                    continue
            return (refined, _solid_volume(refined)) if applied else (None, 0.0)

        old_err = abs(cad_vol - mesh_volume) / (mesh_volume + 1e-15)
        from mesh2param.pipeline.iou_refine import (
            exact_iou_available, mesh_iou_exact)
        # Pick the op-set by SHAPE (exact IoU) WHERE THAT METRIC IS RELIABLE
        # (Linux/container: alpha-wrap available -> true exact IoU; this is the
        # deployment target and where we optimize to beat CADFit). On Windows
        # alpha-wrap is missing so exact IoU degrades to a noisy voxel fallback
        # that mis-ranks candidates — there we keep the validated VOLUME arbiter
        # (no regression).
        #
        # Build + SCORE one candidate at a time, keeping ONLY the running best
        # solid. Holding every candidate solid at once (up to 4: res64/res96 x
        # shape/box op-sets) plus each tessellation is the build's peak-memory
        # spike; on a memory-tight WSL2 backend that spike can crash the Docker
        # engine. Score-and-discard caps it at ~one extra solid. Selection is
        # unchanged: best exact-IoU (Linux) or lowest vol-err (Windows), kept
        # only if it beats the pre-refine solid.
        use_iou = exact_iou_available()
        iou0 = mesh_iou_exact(target, recon) if use_iou else None
        best = None      # (score, solid, vol); higher score is better
        for ops in op_sets:
            cand_solid, cand_vol = _apply(ops)
            if cand_solid is None:
                continue
            if use_iou:
                ctv, ctf = _tessellate_solid(cand_solid)
                if ctv is None or len(ctv) == 0:
                    continue
                cmesh = trimesh.Trimesh(
                    vertices=np.asarray(ctv, float).reshape(-1, 3),
                    faces=np.asarray(ctf, np.int64).reshape(-1, 3),
                    process=False)
                score = mesh_iou_exact(target, cmesh)
            else:
                score = -abs(cand_vol - mesh_volume) / (mesh_volume + 1e-15)
            if best is None or score > best[0]:
                best = (score, cand_solid, cand_vol, ops)
        if best is None:
            return solid, cad_vol

        # FEATURE EDIT: when the result is still well off the mesh, don't
        # leave a too-small cavity prism for the arbiter to paper over
        # with an opaque swap or patch boxes — re-optimize the LARGEST
        # prism cut's own parameters (footprint dilation, depth
        # extension), scored by exact IoU against the mesh. One edited
        # feature, not eight patches.
        if use_iou:
            try:
                _err_now = abs(best[2] - mesh_volume) / (mesh_volume + 1e-15)
                _prisms = [(i, op) for i, op in enumerate(best[3])
                           if op.kind == "prism" and op.op == "cut"]
                if _err_now > 0.06 and _prisms:
                    from shapely.geometry import Polygon as _EP
                    import copy as _copy

                    def _p_vol(op):
                        try:
                            return (_EP(op.polygon).area
                                    * float(op.hi - op.lo))
                        except Exception:
                            return 0.0

                    _i0, _op0 = max(_prisms, key=lambda t: _p_vol(t[1]))

                    def _score_edit(dil, ext):
                        _op2 = _copy.deepcopy(_op0)
                        try:
                            if dil > 0:
                                _p2 = _EP(_op0.polygon).buffer(
                                    dil, join_style=1)
                                if (_p2.geom_type != "Polygon"
                                        or _p2.is_empty):
                                    return None
                                _op2.polygon = [
                                    (float(x), float(y))
                                    for x, y in _p2.exterior.coords[:-1]]
                            _op2.lo = float(_op0.lo) - ext
                            _op2.hi = float(_op0.hi) + ext
                            _ops2 = list(best[3])
                            _ops2[_i0] = _op2
                            _s2, _v2 = _apply(_ops2)
                            if _s2 is None:
                                return None
                            tv2, tf2 = _tessellate_solid(_s2)
                            m2 = trimesh.Trimesh(
                                vertices=np.asarray(
                                    tv2, float).reshape(-1, 3),
                                faces=np.asarray(
                                    tf2, np.int64).reshape(-1, 3),
                                process=False)
                            return (mesh_iou_exact(target, m2),
                                    _s2, _v2, _ops2)
                        except Exception:
                            return None

                    _cur = best
                    _bdil = 0.0
                    for _dil in (0.8, 1.6, 2.4):
                        _r = _score_edit(_dil, 0.0)
                        logger.debug("prism edit dil=%.1f -> %s", _dil,
                                     "fail" if _r is None else _r[0])
                        if _r is not None and _r[0] > _cur[0] + 1e-3:
                            _cur = _r
                            _bdil = _dil
                    for _ext in (1.0, 2.0, 4.0):
                        _r = _score_edit(_bdil, _ext)
                        logger.debug("prism edit dil=%.1f ext=%.1f -> %s",
                                     _bdil, _ext,
                                     "fail" if _r is None else _r[0])
                        if _r is not None and _r[0] > _cur[0] + 1e-3:
                            _cur = _r
                    if _cur is not best:
                        logger.info(
                            "Refine prism EDIT: IoU %.3f -> %.3f "
                            "(footprint/depth re-fit of the largest "
                            "prism cut)", best[0], _cur[0])
                        best = _cur
            except Exception:
                logger.debug("prism edit pass failed", exc_info=True)

        _op_sketch_segments = _refine_op_sketch_segments

        def _emit_steps(ops):
            # Replay the WINNING op-set so each refine operation appears as
            # its own feature-tree step (user ask: refine features must not
            # be an opaque blob). Only ops that change volume materially get
            # a step; the rest fold into the final Refined Model step.
            if steps_out is None:
                return
            try:
                cur = solid
                prev_vol = _solid_volume(cur)
                _emitted = 0
                for op in ops:
                    patch = _refine_op_to_solid(op, cq)
                    if patch is None:
                        continue
                    if op.op == "add":
                        try:
                            if _solid_volume(patch) < 0.004 * mesh_volume:
                                continue
                        except Exception:
                            pass
                    try:
                        nxt = (cur.union(patch) if op.op == "add"
                               else cur.cut(patch))
                    except Exception:
                        continue
                    vol = _solid_volume(nxt)
                    dv = abs(vol - prev_vol)
                    # Cap the ITEMIZED steps: a badly-fit part can accept
                    # 30+ refine boxes (the pillow block hit 34) — each
                    # emitted pair costs two tessellations and floods the
                    # tree. The largest ops come first (greedy); the tail
                    # folds into the following steps' geometry.
                    if dv >= 0.004 * mesh_volume and _emitted < 8:
                        _emitted += 1
                        try:
                            _seg = _op_sketch_segments(op, patch)
                            if _seg is not None and len(_seg):
                                v2, f2 = _tessellate_solid(cur, fast=True)
                                steps_out.append(ConstructionStep(
                                    name=("Sketch Cut" if op.op != "add"
                                          else "Sketch Join"),
                                    tess_vertices=v2, tess_faces=f2,
                                    volume=prev_vol, sketch_overlay=_seg,
                                ))
                            v3, f3 = _tessellate_solid(nxt, fast=True)
                            steps_out.append(ConstructionStep(
                                name=(f"Refine {'Add' if op.op == 'add' else 'Cut'} "
                                      f"({op.kind}, {dv:,.0f}mm³)"),
                                tess_vertices=v3, tess_faces=f3, volume=vol,
                            ))
                        except Exception:
                            pass
                    cur = nxt
                    prev_vol = vol
            except Exception:
                logger.debug("refine step replay failed", exc_info=True)

        if use_iou:
            if best[0] > iou0 + 1e-3:
                logger.info("Residual refine: shape IoU %.3f -> %.3f",
                            iou0, best[0])
                _emit_steps(best[3])
                return best[1], best[2]
            return solid, cad_vol

        # Volume arbiter (Windows / no reliable exact IoU).
        best_vol_err = -best[0]
        if best_vol_err < old_err - 1e-4:
            logger.info("Residual refine: vol err %.2f%% -> %.2f%%",
                        old_err * 100, best_vol_err * 100)
            _emit_steps(best[3])
            return best[1], best[2]
        return solid, cad_vol
    except Exception:
        logger.debug("Residual refine post-pass failed", exc_info=True)
        return solid, cad_vol


def _fillet_refine_solid(solid, cleaned: CleanedMesh, *,
                         max_radius_frac: float = 0.04,
                         n_radii: int = 6):
    """Opt-in fillet detection by IoU sweep (CADFit `refine_edges` style).

    The blend-graph + stitching fillet path (peel_via_topology +
    _apply_recognized_fillets_to_solid) MEASURES fillet radii from mesh
    curvature and maps them onto solid edges — the fragile, bug-prone path the
    fillet handoffs document. This is the alternative: don't measure, SEARCH.
    Sweep a global fillet radius over the solid's edges; build each; score by
    voxel IoU against the target mesh; keep the radius that best improves it.
    A sharp-edged target loses IoU when filleted, so it self-selects to no
    fillet. Dissolves the fillet-vs-designed-radius ambiguity by construction.

    OFF by default: it rebuilds + voxelizes the solid once per radius (slow),
    and v1 applies ONE global radius to all edges (good for uniformly-filleted
    parts, crude for mixed). Per-edge radii + a local (per-edge) IoU score are
    the next increment (docs/CADFIT_PORTING_PLAN.md, Phase 2.3). Volume is a
    weak fillet signal, so we score on voxel IoU at elevated resolution, not
    bulk volume. Returns the (possibly) filleted solid; any failure or
    non-improvement returns the input unchanged.
    """
    try:
        import numpy as _np
        from mesh2param.pipeline.iou_refine import voxel_iou
    except Exception:
        return solid
    if solid is None:
        return solid
    try:
        target = trimesh.Trimesh(vertices=cleaned.vertices,
                                 faces=cleaned.faces, process=False)
        diag = float(_np.linalg.norm(target.bounds[1] - target.bounds[0])) or 1.0

        def _iou_of(s):
            tv, tf = _tessellate_solid(s)
            if tv is None or tf is None or len(tv) == 0:
                return -1.0
            recon = trimesh.Trimesh(
                vertices=_np.asarray(tv, float).reshape(-1, 3),
                faces=_np.asarray(tf, _np.int64).reshape(-1, 3), process=False)
            return voxel_iou(target, recon, res=96)

        base_iou = _iou_of(solid)
        if base_iou < 0:
            return solid
        best_solid, best_iou = solid, base_iou
        # Largest radius first: a too-big fillet fails fast in OCC.
        for r in sorted(_np.linspace(0.01 * diag, max_radius_frac * diag,
                                     n_radii), reverse=True):
            try:
                cand = solid.edges().fillet(float(r))
            except Exception:
                continue
            iou = _iou_of(cand)
            if iou > best_iou + 1e-3:
                best_solid, best_iou = cand, iou
        if best_iou > base_iou + 1e-3:
            logger.info("Fillet refine: IoU %.3f -> %.3f", base_iou, best_iou)
        return best_solid
    except Exception:
        logger.debug("Fillet refine failed", exc_info=True)
        return solid


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
