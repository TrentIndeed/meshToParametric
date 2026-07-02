"""Job queue for concurrent mesh-to-CAD pipeline runs.

Each upload creates a Job with a UUID. A background worker thread
processes jobs one at a time. Frontend polls by job ID.

Completed jobs are persisted to SQLite so they survive server restarts.
On startup, completed jobs are loaded from the DB back into memory.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue

import numpy as np

logger = logging.getLogger(__name__)

MAX_JOBS = 50  # max completed jobs to keep in memory


def _ingest_streamed_steps(job, stream_dir, already: int) -> int:
    """Read construction-step files the CAD subprocess has streamed since the
    last poll and append them to `job.steps`, bumping `job.steps_version`.

    Files are written atomically by export_cadquery as `{NNNN}.pkl` in build
    order. We only read contiguous indices starting at `already`, so a file
    mid-write (its index not yet present) just defers to the next poll tick.
    Returns the new count of ingested steps. Best-effort: never raises into
    the build loop.
    """
    import pickle
    try:
        from mesh2param.pipeline.export_cadquery import ConstructionStep
    except Exception:
        return already
    n = already
    while True:
        f = stream_dir / f"{n:04d}.pkl"
        if not f.exists():
            break
        try:
            with open(f, "rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            # Possibly caught mid-rename; retry on the next tick.
            break
        try:
            job.steps.append(ConstructionStep(
                name=payload.get("name", f"Step {n + 1}"),
                tess_vertices=payload.get("tess_vertices"),
                tess_faces=payload.get("tess_faces"),
                volume=float(payload.get("volume", 0.0) or 0.0),
                sketch_overlay=payload.get("sketch_overlay"),
                overlay_arcs=payload.get("overlay_arcs"),
            ))
            n += 1
        except Exception:
            logger.debug("Failed to ingest streamed step %d", n, exc_info=True)
            break
    if n != already:
        job.steps_version += 1
        job.last_activity_at = time.time()
        logger.info("Live CAD build: %d step(s) streamed for job %s",
                    n, job.id)
    return n


def _project_vertex_to_primitive(vert: np.ndarray, fitted) -> np.ndarray:
    """Snap a vertex onto its region's fitted analytic surface.

    Used for the "render only the primitives" view: each region's
    triangulation is preserved, but every vertex is moved to lie
    exactly on the fitted plane / cylinder / cone / sphere. Shared
    vertices duplicated per region — boundaries become small visible
    gaps that highlight where one primitive ends and another begins.
    """
    from mesh2param.models.region import SurfaceType
    params = fitted.params
    stype = fitted.surface_type
    if stype == SurfaceType.PLANE:
        normal = np.asarray(params["normal"], dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        point = np.asarray(params["point"], dtype=float)
        signed_dist = float(np.dot(vert - point, normal))
        return vert - signed_dist * normal
    if stype == SurfaceType.SPHERE:
        center = np.asarray(params["center"], dtype=float)
        radius = float(params["radius"])
        d = vert - center
        n = np.linalg.norm(d)
        if n < 1e-15:
            return vert
        return center + d * (radius / n)
    if stype == SurfaceType.CYLINDER:
        axis_point = np.asarray(params["axis_point"], dtype=float)
        axis_dir = np.asarray(params["axis_dir"], dtype=float)
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)
        radius = float(params["radius"])
        rel = vert - axis_point
        axial = float(np.dot(rel, axis_dir)) * axis_dir
        radial = rel - axial
        rn = np.linalg.norm(radial)
        if rn < 1e-15:
            return vert
        return axis_point + axial + radial * (radius / rn)
    if stype == SurfaceType.CONE:
        apex = np.asarray(params["apex"], dtype=float)
        axis_dir = np.asarray(params["axis_dir"], dtype=float)
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)
        half_angle = float(np.radians(params["half_angle"]))
        rel = vert - apex
        t = float(np.dot(rel, axis_dir))
        radial = rel - t * axis_dir
        rn = np.linalg.norm(radial)
        expected_r = abs(t) * np.tan(half_angle)
        if rn < 1e-15 or expected_r < 1e-15:
            return vert
        return apex + t * axis_dir + radial * (expected_r / rn)
    return vert


def _build_primitive_mesh(regions, sub_cleaned, fitted_list, color_for_region):
    """Build a (vertices, faces, per-face-colors) mesh by projecting
    each region's vertices onto its fitted primitive.

    Vertices are duplicated per region so each region renders on its
    own exact primitive surface. Returns three numpy arrays ready to
    flatten into the viz wire format.
    """
    fitted_lookup = {f.region_id: f for f in fitted_list}
    new_vertices: list = []
    new_faces: list = []
    new_colors: list = []
    for region in regions:
        f = fitted_lookup.get(region.region_id)
        if f is None or len(region.face_indices) == 0:
            continue
        color = color_for_region.get(region.region_id, [0.6, 0.6, 0.6])
        # Snap CLAMP: a CYLINDER/CONE/SPHERE fitted to a noisy or barely-curved
        # region (e.g. the wall-split leftover) has a wild axis/radius, so
        # projecting the region's vertices onto it flings them far from the mesh
        # — the "hundreds of weird shards" in the primitive view. Cap how far a
        # vertex may move to a fraction of the region's own size; a vertex that
        # would move further is left in place (the fit is bad for it). Good
        # fits (planes, real bores) move vertices only by the noise amplitude,
        # well under the cap, so they're unaffected.
        rvi = np.unique(sub_cleaned.faces[np.asarray(region.face_indices)])
        rverts = sub_cleaned.vertices[rvi]
        rdiag = float(np.linalg.norm(rverts.max(0) - rverts.min(0))) if len(rverts) else 0.0
        # Tight cap: a good fit (plane, real bore) moves a vertex by roughly the
        # mesh noise amplitude (<~1mm), so this keeps every good fit exact while
        # a garbage curved fit (wild axis/radius) gets its vertices held at the
        # mesh — the region then renders as the (correct-looking) mesh patch.
        max_move = max(0.04 * rdiag, 1.0)
        # Per-region vertex projection cache: map original vertex
        # index -> new index in this region's projected vertex set.
        local_remap: dict[int, int] = {}
        local_verts: list[np.ndarray] = []
        for fi in region.face_indices:
            tri = sub_cleaned.faces[fi]
            new_tri = []
            for vi in tri:
                vi_int = int(vi)
                if vi_int not in local_remap:
                    orig = sub_cleaned.vertices[vi_int]
                    projected = _project_vertex_to_primitive(orig, f)
                    if float(np.linalg.norm(projected - orig)) > max_move:
                        projected = orig          # bad fit for this vertex
                    local_remap[vi_int] = len(local_verts)
                    local_verts.append(projected)
                new_tri.append(local_remap[vi_int])
            base_idx = len(new_vertices)
            new_faces.append([base_idx + new_tri[0],
                              base_idx + new_tri[1],
                              base_idx + new_tri[2]])
            new_colors.append(color)
        new_vertices.extend(local_verts)
    if not new_vertices:
        return None
    verts_arr = np.asarray(new_vertices, dtype=np.float32)
    faces_arr = np.asarray(new_faces, dtype=np.int32)
    colors_arr = np.asarray(new_colors, dtype=np.float32)
    return verts_arr, faces_arr, colors_arr


def _color_regions_no_adjacent_match(regions, sub_cleaned, palette):
    """Assign each region a color from the palette such that no two
    mesh-adjacent regions share the same color.

    Greedy graph-coloring: visit regions in size order (largest
    first), and pick the first palette entry that none of its
    already-colored adjacent neighbors are using. If every palette
    color is taken (shouldn't happen with a 12-color palette since
    a 3D mesh region rarely has 12 distinct neighbors), fall back
    to the next palette index by region order.

    Returns a dict region_id -> [r, g, b].
    """
    from collections import defaultdict

    # Build region adjacency from face adjacency.
    face_to_region: dict[int, int] = {}
    for r in regions:
        for fi in r.face_indices:
            face_to_region[int(fi)] = r.region_id

    # Edge-shared face adjacency
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(sub_cleaned.faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            edge_to_faces[edge].append(fi)

    region_neighbors: dict[int, set[int]] = defaultdict(set)
    for face_list in edge_to_faces.values():
        for i in range(len(face_list)):
            ri = face_to_region.get(face_list[i])
            if ri is None:
                continue
            for j in range(i + 1, len(face_list)):
                rj = face_to_region.get(face_list[j])
                if rj is None or rj == ri:
                    continue
                region_neighbors[ri].add(rj)
                region_neighbors[rj].add(ri)

    # 2-hop neighbors: regions that share a common direct neighbor.
    # This catches the "bore + rolled-edge separated by a thin rim
    # fillet" case — the bore and the rolled edge aren't mesh-
    # adjacent, but visually they read as continuous through the
    # narrow rim ring. Forcing them to different colors makes the
    # rim's separation visible at a glance.
    region_two_hop: dict[int, set[int]] = defaultdict(set)
    for ri, neighbors in region_neighbors.items():
        for n in neighbors:
            for nn in region_neighbors.get(n, set()):
                if nn != ri:
                    region_two_hop[ri].add(nn)

    # Compute HSL features for each palette entry. The previous
    # RGB-distance metric let visually-similar pastels (e.g. cyan
    # next to green: both light, both green-dominant, only the
    # B channel differs) end up adjacent — they read as the same
    # color under 3D shading. HSL splits the perceptual axes that
    # matter: hue (where on the color wheel) and lightness (how
    # bright). We score against neighbors using both, weighted
    # so that two LIGHT colors next to each other or two
    # SAME-HUE colors next to each other are both penalised.
    import colorsys
    palette_hsl: list[tuple[float, float, float]] = []
    for r, g, b in palette:
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        palette_hsl.append((h, l, s))

    def _hue_dist(a: int, b: int) -> float:
        # Circular hue distance in [0, 0.5]
        ha, hb = palette_hsl[a][0], palette_hsl[b][0]
        d = abs(ha - hb)
        return min(d, 1.0 - d)

    def _light_dist(a: int, b: int) -> float:
        return abs(palette_hsl[a][1] - palette_hsl[b][1])

    # Minimum hue separation that reads as "different hue family"
    # at a glance. 0.20 = 72° on the color wheel — well past the
    # 60° boundary where adjacent hues (e.g. red and pink-magenta,
    # red and orange) start blending visually. The palette is
    # designed with 4 hue families at 90° apart so any two
    # distinct hues exceed this threshold.
    HUE_SEPARATION = 0.20

    def _hue_far_enough(a: int, b: int) -> bool:
        return _hue_dist(a, b) >= HUE_SEPARATION

    def _perceptual_dist(a: int, b: int) -> float:
        # Hue × 4, lightness × 1 — hue dominates because human
        # vision distinguishes hues much faster than lightness.
        return 4.0 * _hue_dist(a, b) + 1.0 * _light_dist(a, b)

    # Greedy color assignment, largest regions first. For each
    # region, pick the palette color that MAXIMIZES the minimum
    # perceptual distance to its 1-hop and 2-hop neighbors. The
    # tiered score prefers (a) not-already-used-by-1-hop, then
    # (b) not-used-by-2-hop, then (c) max perceptual distance
    # against 1-hop, (d) against 2-hop. This guarantees adjacent
    # regions get colors with different hues AND different
    # lightness — preventing both same-color and same-shade
    # collisions.
    sorted_regions = sorted(regions, key=lambda r: -r.num_faces)
    color_idx_for: dict[int, int] = {}
    for r in sorted_regions:
        rid = r.region_id
        one_hop_used = [
            color_idx_for[n]
            for n in region_neighbors.get(rid, set())
            if n in color_idx_for
        ]
        two_hop_used = [
            color_idx_for[n]
            for n in region_two_hop.get(rid, set())
            if n in color_idx_for
        ]
        # Score tiers (highest priority first):
        #   1. hue-distinct from every 1-hop neighbor
        #   2. hue-distinct from every 2-hop neighbor
        #   3. max of MIN perceptual distance over 1-hop ∪ 2-hop
        #      (the "weakest link" — guarantees no neighbor at any
        #      hop is too close)
        #   4. max sum of distances (tiebreaker)
        # The weakest-link metric prevents the failure mode where
        # min_dist_1hop dominated and let a candidate with
        # min_dist_2hop = 0 win — which produced bore + tab-top
        # both getting THE SAME palette index.
        all_used = one_hop_used + two_hop_used
        best_ci, best_score = 0, (-1, -1, -1.0, -1.0)
        for ci in range(len(palette)):
            hue_ok_1hop = (
                1 if all(_hue_far_enough(ci, c) for c in one_hop_used) else 0
            )
            hue_ok_2hop = (
                1 if all(_hue_far_enough(ci, c) for c in two_hop_used) else 0
            )
            if all_used:
                weakest = min(_perceptual_dist(ci, c) for c in all_used)
                total = sum(_perceptual_dist(ci, c) for c in all_used)
            else:
                weakest, total = 4.0, 4.0
            score = (hue_ok_1hop, hue_ok_2hop, weakest, total)
            if score > best_score:
                best_score = score
                best_ci = ci
        color_idx_for[rid] = best_ci
    return {rid: palette[ci] for rid, ci in color_idx_for.items()}


# Ordered list of pipeline stages that support a pause/breakpoint, as
# (stage_key, display_name) pairs. The frontend's "pause at stage"
# dropdown is populated from this list (kept in sync in app.js). The
# stage_key is what gets stored in Job.pause_at_stage and matched in
# _pause_for_review.
PAUSE_STAGES: list[tuple[str, str]] = [
    ("scan_cleanup", "Scan cleanup (denoise + align)"),
    ("denoise", "Denoise (feature-preserving)"),
    ("align", "Align to canonical frame"),
    ("regularize", "Regularize (flatten faces, snap 90°)"),
    ("decompose", "Decompose bodies"),
    ("segment", "Segmentation"),
    ("fit", "Primitive fitting"),
    ("topology", "Topology & edges"),
    ("blend_graph", "Blend graph"),
    ("stitched", "Stitched fillets (debug)"),
    ("features", "Feature recognition"),
    # build / intent / apply checkpoints removed — the two-phase right-pane
    # build now shows the model accreting and fillets applying live, so these
    # left-pane re-shows were redundant. Reconcile stays (a real diagnostic).
    ("reconcile", "Blend reconciliation (debug)"),
]
PAUSE_STAGE_KEYS = {k for k, _ in PAUSE_STAGES}


@dataclass
class Job:
    id: str
    filename: str
    file_path: str  # temp file path
    client_ip: str = ""  # for per-IP billing
    status: str = "queued"  # queued | processing | done | error
    progress: int = 0
    progress_stage: str = "Queued"
    paused: bool = False  # True when waiting for user to press Continue
    step_through: bool = True  # pause after each visual stage
    # One-shot breakpoint: when step_through is off, pause at exactly this
    # stage key (see PAUSE_STAGES). Cleared once the user continues past it,
    # so the rest of the pipeline runs without further pauses.
    pause_at_stage: str | None = None
    error: str | None = None
    # Worker-watchdog timestamp. Refreshed every time the worker calls
    # `_check_cancelled()` (which happens at most stage transitions and
    # inside the long polls). When this falls too far behind real time
    # while status is "processing", the worker is hung and the queue
    # auto-force-resets it.
    last_activity_at: float = field(default_factory=time.time)
    # Pipeline results
    features: object = None
    cleaned: object = None
    # Durable pickle of (tree, cleaned, body_results, decomposition) written at
    # build time so the feature-tree editor can re-run export on a mutated tree.
    cad_bundle_path: str | None = None
    # STEP of the current built solid (refreshed on each edit) — source for
    # pickable-edge extraction and the edited-model download.
    cad_step_path: str | None = None
    # Cached BASE solid (tree minus current deletes, no finishes) so finish-only
    # edits skip the export. _edit_base_deletes tracks which delete set it holds.
    cad_editbase_path: str | None = None
    steps: list = field(default_factory=list)
    variants: list = field(default_factory=list)  # list of {name, strategy, steps}
    active_variant: int = 0  # index of currently selected variant
    mesh_volume: float = 0.0
    cad_volume: float = 0.0
    # Progress visualization — updated per stage for live rendering
    progress_vertices: list | None = None  # flat float list
    progress_faces: list | None = None     # flat int list
    progress_colors: list | None = None    # flat float list (per-face RGB)
    progress_colors_per_face: bool = False
    progress_edges: list | None = None     # flat float list (line segment pairs)
    progress_edge_colors: list | None = None  # flat float list (per-edge-vertex RGB)
    progress_version: int = 0  # incremented on each viz update
    # Live CAD construction streaming — bumped each time a new construction
    # step is ingested from the CAD subprocess while it's still building, so
    # the dual-pane UI's right pane accretes features in real time.
    steps_version: int = 0
    building_live: bool = False  # True while CAD steps are streaming in
    bbox_min: list = field(default_factory=lambda: [0, 0, 0])
    bbox_max: list = field(default_factory=lambda: [0, 0, 0])
    token_cost: int = 0
    created_at: float = field(default_factory=time.time)
    # Per-stage debug summary lines, e.g.
    #   "[Body 0] Stage 2: 17 regions (5 plates, 10 curves, 2 cyl) median=42f"
    # Populated as the pipeline runs and exposed via /api/job/{id}/status
    # so the frontend Debug toggle can show them.
    debug_summary: list = field(default_factory=list)


class JobQueue:
    """Thread-safe job queue with background worker."""

    def __init__(self):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: Queue[str] = Queue()
        self._lock = threading.Lock()
        self._worker = None
        self._tokens_spent = 0
        self._on_complete = None  # callback: fn(job) called when job finishes
        self._db_loaded = False  # True after initial DB load
        # Background watchdog: catches the "worker thread is alive but
        # not making progress" case even when nobody submits a new job.
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="job-queue-watchdog",
        )
        self._watchdog.start()
        # Cloud (USE_MODAL=1): poll the parallel Modal calls and finalize jobs
        # as they complete. No-op cost when disabled (thread just sleeps).
        self._modal_poller = threading.Thread(
            target=self._modal_poller_loop, daemon=True, name="modal-poller",
        )
        self._modal_poller.start()

    def _modal_poller_loop(self):
        """Finalize jobs dispatched to Modal as their isolated containers finish."""
        from mesh2param.viz_server import modal_runner
        while True:
            time.sleep(1.0)
            if not modal_runner.is_enabled():
                continue
            for job in list(self._jobs.values()):
                if job.status == "processing" and getattr(job, "_modal_call_id", None) \
                        and not getattr(job, "_modal_done", False):
                    try:
                        if modal_runner.poll_and_finalize(job):
                            job._modal_done = True
                            if job.status == "done":
                                # Parity with the local completion path: account
                                # tokens + persist to SQLite so cloud jobs survive
                                # restarts and show in history.
                                try:
                                    self._tokens_spent += job.token_cost
                                    self._save_to_db(job)
                                except Exception:
                                    logger.exception("Modal job %s: token/DB save failed", job.id)
                                if self._on_complete:
                                    try:
                                        self._on_complete(job)
                                    except Exception:
                                        logger.debug("on_complete failed", exc_info=True)
                    except Exception:
                        logger.exception("Modal poller error on job %s", job.id)

    def set_on_complete(self, fn):
        """Set a callback for when a job completes. fn(job) is called."""
        self._on_complete = fn

    @property
    def tokens_spent(self) -> int:
        return self._tokens_spent

    def submit(self, filename: str, file_path: str, client_ip: str = "", num_variants: int = 1) -> str:
        """Submit a new job. Returns job ID."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, filename=filename, file_path=file_path, client_ip=client_ip)
        job._num_variants = max(1, min(4, num_variants))
        job._target_dims = None
        logger.info("Job submitted: id=%s filename=%r ip=%s", job_id, filename, client_ip)

        with self._lock:
            self._jobs[job_id] = job
            # Evict oldest completed jobs if over limit
            if len(self._jobs) > MAX_JOBS:
                to_remove = []
                for jid, j in self._jobs.items():
                    if j.status in ("done", "error") and len(self._jobs) - len(to_remove) > MAX_JOBS:
                        to_remove.append(jid)
                for jid in to_remove:
                    del self._jobs[jid]
                if to_remove:
                    logger.debug("Evicted %d old jobs from memory", len(to_remove))

        self._queue.put(job_id)
        self._ensure_worker()
        return job_id

    # Hard ceiling on how long a job can run without a worker heartbeat
    # before the queue declares the worker thread hung and spawns a
    # fresh one. The old thread becomes an orphan daemon — Python can't
    # kill threads cleanly, but it won't hold up the queue.
    STUCK_THRESHOLD_S = 180.0

    def _ensure_worker(self):
        """Start the worker thread if not already running, AND force-
        restart it if the current processing job has gone silent past
        the stuck threshold.

        Called by `submit()` and by the background watchdog. Cheap and
        safe to call often — if nothing's wrong it's a no-op."""
        if self._worker is not None and self._worker.is_alive():
            stuck_job = self._find_stuck_job()
            if stuck_job is not None:
                self._force_new_worker(stuck_job)
                return
            return  # worker fine
        # No live worker — spawn one (also covers the initial-startup case).
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _find_stuck_job(self):
        """Return the processing job that's gone past STUCK_THRESHOLD_S
        with no `_check_cancelled` heartbeat, else None. Paused jobs
        (`job.paused == True`) are excluded — those are waiting on the
        user, not stuck."""
        now = time.time()
        with self._lock:
            for j in self._jobs.values():
                if j.status != "processing" or j.paused:
                    continue
                if now - j.last_activity_at > self.STUCK_THRESHOLD_S:
                    return j
        return None

    def _force_new_worker(self, stuck_job=None) -> None:
        """Mark any stuck job as errored, terminate every tracked
        subprocess (across all jobs, just to be safe), and spawn a
        fresh worker thread. The previous thread becomes orphan —
        Python can't reliably kill threads, but daemonised it'll exit
        with the process and meanwhile won't block the queue."""
        with self._lock:
            jobs_snapshot = list(self._jobs.values())
        for j in jobs_snapshot:
            proc = getattr(j, "_cad_subprocess", None)
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
                j._cad_subprocess = None
            if j is stuck_job:
                j.status = "error"
                j.paused = False
                if not j.error:
                    j.error = "Worker stuck — auto-reset"
        logger.warning(
            "Force-resetting worker thread (stuck job: %s)",
            stuck_job.id if stuck_job else "(none)",
        )
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def force_reset(self) -> dict:
        """Manual reset entry point exposed via the API. Kills any
        in-flight subprocess, marks the active job as errored, and
        replaces the worker thread."""
        with self._lock:
            active = next(
                (j for j in self._jobs.values() if j.status == "processing"),
                None,
            )
        self._force_new_worker(active)
        return {"reset": True, "killed_job_id": active.id if active else None}

    def _watchdog_loop(self):
        """Background thread: every 15 s, check for a hung worker and
        force-restart if so. Catches the case where the user is just
        waiting (no new submissions) and so `_ensure_worker` never
        runs to trigger the auto-reset."""
        while True:
            time.sleep(15.0)
            try:
                stuck = self._find_stuck_job()
                if stuck is not None:
                    self._force_new_worker(stuck)
            except Exception:
                logger.exception("Watchdog crashed; continuing")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        # Try loading from DB if not in memory
        return self._load_single_from_db(job_id)

    def _worker_loop(self):
        """Process jobs one at a time."""
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._run_pipeline(job)
            except Exception as e:
                # Preserve "Cancelled by user" if it was set by the cancel endpoint
                if job.error != "Cancelled by user":
                    job.error = str(e)
                job.status = "error"
                logger.exception("Job %s failed", job_id)
            finally:
                # Keep temp file for cancelled jobs so user can re-run,
                # and for successfully done jobs so the UI's Reanalyze
                # button can re-trigger the pipeline on the same mesh.
                # Only delete on hard error (file is probably already
                # broken or unrecognized and re-running won't help).
                cancelled = (
                    job.error == "Cancelled by user"
                    or job.error == "Job cancelled"
                    or "cancel" in (job.error or "").lower()
                )
                # Modal jobs are still "processing" right after dispatch (the
                # poller finalizes later) — keep their file so re-analyze works.
                keep = (cancelled or job.status == "done"
                        or getattr(job, "_modal_call_id", None) is not None)
                if not keep:
                    Path(job.file_path).unlink(missing_ok=True)

    def _launch_cad_subprocess(self, job, num_variants):
        """Pre-warm + Popen the CadQuery build subprocess; return its handle
        bundle. Input is provided LATER via `_write_cad_input`.

        Launched EARLY (at ingest) so the slow `import cadquery` (OCC load,
        ~15 s) overlaps the analysis stages instead of blocking the build. The
        subprocess imports CadQuery, then POLLS for its input pickle, which the
        parent drops once recognition has produced the feature tree — so the
        build starts the instant the data is ready, with CadQuery already
        loaded, and its steps stream during the remaining analysis viz. It then
        waits at `clusters_path` for the stitched fillets (phase two).
        """
        import tempfile
        cad_input_path = Path("data") / f"cad_input_{job.id}.pkl"
        # Persist the built solid as STEP so the feature-tree editor can extract
        # pickable edges fast and offer a download without re-running the build.
        cad_solid_path = Path("data") / f"cad_solid_{job.id}.step"
        # Separate cached BASE solid (no user finishes) so finish-only edits can
        # skip the export and just re-fillet — see rebuild.rebuild_from_bundle.
        cad_editbase_path = Path("data") / f"cad_editbase_{job.id}.step"
        for _p in (cad_solid_path, cad_editbase_path):
            try:
                Path(_p).unlink(missing_ok=True)
            except Exception:
                pass
        job.cad_step_path = str(cad_solid_path)
        job.cad_editbase_path = str(cad_editbase_path)
        try:
            cad_input_path.unlink(missing_ok=True)
        except Exception:
            pass
        tmp_out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp_out.close()

        clusters_path = Path("data") / f"cad_clusters_{job.id}.pkl"
        try:
            clusters_path.unlink(missing_ok=True)
        except Exception:
            pass
        job._cad_clusters_path = str(clusters_path)

        cad_stream_dir = Path("data") / f"cad_stream_{job.id}"
        try:
            if cad_stream_dir.exists():
                import shutil as _sh
                _sh.rmtree(cad_stream_dir, ignore_errors=True)
            cad_stream_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("Could not prepare CAD stream dir", exc_info=True)
        job.steps = []
        job.steps_version = 0
        # NOT building_live yet — the subprocess is only PRE-WARMING (loading
        # CadQuery) and has no input. Showing the right pane now would leave it
        # blank for the whole analysis. _write_cad_input flips it on when the
        # build actually starts.
        job.building_live = False
        job._cad_stream_ingested = 0

        script = f"""
import pickle, sys, os, time, logging, traceback, faulthandler
faulthandler.enable()
logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format='%(levelname)s %(name)s: %(message)s',
)
sys.path.insert(0, '.')
try:
    # Pre-warm: pay the slow OCC import NOW, while the parent is still
    # analysing, so the build starts instantly once the input arrives.
    import cadquery  # noqa: F401
    print('CAD engine ready; waiting for feature tree...', flush=True)
    _in = r'{cad_input_path}'
    # Wait generously: in STEP MODE the user pauses at each analysis stage, so
    # the gap from pre-warm to feature-tree-ready can be many minutes. The old
    # 300s timeout fired during those pauses and surfaced as a bogus "CadQuery
    # failed". 1 hour covers any realistic step-through.
    _dl = time.time() + 3600
    while time.time() < _dl:
        if os.path.exists(_in) and os.path.getsize(_in) > 0:
            time.sleep(0.15)   # let the parent's atomic rename settle
            break
        time.sleep(0.1)
    if not (os.path.exists(_in) and os.path.getsize(_in) > 0):
        print('No feature tree after wait window; exiting cleanly', flush=True)
        sys.exit(0)
    with open(_in, 'rb') as f:
        tree, cleaned, all_body_results, decomposition, stitched_clusters = pickle.load(f)
    from mesh2param.pipeline.export_cadquery import export_cadquery
    result = export_cadquery(tree, cleaned, use_llm=False, num_variants={num_variants},
                             body_results=all_body_results if decomposition.is_multi_body else None,
                             stitched_clusters=stitched_clusters,
                             stream_dir=r'{cad_stream_dir}',
                             stitched_clusters_path=r'{clusters_path}')
    try:
        if result.solid is not None:
            import cadquery as _cq
            _cq.exporters.export(result.solid, r'{cad_solid_path}')
            import shutil as _sh
            _sh.copyfile(r'{cad_solid_path}', r'{cad_editbase_path}')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    result.solid = None
    with open(r'{tmp_out.name}', 'wb') as f:
        pickle.dump(result, f)
except BaseException as _exc:
    print('=== SUBPROCESS EXCEPTION ===', flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    sys.exit(2)
"""
        cad_log_path = Path("data") / f"cad_subproc_{job.id}.log"
        cad_log_path.parent.mkdir(parents=True, exist_ok=True)
        cad_err_path = Path("data") / f"cad_subproc_{job.id}.err.log"
        cad_log_fh = open(cad_log_path, "w", encoding="utf-8")
        cad_err_fh = open(cad_err_path, "w", encoding="utf-8")
        proc_handle = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=cad_log_fh, stderr=cad_err_fh, text=True,
        )
        job._cad_subprocess = proc_handle
        logger.info("CAD build subprocess pre-warmed (pid=%s) for job %s",
                    proc_handle.pid, job.id)
        return {
            'proc_handle': proc_handle, 'cad_input_path': cad_input_path,
            'tmp_out': tmp_out, 'cad_log_path': cad_log_path,
            'cad_err_path': cad_err_path, 'cad_log_fh': cad_log_fh,
            'cad_err_fh': cad_err_fh, 'cad_stream_dir': cad_stream_dir,
            'deadline': time.time() + 600.0, 'clusters_path': clusters_path,
            'input_written': False, 'job': job,
        }

    def _write_cad_input(self, bundle, tree, cleaned, all_body_results,
                         decomposition):
        """Hand the pre-warmed CAD subprocess its inputs (atomic write), so it
        unblocks and starts building. Returns True on success."""
        import os
        import pickle
        if bundle is None or bundle.get('input_written'):
            return False
        p = str(bundle['cad_input_path'])
        try:
            with open(p + ".tmp", "wb") as f:
                pickle.dump(
                    (tree, cleaned, all_body_results, decomposition, []), f)
            os.replace(p + ".tmp", p)
            bundle['input_written'] = True
            bundle['job'].building_live = True  # build is starting -> show pane
            # Durable copy for interactive re-builds (feature tree editing): the
            # /api/job/{id}/edit endpoint loads this, mutates the tree, and
            # re-runs export. Kept separate from cad_input (which gets unlinked).
            try:
                dur = Path("data") / f"cad_bundle_{bundle['job'].id}.pkl"
                with open(str(dur) + ".tmp", "wb") as f:
                    pickle.dump(
                        (tree, cleaned, all_body_results, decomposition), f)
                os.replace(str(dur) + ".tmp", str(dur))
                bundle['job'].cad_bundle_path = str(dur)
            except Exception:
                logger.debug("Could not persist durable CAD bundle", exc_info=True)
            return True
        except Exception:
            logger.exception("Failed to hand input to CAD subprocess")
            return False

    def _run_pipeline(self, job: Job):
        """Run the full mesh-to-CAD pipeline for a job."""
        import pickle
        from mesh2param.config import (
            IngestConfig,
        )
        from mesh2param.pipeline.ingest import ingest_mesh

        # Cloud path (USE_MODAL=1): SPAWN the build in an isolated Modal container
        # and return immediately — the worker fans the queue out to parallel
        # containers, and the poller thread finalizes each as it completes. With
        # USE_MODAL unset this is skipped entirely and the local path below runs.
        from mesh2param.viz_server import modal_runner
        if modal_runner.is_enabled():
            modal_runner.dispatch(job)
            return

        job.status = "processing"
        logger.info("Pipeline started for job %s (file=%r)", job.id, job.filename)

        def _check_cancelled():
            # Worker-watchdog heartbeat: any call into _check_cancelled
            # means the worker is making forward progress in user code.
            # If this stops getting called, the watchdog will eventually
            # force-restart the worker thread.
            job.last_activity_at = time.time()
            if job.status == "error":
                raise RuntimeError("Job cancelled")

        def _pause_for_review(stage_key: str | None = None):
            """Pause pipeline so the user can review the current visualization.

            Pauses when either step-through mode is on (pause at every
            stage) OR a one-shot breakpoint targets this stage_key. After
            the user continues past a breakpoint, it's cleared so the rest
            of the run proceeds without pausing.
            """
            breakpoint_hit = (
                stage_key is not None
                and job.pause_at_stage is not None
                and stage_key == job.pause_at_stage
            )
            if not job.step_through and not breakpoint_hit:
                return
            job.paused = True
            while job.paused:
                if job.status == "error":
                    break
                time.sleep(0.2)
            job.paused = False
            if breakpoint_hit:
                job.pause_at_stage = None  # one-shot — don't pause again

        def _dwell_if_not_step_through(stage_key: str | None = None, seconds: float = 2.5):
            """Hold the current progress viz long enough for the frontend's
            poll loop to catch it. Called after EVERY stage's viz
            emission: stages whose follow-on computation is fast would
            otherwise flash for a fraction of a second in non-step-through
            mode. In step-through mode (or when this stage is the
            breakpoint target) the explicit pause handles visibility
            instead.
            """
            if job.step_through:
                return
            if stage_key is not None and stage_key == job.pause_at_stage:
                return  # about to pause here anyway
            elapsed = 0.0
            while elapsed < seconds:
                if job.status == "error":
                    break
                # If the CAD build was launched early and is building
                # concurrently, ingest its streamed steps during this dwell so
                # the right pane accretes features while the left still shows
                # analysis stages.
                _ln = getattr(job, "_cad_launch", None)
                if _ln is not None:
                    try:
                        job._cad_stream_ingested = _ingest_streamed_steps(
                            job, _ln["cad_stream_dir"],
                            getattr(job, "_cad_stream_ingested", 0))
                    except Exception:
                        pass
                time.sleep(0.25)
                elapsed += 0.25

        # Helper: set mesh visualization on the job for live progress
        def _set_mesh_viz(cleaned_mesh, face_colors=None, edge_pts=None, edge_colors=None):
            """Update the job's progress geometry for the frontend to render."""
            _check_cancelled()
            job.progress_vertices = cleaned_mesh.vertices.flatten().tolist()
            job.progress_faces = cleaned_mesh.faces.flatten().tolist()
            if face_colors is not None:
                # Send per-FACE colors (num_faces, 3) — frontend builds
                # non-indexed geometry and assigns each face's 3 vertices
                # the same color for sharp per-face coloring.
                job.progress_colors = face_colors.flatten().tolist()
                job.progress_colors_per_face = True
            else:
                job.progress_colors = None
                job.progress_colors_per_face = False
            job.progress_edges = edge_pts.flatten().tolist() if edge_pts is not None else None
            job.progress_edge_colors = edge_colors.flatten().tolist() if edge_colors is not None else None
            job.progress_version += 1
            logger.info("Progress viz v%d: %d verts, %d faces, colors=%s, edges=%s", job.progress_version,
                        len(job.progress_vertices), len(job.progress_faces),
                        'yes' if job.progress_colors else 'no',
                        'yes' if job.progress_edges else 'no')

        def _set_body_viz(body_colors_list):
            """Unified viz for one or more bodies. Each entry is (body, sub_cleaned, face_colors).
            Single body: renders on its own mesh. Multi body: renders each as a piece."""
            # Clear the PREVIOUS stage's overlays — leftover edge overlays
            # (topology) and piece renders (decompose) stacked under the
            # new stage's colors (the stitched-fillets view rendered on
            # top of the primitive-fits visual).
            job.progress_edges = None
            job.progress_edge_colors = None
            if len(body_colors_list) == 1:
                body, sub_cleaned, fc = body_colors_list[0]
                job._decomp_pieces = None
                _set_mesh_viz(sub_cleaned, face_colors=fc)
            else:
                pieces = []
                for bi, (body, sub_cleaned, fc) in enumerate(body_colors_list):
                    pieces.append({
                        "vertices": sub_cleaned.vertices.flatten().tolist(),
                        "faces": sub_cleaned.faces.flatten().tolist(),
                        "colors": fc.flatten().tolist() if fc is not None else None,
                        "color": body_colors[bi % len(body_colors)],
                    })
                job.progress_colors = None
                job.progress_colors_per_face = False
                job._decomp_pieces = pieces
                job.progress_version += 1

        # Stage 1: Ingest
        job.progress = 5
        job.progress_stage = "Ingesting mesh..."
        logger.debug("Job %s: stage=ingest file=%s", job.id, job.file_path)
        # scan_auto: dense scans (>>30k faces) auto-get the scan
        # front-end (feature-preserving denoise + canonical-frame
        # alignment); clean CAD passes through untouched.
        # Capture the cleanup front-end's intermediate meshes (post-denoise,
        # post-align) so each becomes its own visual stage below.
        _cleanup_inter: dict = {}
        cleaned = ingest_mesh(
            job.file_path, IngestConfig(remesh=False, scan_auto=True),
            on_stage=lambda name, m: _cleanup_inter.__setitem__(name, m))

        # Resize mesh if target dimensions were specified
        target_dims = getattr(job, '_target_dims', None)
        if target_dims and any(v for v in target_dims.values() if v):
            tw, th, tt = target_dims.get('width'), target_dims.get('height'), target_dims.get('thickness')
            if any([tw, th, tt]):
                bbox_size = cleaned.bbox[1] - cleaned.bbox[0]
                target = sorted([tw or 0, th or 0, tt or 0], reverse=True)
                actual = sorted(bbox_size.tolist(), reverse=True)
                scales = [t / a if t and a > 0 else 1.0 for t, a in zip(target, actual)]
                axis_order = sorted(range(3), key=lambda i: -bbox_size[i])
                scale_xyz = np.array([1.0, 1.0, 1.0])
                for i, ax in enumerate(axis_order):
                    scale_xyz[ax] = scales[i]
                center = (cleaned.bbox[0] + cleaned.bbox[1]) / 2
                cleaned.vertices = (cleaned.vertices - center) * scale_xyz + center
                cleaned.bbox = np.array([cleaned.vertices.min(axis=0), cleaned.vertices.max(axis=0)])
                logger.info("Resized mesh after cleaning: scales=%s new_dims=%s",
                            scale_xyz.tolist(), (cleaned.bbox[1] - cleaned.bbox[0]).tolist())

        _set_mesh_viz(cleaned)

        # PRE-WARM the CAD build subprocess NOW (right after ingest) so its slow
        # `import cadquery` overlaps decompose + recognition + the viz stages.
        # It blocks waiting for the feature tree, which we hand it the instant
        # recognition finishes — so the right pane starts building during the
        # analysis stages instead of sitting blank until the CAD stage.
        job._cad_launch = None
        try:
            job._cad_launch = self._launch_cad_subprocess(
                job, getattr(job, '_num_variants', 1))
        except Exception:
            logger.exception("CAD pre-warm failed; will build synchronously")
            job._cad_launch = None

        # Stage 1b: cleanup front-end visuals for SCAN / AI ("dirty") inputs —
        # one verifiable stage per distinct transformation (Denoise, Align;
        # Regularize comes after fitting). Clean CAD skips all of these (the
        # front-end never ran).
        if cleaned.metadata.get("scan_mode"):
            rf = int(cleaned.metadata.get("raw_faces", 0))
            am = cleaned.metadata.get("align_method", "?")
            src = cleaned.metadata.get("cleanup_source", "scan")
            kind = "AI/dirty mesh" if src == "dirty_mesh" else "scan"
            job.progress = 5
            # Denoise: feature-preserving surface smoothing (no faceting).
            if "denoise" in _cleanup_inter:
                _set_mesh_viz(_cleanup_inter["denoise"])
                job.progress_stage = (
                    f"Denoise ({kind}): feature-preserving surface smoothing")
                job.debug_summary.append(
                    f"Denoise: {rf:,} raw faces, surface noise smoothed "
                    f"(edges preserved)")
                job.progress_version += 1
                _dwell_if_not_step_through("denoise")
                _pause_for_review("denoise")
            # Align: rotate onto the part's own orthogonal frame.
            ext = np.round(cleaned.bbox[1] - cleaned.bbox[0], 1).tolist()
            if "align" in _cleanup_inter:
                _set_mesh_viz(_cleanup_inter["align"])
                job.progress_stage = (
                    f"Align: canonical {am} frame — extents {ext} mm")
                job.debug_summary.append(
                    f"Align: rotated onto {am} frame, extents {ext} mm")
                job.progress_version += 1
                _dwell_if_not_step_through("align")
                _pause_for_review("align")
            _set_mesh_viz(cleaned)

        # Whole-mesh segment + peel was REMOVED. It modified `cleaned`
        # in-place before decomposition, so per-body segmentation saw
        # an already-peeled mesh with no fillets left to find. Blend
        # records now come from the per-body pass and get cross-body
        # grouping by the stitched-fillets stage.

        # The stitched-fillets stage populates this with the cluster
        # payload (group_id + face-vertex array) consumed by the CAD
        # apply pass. Stays empty if the stitched stage didn't run.
        job._stitched_clusters = []
        # uid → group_id, used by the reconcile-stage viz at the end
        # of the run to recolor each fillet face by apply-stage fate.
        job._uid_to_group = {}

        # Stage 1.5: Body Decomposition — for genuinely DISCONNECTED
        # shells only (a true assembly of separate parts). A connected
        # solid is ONE part however many extrusions it takes: the
        # extrusion engine decomposes it into base + JOIN extrudes in a
        # single feature tree (RackMount = plus-base with its 4 tabs +
        # two side plates), instead of virtual body cuts + recombination.
        from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
        from mesh2param.config import DecomposeConfig
        from mesh2param.models.feature import FeatureType
        job.progress = 6
        job.progress_stage = "Decomposing bodies..."
        import trimesh as _conn_tm
        _conn_mesh = _conn_tm.Trimesh(
            vertices=cleaned.vertices, faces=cleaned.faces, process=False,
        )
        try:
            _n_components = len(_conn_mesh.split(only_watertight=False))
        except Exception:
            _n_components = 1
        decomposition = decompose_bodies(
            cleaned, DecomposeConfig(enabled=_n_components > 1),
        )

        # Fixed rotation: warm/cool alternating, punchy saturated.
        # Defined unconditionally — the fit/viz stages below index into
        # it for single-body parts too (defining it only in the
        # multi-body branch made every single-body job crash with
        # UnboundLocalError).
        body_colors = [
            [0.0, 0.55, 1.0],  # 1 vivid blue
            [1.0, 0.35, 0.0],  # 2 orange (no green channel = no yellow)
            [0.0, 1.0, 0.25],  # 3 neon green
            [1.0, 0.0, 0.55],  # 4 hot pink
            [0.0, 0.15, 0.9],  # 5 deep blue
            [0.0, 0.9, 0.8],   # 6 cyan
            [0.7, 0.0, 1.0],   # 7 purple
            [1.0, 0.2, 0.0],   # 8 red-orange
        ]

        if decomposition.is_multi_body:
            # Send the cut pieces as separate meshes to the frontend
            pieces = []
            for bi, body in enumerate(decomposition.bodies):
                piece = body.metadata.get("piece_mesh")
                if piece is None:
                    continue
                pieces.append({
                    "vertices": piece.vertices.flatten().tolist(),
                    "faces": piece.faces.flatten().tolist(),
                    "color": body_colors[bi % len(body_colors)],
                })
            job.progress_stage = f"Decomposed into {len(decomposition.bodies)} bodies"
            job.progress_vertices = cleaned.vertices.flatten().tolist()
            job.progress_faces = cleaned.faces.flatten().tolist()
            job.progress_colors = None
            job.progress_colors_per_face = False
            job._decomp_pieces = pieces  # separate meshes for frontend
            job.progress_version += 1
            logger.info("Multi-body: %d bodies detected", len(decomposition.bodies))
            if job.step_through or job.pause_at_stage == "decompose":
                _dwell_if_not_step_through("decompose")
                _pause_for_review("decompose")
            else:
                # Step mode off — still let the explode animation play (3s)
                for _ in range(15):
                    if job.status == "error":
                        break
                    time.sleep(0.2)
        else:
            logger.info("Single body detected")

        # Keep decomp pieces visible until segmentation viz overwrites them
        # (_set_body_viz will replace them with stage-colored pieces)
        # Store decomposition for export stage
        job._decomposition = decomposition

        # ── Stages 2-5: run per body (or full mesh for single-body) ──

        # 4 distinct hue families × 3 lightness tiers = 12 colors.
        # Hues at 90° intervals (red, lime, cyan, purple) so any
        # two distinct hues land in clearly different families —
        # never "red-orange" next to "pink-magenta" the way 60°
        # gaps allowed. Lightness alternates across indices so
        # consecutive picks differ in BOTH hue and brightness.
        palette = [
            [1.00, 0.55, 0.55],  # 0  light red    (hue   0°, L 0.78)
            [0.45, 0.05, 0.05],  # 1  dark red     (hue   0°, L 0.25)
            [0.65, 1.00, 0.40],  # 2  light lime   (hue  90°, L 0.70)
            [0.20, 0.50, 0.00],  # 3  dark olive   (hue  90°, L 0.25)
            [0.45, 0.95, 1.00],  # 4  light cyan   (hue 188°, L 0.73)
            [0.00, 0.40, 0.55],  # 5  dark teal    (hue 196°, L 0.27)
            [0.85, 0.55, 1.00],  # 6  light purple (hue 273°, L 0.78)
            [0.30, 0.00, 0.55],  # 7  dark purple  (hue 273°, L 0.27)
            [1.00, 0.85, 0.55],  # 8  cream        (hue  40°, L 0.78)
            [0.55, 0.30, 0.00],  # 9  dark amber   (hue  33°, L 0.27)
            [0.40, 1.00, 0.85],  # 10 mint         (hue 158°, L 0.70)
            [0.10, 0.30, 0.65],  # 11 navy         (hue 218°, L 0.38)
        ]
        edge_type_colors = {
            'SHARP': [1.0, 1.0, 1.0], 'FILLET': [0.35, 0.65, 1.0],
            'CHAMFER': [1.0, 0.85, 0.25], 'SMOOTH': [0.5, 0.5, 0.5],
        }
        feature_type_colors = {
            FeatureType.BASE_EXTRUSION: [0.0, 0.55, 1.0],
            FeatureType.THROUGH_HOLE: [0.0, 1.0, 0.25],
            FeatureType.BLIND_HOLE: [0.0, 0.9, 0.8],
            FeatureType.CHAMFER: [1.0, 0.35, 0.0],
            FeatureType.FILLET: [1.0, 0.0, 0.55],
            FeatureType.POCKET: [0.7, 0.0, 1.0],
            FeatureType.BOSS: [1.0, 0.2, 0.0],
        }

        bodies_to_process = decomposition.bodies
        all_body_results = []  # list of (body, sub_cleaned, tree)

        def _face_edges(faces):
            """Get set of edge tuples (v0, v1) from face array."""
            edges_set = set()
            for f in faces:
                for i in range(3):
                    edges_set.add(tuple(sorted((int(f[i]), int(f[(i+1) % 3])))))
            return edges_set

        # ── Process each body through the single-body pipeline ──
        # Run all bodies in parallel, then show visualizations per stage.

        def _process_single_body(bi, body):
            # SHARED CORE: recognize_body() is the single source of truth for
            # segment -> fit -> (regularize -> re-seg/fit) -> topology -> blends
            # -> recognize. Standalone scripts/tests call the SAME function, so
            # a reproduction always matches what the app produces (no drift).
            from mesh2param.pipeline.convert import recognize_body
            sub_cleaned = extract_sub_mesh(cleaned, body.face_indices, body=body)
            _check_cancelled()
            br = recognize_body(
                sub_cleaned, bool(cleaned.metadata.get("scan_mode")))
            _check_cancelled()
            # Element [1] is the VIZ mesh: the mesh the regions/fitted/edges/
            # blends actually live on. For dirty AI/scan parts that's a
            # DECIMATED copy (br.seg_mesh) — drawing the regions on it shows
            # clean flats instead of the shattered full-mesh segmentation, and
            # it matches the ~10K-triangle regime segmentation was tuned for.
            # The full-density mesh is rebuilt where the CAD export needs it.
            return (body, br.seg_mesh, br.regions, br.fitted, br.graph,
                    br.edges, br.tree, br.blends)

        job.progress = 8
        job.progress_stage = "Analyzing surfaces..."

        # Sequential body processing. Earlier this used a
        # ThreadPoolExecutor with one worker per body, but scipy's
        # least_squares / minpack and sklearn's KMeans/DBSCAN call
        # into MKL/BLAS routines that are not thread-safe on
        # Windows — concurrent calls across bodies produced
        # "Windows fatal exception: access violation" crashes
        # (sometimes in _fit_sphere, sometimes in _fit_cone, and
        # they persisted even with a lock around fit_surfaces
        # because segmentation on the other threads was still
        # hitting BLAS at the same time). Sequential processing
        # adds ~1-2s on a 3-body part but eliminates the crash.
        body_stage_data = []
        for bi, b in enumerate(bodies_to_process):
            _check_cancelled()
            body_stage_data.append(_process_single_body(bi, b))

        # No upstream radius reconciliation — the stitched stage
        # collapses cross-body fragments into one cluster, and the
        # apply stage measures the radius per CAD edge from the same
        # vertex set. The per-body record radii are never read.

        # The CAD build needs the FULL-density per-body mesh (decimation hurts
        # the build — blocky screws, bad profiles). body_stage_data[*][1] is
        # now the decimated viz mesh, so rebuild the full sub-mesh here.
        all_body_results = [
            (body, extract_sub_mesh(cleaned, body.face_indices, body=body),
             body_tree)
            for body, _viz_mesh, _, _, _, _, body_tree, _ in body_stage_data]

        # ── Phase one: hand the pre-warmed build its inputs NOW ──
        # The CAD subprocess was pre-warmed at ingest (CadQuery already loaded)
        # and is blocking for the feature tree. Drop it in so the prismatic
        # build starts immediately and streams CONCURRENTLY with every analysis
        # viz stage that follows (regularize, segment, fit, topology, blend,
        # stitched, features). It then waits for the fillet clusters (phase two,
        # dropped at clusters_path). The dwell loop ingests its streamed steps.
        if all_body_results and job._cad_launch is not None:
            try:
                self._write_cad_input(
                    job._cad_launch, all_body_results[0][2], cleaned,
                    all_body_results, decomposition)
            except Exception:
                logger.exception("Handing input to CAD subprocess failed")

        # Stage 1c: Regularize — for SCAN / AI inputs the per-body processing
        # projected the wobbly vertices onto the recovered consensus planes
        # (flat faces, exact 90° corners). Show the crisped geometry of the
        # primary body as its own verifiable stage.
        if cleaned.metadata.get("scan_mode") and body_stage_data:
            reg_mesh = body_stage_data[0][1]   # regularized sub_cleaned (body 0)
            _set_mesh_viz(reg_mesh)
            ext = np.round(reg_mesh.bbox[1] - reg_mesh.bbox[0], 1).tolist()
            job.progress_stage = (
                "Regularize: faces flattened, corners snapped to 90° — "
                f"extents {ext} mm")
            job.debug_summary.append(
                "Regularize: vertices projected onto consensus planes "
                "(flat faces, exact right angles)")
            job.progress_version += 1
            _dwell_if_not_step_through("regularize")
            _pause_for_review("regularize")

        # ── Show per-stage visualizations ──
        # Each body's sub_cleaned is the mesh used for that body's pipeline.
        # Colors are per-face on that sub_cleaned.

        # Stage 2: Segmentation
        job.progress = 15
        job.progress_stage = "Segmenting surfaces..."
        seg_colors = []
        for bi, (body, sub_cleaned, regions, fitted, graph, edges_list, body_tree, _topo_blends) in enumerate(body_stage_data):
            fc = np.full((len(sub_cleaned.faces), 3), 0.55)
            color_for_region = _color_regions_no_adjacent_match(
                regions, sub_cleaned, palette,
            )
            for region in regions:
                idx = np.asarray(region.face_indices, dtype=np.int64)
                fc[idx[idx < len(fc)]] = color_for_region[region.region_id]
            seg_colors.append((body, sub_cleaned, fc))
            # Debug summary: kind breakdown by mean-normal magnitude.
            n_plate = n_curve = n_cyl = 0
            face_counts = []
            for r in regions:
                face_counts.append(r.num_faces)
                mn = sub_cleaned.normals[r.face_indices].mean(axis=0)
                mag = float(np.linalg.norm(mn))
                if mag > 0.95:
                    n_plate += 1
                elif mag < 0.3:
                    n_cyl += 1
                else:
                    n_curve += 1
            median_f = int(sorted(face_counts)[len(face_counts) // 2]) if face_counts else 0
            job.debug_summary.append(
                f"[Body {bi}] Stage 2: {len(regions)} regions "
                f"({n_plate} plate, {n_curve} curve, {n_cyl} cyl) median={median_f}f"
            )
        _set_body_viz(seg_colors)
        _dwell_if_not_step_through("segment")
        _pause_for_review("segment")

        # Stage 3: Fitting
        # Render PRIMITIVE GEOMETRY: each region's vertices are
        # projected onto its fitted plane / cylinder / cone / sphere.
        # The triangulation is preserved but the surface itself is
        # mathematically exact. Region boundaries become small
        # visible gaps because vertices duplicated per region project
        # to different points — that gap visually highlights where
        # one primitive ends and the next begins, which is exactly
        # the "show me only the primitives" view.
        job.progress = 20
        job.progress_stage = "Fitting primitives..."
        fit_pieces = []
        from mesh2param.config import FitConfig as _FitCfg
        fit_rms_tol_ratio = _FitCfg().rms_tolerance_ratio
        for bi, (body, sub_cleaned, regions, fitted, graph, edges_list, body_tree, _topo_blends) in enumerate(body_stage_data):
            color_for_region = _color_regions_no_adjacent_match(
                regions, sub_cleaned, palette,
            )
            built = _build_primitive_mesh(
                regions, sub_cleaned, fitted, color_for_region,
            )
            if built is None:
                fit_pieces.append({
                    "vertices": sub_cleaned.vertices.flatten().tolist(),
                    "faces": sub_cleaned.faces.flatten().tolist(),
                    "colors": None,
                    "color": body_colors[len(fit_pieces) % len(body_colors)],
                })
            else:
                verts_arr, faces_arr, colors_arr = built
                fit_pieces.append({
                    "vertices": verts_arr.flatten().tolist(),
                    "faces": faces_arr.flatten().tolist(),
                    "colors": colors_arr.flatten().tolist(),
                    "color": body_colors[len(fit_pieces) % len(body_colors)],
                })
            # Debug summary: fit-type counts and misfits.
            type_counts = {}
            misfits = []
            bbox_diag = float(np.linalg.norm(
                sub_cleaned.bbox[1] - sub_cleaned.bbox[0]
            ))
            tol = fit_rms_tol_ratio * bbox_diag
            for f in fitted:
                t = f.surface_type.name
                type_counts[t] = type_counts.get(t, 0) + 1
                if f.rms_error > tol:
                    misfits.append(f"rid {f.region_id} {t} RMS={f.rms_error:.2f}")
            type_str = " ".join(
                f"{n}{k[:3].lower()}" for k, n in sorted(type_counts.items())
            )
            misfit_str = (
                f", {len(misfits)} misfit (worst: {misfits[0]})"
                if misfits else ""
            )
            job.debug_summary.append(
                f"[Body {bi}] Stage 3: {len(fitted)} fits ({type_str}){misfit_str}"
            )
        # Always use the pieces path (works for 1 or multiple bodies),
        # since projecting per-region requires per-region vertex sets
        # that don't fit a single CleanedMesh.
        job.progress_colors = None
        job.progress_colors_per_face = False
        job._decomp_pieces = fit_pieces
        job.progress_version += 1
        _dwell_if_not_step_through("fit")
        _pause_for_review("fit")

        # Stage 4: Topology — same face colors as fitting + edge lines
        job.progress = 23
        job.progress_stage = "Building topology..."
        edge_pts_list = []
        edge_col_list = []
        for bi, (body, sub_cleaned, regions, fitted, graph, edges_list, body_tree, _topo_blends) in enumerate(body_stage_data):
            region_lookup = {r.region_id: r for r in regions}
            edge_type_counts = {}
            for edge in edges_list:
                edge_type_counts[edge.transition_type.name] = edge_type_counts.get(
                    edge.transition_type.name, 0
                ) + 1
                ra = region_lookup.get(edge.surface_a_id)
                rb = region_lookup.get(edge.surface_b_id)
                if ra is None or rb is None:
                    continue
                faces_a = sub_cleaned.faces[ra.face_indices] if len(ra.face_indices) > 0 else np.array([])
                faces_b = sub_cleaned.faces[rb.face_indices] if len(rb.face_indices) > 0 else np.array([])
                if len(faces_a) == 0 or len(faces_b) == 0:
                    continue
                shared_edges = _face_edges(faces_a) & _face_edges(faces_b)
                if not shared_edges:
                    continue
                color = edge_type_colors.get(edge.transition_type.name, [0.5, 0.5, 0.5])
                for v0, v1 in shared_edges:
                    edge_pts_list.extend([sub_cleaned.vertices[v0], sub_cleaned.vertices[v1]])
                    edge_col_list.extend([color, color])
            type_str = ", ".join(
                f"{n} {k.lower()}" for k, n in sorted(
                    edge_type_counts.items(), key=lambda kv: -kv[1]
                )
            ) or "0"
            job.debug_summary.append(
                f"[Body {bi}] Stage 4: {len(edges_list)} edges ({type_str})"
            )
        edge_pts = np.array(edge_pts_list) if edge_pts_list else None
        edge_cols = np.array(edge_col_list) if edge_col_list else None
        job.progress_edges = edge_pts.flatten().tolist() if edge_pts is not None else None
        job.progress_edge_colors = edge_cols.flatten().tolist() if edge_cols is not None else None
        # Topology gets its OWN base geometry: the raw cleaned mesh in
        # uniform neutral gray. Reusing the fit stage's primitive
        # projection (even recolored) kept the two stages visually
        # entangled — projection gaps and region seams read as topology
        # artifacts. Here the classified edge lines are the only
        # colored content, drawn over the true mesh they were computed
        # on. Keep PER-FACE colors (all neutral) rather than
        # colors=None — the colorless piece path is the decompose viz
        # (translucent + explode animation), wrong for this stage.
        topo_pieces = []
        for body, sub_cleaned, *_rest in body_stage_data:
            topo_pieces.append({
                "vertices": sub_cleaned.vertices.flatten().tolist(),
                "faces": sub_cleaned.faces.flatten().tolist(),
                "colors": [0.62, 0.64, 0.68] * len(sub_cleaned.faces),
                "color": None,
            })
        job.progress_colors = None
        job.progress_colors_per_face = False
        job._decomp_pieces = topo_pieces
        job.progress_version += 1
        _dwell_if_not_step_through("topology")
        _pause_for_review("topology")

        # Stage 4.5: Blend graph — highlight each plane-to-plane
        # connection that was classified as a fillet or chamfer. Unlike
        # the topology viz which draws every inter-region edge, this
        # stage paints the BLEND REGIONS themselves (the small cylinder
        # or narrow plane sitting between two larger planes), so it's
        # obvious which surfaces are going to be handled as blends.
        try:
            blend_face_colors_by_body = []
            total_fil = 0
            total_cha = 0
            for body, sub_cleaned, regions, fitted, graph, edges_list, body_tree, topo_blends in body_stage_data:
                fc = np.full((len(sub_cleaned.faces), 3), 0.60)
                for rec in topo_blends:
                    idx = rec.get("face_indices")
                    if idx is None or len(idx) == 0:
                        continue
                    if rec.get("kind") == "fillet":
                        fc[idx] = [0.0, 0.9, 0.95]   # cyan
                        total_fil += 1
                    else:
                        fc[idx] = [1.0, 0.55, 0.0]   # orange
                        total_cha += 1
                blend_face_colors_by_body.append((body, sub_cleaned, fc))

            job.progress = 25
            job.progress_stage = (
                f"Blend graph: {total_fil} fillet + {total_cha} chamfer "
                f"plane-to-plane connection(s)"
            )
            job.progress_edges = None
            job.progress_edge_colors = None
            _set_body_viz(blend_face_colors_by_body)
            _dwell_if_not_step_through("blend_graph")
            _pause_for_review("blend_graph")
        except Exception as _bg_exc:
            logger.debug("Blend graph viz failed: %s", _bg_exc)

        # --- Debug stage: Stitched fillets ---
        # Endpoint-graph stitcher. Every fillet from the blend graph is
        # kept as-is (no reclassification, no dropping). Two records
        # are stitched into the same group if either rule fires:
        #
        #   1. **Direct contact** — any face vertex of A is within
        #      `DIRECT_EPS` (2 mm) of any face vertex of B. Catches
        #      duplicate vertices at body-decomposition seams and
        #      curved-segment junctions where mesh edges touch.
        #
        #   2. **Collinear continuation** — A and B have aligned axes
        #      (|cos| ≥ 0.5) AND some endpoint pair (EA, EB) lies on
        #      the same axial line: the perpendicular offset of
        #      (EB − EA) from A's axis is below `eps_perp`, and the
        #      axial gap is below `eps_axial`. Thresholds scale with
        #      radius so small fillets aren't merged across big plates
        #      and big rim fillets aren't broken up. The perpendicular
        #      check rejects parallel fillets on opposite plate edges
        #      (their connection vector is perpendicular to the axis,
        #      not along it).
        #
        # Chamfers paint in their blend-graph orange so nothing visible
        # there disappears here. The CAD apply pipeline is untouched —
        # this stitching is viz-only.
        try:
            from mesh2param.pipeline.convert import stitch_fillet_clusters
            _stitched = stitch_fillet_clusters(body_stage_data)
            stitched_clusters_payload = _stitched.clusters
            uid_to_group = _stitched.uid_to_group
            stitched_face_colors = _stitched.face_colors
            n_records = _stitched.n_records
            n_groups = _stitched.n_groups
            n_multi = _stitched.n_multi
            n_singletons = _stitched.n_singletons
            n_direct_unions = _stitched.n_direct_unions
            n_axial_unions = _stitched.n_axial_unions
            n_chamfers = _stitched.n_chamfers
            job._stitched_clusters = stitched_clusters_payload
            job._uid_to_group = dict(uid_to_group)

            # ── Phase two unlock ──
            # The build subprocess (launched right after recognition) has
            # already built the prismatic model and is now blocked waiting for
            # these fillet clusters. Drop them atomically so it applies
            # fillets/chamfers as the second phase.
            _cp = getattr(job, '_cad_clusters_path', None)
            if _cp:
                try:
                    import os as _os
                    import pickle as _pk
                    _tmp = _cp + ".tmp"
                    with open(_tmp, "wb") as _f:
                        _pk.dump(stitched_clusters_payload, _f)
                    _os.replace(_tmp, _cp)
                    logger.info(
                        "Phase two: handed %d fillet cluster(s) to the build",
                        len(stitched_clusters_payload))
                except Exception:
                    logger.exception(
                        "Failed to hand fillet clusters to the build subprocess")

            # --- Ground-truth dump for the stitched stage ---
            # Write what the ACTUAL server produced to a file so it
            # can be inspected directly (logger.info from the worker
            # thread doesn't always reach stdout). One row per
            # cluster + per-cluster coverage near the two known
            # rackmount through-hole centers, so we can confirm
            # what category each hole-edge fillet actually got.
            try:
                import json as _json
                _diag_path = Path("data") / f"stitched_diag_{job.id}.json"
                _diag_path.parent.mkdir(parents=True, exist_ok=True)
                _hole_centers = [
                    [-40.4, 125.5, -133.4],
                    [42.4, 125.5, -133.4],
                ]
                _rows = []
                for c in stitched_clusters_payload:
                    cv = np.asarray(c["vertices"], dtype=float)
                    ctr = cv.mean(axis=0)
                    # min vertex distance to each hole center
                    hole_min = [
                        float(np.min(np.linalg.norm(cv - np.asarray(hc), axis=1)))
                        for hc in _hole_centers
                    ]
                    _rows.append({
                        "group_id": int(c["group_id"]),
                        "category": c["category"],
                        "length_mm": round(float(c.get("length", 0)), 1),
                        "coverage_deg": round(float(c.get("coverage_deg", 0)), 0),
                        "inplane_mm": round(float(c.get("inplane_mm", 0)), 1),
                        "radius": round(float(c["radius"]), 2),
                        "n_records": int(c["n_records"]),
                        "centroid": [round(float(x), 1) for x in ctr],
                        "min_dist_to_holeA": round(hole_min[0], 1),
                        "min_dist_to_holeB": round(hole_min[1], 1),
                    })
                with open(_diag_path, "w", encoding="utf-8") as _fh:
                    _json.dump({
                        "n_records": n_records,
                        "n_groups": n_groups,
                        "clusters": _rows,
                    }, _fh, indent=2)
                logger.info("Stitched diag written to %s", _diag_path)
            except Exception as _diag_exc:
                logger.warning("Stitched diag dump failed: %s", _diag_exc)

            _check_cancelled()
            job.progress = 25
            # Per-category cluster breakdown for the viz banner.
            cat_counts: dict[str, int] = {}
            for c in stitched_clusters_payload:
                k = c.get("category", "curve_edge")
                cat_counts[k] = cat_counts.get(k, 0) + 1
            cat_summary = ", ".join(
                f"{n} {k}" for k, n in sorted(
                    cat_counts.items(), key=lambda kv: -kv[1]
                )
            ) or "—"
            job.progress_stage = (
                f"Stitched fillets: {n_records} records → {n_groups} groups "
                f"({n_multi} multi-record, {n_singletons} singletons); "
                f"contact={n_direct_unions} axial={n_axial_unions}; "
                f"categories: {cat_summary}; chamfers={n_chamfers} orange"
            )
            job.progress_edges = None
            job.progress_edge_colors = None
            _set_body_viz(stitched_face_colors)
            job.debug_summary.append(
                f"Stitched fillets: {n_records} → {n_groups} groups "
                f"({n_multi} multi-record, {n_singletons} singletons)"
            )
            _dwell_if_not_step_through("stitched")
            _pause_for_review("stitched")
        except Exception as _st_exc:
            logger.exception("Stitched fillets stage failed: %s", _st_exc)

        # Stage 5: Feature recognition
        job.progress = 26
        job.progress_stage = "Recognizing features..."
        job.progress_edges = None
        job.progress_edge_colors = None
        feat_colors = []
        for bi, (body, sub_cleaned, regions, fitted, graph, edges_list, body_tree, _topo_blends) in enumerate(body_stage_data):
            fc = np.full((len(sub_cleaned.faces), 3), 0.55)
            feat_type_counts = {}
            for feat in body_tree.features:
                tname = getattr(feat.type, 'name', str(feat.type))
                feat_type_counts[tname] = feat_type_counts.get(tname, 0) + 1
                color = feature_type_colors.get(feat.type, [0.55, 0.55, 0.55])
                try:
                    fi_arr = feat.face_indices
                    if fi_arr is not None and hasattr(fi_arr, '__len__') and len(fi_arr) > 0:
                        for fi in fi_arr:
                            if 0 <= int(fi) < len(fc):
                                fc[int(fi)] = color
                except Exception:
                    pass
            feat_colors.append((body, sub_cleaned, fc))
            type_str = ", ".join(
                f"{n} {k.lower().replace('_', ' ')}"
                for k, n in sorted(feat_type_counts.items(), key=lambda kv: -kv[1])
            ) or "0"
            job.debug_summary.append(
                f"[Body {bi}] Stage 5: {len(body_tree.features)} features ({type_str})"
            )
        _set_body_viz(feat_colors)
        _dwell_if_not_step_through("features")
        _pause_for_review("features")

        tree = all_body_results[0][2]
        job.features = tree
        job.cleaned = cleaned
        job.mesh_volume = self._estimate_volume(cleaned)
        job.bbox_min = cleaned.bbox[0].tolist()
        job.bbox_max = cleaned.bbox[1].tolist()

        # Clear edges for build stage, keep feature colors
        job.progress_edges = None
        job.progress_edge_colors = None
        job.progress = 30
        _check_cancelled()
        job.progress_stage = "Building CAD model..."
        logger.debug("Job %s: stage=cadquery features=%d bodies=%d",
                      job.id, tree.num_features, len(all_body_results))

        # CadQuery runs in a subprocess. The fillet feature list comes
        # straight from the stitched stage — one cluster per physical
        # fillet, each carrying the face-vertex set so the apply pass
        # can do vertex-proximity edge matching with per-edge measured
        # radii. No upstream radius value gets passed through.
        stitched_clusters = list(getattr(job, '_stitched_clusters', []) or [])
        logger.info(
            "stitched-clusters → export: %d group(s) feeding fillet apply",
            len(stitched_clusters),
        )
        # The build was pre-warmed at ingest and handed its input after
        # recognition, so it built concurrently with the analysis viz. Reuse
        # that handle bundle; if pre-warm was skipped/failed, launch + feed it
        # now (synchronous fallback).
        _launch = getattr(job, '_cad_launch', None)
        if _launch is None:
            _launch = self._launch_cad_subprocess(
                job, getattr(job, '_num_variants', 1))
        # Make sure the subprocess has its input (no-op if already handed over).
        self._write_cad_input(_launch, tree, cleaned, all_body_results,
                              decomposition)
        job._cad_launch = None
        proc_handle = _launch['proc_handle']
        cad_input_path = _launch['cad_input_path']
        tmp_out = _launch['tmp_out']
        cad_log_path = _launch['cad_log_path']
        cad_err_path = _launch['cad_err_path']
        cad_log_fh = _launch['cad_log_fh']
        cad_err_fh = _launch['cad_err_fh']
        cad_stream_dir = _launch['cad_stream_dir']
        deadline = _launch['deadline']
        # Guarantee phase-two clusters are delivered before we block on the
        # build. If the stitched stage already dropped them the file exists; if
        # it errored (or this is a synchronous fallback launch) drop what we
        # have now so the waiting subprocess proceeds instead of timing out.
        _cp2 = _launch.get('clusters_path')
        if _cp2 is not None and not Path(_cp2).exists():
            try:
                import os as _os2
                with open(str(_cp2) + ".tmp", "wb") as _f:
                    pickle.dump(stitched_clusters, _f)
                _os2.replace(str(_cp2) + ".tmp", str(_cp2))
            except Exception:
                logger.debug("Fallback clusters write failed", exc_info=True)
        # Continue ingesting from wherever the dwell loop's concurrent ingest
        # left off, so streamed steps aren't double-counted.
        _stream_ingested = getattr(job, '_cad_stream_ingested', 0)
        try:
            try:
                while True:
                    if job.status == "error":
                        # Cancel hit while subprocess running — kill it
                        # so the worker can move on to the next job.
                        try:
                            proc_handle.terminate()
                            proc_handle.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc_handle.kill()
                        Path(cad_input_path).unlink(missing_ok=True)
                        Path(tmp_out.name).unlink(missing_ok=True)
                        raise RuntimeError("Cancelled by user")
                    # Heartbeat so the watchdog doesn't classify the
                    # running subprocess as a stuck worker and kill it.
                    # The subprocess can legitimately take longer than
                    # STUCK_THRESHOLD_S; only its internal `deadline`
                    # below should bound it.
                    job.last_activity_at = time.time()
                    # Ingest any construction steps the subprocess has streamed
                    # to disk since we last looked, so the UI's right pane
                    # accretes features live while the build is still running.
                    _stream_ingested = _ingest_streamed_steps(
                        job, cad_stream_dir, _stream_ingested)
                    # Surface the build SUB-PHASE the subprocess writes to
                    # _phase.txt (Applying fillets… / Refining… / Validating…)
                    # so the silent post-step minutes show what's actually
                    # running instead of a frozen "Building CAD model…".
                    try:
                        _phase_f = cad_stream_dir / "_phase.txt"
                        if _phase_f.exists():
                            _ph = _phase_f.read_text(encoding="utf-8").strip()
                            if _ph and job.progress_stage != _ph:
                                job.progress_stage = _ph
                    except Exception:
                        pass
                    try:
                        proc_handle.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if time.time() > deadline:
                        proc_handle.kill()
                        try:
                            proc_handle.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                        # Don't unlink tmp_in here — let the failure
                        # handler preserve it for offline replay.
                        Path(tmp_out.name).unlink(missing_ok=True)
                        raise RuntimeError(
                            f"CadQuery subprocess timed out after 600s; "
                            f"see {cad_log_path}"
                        )
            finally:
                job._cad_subprocess = None
                try:
                    cad_log_fh.close()
                except Exception:
                    pass
                try:
                    cad_err_fh.close()
                except Exception:
                    pass

            class _ProcResult:
                pass
            proc = _ProcResult()
            proc.returncode = proc_handle.returncode
            try:
                proc.stdout = cad_log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                proc.stdout = ""
            try:
                proc.stderr = cad_err_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                proc.stderr = ""
        except Exception:
            logger.exception("Job %s: CadQuery subprocess failed", job.id)
            Path(cad_input_path).unlink(missing_ok=True)
            Path(tmp_out.name).unlink(missing_ok=True)
            raise
        Path(cad_input_path).unlink(missing_ok=True)

        if proc.returncode != 0:
            # Preserve the input pickle for offline replay.
            preserved = None
            try:
                preserved = Path("data") / f"cad_subproc_{job.id}_input.pkl"
                preserved.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(str(cad_input_path), str(preserved))
            except Exception as _cpy_exc:
                logger.warning("Failed to preserve subproc input pkl: %s", _cpy_exc)
            logger.error(
                "Job %s: CadQuery subprocess failed (exit %d); stdout at %s; stderr at %s; input pkl: %s\n"
                "stderr tail: %s\nstdout tail: %s",
                job.id, proc.returncode, cad_log_path, cad_err_path, preserved,
                (proc.stderr or "")[-600:],
                (proc.stdout or "")[-600:],
            )
            Path(tmp_out.name).unlink(missing_ok=True)
            raise RuntimeError(
                f"CadQuery failed (exit {proc.returncode}); see {cad_log_path}"
            )

        # Forward fillet-matching lines from the subprocess into the
        # main server log so the matcher's per-edge decisions are
        # visible without running the pipeline standalone. Filter to
        # the lines we actually want to see — the subprocess can be
        # very chatty otherwise.
        try:
            interesting = (
                "Edge enumeration", "Fillet matching", "Fillet pass",
                "Bucket ", "Stitched fillets", "nearest-edge dump",
                "cluster ", "CHECKPOINT", "SUBPROCESS EXCEPTION",
            )
            for line in (proc.stdout or "").splitlines():
                if any(tag in line for tag in interesting):
                    logger.info("[cad] %s", line)
        except Exception:
            pass

        try:
            with open(tmp_out.name, "rb") as f:
                result = pickle.load(f)
        except Exception:
            logger.exception("Job %s: failed to deserialize CadQuery output", job.id)
            raise
        finally:
            Path(tmp_out.name).unlink(missing_ok=True)

        # Persist exactly what the per-user Onshape export needs (feature
        # tree, cleaned mesh, as-built holes/blends) keyed by job id, so
        # "Export to Onshape" works even after the in-memory job is
        # evicted (the endpoint is api/onshape_oauth.py::export_job).
        try:
            _ox_dir = Path("data")
            _ox_dir.mkdir(exist_ok=True)
            with open(_ox_dir / f"onshape_export_{job.id}.pkl", "wb") as _oxf:
                pickle.dump({
                    "tree": tree,
                    "cleaned": cleaned,
                    "as_built": getattr(result, "as_built", {}) or {},
                }, _oxf)
        except Exception:
            logger.debug("Could not persist Onshape export data for job %s",
                         job.id, exc_info=True)

        # The streamed steps were a live preview; the result's construction
        # steps are authoritative (fully tessellated, post-reconciliation).
        # Swap to them and bump the version so the UI loads the final geometry.
        job.steps = result.construction_steps
        job.cad_volume = result.cad_volume
        job.building_live = False
        job.steps_version += 1
        try:
            import shutil as _sh
            _sh.rmtree(cad_stream_dir, ignore_errors=True)
            _cp_done = getattr(job, '_cad_clusters_path', None)
            if _cp_done:
                Path(_cp_done).unlink(missing_ok=True)
        except Exception:
            pass
        # Store variants
        job.variants = []
        for v in getattr(result, 'variants', []):
            job.variants.append({
                'name': v.name,
                'strategy': v.strategy,
                'steps': v.construction_steps,
                'cad_volume': v.cad_volume,
            })
        if job.variants:
            job.steps = job.variants[0]['steps']  # default to first variant

        # (Removed: the "3D model built (before fillets)" pre-blend checkpoint
        # stage — the two-phase right-pane build now shows the un-blended solid
        # accreting, then the fillets going on, so this left-pane snapshot was
        # redundant.)

        # (Removed: the "Fillet & Chamfer Intent Overlay" left-pane viz stage —
        # consolidated into the two-phase right-pane build, which shows the real
        # fillets going on after the stitched stage.)

        # (Removed: the "Apply fillets & chamfers" left-pane viz stage. The
        # two-phase build now applies fillets live in the RIGHT build pane right
        # after the stitched stage, so re-showing the post-apply solid on the
        # left was redundant. The reconciliation fate overlay below stays — it's
        # a real diagnostic (which groups matched/applied), not a re-show.)

        # --- Debug stage: Blend reconciliation fate overlay ---
        # Paint every fillet face by what the apply stage did with its
        # stitched group:
        #   cyan = group's CAD edges matched and were filleted
        #   red  = group exists but no CAD edge matched (apply skipped it)
        # No "merged" / "badfit" / "filtered" categories — the new
        # pipeline doesn't drop records or fit radii, so those buckets
        # are gone.
        try:
            diag = getattr(result, 'fillet_match_diag', None) or []
            diag_by_group = {int(d.get("group_id", -1)): d for d in diag}
            uid_to_group_map = dict(getattr(job, "_uid_to_group", {}) or {})

            BASE = [0.58, 0.60, 0.63]
            CYAN = [0.0, 0.85, 0.95]
            RED = [0.93, 0.18, 0.18]
            counts = {"matched": 0, "unmatched": 0}
            fate_colors_by_body = []
            for body, sub_cleaned, regions, fitted, graph, edges_list, body_tree, topo_blends in body_stage_data:
                fc = np.tile(BASE, (len(sub_cleaned.faces), 1))
                for rec in topo_blends:
                    if rec.get("kind") != "fillet":
                        continue
                    idx = rec.get("face_indices")
                    if idx is None or len(idx) == 0:
                        continue
                    uid = int(rec.get("stitch_uid", -1))
                    gi = uid_to_group_map.get(uid)
                    d = diag_by_group.get(int(gi)) if gi is not None else None
                    if d is not None and d.get("matched"):
                        fc[idx] = CYAN
                        counts["matched"] += 1
                    else:
                        fc[idx] = RED
                        counts["unmatched"] += 1
                fate_colors_by_body.append((body, sub_cleaned, fc))

            _check_cancelled()
            job.progress = 99
            job.progress_stage = (
                "Blend reconciliation — "
                f"{counts['matched']} matched, {counts['unmatched']} unmatched"
            )
            job.progress_edges = None
            job.progress_edge_colors = None
            job._decomp_pieces = None
            _set_body_viz(fate_colors_by_body)
            job.debug_summary.append(
                "Reconcile: " + ", ".join(f"{k}={v}" for k, v in counts.items())
            )
            _dwell_if_not_step_through("reconcile")
            _pause_for_review("reconcile")
        except Exception as _rec_exc:
            logger.debug("Blend reconciliation stage failed: %s", _rec_exc)

        job.progress = 100
        job.progress_stage = "Done"

        # Compute token cost — tuned so 2×Mesh1 + 1×Mesh2 ≈ 100 free tokens
        cost = max(1, cleaned.num_faces // 150)  # base: face complexity
        for feat in tree.features:
            if feat.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
                cost += 2
            elif feat.type in (FeatureType.CHAMFER, FeatureType.FILLET):
                cost += 2
            elif feat.type in (FeatureType.POCKET, FeatureType.BOSS, FeatureType.SLOT):
                cost += 3
        # Round to nearest 5
        cost = max(5, round(cost / 5) * 5)
        job.token_cost = cost
        self._tokens_spent += cost

        job.status = "done"
        logger.info("Pipeline completed for job %s: %d steps, token_cost=%d",
                     job.id, len(job.steps), job.token_cost)

        # Persist to SQLite
        self._save_to_db(job)

        # Notify callback (for per-IP billing)
        if self._on_complete:
            try:
                self._on_complete(job)
            except Exception:
                logger.exception("Job %s: on_complete callback failed", job.id)

    # ── Database persistence ──

    def load_completed_from_db(self):
        """Load completed jobs from SQLite into memory on startup."""
        if self._db_loaded:
            return
        self._db_loaded = True
        try:
            from mesh2param.db.database import SessionLocal, MeshJob as DBMeshJob, JobStep as DBJobStep
            from mesh2param.pipeline.export_cadquery import ConstructionStep
            db = SessionLocal()
            try:
                db_jobs = (
                    db.query(DBMeshJob)
                    .filter(DBMeshJob.status == "done", DBMeshJob.job_uuid.isnot(None))
                    .order_by(DBMeshJob.created_at.desc())
                    .limit(MAX_JOBS)
                    .all()
                )
                loaded = 0
                for db_job in reversed(db_jobs):  # oldest first so OrderedDict order is correct
                    job_id = db_job.job_uuid
                    with self._lock:
                        if job_id in self._jobs:
                            continue

                    job = Job(
                        id=job_id,
                        filename=db_job.filename or "",
                        file_path="",
                        status="done",
                        progress=100,
                        progress_stage="Done",
                        mesh_volume=db_job.mesh_volume or 0.0,
                        cad_volume=db_job.cad_volume or 0.0,
                        token_cost=db_job.token_cost or 0,
                        created_at=db_job.created_at.timestamp() if db_job.created_at else time.time(),
                    )

                    # Restore bbox
                    try:
                        job.bbox_min = json.loads(db_job.bbox_min_json) if db_job.bbox_min_json else [0, 0, 0]
                        job.bbox_max = json.loads(db_job.bbox_max_json) if db_job.bbox_max_json else [0, 0, 0]
                    except Exception:
                        logger.debug("Failed to parse bbox JSON for job %s, using defaults", job_id)

                    # Restore construction steps
                    db_steps = (
                        db.query(DBJobStep)
                        .filter(DBJobStep.job_id == db_job.id)
                        .order_by(DBJobStep.step_index)
                        .all()
                    )
                    steps = []
                    for db_step in db_steps:
                        tess_verts = None
                        tess_faces = None
                        if db_step.vertices_blob and db_step.vertices_shape:
                            try:
                                shape = tuple(int(x) for x in db_step.vertices_shape.split(","))
                                tess_verts = np.frombuffer(db_step.vertices_blob, dtype=np.float64).reshape(shape)
                            except Exception:
                                logger.debug("Failed to deserialize vertices for job %s step %s",
                                             job_id, db_step.step_index)
                        if db_step.faces_blob and db_step.faces_shape:
                            try:
                                shape = tuple(int(x) for x in db_step.faces_shape.split(","))
                                tess_faces = np.frombuffer(db_step.faces_blob, dtype=np.int64).reshape(shape)
                            except Exception:
                                logger.debug("Failed to deserialize faces for job %s step %s",
                                             job_id, db_step.step_index)
                        overlay = None
                        if getattr(db_step, "sketch_overlay_blob", None) and getattr(db_step, "sketch_overlay_shape", None):
                            try:
                                shape = tuple(int(x) for x in db_step.sketch_overlay_shape.split(","))
                                overlay = np.frombuffer(db_step.sketch_overlay_blob, dtype=np.float64).reshape(shape)
                            except Exception:
                                pass
                        steps.append(ConstructionStep(
                            name=db_step.name,
                            tess_vertices=tess_verts if tess_verts is not None else np.array([]),
                            tess_faces=tess_faces if tess_faces is not None else np.array([]),
                            volume=db_step.volume or 0.0,
                            sketch_overlay=overlay,
                        ))
                    job.steps = steps

                    # Restore variants from pickled blob
                    if getattr(db_job, 'variants_blob', None):
                        try:
                            import pickle as _pkl
                            job.variants = _pkl.loads(db_job.variants_blob)
                            if job.variants:
                                job.steps = job.variants[0]['steps']
                        except Exception:
                            logger.debug("Failed to restore variants for job %s", job_id)

                    with self._lock:
                        self._jobs[job_id] = job
                    self._tokens_spent += job.token_cost
                    loaded += 1

                if loaded:
                    logger.info("Loaded %d completed jobs from database", loaded)
            finally:
                db.close()
        except Exception:
            logger.exception("Failed to load jobs from database")

    def _save_to_db(self, job: Job):
        """Persist a completed job to SQLite."""
        try:
            from mesh2param.db.database import SessionLocal, MeshJob as DBMeshJob, JobStep as DBJobStep
            db = SessionLocal()
            try:
                # Check if already saved
                existing = db.query(DBMeshJob).filter(DBMeshJob.job_uuid == job.id).first()
                if existing:
                    return

                num_features = 0
                features_summary = None
                if job.features and hasattr(job.features, "features"):
                    num_features = len(job.features.features)
                    try:
                        features_summary = json.dumps([
                            {"type": f.type.value if hasattr(f.type, "value") else str(f.type),
                             "name": getattr(f, "name", ""),
                             "params": {k: v for k, v in (getattr(f, "params", {}) or {}).items()
                                        if isinstance(v, (int, float, str, bool))}}
                            for f in job.features.features
                        ])
                    except Exception:
                        pass

                # Pickle variants for DB persistence
                variants_blob = None
                if job.variants:
                    try:
                        import pickle as _pkl
                        variants_blob = _pkl.dumps(job.variants)
                    except Exception:
                        logger.warning("Failed to pickle variants for job %s", job.id)

                db_job = DBMeshJob(
                    user_id=1,  # default user; TODO: pass actual user from request
                    job_uuid=job.id,
                    filename=job.filename,
                    status=job.status,
                    error=job.error,
                    mesh_volume=job.mesh_volume,
                    cad_volume=job.cad_volume,
                    num_features=num_features,
                    num_steps=len(job.steps),
                    token_cost=job.token_cost,
                    ai_tokens_used=job.token_cost,
                    bbox_min_json=json.dumps(job.bbox_min),
                    bbox_max_json=json.dumps(job.bbox_max),
                    features_json=features_summary,
                    variants_blob=variants_blob,
                    completed_at=datetime.utcnow(),
                )
                db.add(db_job)
                db.flush()  # get db_job.id

                for i, step in enumerate(job.steps):
                    verts_blob = None
                    verts_shape = None
                    faces_blob = None
                    faces_shape = None
                    if step.tess_vertices is not None and len(step.tess_vertices) > 0:
                        arr = np.asarray(step.tess_vertices, dtype=np.float64)
                        verts_blob = arr.tobytes()
                        verts_shape = ",".join(str(s) for s in arr.shape)
                    if step.tess_faces is not None and len(step.tess_faces) > 0:
                        arr = np.asarray(step.tess_faces, dtype=np.int64)
                        faces_blob = arr.tobytes()
                        faces_shape = ",".join(str(s) for s in arr.shape)

                    overlay_blob = None
                    overlay_shape = None
                    if getattr(step, "sketch_overlay", None) is not None and len(step.sketch_overlay) > 0:
                        arr = np.asarray(step.sketch_overlay, dtype=np.float64)
                        overlay_blob = arr.tobytes()
                        overlay_shape = ",".join(str(s) for s in arr.shape)

                    db_step = DBJobStep(
                        job_id=db_job.id,
                        step_index=i,
                        name=step.name,
                        volume=step.volume,
                        is_sketch="sketch" in step.name.lower(),
                        vertices_blob=verts_blob,
                        vertices_shape=verts_shape,
                        faces_blob=faces_blob,
                        faces_shape=faces_shape,
                        sketch_overlay_blob=overlay_blob,
                        sketch_overlay_shape=overlay_shape,
                    )
                    db.add(db_step)

                db.commit()
                logger.info("Saved job %s to database (%d steps)", job.id, len(job.steps))
            finally:
                db.close()
        except Exception:
            logger.exception("Failed to save job %s to database", job.id)

    def _load_single_from_db(self, job_id: str) -> Job | None:
        """Try to load a single job from the DB (cache miss)."""
        try:
            from mesh2param.db.database import SessionLocal, MeshJob as DBMeshJob, JobStep as DBJobStep
            from mesh2param.pipeline.export_cadquery import ConstructionStep
            db = SessionLocal()
            try:
                db_job = db.query(DBMeshJob).filter(
                    DBMeshJob.job_uuid == job_id, DBMeshJob.status == "done"
                ).first()
                if db_job is None:
                    return None

                job = Job(
                    id=job_id,
                    filename=db_job.filename or "",
                    file_path="",
                    status="done",
                    progress=100,
                    progress_stage="Done",
                    mesh_volume=db_job.mesh_volume or 0.0,
                    cad_volume=db_job.cad_volume or 0.0,
                    token_cost=db_job.token_cost or 0,
                    created_at=db_job.created_at.timestamp() if db_job.created_at else time.time(),
                )
                try:
                    job.bbox_min = json.loads(db_job.bbox_min_json) if db_job.bbox_min_json else [0, 0, 0]
                    job.bbox_max = json.loads(db_job.bbox_max_json) if db_job.bbox_max_json else [0, 0, 0]
                except Exception:
                    logger.debug("Failed to parse bbox JSON for single-load job %s", job_id)

                db_steps = (
                    db.query(DBJobStep)
                    .filter(DBJobStep.job_id == db_job.id)
                    .order_by(DBJobStep.step_index)
                    .all()
                )
                steps = []
                for db_step in db_steps:
                    tess_verts = None
                    tess_faces = None
                    if db_step.vertices_blob and db_step.vertices_shape:
                        try:
                            shape = tuple(int(x) for x in db_step.vertices_shape.split(","))
                            tess_verts = np.frombuffer(db_step.vertices_blob, dtype=np.float64).reshape(shape)
                        except Exception:
                            logger.debug("Failed to deserialize vertices for job %s step %s (single-load)",
                                         job_id, db_step.step_index)
                    if db_step.faces_blob and db_step.faces_shape:
                        try:
                            shape = tuple(int(x) for x in db_step.faces_shape.split(","))
                            tess_faces = np.frombuffer(db_step.faces_blob, dtype=np.int64).reshape(shape)
                        except Exception:
                            logger.debug("Failed to deserialize faces for job %s step %s (single-load)",
                                         job_id, db_step.step_index)
                    steps.append(ConstructionStep(
                        name=db_step.name,
                        tess_vertices=tess_verts if tess_verts is not None else np.array([]),
                        tess_faces=tess_faces if tess_faces is not None else np.array([]),
                        volume=db_step.volume or 0.0,
                    ))
                job.steps = steps

                # Cache in memory
                with self._lock:
                    self._jobs[job_id] = job
                return job
            finally:
                db.close()
        except Exception:
            logger.exception("Failed to load job %s from database", job_id)
            return None

    @staticmethod
    def _estimate_volume(cleaned) -> float:
        import trimesh
        try:
            mesh = trimesh.Trimesh(
                vertices=cleaned.vertices, faces=cleaned.faces, process=False,
            )
            if mesh.is_watertight:
                return abs(float(mesh.volume))
            return abs(float(mesh.convex_hull.volume))
        except Exception:
            logger.exception("Failed to estimate mesh volume")
            return 0.0


# Singleton
queue = JobQueue()
