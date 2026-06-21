"""Optional Modal dispatch for pipeline jobs (set USE_MODAL=1 to enable).

When enabled, a job runs in an ISOLATED Modal container (the deployed `build_mesh`
function) instead of the local worker — so concurrent customer jobs run in
PARALLEL, each isolated (no pymeshlab-can't-run-concurrently wall, the same one
that blocks local parallelism). The local worker just SPAWNS each job's Modal
call and moves on; a poller thread finalizes them as they complete.

Live build streaming: the container also pushes each construction step to a
shared modal.Queue, partitioned by the job id. The poller drains that partition
every tick and appends the steps to `job.steps` live — so the cloud build streams
into the dual-pane viewer step-by-step, just like a local build. If the queue is
unavailable (or a step is too big), the build still finishes from the final
result; streaming just degrades to one step.

Default OFF: with USE_MODAL unset the local path is 100% unchanged.

Server env needed when enabled: `modal` installed + MODAL_TOKEN_ID and
MODAL_TOKEN_SECRET (or a mounted ~/.modal.toml). The function must be deployed:
`modal deploy modal_app.py`. See docs/MODAL_DEPLOY.md.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "mesh2param-pipeline"
FUNC_NAME = "build_mesh"
STEP_QUEUE_NAME = "mesh2param-steps"

_fn = None
_queue = None


def is_enabled() -> bool:
    return os.environ.get("USE_MODAL", "").strip().lower() in ("1", "true", "yes")


def _function():
    global _fn
    if _fn is None:
        import modal
        _fn = modal.Function.from_name(APP_NAME, FUNC_NAME)
    return _fn


def _step_queue():
    """The shared step Queue (created on first use). May raise; callers guard."""
    global _queue
    if _queue is None:
        import modal
        _queue = modal.Queue.from_name(STEP_QUEUE_NAME, create_if_missing=True)
    return _queue


def dispatch(job) -> bool:
    """Spawn the Modal build for this job (non-blocking). The poller finalizes it.
    Returns True if dispatched, False on error (job marked errored)."""
    try:
        data = Path(job.file_path).read_bytes()
        # stream_key = job.id: the container pushes steps to this Queue partition.
        # editor=True: server jobs need the heavy rebuild bundle + STEP bytes back
        # (edge-picking, download, Compare, /edit). (mesh_bytes, filename,
        # want_geometry, stream_key, editor)
        call = _function().spawn(data, job.filename, True, job.id, True)
        job._modal_call_id = call.object_id
        job._modal_stream_key = job.id
        job._modal_streamed = False
        job.status = "processing"
        job.progress = 40
        job.progress_stage = "Building in cloud"
        job.building_live = False
        logger.info("Job %s -> Modal call %s", job.id, call.object_id)
        return True
    except Exception:
        logger.exception("Modal dispatch failed for job %s", job.id)
        job.status = "error"
        job.error = "Cloud dispatch failed (is USE_MODAL configured + the function deployed?)"
        return False


def _drain_steps(job) -> bool:
    """Pull any streamed construction steps for this job and append them live,
    bumping steps_version so the UI accretes them. Returns True if the
    end-of-stream sentinel was seen. Best-effort: never raises into the poller."""
    key = getattr(job, "_modal_stream_key", None)
    if not key:
        return False
    try:
        import numpy as np
        from mesh2param.pipeline.export_cadquery import ConstructionStep
        q = _step_queue()
        msgs = q.get_many(64, block=False, partition=key)
    except Exception:
        return False
    done = False
    appended = 0
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        if m.get("__done__"):
            done = True
            continue
        try:
            job.steps.append(ConstructionStep(
                name=m.get("name", f"Step {len(job.steps) + 1}"),
                tess_vertices=np.asarray(m["tess_vertices"], np.float64).reshape(-1, 3),
                tess_faces=np.asarray(m["tess_faces"], np.int64).reshape(-1, 3),
                volume=float(m.get("volume", 0.0) or 0.0),
            ))
            appended += 1
        except Exception:
            logger.debug("Bad streamed step for job %s", job.id, exc_info=True)
    if appended:
        job._modal_streamed = True
        job.building_live = True
        job.steps_version = getattr(job, "steps_version", 0) + 1
        job.progress = min(95, getattr(job, "progress", 40) + appended)
        job.progress_stage = f"Building model... {len(job.steps)} step(s)"
        job.last_activity_at = time.time()
        logger.info("Cloud build: %d step(s) streamed for job %s (%d total)",
                    appended, job.id, len(job.steps))
    return done


def _clear_partition(job) -> None:
    key = getattr(job, "_modal_stream_key", None)
    if not key:
        return
    try:
        _step_queue().clear(partition=key)
    except Exception:
        logger.debug("Could not clear step partition for job %s", job.id,
                     exc_info=True)


def poll_and_finalize(job) -> bool:
    """Check the job's Modal call. Returns True when finished (done OR error) so
    the caller can stop polling it; False while still running. Drains streamed
    steps each tick so the build streams into the UI as it runs."""
    cid = getattr(job, "_modal_call_id", None)
    if not cid:
        return True
    # Stream any steps produced so far BEFORE checking for completion.
    _drain_steps(job)
    try:
        import modal
        call = modal.FunctionCall.from_id(cid)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            return False  # not ready yet — keep polling (and streaming)
    except Exception:
        logger.exception("Modal poll failed for job %s", job.id)
        job.status = "error"
        job.error = "Cloud build failed"
        _clear_partition(job)
        return True
    # Result is ready: pick up any final steps still in the queue, then finalize.
    _drain_steps(job)
    _finalize(job, result)
    return True


def _finalize(job, r: dict) -> None:
    import numpy as np
    from mesh2param.pipeline.export_cadquery import ConstructionStep

    if not r or not r.get("ok"):
        job.status = "error"
        job.error = (r or {}).get("error", "Cloud build returned no result")
        job.building_live = False
        _clear_partition(job)
        return

    # If steps already streamed in, keep them — don't collapse to a single
    # "Model" step. Only synthesize one final step when nothing streamed.
    streamed = bool(getattr(job, "_modal_streamed", False)) and len(job.steps) > 0
    if not streamed:
        tv, tf = r.get("tess_vertices"), r.get("tess_faces")
        if not tv or not tf:
            job.status = "error"
            job.error = "Cloud build produced no geometry"
            job.building_live = False
            _clear_partition(job)
            return
        job.steps = [ConstructionStep(
            name="Model",
            tess_vertices=np.asarray(tv, np.float64).reshape(-1, 3),
            tess_faces=np.asarray(tf, np.int64).reshape(-1, 3),
            volume=float(r.get("cad_volume", 0.0)),
        )]
        job.steps_version = getattr(job, "steps_version", 0) + 1

    job.cad_volume = float(r.get("cad_volume", 0.0))
    job.mesh_volume = float(r.get("mesh_volume", getattr(job, "mesh_volume", 0.0)) or 0.0)

    # Editor parity: land the rebuild bundle + STEP on disk so the feature-tree
    # editor, edge-picking, download-step, and the mesh Compare view all work on
    # cloud jobs exactly as they do locally.
    _restore_editor_artifacts(job, r)

    job.building_live = False
    job.progress = 100
    job.progress_stage = "Done"
    job.status = "done"
    _clear_partition(job)
    logger.info("Job %s finalized from Modal (IoU=%s, %s feats, %s steps, %ss)",
                job.id, r.get("iou"), r.get("n_features"), len(job.steps),
                r.get("elapsed"))


def _restore_editor_artifacts(job, r: dict) -> None:
    """Reconstitute the local-job server state from the cloud result: write the
    durable rebuild bundle + STEP (+ a cached edit base), and recover the feature
    tree / cleaned mesh / bbox / token cost. Best-effort; the model still shows
    if any of this fails (only editing/compare degrade)."""
    import shutil

    data = Path("data")
    try:
        data.mkdir(exist_ok=True)
    except Exception:
        pass

    bundle_bytes = r.get("bundle_bytes")
    if bundle_bytes:
        try:
            bpath = data / f"cad_bundle_{job.id}.pkl"
            bpath.write_bytes(bundle_bytes)
            job.cad_bundle_path = str(bpath)
        except Exception:
            logger.debug("Could not write cloud rebuild bundle", exc_info=True)
        # Recover tree + cleaned mesh for the editor list, Compare view, bbox,
        # and token cost (same fields the local worker sets).
        try:
            import pickle
            tree, cleaned, _, _ = pickle.loads(bundle_bytes)
            job.features = tree
            job.cleaned = cleaned
            _set_bbox_and_tokens(job, tree, cleaned)
        except Exception:
            logger.debug("Could not recover tree/cleaned from bundle", exc_info=True)

    step_bytes = r.get("step_bytes")
    if step_bytes:
        try:
            spath = data / f"cad_solid_{job.id}.step"
            spath.write_bytes(step_bytes)
            job.cad_step_path = str(spath)
            ebase = data / f"cad_editbase_{job.id}.step"
            shutil.copyfile(spath, ebase)
            job.cad_editbase_path = str(ebase)
            job._edit_base_deletes = frozenset()  # base = full model, no deletes
        except Exception:
            logger.debug("Could not write cloud STEP", exc_info=True)


def _set_bbox_and_tokens(job, tree, cleaned) -> None:
    """Mirror the local worker's bbox + token-cost computation for a cloud job."""
    try:
        import numpy as np
        v = np.asarray(cleaned.vertices, float)
        if len(v):
            job.bbox_min = [float(x) for x in v.min(axis=0)]
            job.bbox_max = [float(x) for x in v.max(axis=0)]
    except Exception:
        pass
    try:
        from mesh2param.models.feature import FeatureType
        cost = max(1, cleaned.num_faces // 150)
        for feat in tree.features:
            if feat.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
                cost += 2
            elif feat.type in (FeatureType.CHAMFER, FeatureType.FILLET):
                cost += 2
            elif feat.type in (FeatureType.POCKET, FeatureType.BOSS, FeatureType.SLOT):
                cost += 3
        job.token_cost = max(5, round(cost / 5) * 5)
    except Exception:
        logger.debug("Could not compute cloud token cost", exc_info=True)
