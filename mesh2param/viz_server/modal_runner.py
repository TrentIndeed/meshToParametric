"""Optional Modal dispatch for pipeline jobs (set USE_MODAL=1 to enable).

When enabled, a job runs in an ISOLATED Modal container (the deployed `build_mesh`
function) instead of the local worker — so concurrent customer jobs run in
PARALLEL, each isolated (no pymeshlab-can't-run-concurrently wall, the same one
that blocks local parallelism). The local worker just SPAWNS each job's Modal
call and moves on; a poller thread finalizes them as they complete.

Default OFF: with USE_MODAL unset the local path is 100% unchanged.

Server env needed when enabled: `modal` installed + MODAL_TOKEN_ID and
MODAL_TOKEN_SECRET (or a mounted ~/.modal.toml). The function must be deployed:
`modal deploy modal_app.py`. See docs/MODAL_DEPLOY.md.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "mesh2param-pipeline"
FUNC_NAME = "build_mesh"

_fn = None


def is_enabled() -> bool:
    return os.environ.get("USE_MODAL", "").strip().lower() in ("1", "true", "yes")


def _function():
    global _fn
    if _fn is None:
        import modal
        _fn = modal.Function.from_name(APP_NAME, FUNC_NAME)
    return _fn


def dispatch(job) -> bool:
    """Spawn the Modal build for this job (non-blocking). The poller finalizes it.
    Returns True if dispatched, False on error (job marked errored)."""
    try:
        data = Path(job.file_path).read_bytes()
        call = _function().spawn(data, job.filename, True)  # want_geometry=True
        job._modal_call_id = call.object_id
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


def poll_and_finalize(job) -> bool:
    """Check the job's Modal call. Returns True when finished (done OR error) so
    the caller can stop polling it; False while still running."""
    cid = getattr(job, "_modal_call_id", None)
    if not cid:
        return True
    try:
        import modal
        call = modal.FunctionCall.from_id(cid)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            return False  # not ready yet
    except Exception:
        logger.exception("Modal poll failed for job %s", job.id)
        job.status = "error"
        job.error = "Cloud build failed"
        return True
    _finalize(job, result)
    return True


def _finalize(job, r: dict) -> None:
    import numpy as np
    from mesh2param.pipeline.export_cadquery import ConstructionStep

    if not r or not r.get("ok"):
        job.status = "error"
        job.error = (r or {}).get("error", "Cloud build returned no result")
        return
    tv, tf = r.get("tess_vertices"), r.get("tess_faces")
    if not tv or not tf:
        job.status = "error"
        job.error = "Cloud build produced no geometry"
        return
    step = ConstructionStep(
        name="Model",
        tess_vertices=np.asarray(tv, np.float64).reshape(-1, 3),
        tess_faces=np.asarray(tf, np.int64).reshape(-1, 3),
        volume=float(r.get("cad_volume", 0.0)),
    )
    job.steps = [step]
    job.steps_version = getattr(job, "steps_version", 0) + 1
    job.cad_volume = float(r.get("cad_volume", 0.0))
    job.progress = 100
    job.progress_stage = "Done"
    job.status = "done"
    logger.info("Job %s finalized from Modal (IoU=%s, %s feats, %ss)",
                job.id, r.get("iou"), r.get("n_features"), r.get("elapsed"))
