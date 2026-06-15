"""Onshape OAuth endpoints — connect / callback / disconnect / status.

The browser flow:
  1. User (logged in, Pro) hits /api/onshape/connect -> 302 to Onshape's
     consent screen with a signed CSRF `state`.
  2. Onshape redirects back to /api/onshape/callback?code&state.
  3. We verify state, exchange the code for tokens, store them encrypted
     on the User, and bounce to the app with a success flag.

Token mechanics live in onshape/oauth.py; this file is just the HTTP
surface + auth/gating.
"""
from __future__ import annotations

import hmac
import logging
import os
import secrets
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from sqlalchemy.orm import Session

from mesh2param.api.auth import (
    DEVELOPER_USERNAMES,
    _b64url,
    _b64url_decode,
    _sign,
    get_user_by_token,
)
from mesh2param.db.database import get_db
from mesh2param.onshape import oauth as os_oauth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/onshape", tags=["onshape"])

_STATE_TTL = 600  # seconds


def _current_user(request: Request, db: Session):
    token = request.cookies.get("paramesh_token", "")
    return get_user_by_token(token, db) if token else None


def _can_use_onshape(user) -> bool:
    if user is None:
        return False
    if user.username in DEVELOPER_USERNAMES:
        return True
    # Pro and up — plus the developer tier, which must have full access.
    return (user.plan or "free") in ("pro", "unlimited", "developer")


def _redirect_uri(request: Request) -> str:
    override = os.environ.get("ONSHAPE_OAUTH_REDIRECT_URI", "").strip()
    if override:
        return override
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get(
        "host", "localhost")
    return f"{proto}://{host}/api/onshape/callback"


def _safe_next(n: str | None) -> str:
    """Only allow same-site path redirects (block //evil.com and absolute
    URLs)."""
    if not n or not n.startswith("/") or n.startswith("//"):
        return "/home"
    return n


def _make_state(username: str, next_path: str = "/home") -> str:
    import json
    obj = {
        "u": username,
        "n": _safe_next(next_path),
        "e": int(time.time()) + _STATE_TTL,
        "x": secrets.token_hex(6),
    }
    raw = _b64url(json.dumps(obj, separators=(",", ":")).encode())
    return f"{raw}.{_sign(raw)}"


def _parse_state(state: str) -> dict | None:
    import json
    try:
        raw, sig = state.rsplit(".", 1)
        if not hmac.compare_digest(_sign(raw), sig):
            return None
        obj = json.loads(_b64url_decode(raw))
        if int(obj["e"]) < time.time():
            return None
        return obj
    except (ValueError, TypeError, KeyError, UnicodeDecodeError):
        return None


def _state_username(state: str) -> str | None:
    obj = _parse_state(state)
    return obj["u"] if obj else None


@router.get("/status")
def status(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    return {
        "configured": os_oauth.is_configured(),
        "eligible": _can_use_onshape(user),
        "connected": bool(user and os_oauth.is_connected(user)),
        "onshape_user_name": getattr(user, "onshape_user_name", None)
        if user else None,
    }


def _bounce(next_path: str, flag: str) -> RedirectResponse:
    sep = "&" if "?" in next_path else "?"
    return RedirectResponse(f"{next_path}{sep}onshape={flag}", status_code=302)


@router.get("/connect")
def connect(request: Request, next: str = "/home",
            db: Session = Depends(get_db)):
    user = _current_user(request, db)
    next_path = _safe_next(next)
    if user is None:
        return RedirectResponse(f"/login?next={next_path}", status_code=302)
    if not os_oauth.is_configured():
        return _bounce(next_path, "unconfigured")
    if not _can_use_onshape(user):
        return _bounce(next_path, "upgrade")
    state = _make_state(user.username, next_path)
    url = os_oauth.build_authorize_url(state, _redirect_uri(request))
    return RedirectResponse(url, status_code=302)


@router.get("/callback")
def callback(request: Request, db: Session = Depends(get_db),
             code: str = "", state: str = "", error: str = ""):
    obj = _parse_state(state)
    next_path = _safe_next(obj.get("n") if obj else None)
    if error:
        logger.info("Onshape OAuth denied/aborted: %s", error)
        return _bounce(next_path, "denied")
    if obj is None or not code:
        return _bounce(next_path, "badstate")
    username = obj["u"]
    from mesh2param.db.database import User
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return _bounce(next_path, "badstate")
    try:
        bundle = os_oauth.exchange_code(code, _redirect_uri(request))
    except os_oauth.OnshapeOAuthError:
        logger.exception("Onshape code exchange failed")
        return _bounce(next_path, "error")
    # identity for "Connected as <name>"
    identity = None
    access = bundle.get("access_token")
    if access:
        from mesh2param.onshape.client import OnshapeClient
        identity = os_oauth.fetch_identity(
            OnshapeClient.from_oauth_token(access))
    os_oauth.store_tokens_on_user(user, bundle, db, identity)
    logger.info("Onshape connected for user %s (%s)", username,
                (identity or {}).get("name"))
    return _bounce(next_path, "connected")


@router.post("/disconnect")
def disconnect(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if user is None:
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    if os_oauth.is_connected(user):
        os_oauth.disconnect_user(user, db)
    return {"status": "ok", "connected": False}


# Seconds between feature adds in the live build — paced so the user
# can watch each one appear in their Onshape viewport.
_LIVE_PACE = 0.45


def _export_context(job_uuid: str, request: Request, db: Session):
    """Shared preamble for the export endpoints. Returns
    (error_response | None, ctx). ctx = {user, job_row, client, data}
    when everything is ready; otherwise error_response is set."""
    import pickle
    from pathlib import Path

    from mesh2param.db.database import MeshJob

    user = _current_user(request, db)
    if user is None:
        return JSONResponse({"error": "not authenticated"},
                            status_code=401), None
    if not _can_use_onshape(user):
        return JSONResponse({"error": "upgrade", "detail":
                             "Onshape export is a Pro feature."},
                            status_code=403), None
    job_row = db.query(MeshJob).filter(MeshJob.job_uuid == job_uuid).first()
    if job_row is None:
        return JSONResponse({"error": "job not found"},
                            status_code=404), None
    if job_row.user_id and job_row.user_id != user.id:
        return JSONResponse({"error": "forbidden"}, status_code=403), None
    client = os_oauth.get_onshape_client_for_user(user, db)
    if client is None:
        return JSONResponse(
            {"error": "not_connected",
             "detail": "Connect your Onshape account first."},
            status_code=409), None
    art = Path("data") / f"onshape_export_{job_uuid}.pkl"
    if not art.exists():
        return JSONResponse(
            {"error": "expired",
             "detail": "Export data has expired; re-run the pipeline."},
            status_code=410), None
    try:
        with open(art, "rb") as f:
            data = pickle.load(f)
    except Exception:
        logger.exception("Failed to load Onshape export data %s", job_uuid)
        return JSONResponse({"error": "load_failed"}, status_code=500), None
    return None, {"user": user, "job_row": job_row,
                  "client": client, "data": data}


def _link_job(job_uuid: str, did: str, wid: str, eid: str, url: str,
              status: str) -> None:
    """Persist the Onshape document link onto the job (own session, so
    it's safe to call from inside a streaming generator)."""
    from datetime import datetime

    from mesh2param.db.database import MeshJob, SessionLocal
    s = SessionLocal()
    try:
        jr = s.query(MeshJob).filter(MeshJob.job_uuid == job_uuid).first()
        if jr is not None:
            jr.onshape_document_id = did
            jr.onshape_workspace_id = wid
            jr.onshape_element_id = eid
            jr.onshape_url = url
            jr.onshape_exported_at = datetime.utcnow()
            jr.onshape_export_status = status
            s.commit()
    finally:
        s.close()


@router.post("/export/{job_uuid}")
def export_job(job_uuid: str, request: Request, force: int = 0,
               db: Session = Depends(get_db)):
    """Export a finished job to the current user's own Onshape account
    (all features at once) and link the document. Idempotent: an already
    linked job returns its url unless force=1."""
    from pathlib import Path

    from mesh2param.config import ExportConfig
    from mesh2param.pipeline.export_onshape import export_to_onshape

    err, ctx = _export_context(job_uuid, request, db)
    if err is not None:
        return err
    job_row = ctx["job_row"]
    if job_row.onshape_url and not force:
        return {"status": "ok", "already": True, "url": job_row.onshape_url}

    data = ctx["data"]
    doc_name = f"{Path(job_row.filename or 'part').stem} (Paramesh)"
    try:
        res = export_to_onshape(
            data["tree"], data["cleaned"],
            ExportConfig(dry_run=False, document_name=doc_name),
            as_built=data.get("as_built"), client=ctx["client"],
        )
    except Exception:
        logger.exception("Onshape export failed for job %s", job_uuid)
        return JSONResponse({"error": "export_failed"}, status_code=502)

    status = ("ok" if res.feature_count == res.expected_features
              else "partial")
    _link_job(job_uuid, res.document_id, res.workspace_id, res.element_id,
              res.document_url, status)
    logger.info("Job %s exported to Onshape: %d/%d -> %s", job_uuid,
                res.feature_count, res.expected_features, res.document_url)
    return {
        "status": "ok",
        "url": res.document_url,
        "features": f"{res.feature_count}/{res.expected_features}",
        "volume_error_pct": round(res.volume_error_pct, 1),
        "export_status": status,
    }


@router.post("/document/new")
def new_document(request: Request, db: Session = Depends(get_db)):
    """Create a fresh EMPTY Onshape document in the user's account and
    return its ids + url. The UI opens it and binds it as the build
    target; subsequent pipeline runs build into it."""
    from mesh2param.pipeline.export_onshape import create_empty_document

    user = _current_user(request, db)
    if user is None:
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    if not _can_use_onshape(user):
        return JSONResponse({"error": "upgrade", "detail":
                             "Onshape is a Pro feature."}, status_code=403)
    client = os_oauth.get_onshape_client_for_user(user, db)
    if client is None:
        return JSONResponse(
            {"error": "not_connected",
             "detail": "Connect your Onshape account first."},
            status_code=409)
    try:
        info = create_empty_document(client, "Paramesh part")
    except Exception:
        logger.exception("Failed to create empty Onshape document")
        return JSONResponse({"error": "create_failed"}, status_code=502)
    logger.info("New empty Onshape document %s for user %s",
                info["document_id"], user.username)
    return {"status": "ok", **info}


@router.post("/export/{job_uuid}/live")
def export_job_live(job_uuid: str, request: Request, force: int = 0,
                    document_id: str = "", workspace_id: str = "",
                    element_id: str = "",
                    db: Session = Depends(get_db)):
    """Streamed LIVE build. With document_id/workspace_id/element_id,
    build into that EXISTING document (clearing it first) — this is how
    a run targets the doc opened by 'New Onshape Document', and how a
    re-analyze of the same mesh rebuilds into the same file. Without a
    target, a fresh document is created. Server-Sent Events:
      {type: document, url, total} | {type: feature, index, name, ok,
      n_ok, total} | {type: done, url, n_ok, total} | {type: error, ...}
    Non-2xx (not authed / upgrade / not connected / expired) come back as
    plain JSON so the UI can react (e.g. start OAuth)."""
    import json
    import time as _t
    from pathlib import Path

    from mesh2param.pipeline.export_onshape import (
        apply_planned_feature, prepare_live_export,
    )

    err, ctx = _export_context(job_uuid, request, db)
    if err is not None:
        return err
    job_row = ctx["job_row"]
    client = ctx["client"]
    data = ctx["data"]
    target = ((document_id, workspace_id, element_id)
              if document_id and workspace_id and element_id else None)
    # When targeting a bound document, ignore the job's old link — always
    # (re)build into the bound doc.
    already_url = None if target else job_row.onshape_url
    doc_name = f"{Path(job_row.filename or 'part').stem} (Paramesh)"

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    def gen():
        # Already built and not forced → just point the UI at it.
        if already_url and not force:
            yield _sse({"type": "document", "url": already_url,
                        "total": 0, "already": True,
                        "document_id": job_row.onshape_document_id or "",
                        "workspace_id": job_row.onshape_workspace_id or "",
                        "element_id": job_row.onshape_element_id or ""})
            yield _sse({"type": "done", "url": already_url, "already": True,
                        "n_ok": 0, "total": 0})
            return
        try:
            prep = prepare_live_export(
                client, data["tree"], data["cleaned"],
                data.get("as_built"), doc_name, target=target)
        except Exception as exc:
            logger.exception("Live export prep failed for %s", job_uuid)
            yield _sse({"type": "error", "detail": str(exc)[:200]})
            return
        if prep is None:
            yield _sse({"type": "error", "detail": "empty export plan"})
            return
        plan = prep["plan"]
        did, wid, eid, url = (prep["did"], prep["wid"],
                              prep["eid"], prep["url"])
        total = len(plan.features)
        logger.info("Live build %s -> %s (%d features, target=%s)",
                    job_uuid, url, total, bool(target))
        yield _sse({"type": "document", "url": url, "total": total,
                    "target": bool(target),
                    "document_id": did, "workspace_id": wid,
                    "element_id": eid})
        # New doc: give the browser a moment to load the empty part
        # studio so the user sees it build from the start. A bound
        # target is already open, so build immediately.
        if not target:
            _t.sleep(1.8)
        n_ok = 0
        feat_log: list[dict] = []
        for i, pf in enumerate(plan.features):
            detail = "OK"
            try:
                ok, _fid, detail = apply_planned_feature(client, did, wid, eid,
                                                         plan, pf)
            except Exception as exc:
                logger.exception("Live feature %s failed", pf.name)
                ok = False
                detail = f"exception: {str(exc)[:300]}"
            if ok:
                n_ok += 1
            feat_log.append({"index": i, "name": pf.name, "ok": ok,
                             "detail": detail})
            yield _sse({"type": "feature", "index": i, "name": pf.name,
                        "ok": ok, "n_ok": n_ok, "total": total})
            _t.sleep(_LIVE_PACE)
        _link_job(job_uuid, did, wid, eid, url,
                  "ok" if n_ok == total else "partial")
        logger.info("Live build %s done: %d/%d features", job_uuid,
                    n_ok, total)
        # Persist a per-build log (incl. final feature states) so a failed
        # export can be debugged from disk — no need to re-run a live build
        # (which spends Onshape API calls).
        try:
            from mesh2param.pipeline.export_onshape import write_build_log
            final_states = []
            try:
                fdata = client.features(did, wid, eid)
                fstates = fdata.get("featureStates", {}) or {}
                for f in fdata.get("features", []):
                    fid = f.get("featureId", "")
                    final_states.append({
                        "name": f.get("name", "?"),
                        "status": (fstates.get(fid) or {}).get(
                            "featureStatus", "?"),
                    })
            except Exception:
                logger.debug("Could not fetch final feature states",
                             exc_info=True)
            cad_vol = client.volume_mm3(did, wid, eid)
            write_build_log({
                "timestamp": str(int(_t.time())),
                "job_uuid": job_uuid,
                "user": getattr(ctx.get("user"), "username", "?"),
                "url": url, "document_id": did, "workspace_id": wid,
                "element_id": eid,
                "n_ok": n_ok, "total": total,
                "skipped": plan.skipped,
                "cad_volume": round(cad_vol, 1) if cad_vol else None,
                "features": feat_log,
                "final_states": final_states,
            }, label=f"job_{job_uuid}")
        except Exception:
            logger.debug("Build-log persistence failed", exc_info=True)
        yield _sse({"type": "done", "url": url, "n_ok": n_ok,
                    "total": total})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
