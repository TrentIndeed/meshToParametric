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
from fastapi.responses import JSONResponse, RedirectResponse
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
    return (user.plan or "free") == "pro"


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


@router.post("/export/{job_uuid}")
def export_job(job_uuid: str, request: Request, force: int = 0,
               db: Session = Depends(get_db)):
    """Export a finished job to the current user's own Onshape account
    and link the document to the job. Idempotent: a job that already has
    a linked document returns it unless force=1."""
    import pickle
    from datetime import datetime
    from pathlib import Path

    from mesh2param.config import ExportConfig
    from mesh2param.db.database import MeshJob
    from mesh2param.pipeline.export_onshape import export_to_onshape

    user = _current_user(request, db)
    if user is None:
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    if not _can_use_onshape(user):
        return JSONResponse({"error": "upgrade", "detail":
                             "Onshape export is a Pro feature."},
                            status_code=403)

    job_row = db.query(MeshJob).filter(
        MeshJob.job_uuid == job_uuid).first()
    if job_row is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    if job_row.user_id and job_row.user_id != user.id:
        return JSONResponse({"error": "forbidden"}, status_code=403)

    # Already linked → return it (don't spawn a duplicate document).
    if job_row.onshape_url and not force:
        return {"status": "ok", "already": True, "url": job_row.onshape_url}

    client = os_oauth.get_onshape_client_for_user(user, db)
    if client is None:
        return JSONResponse(
            {"error": "not_connected",
             "detail": "Connect your Onshape account first."},
            status_code=409)

    art = Path("data") / f"onshape_export_{job_uuid}.pkl"
    if not art.exists():
        return JSONResponse(
            {"error": "expired",
             "detail": "Export data has expired; re-run the pipeline."},
            status_code=410)
    try:
        with open(art, "rb") as f:
            data = pickle.load(f)
    except Exception:
        logger.exception("Failed to load Onshape export data %s", job_uuid)
        return JSONResponse({"error": "load_failed"}, status_code=500)

    doc_name = f"{Path(job_row.filename or 'part').stem} (Paramesh)"
    try:
        res = export_to_onshape(
            data["tree"], data["cleaned"],
            ExportConfig(dry_run=False, document_name=doc_name),
            as_built=data.get("as_built"), client=client,
        )
    except Exception:
        logger.exception("Onshape export failed for job %s", job_uuid)
        return JSONResponse({"error": "export_failed"}, status_code=502)

    job_row.onshape_document_id = res.document_id
    job_row.onshape_workspace_id = res.workspace_id
    job_row.onshape_element_id = res.element_id
    job_row.onshape_url = res.document_url
    job_row.onshape_exported_at = datetime.utcnow()
    job_row.onshape_export_status = (
        "ok" if res.feature_count == res.expected_features else "partial")
    db.add(job_row)
    db.commit()
    logger.info("Job %s exported to Onshape: %d/%d features -> %s",
                job_uuid, res.feature_count, res.expected_features,
                res.document_url)
    return {
        "status": "ok",
        "url": res.document_url,
        "features": f"{res.feature_count}/{res.expected_features}",
        "volume_error_pct": round(res.volume_error_pct, 1),
        "export_status": job_row.onshape_export_status,
    }
