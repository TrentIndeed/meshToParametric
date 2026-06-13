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


def _make_state(username: str) -> str:
    expiry = int(time.time()) + _STATE_TTL
    nonce = secrets.token_hex(8)
    payload = f"{_b64url(username.encode())}.{expiry}.{nonce}"
    return f"{payload}.{_sign(payload)}"


def _state_username(state: str) -> str | None:
    try:
        user_b64, expiry, nonce, sig = state.split(".")
        payload = f"{user_b64}.{expiry}.{nonce}"
        if not hmac.compare_digest(_sign(payload), sig):
            return None
        if int(expiry) < time.time():
            return None
        return _b64url_decode(user_b64).decode()
    except (ValueError, TypeError, UnicodeDecodeError):
        return None


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


@router.get("/connect")
def connect(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if user is None:
        return RedirectResponse("/login?next=/home", status_code=302)
    if not os_oauth.is_configured():
        return RedirectResponse("/home?onshape=unconfigured", status_code=302)
    if not _can_use_onshape(user):
        return RedirectResponse("/home?onshape=upgrade", status_code=302)
    state = _make_state(user.username)
    url = os_oauth.build_authorize_url(state, _redirect_uri(request))
    return RedirectResponse(url, status_code=302)


@router.get("/callback")
def callback(request: Request, db: Session = Depends(get_db),
             code: str = "", state: str = "", error: str = ""):
    if error:
        logger.info("Onshape OAuth denied/aborted: %s", error)
        return RedirectResponse("/home?onshape=denied", status_code=302)
    username = _state_username(state)
    if not username or not code:
        return RedirectResponse("/home?onshape=badstate", status_code=302)
    from mesh2param.db.database import User
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return RedirectResponse("/home?onshape=badstate", status_code=302)
    try:
        bundle = os_oauth.exchange_code(code, _redirect_uri(request))
    except os_oauth.OnshapeOAuthError:
        logger.exception("Onshape code exchange failed")
        return RedirectResponse("/home?onshape=error", status_code=302)
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
    return RedirectResponse("/home?onshape=connected", status_code=302)


@router.post("/disconnect")
def disconnect(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if user is None:
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    if os_oauth.is_connected(user):
        os_oauth.disconnect_user(user, db)
    return {"status": "ok", "connected": False}
