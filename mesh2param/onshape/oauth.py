"""Onshape OAuth 2.0 — per-user account linking.

The live export engine (client.py / feature_builder.py) is account-
agnostic; this module is what makes each user export to THEIR OWN
Onshape account instead of the server owner's API keys:

  - build the authorize URL (with a signed CSRF state),
  - exchange the returned code for access + refresh tokens,
  - store them ENCRYPTED on the User row (Fernet),
  - hand callers an authenticated OnshapeClient, transparently
    refreshing an expired access token.

App credentials come from ONSHAPE_OAUTH_CLIENT_ID / _CLIENT_SECRET
(env / .env). Token encryption key from ONSHAPE_TOKEN_KEY, falling back
to AUTH_SECRET — rotating either just forces users to reconnect.

Design + decisions: docs/ONSHAPE_INTEGRATION_PLAN.md.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests
from cryptography.fernet import Fernet, InvalidToken

from mesh2param.onshape.client import (
    DEFAULT_BASE_URL,
    OnshapeApiError,
    OnshapeClient,
    _load_dotenv_fallback,
)

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://oauth.onshape.com/oauth/authorize"
TOKEN_URL = "https://oauth.onshape.com/oauth/token"
# Scopes must be a subset of those enabled on the registered app.
DEFAULT_SCOPE = "OAuth2Read OAuth2Write OAuth2Profile"
# Refresh this many seconds BEFORE the access token actually expires.
REFRESH_SKEW_SECONDS = 90


class OnshapeOAuthError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(_client_id() and _client_secret())


def _client_id() -> str:
    cid = os.environ.get("ONSHAPE_OAUTH_CLIENT_ID", "").strip()
    if not cid:
        _load_dotenv_fallback()
        cid = os.environ.get("ONSHAPE_OAUTH_CLIENT_ID", "").strip()
    return cid


def _client_secret() -> str:
    cs = os.environ.get("ONSHAPE_OAUTH_CLIENT_SECRET", "").strip()
    if not cs:
        _load_dotenv_fallback()
        cs = os.environ.get("ONSHAPE_OAUTH_CLIENT_SECRET", "").strip()
    return cs


# ── Token encryption at rest ──────────────────────────────────────────

def _fernet() -> Fernet:
    secret = (
        os.environ.get("ONSHAPE_TOKEN_KEY")
        or os.environ.get("AUTH_SECRET")
    )
    if not secret:
        _load_dotenv_fallback()
        secret = (
            os.environ.get("ONSHAPE_TOKEN_KEY")
            or os.environ.get("AUTH_SECRET")
        )
    if not secret:
        # Dev-only ephemeral key: tokens won't survive a restart, so the
        # user just reconnects. Never hit in prod (AUTH_SECRET is set).
        logger.warning(
            "No ONSHAPE_TOKEN_KEY/AUTH_SECRET; using an ephemeral token "
            "encryption key (stored Onshape tokens will not persist).",
        )
        secret = "mesh2param-dev-ephemeral"
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt_token(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_token(ciphertext: str | None) -> str | None:
    if not ciphertext:
        return None
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except (InvalidToken, ValueError):
        logger.warning("Stored Onshape token could not be decrypted "
                       "(key rotated?); treating as disconnected.")
        return None


# ── OAuth flow ────────────────────────────────────────────────────────

def build_authorize_url(state: str, redirect_uri: str,
                        scope: str = DEFAULT_SCOPE) -> str:
    if not is_configured():
        raise OnshapeOAuthError("Onshape OAuth app is not configured "
                                "(ONSHAPE_OAUTH_CLIENT_ID/SECRET).")
    params = {
        "response_type": "code",
        "client_id": _client_id(),
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": scope,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def _token_request(data: dict) -> dict:
    data = {**data, "client_id": _client_id(),
            "client_secret": _client_secret()}
    r = requests.post(TOKEN_URL, data=data, timeout=30, headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    })
    if r.status_code >= 400:
        raise OnshapeOAuthError(f"token endpoint {r.status_code}: "
                                f"{r.text[:300]}")
    return r.json()


def exchange_code(code: str, redirect_uri: str) -> dict:
    """Authorization code -> token bundle (access/refresh/expires_in)."""
    return _token_request({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    })


def refresh_access_token(refresh_token: str) -> dict:
    return _token_request({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    })


# ── User persistence ──────────────────────────────────────────────────

def store_tokens_on_user(user, token_bundle: dict, db,
                         identity: dict | None = None) -> None:
    """Encrypt + persist a token bundle (and optional identity) on a
    User. Caller commits."""
    access = token_bundle.get("access_token")
    refresh = token_bundle.get("refresh_token")
    expires_in = int(token_bundle.get("expires_in", 0) or 0)
    if access:
        user.onshape_access_token_enc = encrypt_token(access)
    if refresh:
        user.onshape_refresh_token_enc = encrypt_token(refresh)
    user.onshape_token_expiry = (
        datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        if expires_in else None
    )
    if identity:
        user.onshape_user_id = str(identity.get("id") or "") or None
        user.onshape_user_name = (
            identity.get("name") or identity.get("email") or None
        )
    if user.onshape_connected_at is None:
        user.onshape_connected_at = datetime.now(timezone.utc)
    db.add(user)
    db.commit()


def disconnect_user(user, db) -> None:
    user.onshape_access_token_enc = None
    user.onshape_refresh_token_enc = None
    user.onshape_token_expiry = None
    user.onshape_user_id = None
    user.onshape_user_name = None
    user.onshape_connected_at = None
    db.add(user)
    db.commit()


def is_connected(user) -> bool:
    return bool(getattr(user, "onshape_access_token_enc", None))


def _expired(user) -> bool:
    exp = getattr(user, "onshape_token_expiry", None)
    if exp is None:
        return False  # unknown expiry — try the token, refresh on 401
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return time.time() >= exp.timestamp() - REFRESH_SKEW_SECONDS


def get_onshape_client_for_user(
    user, db, base_url: str = DEFAULT_BASE_URL,
) -> OnshapeClient | None:
    """Authenticated per-user client, refreshing the access token when
    it is near/over expiry. None if the user hasn't connected or the
    refresh failed (caller should prompt 'reconnect Onshape')."""
    if not is_connected(user):
        return None
    access = decrypt_token(user.onshape_access_token_enc)
    if access is None:
        return None
    if _expired(user):
        refresh = decrypt_token(user.onshape_refresh_token_enc)
        if not refresh:
            return None
        try:
            bundle = refresh_access_token(refresh)
        except OnshapeOAuthError:
            logger.warning("Onshape token refresh failed for user %s",
                           getattr(user, "id", "?"))
            return None
        store_tokens_on_user(user, bundle, db)
        access = decrypt_token(user.onshape_access_token_enc)
        if access is None:
            return None
    return OnshapeClient.from_oauth_token(access, base_url=base_url)


def fetch_identity(client: OnshapeClient) -> dict | None:
    try:
        return client.sessioninfo()
    except OnshapeApiError:
        return None
