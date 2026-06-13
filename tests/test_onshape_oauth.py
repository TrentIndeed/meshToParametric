"""Onshape OAuth layer: token crypto, Bearer client, authorize-URL,
code/refresh exchange (mocked), per-user client selection, and the
router's signed CSRF state. No network."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mesh2param.onshape import oauth as o
from mesh2param.onshape.client import OnshapeClient


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ONSHAPE_TOKEN_KEY", "unit-test-key")
    monkeypatch.setenv("ONSHAPE_OAUTH_CLIENT_ID", "CID")
    monkeypatch.setenv("ONSHAPE_OAUTH_CLIENT_SECRET", "CS")


# ── token crypto ──

def test_token_crypto_roundtrip():
    ct = o.encrypt_token("access-xyz")
    assert ct != "access-xyz"
    assert o.decrypt_token(ct) == "access-xyz"


def test_decrypt_handles_none_and_garbage():
    assert o.decrypt_token(None) is None
    assert o.decrypt_token("not-a-fernet-token") is None


def test_decrypt_with_rotated_key_returns_none(monkeypatch):
    ct = o.encrypt_token("access-xyz")
    monkeypatch.setenv("ONSHAPE_TOKEN_KEY", "a-different-key")
    assert o.decrypt_token(ct) is None  # graceful -> reconnect


# ── bearer client ──

def test_bearer_client_header():
    c = OnshapeClient.from_oauth_token("tok1")
    assert c._session.headers["Authorization"] == "Bearer tok1"
    c.set_oauth_token("tok2")
    assert c._session.headers["Authorization"] == "Bearer tok2"


# ── authorize url ──

def test_authorize_url_has_required_params():
    u = o.build_authorize_url("STATE", "https://x/cb")
    for frag in ("response_type=code", "client_id=CID", "state=STATE"):
        assert frag in u
    # Onshape rejects a scope query param ("Invalid request") — scopes
    # come from the registered app's permissions. Must NOT be sent.
    assert "scope=" not in u


def test_authorize_url_unconfigured_raises(monkeypatch):
    monkeypatch.delenv("ONSHAPE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("ONSHAPE_OAUTH_CLIENT_SECRET", raising=False)
    # also defeat the .env fallback the helper performs
    monkeypatch.setattr(o, "_load_dotenv_fallback", lambda: None)
    assert not o.is_configured()
    with pytest.raises(o.OnshapeOAuthError):
        o.build_authorize_url("S", "https://x/cb")


# ── code / refresh exchange (mocked) ──

class _Resp:
    def __init__(self, code, payload):
        self.status_code = code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_exchange_and_refresh(monkeypatch):
    calls = {}

    def fake_post(url, data=None, timeout=None, headers=None):
        calls["url"] = url
        calls["data"] = data
        return _Resp(200, {"access_token": "A", "refresh_token": "R",
                           "expires_in": 3600})

    monkeypatch.setattr(o.requests, "post", fake_post)
    bundle = o.exchange_code("CODE", "https://x/cb")
    assert bundle["access_token"] == "A"
    assert calls["data"]["grant_type"] == "authorization_code"
    assert calls["data"]["client_id"] == "CID"
    assert calls["data"]["client_secret"] == "CS"

    o.refresh_access_token("R")
    assert calls["data"]["grant_type"] == "refresh_token"


def test_token_endpoint_error_raises(monkeypatch):
    monkeypatch.setattr(o.requests, "post",
                        lambda *a, **k: _Resp(400, "bad"))
    with pytest.raises(o.OnshapeOAuthError):
        o.exchange_code("CODE", "https://x/cb")


# ── persistence + per-user client ──

def _fake_user(**kw):
    base = dict(
        id=1, username="u",
        onshape_access_token_enc=None, onshape_refresh_token_enc=None,
        onshape_token_expiry=None, onshape_user_id=None,
        onshape_user_name=None, onshape_connected_at=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _FakeDB:
    def add(self, *_):
        pass

    def commit(self):
        pass


def test_store_and_disconnect():
    u = _fake_user()
    db = _FakeDB()
    o.store_tokens_on_user(
        u, {"access_token": "A", "refresh_token": "R", "expires_in": 3600},
        db, identity={"id": "oid", "name": "Trenton"},
    )
    assert o.is_connected(u)
    assert u.onshape_user_name == "Trenton"
    assert o.decrypt_token(u.onshape_access_token_enc) == "A"
    assert u.onshape_token_expiry is not None
    o.disconnect_user(u, db)
    assert not o.is_connected(u)
    assert u.onshape_access_token_enc is None


def test_client_for_user_none_when_disconnected():
    assert o.get_onshape_client_for_user(_fake_user(), _FakeDB()) is None


def test_client_for_user_valid_token():
    u = _fake_user(
        onshape_access_token_enc=o.encrypt_token("A"),
        onshape_token_expiry=datetime.now(timezone.utc)
        + timedelta(hours=1),
    )
    c = o.get_onshape_client_for_user(u, _FakeDB())
    assert c is not None
    assert c._session.headers["Authorization"] == "Bearer A"


def test_client_for_user_refreshes_when_expired(monkeypatch):
    u = _fake_user(
        onshape_access_token_enc=o.encrypt_token("OLD"),
        onshape_refresh_token_enc=o.encrypt_token("R"),
        onshape_token_expiry=datetime.now(timezone.utc)
        - timedelta(minutes=5),  # already expired
    )
    monkeypatch.setattr(o, "refresh_access_token",
                        lambda r: {"access_token": "NEW",
                                   "refresh_token": "R2",
                                   "expires_in": 3600})
    c = o.get_onshape_client_for_user(u, _FakeDB())
    assert c is not None
    assert c._session.headers["Authorization"] == "Bearer NEW"
    assert o.decrypt_token(u.onshape_access_token_enc) == "NEW"


def test_client_for_user_refresh_failure_returns_none(monkeypatch):
    u = _fake_user(
        onshape_access_token_enc=o.encrypt_token("OLD"),
        onshape_refresh_token_enc=o.encrypt_token("R"),
        onshape_token_expiry=datetime.now(timezone.utc)
        - timedelta(minutes=5),
    )

    def boom(_):
        raise o.OnshapeOAuthError("revoked")

    monkeypatch.setattr(o, "refresh_access_token", boom)
    assert o.get_onshape_client_for_user(u, _FakeDB()) is None


# ── router CSRF state ──

def test_state_roundtrip_and_tamper():
    from mesh2param.api.onshape_oauth import _make_state, _state_username
    s = _make_state("trenton")
    assert _state_username(s) == "trenton"
    assert _state_username(s + "x") is None  # tampered sig
    assert _state_username("garbage") is None


def test_state_expiry(monkeypatch):
    from mesh2param.api import onshape_oauth as r
    s = r._make_state("trenton")  # expiry = now + _STATE_TTL
    future = time.time() + r._STATE_TTL + 100
    monkeypatch.setattr(time, "time", lambda: future)
    assert r._state_username(s) is None
