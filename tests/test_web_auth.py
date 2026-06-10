"""Web-layer tests: auth endpoints, password hashing, tokens, settings,
and the cleaner download endpoint (path traversal).

Uses a throwaway SQLite database via DATABASE_URL, set before any
mesh2param.db import so the module-level engine binds to it.
"""

import os
import tempfile

import pytest

_TEST_DB = os.path.join(tempfile.gettempdir(), "mesh2param_test_web.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
os.environ.setdefault("AUTH_SECRET", "test-secret-not-for-production")

from fastapi.testclient import TestClient  # noqa: E402

from mesh2param.api import auth  # noqa: E402
from mesh2param.db.database import Base, engine  # noqa: E402
from mesh2param.viz_server.app import app  # noqa: E402


@pytest.fixture()
def client():
    Base.metadata.create_all(engine)
    auth._auth_attempts.clear()  # reset rate limiter between tests
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(engine)


# ── Password hashing ──

def test_hash_password_roundtrip():
    stored = auth.hash_password("hunter2")
    assert stored.startswith("pbkdf2_sha256$")
    assert auth.verify_password("hunter2", stored)
    assert not auth.verify_password("wrong", stored)


def test_hashes_are_salted():
    assert auth.hash_password("same") != auth.hash_password("same")


def test_legacy_sha256_hash_still_verifies():
    legacy = auth._legacy_hash_password("oldpass")
    assert auth.verify_password("oldpass", legacy)
    assert not auth.verify_password("wrong", legacy)
    assert auth.password_needs_rehash(legacy)
    assert not auth.password_needs_rehash(auth.hash_password("new"))


def test_verify_password_garbage_hash():
    assert not auth.verify_password("x", "")
    assert not auth.verify_password("x", "pbkdf2_sha256$notanumber$zz$yy")


# ── Tokens ──

def test_token_roundtrip():
    token = auth._make_token("alice")
    assert auth.token_username(token) == "alice"


def test_token_tampered_signature_rejected():
    token = auth._make_token("alice")
    head, _, _ = token.rpartition(".")
    assert auth.token_username(head + ".AAAA") is None


def test_token_expired_rejected():
    token = auth._make_token("alice", ttl_seconds=-1)
    assert auth.token_username(token) is None


def test_token_garbage_rejected():
    assert auth.token_username("") is None
    assert auth.token_username("not.a.token") is None
    assert auth.token_username("v1.abc.123.def") is None


# ── Endpoints ──

def test_signup_login_verify_flow(client):
    r = client.post("/auth/signup", json={"username": "bob", "password": "secret123"})
    assert r.status_code == 200
    token = r.json()["token"]
    assert auth.token_username(token) == "bob"

    r = client.post("/auth/login", json={"username": "bob", "password": "secret123"})
    assert r.status_code == 200

    r = client.get("/auth/verify", params={"token": r.json()["token"]})
    assert r.status_code == 200
    assert r.json()["username"] == "bob"


def test_login_wrong_password_rejected(client):
    client.post("/auth/signup", json={"username": "carol", "password": "right"})
    r = client.post("/auth/login", json={"username": "carol", "password": "wrong"})
    assert r.status_code == 401


def test_duplicate_signup_rejected(client):
    client.post("/auth/signup", json={"username": "dave", "password": "pw1234"})
    r = client.post("/auth/signup", json={"username": "dave", "password": "pw1234"})
    assert r.status_code == 409


def test_verify_invalid_token_rejected(client):
    r = client.get("/auth/verify", params={"token": "bogus"})
    assert r.status_code == 401


def test_legacy_hash_upgraded_on_login(client):
    from mesh2param.db.database import SessionLocal, User

    client.post("/auth/signup", json={"username": "erin", "password": "legacy-pw"})
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == "erin").first()
        user.password_hash = auth._legacy_hash_password("legacy-pw")
        db.commit()
    finally:
        db.close()

    r = client.post("/auth/login", json={"username": "erin", "password": "legacy-pw"})
    assert r.status_code == 200

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == "erin").first()
        assert user.password_hash.startswith("pbkdf2_sha256$")
    finally:
        db.close()


def test_login_rate_limited(client):
    for _ in range(auth._AUTH_RATE_LIMIT):
        client.post("/auth/login", json={"username": "nobody", "password": "x"})
    r = client.post("/auth/login", json={"username": "nobody", "password": "x"})
    assert r.status_code == 429


def test_change_password_flow(client):
    client.post("/auth/signup", json={"username": "fred", "password": "first"})
    auth._auth_attempts.clear()
    r = client.post("/settings/change-password", json={
        "username": "fred", "current_password": "first", "new_password": "second",
    })
    assert r.status_code == 200
    r = client.post("/auth/login", json={"username": "fred", "password": "second"})
    assert r.status_code == 200
    r = client.post("/auth/login", json={"username": "fred", "password": "first"})
    assert r.status_code == 401


# ── Cleaner download: path traversal ──

def test_download_rejects_traversal(client):
    r = client.get("/api/clean/download/..%2F..%2Fetc%2Fpasswd.stl")
    assert r.status_code == 404
    r = client.get("/api/clean/download/..%5C..%5Cwindows%5Cwin.ini.stl")
    assert r.status_code == 404


def test_download_rejects_non_stl(client):
    r = client.get("/api/clean/download/something.exe")
    assert r.status_code == 404


def test_download_serves_temp_stl(client):
    fd, path = tempfile.mkstemp(suffix=".stl")
    os.write(fd, b"solid test\nendsolid test\n")
    os.close(fd)
    try:
        r = client.get(f"/api/clean/download/{os.path.basename(path)}")
        assert r.status_code == 200
        assert b"solid test" in r.content
    finally:
        os.unlink(path)


# ── Upload suffix whitelist ──

def test_preview_rejects_bad_suffix(client):
    r = client.post("/api/preview", files={"file": ("evil.phtml", b"data")})
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["error"]
