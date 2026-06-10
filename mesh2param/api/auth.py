"""Auth endpoints — signup, login, verify.

Passwords are hashed with PBKDF2-HMAC-SHA256 (per-user salt). Legacy
unsalted SHA-256 hashes are still accepted on login and upgraded in place.

Tokens are stateless, signed, and expiring:
    v2.<b64url(username)>.<expiry_unix>.<b64url(hmac_sha256(secret, payload))>
so verification is an O(1) signature check + direct username lookup.
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import deque

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from mesh2param.db.database import User, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_SECRET = os.getenv("AUTH_SECRET", secrets.token_hex(32))
TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_DAYS", "30")) * 86400
PBKDF2_ITERATIONS = 600_000
_PBKDF2_PREFIX = "pbkdf2_sha256"

# Comma-separated usernames that get the developer plan (unlimited tokens).
DEVELOPER_USERNAMES = {
    name.strip() for name in os.getenv("DEVELOPER_USERNAMES", "").split(",") if name.strip()
}


# ── Password hashing ──

def _hash_password(password: str, *, salt_hex: str | None = None,
                   iterations: int = PBKDF2_ITERATIONS) -> str:
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), iterations
    )
    return f"{_PBKDF2_PREFIX}${iterations}${salt_hex}${dk.hex()}"


def _legacy_hash_password(password: str) -> str:
    return hashlib.sha256(f"{AUTH_SECRET}:{password}".encode()).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """Timing-safe check against either a PBKDF2 or a legacy SHA-256 hash."""
    if not stored_hash:
        return False
    if stored_hash.startswith(_PBKDF2_PREFIX + "$"):
        try:
            _, iterations, salt_hex, _ = stored_hash.split("$", 3)
            candidate = _hash_password(password, salt_hex=salt_hex, iterations=int(iterations))
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(candidate, stored_hash)
    return hmac.compare_digest(_legacy_hash_password(password), stored_hash)


def password_needs_rehash(stored_hash: str) -> bool:
    return not stored_hash.startswith(_PBKDF2_PREFIX + "$")


def hash_password(password: str) -> str:
    """Hash a password for storage (public alias)."""
    return _hash_password(password)


# ── Tokens ──

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _sign(payload: str) -> str:
    return _b64url(hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).digest())


def _make_token(username: str, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    expiry = int(time.time()) + ttl_seconds
    payload = f"{_b64url(username.encode())}.{expiry}"
    return f"v2.{payload}.{_sign(payload)}"


def token_username(token: str) -> str | None:
    """Return the username for a valid, unexpired token, else None."""
    try:
        version, user_b64, expiry_str, sig = token.split(".")
        if version != "v2":
            return None
        payload = f"{user_b64}.{expiry_str}"
        if not hmac.compare_digest(_sign(payload), sig):
            return None
        if int(expiry_str) < time.time():
            return None
        return _b64url_decode(user_b64).decode()
    except (ValueError, TypeError, UnicodeDecodeError):
        return None


# ── Rate limiting (in-memory, per-IP sliding window) ──

_AUTH_RATE_LIMIT = 10       # attempts
_AUTH_RATE_WINDOW = 60.0    # seconds
_auth_attempts: dict[str, deque] = {}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.monotonic()
    attempts = _auth_attempts.setdefault(ip, deque())
    while attempts and now - attempts[0] > _AUTH_RATE_WINDOW:
        attempts.popleft()
    if len(attempts) >= _AUTH_RATE_LIMIT:
        logger.warning("Auth rate limit hit for ip=%s", ip)
        raise HTTPException(status_code=429, detail="Too many attempts. Try again in a minute.")
    attempts.append(now)


# ── Endpoints ──

@router.post("/signup")
def signup(body: dict, request: Request, db: Session = Depends(get_db)):
    _check_rate_limit(request)
    username = body.get("username", "").strip()
    password = body.get("password", "")
    email = body.get("email", "").strip()
    referral = body.get("referral", "").strip()
    logger.debug("Signup attempt for username=%r email=%r referral=%r", username, email, referral)

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    if len(password) < 3:
        raise HTTPException(status_code=400, detail="Password must be at least 3 characters")

    try:
        existing = db.query(User).filter(User.username == username).first()
    except Exception:
        logger.exception("Database error checking existing user %r", username)
        raise HTTPException(status_code=500, detail="Database error during signup")
    if existing:
        logger.debug("Signup rejected: username %r already taken", username)
        raise HTTPException(status_code=409, detail="Username already taken")

    try:
        user = User(
            username=username,
            email=email or None,
            password_hash=hash_password(password),
            referral=referral or None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    except Exception:
        logger.exception("Database error creating user %r", username)
        raise HTTPException(status_code=500, detail="Database error during signup")

    logger.info("User signed up: %r (id=%d)", username, user.id)
    token = _make_token(username)
    return {"token": token, "status": "ok", "username": username}


@router.post("/login")
def login(body: dict, request: Request, db: Session = Depends(get_db)):
    _check_rate_limit(request)
    username = body.get("username", "").strip()
    password = body.get("password", "")
    logger.debug("Login attempt for username=%r", username)

    try:
        user = db.query(User).filter(User.username == username).first()
    except Exception:
        logger.exception("Database error during login for %r", username)
        raise HTTPException(status_code=500, detail="Database error during login")
    if not user or not verify_password(password, user.password_hash):
        logger.debug("Login failed for username=%r", username)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Upgrade legacy SHA-256 hashes to PBKDF2 now that we have the plaintext
    if password_needs_rehash(user.password_hash):
        try:
            user.password_hash = hash_password(password)
            db.commit()
            logger.info("Upgraded password hash to PBKDF2 for %r", username)
        except Exception:
            logger.exception("Failed to upgrade password hash for %r", username)
            db.rollback()

    logger.info("User logged in: %r", username)
    token = _make_token(username)
    return {"token": token, "status": "ok", "username": username}


@router.get("/verify")
def verify_token(token: str, db: Session = Depends(get_db)):
    logger.debug("Token verification attempt")
    user = get_user_by_token(token, db)
    if user is None:
        logger.debug("Token verification failed")
        raise HTTPException(status_code=401, detail="Invalid token")
    logger.debug("Token verified for user=%r plan=%s", user.username, user.plan)
    return {"status": "ok", "username": user.username, "plan": user.plan or "free"}


def get_user_by_token(token: str, db: Session) -> User | None:
    """Look up a user by their auth token. Returns None if invalid/expired."""
    username = token_username(token)
    if username is None:
        return None
    try:
        return db.query(User).filter(User.username == username).first()
    except Exception:
        logger.exception("Database error looking up user by token")
        return None


def get_user_plan(token: str, db: Session) -> str:
    """Return the plan for the user identified by token."""
    user = get_user_by_token(token, db)
    if user is None:
        return "free"
    # Auto-promote developer accounts (configured via DEVELOPER_USERNAMES env var)
    if user.username in DEVELOPER_USERNAMES and user.plan != "developer":
        logger.info("Auto-promoting developer account: %r", user.username)
        try:
            user.plan = "developer"
            db.commit()
        except Exception:
            logger.exception("Failed to auto-promote developer account %r", user.username)
    return user.plan or "free"
