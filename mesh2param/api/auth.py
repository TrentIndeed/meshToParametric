"""Auth endpoints — signup, login, verify.

Adapted from operatorDashboard/backend/main.py auth section.
Uses SHA-256 hashing with AUTH_SECRET for passwords and tokens.
"""

import hashlib
import logging
import os
import secrets

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from mesh2param.db.database import User, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_SECRET = os.getenv("AUTH_SECRET", secrets.token_hex(32))


def _hash_password(password: str) -> str:
    return hashlib.sha256(f"{AUTH_SECRET}:{password}".encode()).hexdigest()


def _make_token(username: str) -> str:
    return hashlib.sha256(f"{AUTH_SECRET}:token:{username}".encode()).hexdigest()


@router.post("/signup")
def signup(body: dict, db: Session = Depends(get_db)):
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
            password_hash=_hash_password(password),
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
def login(body: dict, db: Session = Depends(get_db)):
    username = body.get("username", "").strip()
    password = body.get("password", "")
    logger.debug("Login attempt for username=%r", username)

    try:
        user = db.query(User).filter(User.username == username).first()
    except Exception:
        logger.exception("Database error during login for %r", username)
        raise HTTPException(status_code=500, detail="Database error during login")
    if not user or user.password_hash != _hash_password(password):
        logger.debug("Login failed for username=%r", username)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    logger.info("User logged in: %r", username)
    token = _make_token(username)
    return {"token": token, "status": "ok", "username": username}


@router.get("/verify")
def verify_token(token: str, db: Session = Depends(get_db)):
    logger.debug("Token verification attempt")
    try:
        users = db.query(User).all()
    except Exception:
        logger.exception("Database error during token verification")
        raise HTTPException(status_code=500, detail="Database error during verification")
    for user in users:
        if _make_token(user.username) == token:
            logger.debug("Token verified for user=%r plan=%s", user.username, user.plan)
            return {"status": "ok", "username": user.username, "plan": user.plan or "free"}
    logger.debug("Token verification failed — no matching user")
    raise HTTPException(status_code=401, detail="Invalid token")


# ── Developer accounts ──
DEVELOPER_USERNAMES = {"123"}


def get_user_by_token(token: str, db: Session) -> User | None:
    """Look up a user by their auth token. Returns None if not found."""
    try:
        for user in db.query(User).all():
            if _make_token(user.username) == token:
                return user
    except Exception:
        logger.exception("Database error looking up user by token")
    return None


def get_user_plan(token: str, db: Session) -> str:
    """Return the plan for the user identified by token."""
    user = get_user_by_token(token, db)
    if user is None:
        return "free"
    # Auto-promote developer accounts
    if user.username in DEVELOPER_USERNAMES and user.plan != "developer":
        logger.info("Auto-promoting developer account: %r", user.username)
        try:
            user.plan = "developer"
            db.commit()
        except Exception:
            logger.exception("Failed to auto-promote developer account %r", user.username)
    return user.plan or "free"
