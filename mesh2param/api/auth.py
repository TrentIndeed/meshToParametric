"""Auth endpoints — signup, login, verify.

Adapted from operatorDashboard/backend/main.py auth section.
Uses SHA-256 hashing with AUTH_SECRET for passwords and tokens.
"""

import hashlib
import os
import secrets

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from mesh2param.db.database import User, get_db

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

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    if len(password) < 3:
        raise HTTPException(status_code=400, detail="Password must be at least 3 characters")

    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(
        username=username,
        email=email or None,
        password_hash=_hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = _make_token(username)
    return {"token": token, "status": "ok", "username": username}


@router.post("/login")
def login(body: dict, db: Session = Depends(get_db)):
    username = body.get("username", "").strip()
    password = body.get("password", "")

    user = db.query(User).filter(User.username == username).first()
    if not user or user.password_hash != _hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = _make_token(username)
    return {"token": token, "status": "ok", "username": username}


@router.get("/verify")
def verify_token(token: str, db: Session = Depends(get_db)):
    users = db.query(User).all()
    for user in users:
        if _make_token(user.username) == token:
            return {"status": "ok", "username": user.username, "plan": user.plan or "free"}
    raise HTTPException(status_code=401, detail="Invalid token")


# ── Developer accounts ──
DEVELOPER_USERNAMES = {"123"}


def get_user_by_token(token: str, db: Session) -> User | None:
    """Look up a user by their auth token. Returns None if not found."""
    for user in db.query(User).all():
        if _make_token(user.username) == token:
            return user
    return None


def get_user_plan(token: str, db: Session) -> str:
    """Return the plan for the user identified by token."""
    user = get_user_by_token(token, db)
    if user is None:
        return "free"
    # Auto-promote developer accounts
    if user.username in DEVELOPER_USERNAMES and user.plan != "developer":
        user.plan = "developer"
        db.commit()
    return user.plan or "free"
