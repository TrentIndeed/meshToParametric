"""User settings — change password, delete account.

Adapted from operatorDashboard/backend/api/settings.py.
"""

import hashlib
import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from mesh2param.db.database import User, MeshJob, JobStep, get_db

router = APIRouter(prefix="/settings", tags=["settings"])


@router.post("/change-password")
def change_password(body: dict, db: Session = Depends(get_db)):
    username = body.get("username", "")
    current_password = body.get("current_password", "")
    new_password = body.get("new_password", "")

    if not username or not current_password or not new_password:
        raise HTTPException(status_code=400, detail="All fields required")
    if len(new_password) < 3:
        raise HTTPException(status_code=400, detail="Password must be at least 3 characters")

    auth_secret = os.getenv("AUTH_SECRET", "")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    current_hash = hashlib.sha256(f"{auth_secret}:{current_password}".encode()).hexdigest()
    if user.password_hash != current_hash:
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    user.password_hash = hashlib.sha256(f"{auth_secret}:{new_password}".encode()).hexdigest()
    db.commit()
    return {"status": "ok", "message": "Password changed"}


@router.post("/delete-account")
def delete_account(body: dict, db: Session = Depends(get_db)):
    username = body.get("username", "")
    password = body.get("password", "")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    auth_secret = os.getenv("AUTH_SECRET", "")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    pw_hash = hashlib.sha256(f"{auth_secret}:{password}".encode()).hexdigest()
    if user.password_hash != pw_hash:
        raise HTTPException(status_code=401, detail="Incorrect password")

    # Delete all user data (jobs + steps cascade)
    db.query(MeshJob).filter(MeshJob.user_id == user.id).delete()
    db.delete(user)
    db.commit()
    return {"status": "ok", "message": "Account and all data deleted"}
