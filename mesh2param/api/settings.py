"""User settings — change password, delete account.

Adapted from operatorDashboard/backend/api/settings.py.
Password hashing/verification is shared with mesh2param.api.auth.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from mesh2param.api.auth import _check_rate_limit, hash_password, verify_password
from mesh2param.db.database import User, MeshJob, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.post("/change-password")
def change_password(body: dict, request: Request, db: Session = Depends(get_db)):
    _check_rate_limit(request)
    username = body.get("username", "")
    current_password = body.get("current_password", "")
    new_password = body.get("new_password", "")
    logger.debug("Password change request for username=%r", username)

    if not username or not current_password or not new_password:
        raise HTTPException(status_code=400, detail="All fields required")
    if len(new_password) < 3:
        raise HTTPException(status_code=400, detail="Password must be at least 3 characters")

    try:
        user = db.query(User).filter(User.username == username).first()
    except Exception:
        logger.exception("Database error looking up user %r for password change", username)
        raise HTTPException(status_code=500, detail="Database error")
    if not user:
        logger.debug("Password change failed: user %r not found", username)
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(current_password, user.password_hash):
        logger.debug("Password change failed: incorrect current password for %r", username)
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    try:
        user.password_hash = hash_password(new_password)
        db.commit()
    except Exception:
        logger.exception("Database error saving new password for %r", username)
        raise HTTPException(status_code=500, detail="Database error saving password")
    logger.info("Password changed for user %r", username)
    return {"status": "ok", "message": "Password changed"}


@router.post("/delete-account")
def delete_account(body: dict, request: Request, db: Session = Depends(get_db)):
    _check_rate_limit(request)
    username = body.get("username", "")
    password = body.get("password", "")
    logger.debug("Account deletion request for username=%r", username)

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    try:
        user = db.query(User).filter(User.username == username).first()
    except Exception:
        logger.exception("Database error looking up user %r for deletion", username)
        raise HTTPException(status_code=500, detail="Database error")
    if not user:
        logger.debug("Account deletion failed: user %r not found", username)
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(password, user.password_hash):
        logger.debug("Account deletion failed: incorrect password for %r", username)
        raise HTTPException(status_code=401, detail="Incorrect password")

    # Delete all user data (jobs + steps cascade)
    try:
        jobs_deleted = db.query(MeshJob).filter(MeshJob.user_id == user.id).delete()
        db.delete(user)
        db.commit()
    except Exception:
        logger.exception("Database error deleting account %r", username)
        raise HTTPException(status_code=500, detail="Database error during account deletion")
    logger.info("Account deleted: %r (removed %d jobs)", username, jobs_deleted)
    return {"status": "ok", "message": "Account and all data deleted"}
