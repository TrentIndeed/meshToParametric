"""Database models and session for mesh2param.

Adapted from operatorDashboard. Uses SQLite by default.
Models: User (auth), MeshJob (pipeline runs), JobStep (construction steps).
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey,
    LargeBinary,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

logger = logging.getLogger(__name__)

# Default to ./data/mesh2param.db
data_dir = Path(__file__).parent.parent.parent / "data"
data_dir.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{data_dir}/mesh2param.db"

logger.debug("Database URL: %s", DATABASE_URL)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ── Auth ──

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False, unique=True)
    email = Column(String)
    password_hash = Column(String, nullable=False)
    plan = Column(String, default="free")  # free | pro
    stripe_customer_id = Column(String)
    stripe_subscription_id = Column(String)
    subscription_status = Column(String, default="none")
    created_at = Column(DateTime, default=datetime.utcnow)

    jobs = relationship("MeshJob", back_populates="user")


# ── Pipeline Jobs ──

class MeshJob(Base):
    """A single mesh-to-parametric conversion run."""
    __tablename__ = "mesh_jobs"
    id = Column(Integer, primary_key=True, index=True)
    job_uuid = Column(String, unique=True, index=True)  # matches Job.id from queue
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    filename = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending | processing | done | error
    error = Column(Text)
    mesh_volume = Column(Float, default=0.0)
    cad_volume = Column(Float, default=0.0)
    volume_error_pct = Column(Float, default=0.0)
    num_features = Column(Integer, default=0)
    num_steps = Column(Integer, default=0)
    ai_tokens_used = Column(Integer, default=0)
    token_cost = Column(Integer, default=0)
    bbox_min_json = Column(Text)  # JSON [x, y, z]
    bbox_max_json = Column(Text)  # JSON [x, y, z]
    features_json = Column(Text)  # JSON summary of feature tree
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    user = relationship("User", back_populates="jobs")
    steps = relationship("JobStep", back_populates="job", cascade="all, delete-orphan")


class JobStep(Base):
    """A construction step within a mesh job (sketch, extrude, hole, etc.)."""
    __tablename__ = "job_steps"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("mesh_jobs.id"), nullable=False)
    step_index = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    volume = Column(Float, default=0.0)
    is_sketch = Column(Boolean, default=False)
    # Geometry stored as binary blobs (numpy .tobytes())
    vertices_blob = Column(LargeBinary)  # float64 (V, 3) flattened
    faces_blob = Column(LargeBinary)     # int64 (F, 3) flattened
    vertices_shape = Column(String)      # "V,3"
    faces_shape = Column(String)         # "F,3"
    # Keep legacy columns for backward compat
    vertices_json = Column(Text)
    faces_json = Column(Text)

    job = relationship("MeshJob", back_populates="steps")


# ── Waitlist ──

class WaitlistEntry(Base):
    __tablename__ = "waitlist"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Support Tickets ──

class SupportTicket(Base):
    __tablename__ = "support_tickets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    status = Column(String, default="open")  # open | replied | closed
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Helpers ──

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        logger.exception("Database session error during request")
        raise
    finally:
        db.close()


def init_db():
    logger.debug("Initializing database (create_all + migrate)")
    try:
        Base.metadata.create_all(bind=engine)
        _migrate_db()
        logger.info("Database initialized successfully")
    except Exception:
        logger.exception("Failed to initialize database")
        raise


def _migrate_db():
    """Add columns that may be missing from older DB schemas (idempotent)."""
    import sqlite3

    # Only applies to SQLite
    if "sqlite" not in DATABASE_URL:
        return

    db_path = DATABASE_URL.replace("sqlite:///", "")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        def _add_col(table, col, coltype, default=None):
            try:
                dflt = f" DEFAULT {default}" if default is not None else ""
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}{dflt}")
            except sqlite3.OperationalError:
                pass  # column already exists

        # MeshJob new columns
        _add_col("mesh_jobs", "job_uuid", "TEXT")
        _add_col("mesh_jobs", "token_cost", "INTEGER", 0)
        _add_col("mesh_jobs", "bbox_min_json", "TEXT")
        _add_col("mesh_jobs", "bbox_max_json", "TEXT")
        _add_col("mesh_jobs", "features_json", "TEXT")

        # JobStep new columns (blob-based tessellation)
        _add_col("job_steps", "vertices_blob", "BLOB")
        _add_col("job_steps", "faces_blob", "BLOB")
        _add_col("job_steps", "vertices_shape", "TEXT")
        _add_col("job_steps", "faces_shape", "TEXT")

        # Make user_id nullable for anonymous jobs
        # (SQLite doesn't support ALTER COLUMN, but new rows can have NULL
        # if the column was added without NOT NULL — the original schema has
        # nullable=False but create_all won't retroactively enforce it)

        conn.commit()
        conn.close()
        logger.debug("Database migration completed")
    except Exception:
        logger.exception("Database migration failed (non-fatal for fresh DBs)")
