"""Database models and session for mesh2param.

Adapted from operatorDashboard. Uses SQLite by default.
Models: User (auth), MeshJob (pipeline runs), JobStep (construction steps).
"""

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Default to ./data/mesh2param.db
data_dir = Path(__file__).parent.parent.parent / "data"
data_dir.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{data_dir}/mesh2param.db"

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
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending | processing | done | error
    error = Column(Text)
    mesh_volume = Column(Float, default=0.0)
    cad_volume = Column(Float, default=0.0)
    volume_error_pct = Column(Float, default=0.0)
    num_features = Column(Integer, default=0)
    num_steps = Column(Integer, default=0)
    ai_tokens_used = Column(Integer, default=0)
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
    # Geometry stored as JSON blobs (vertices/faces arrays)
    vertices_json = Column(Text)
    faces_json = Column(Text)

    job = relationship("MeshJob", back_populates="steps")


# ── Helpers ──

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
