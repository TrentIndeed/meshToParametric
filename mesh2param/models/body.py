"""Dataclasses for body decomposition (Stage 1.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PrismaticBody:
    """A prismatic sub-body identified by decomposition."""

    body_id: int
    face_indices: np.ndarray       # (N,) int — indices into original mesh faces
    extrude_direction: np.ndarray  # (3,) unit vector
    depth: float                   # extent along extrude_direction
    is_base: bool = False          # True for the first/largest body
    operation: str = "UNION"       # UNION or CUT relative to base
    metadata: dict = field(default_factory=dict)


@dataclass
class BodyDecomposition:
    """Output of Stage 1.5: Body Decomposition."""

    bodies: list[PrismaticBody]
    is_multi_body: bool  # True if >1 body found
    metadata: dict = field(default_factory=dict)
