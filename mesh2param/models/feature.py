from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    BASE_EXTRUSION = "BASE_EXTRUSION"
    REVOLVE = "REVOLVE"
    POCKET = "POCKET"
    THROUGH_HOLE = "THROUGH_HOLE"
    BLIND_HOLE = "BLIND_HOLE"
    BOSS = "BOSS"
    SLOT = "SLOT"
    FILLET = "FILLET"
    CHAMFER = "CHAMFER"


@dataclass
class Feature:
    """A recognized manufacturing feature.

    Output of Stage 5: Feature Recognition.
    """

    type: FeatureType
    face_indices: np.ndarray  # indices into the original mesh
    sketch_geometry: dict = field(default_factory=dict)
    depth: float = 0.0
    direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1.0]))
    operation: str = "NEW"  # NEW, ADD, REMOVE
    dependencies: list[int] = field(default_factory=list)
    onshape_query: dict = field(default_factory=dict)

    def __post_init__(self):
        logger.debug(
            "Feature created: type=%s, depth=%.4f, operation=%s, %d faces, deps=%s",
            self.type.value,
            self.depth,
            self.operation,
            len(self.face_indices),
            self.dependencies,
        )
        if len(self.face_indices) == 0:
            logger.warning("Feature type=%s has zero face_indices", self.type.value)

    def params_dict(self) -> dict:
        return {
            "type": self.type.value,
            "depth": self.depth,
            "direction": self.direction.tolist(),
            "operation": self.operation,
            "num_faces": len(self.face_indices),
        }

    def to_dict(self) -> dict:
        try:
            return {
                "type": self.type.value,
                "num_faces": len(self.face_indices),
                "depth": self.depth,
                "operation": self.operation,
                "dependencies": self.dependencies,
            }
        except Exception:
            logger.exception("Feature.to_dict failed for type=%s", self.type)
            raise
