from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SurfaceType(Enum):
    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    TORUS = "torus"
    UNKNOWN = "unknown"


@dataclass
class FaceRegion:
    """A cluster of mesh faces belonging to a single analytic surface.

    Output of Stage 2: Surface Segmentation.
    """

    region_id: int
    face_indices: np.ndarray  # (N,) int — indices into the original mesh face array
    surface_type: SurfaceType
    vertices: np.ndarray  # (M, 3) float64 — unique vertices in this region
    normals: np.ndarray  # (N, 3) float64 — per-face normals for faces in this region
    boundary_edges: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        logger.debug(
            "FaceRegion %d created: type=%s, %d faces, %d vertices, %d boundary edges",
            self.region_id,
            self.surface_type.value,
            len(self.face_indices),
            len(self.vertices),
            len(self.boundary_edges),
        )
        if len(self.face_indices) == 0:
            logger.warning("FaceRegion %d has zero faces", self.region_id)

    @property
    def num_faces(self) -> int:
        return len(self.face_indices)

    def to_dict(self) -> dict:
        try:
            return {
                "region_id": self.region_id,
                "num_faces": self.num_faces,
                "surface_type": self.surface_type.value,
                "num_vertices": len(self.vertices),
                "num_boundary_edges": len(self.boundary_edges),
            }
        except Exception:
            logger.exception("FaceRegion.to_dict failed for region_id=%s", self.region_id)
            raise
