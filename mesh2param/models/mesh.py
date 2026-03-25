from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CleanedMesh:
    """Output of Stage 1: a validated, repaired mesh ready for segmentation."""

    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int
    normals: np.ndarray  # (F, 3) float64 — per-face normals
    is_watertight: bool
    bbox: np.ndarray  # (2, 3) — min corner, max corner
    metadata: dict = field(default_factory=dict)

    @property
    def bbox_diagonal(self) -> float:
        return float(np.linalg.norm(self.bbox[1] - self.bbox[0]))

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    def to_dict(self) -> dict:
        return {
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "is_watertight": self.is_watertight,
            "bbox_min": self.bbox[0].tolist(),
            "bbox_max": self.bbox[1].tolist(),
            "bbox_diagonal": self.bbox_diagonal,
            "metadata": self.metadata,
        }
