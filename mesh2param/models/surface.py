from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mesh2param.models.region import SurfaceType

logger = logging.getLogger(__name__)


@dataclass
class FittedSurface:
    """A fitted analytic surface for a mesh region.

    Output of Stage 3: Primitive Fitting.
    """

    region_id: int
    surface_type: SurfaceType
    params: dict  # surface-specific parameters
    rms_error: float
    snapped_params: dict = field(default_factory=dict)

    def __post_init__(self):
        logger.debug(
            "FittedSurface created: region_id=%d, type=%s, rms_error=%.6f",
            self.region_id,
            self.surface_type.value,
            self.rms_error,
        )
        if self.rms_error < 0:
            logger.error("FittedSurface region_id=%d has negative rms_error=%.6f", self.region_id, self.rms_error)

    def to_dict(self) -> dict:
        try:
            return {
                "region_id": self.region_id,
                "surface_type": self.surface_type.value,
                "params": self.params,
                "rms_error": self.rms_error,
                "snapped_params": self.snapped_params,
            }
        except Exception:
            logger.exception("FittedSurface.to_dict failed for region_id=%s", self.region_id)
            raise


class TransitionType(Enum):
    SHARP = "sharp"
    FILLET = "fillet"
    CHAMFER = "chamfer"
    SMOOTH = "smooth"


@dataclass
class SurfaceEdge:
    """A classified edge between two adjacent fitted surfaces.

    Output of Stage 4: Topology & Edge Classification.
    """

    surface_a_id: int
    surface_b_id: int
    transition_type: TransitionType
    dihedral_angle: float
    curve_params: dict = field(default_factory=dict)
    edge_vertices: np.ndarray | None = None  # (N, 3) line segments

    def __post_init__(self):
        logger.debug(
            "SurfaceEdge created: %d<->%d, transition=%s, dihedral=%.2f deg",
            self.surface_a_id,
            self.surface_b_id,
            self.transition_type.value,
            self.dihedral_angle,
        )

    def to_dict(self) -> dict:
        try:
            return {
                "surface_a_id": self.surface_a_id,
                "surface_b_id": self.surface_b_id,
                "transition_type": self.transition_type.value,
                "dihedral_angle": self.dihedral_angle,
            }
        except Exception:
            logger.exception(
                "SurfaceEdge.to_dict failed for edge %s<->%s",
                self.surface_a_id,
                self.surface_b_id,
            )
            raise
