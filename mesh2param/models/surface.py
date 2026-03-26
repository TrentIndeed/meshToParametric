from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mesh2param.models.region import SurfaceType


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

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "surface_type": self.surface_type.value,
            "params": self.params,
            "rms_error": self.rms_error,
            "snapped_params": self.snapped_params,
        }


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

    def to_dict(self) -> dict:
        return {
            "surface_a_id": self.surface_a_id,
            "surface_b_id": self.surface_b_id,
            "transition_type": self.transition_type.value,
            "dihedral_angle": self.dihedral_angle,
        }
