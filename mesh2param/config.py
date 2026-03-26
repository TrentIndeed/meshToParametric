from pydantic import BaseModel


class IngestConfig(BaseModel):
    """Configuration for Stage 1: Mesh Ingestion & Cleanup."""

    merge_tolerance: float = 1e-8
    min_face_area: float = 1e-10
    remesh: bool = True
    target_edge_ratio: float = 0.005
    max_hole_edges: int = 50


class SegmentConfig(BaseModel):
    """Configuration for Stage 2: Surface Segmentation."""

    plane_angle_threshold_deg: float = 15.0
    curved_angle_threshold_deg: float = 30.0
    min_region_faces: int = 10
    curvature_variance_threshold: float = 0.1
    max_kmeans_clusters: int = 5


class FitConfig(BaseModel):
    """Configuration for Stage 3: Primitive Fitting."""

    rms_tolerance_ratio: float = 0.005  # max RMS as fraction of bbox diagonal
    ransac_iterations: int = 1000
    ransac_inlier_threshold_ratio: float = 0.01  # inlier distance as fraction of bbox diagonal
    snap_tolerance: float = 0.05  # snap params to round values within this abs tolerance
    axis_snap_angle_deg: float = 2.0  # snap axis to principal axis if within this angle
    try_alternative_types: bool = True  # retry with other surface types if RMS too high
