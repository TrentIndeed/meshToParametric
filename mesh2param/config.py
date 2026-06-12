import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class IngestConfig(BaseModel):
    """Configuration for Stage 1: Mesh Ingestion & Cleanup."""

    merge_tolerance: float = 1e-8
    min_face_area: float = 1e-10
    remesh: bool = True
    target_edge_ratio: float = 0.005
    max_hole_edges: int = 50
    # Decimate meshes above this face count down to it (quadric edge
    # collapse). Scans arrive at 300K-2.5M faces; segmentation cost is
    # superlinear in faces, so cap before Stage 2. 0 disables.
    max_faces: int = 30_000
    # Screened-Poisson surface reconstruction for ragged scan input. Only
    # runs when the cleaned mesh is NOT watertight after normal repair —
    # clean CAD meshes are left untouched, because Poisson SMOOTHS the
    # surface and would round sharp prismatic edges. Closing the open
    # boundaries of a scan gives it a real volume, which the
    # average-cross-section base-profile principle and the volume metric
    # both need. Uses pymeshlab's built-in filter (already a dependency).
    poisson_reconstruct: bool = True
    poisson_depth: int = 8
    # Scale normalization: the pipeline's absolute thresholds (snap
    # tolerances, fillet radii, plate-thickness checks) are tuned for
    # mm-scale parts with a bbox diagonal around 100-500. Parts far
    # outside that regime (inch-unit exports read as numbers, 10x
    # scans) are scaled to canonical_diag at ingest;
    # CleanedMesh.metadata["unit_scale"] records
    # mesh_units = original_units * unit_scale, and the STEP export
    # scales the solid back to original units. In-range parts pass
    # through untouched.
    normalize_scale: bool = True
    canonical_diag: float = 200.0
    normalize_min_diag: float = 25.0
    normalize_max_diag: float = 2000.0

    def model_post_init(self, __context) -> None:
        logger.debug(
            "IngestConfig loaded: remesh=%s, target_edge_ratio=%.4f, max_hole_edges=%d",
            self.remesh,
            self.target_edge_ratio,
            self.max_hole_edges,
        )


class SegmentConfig(BaseModel):
    """Configuration for Stage 2: Surface Segmentation."""

    plane_angle_threshold_deg: float = 15.0
    curved_angle_threshold_deg: float = 30.0
    min_region_faces: int = 10
    curvature_variance_threshold: float = 0.1
    max_kmeans_clusters: int = 5

    def model_post_init(self, __context) -> None:
        logger.debug(
            "SegmentConfig loaded: plane_angle=%.1f deg, curved_angle=%.1f deg, min_faces=%d",
            self.plane_angle_threshold_deg,
            self.curved_angle_threshold_deg,
            self.min_region_faces,
        )


class FitConfig(BaseModel):
    """Configuration for Stage 3: Primitive Fitting."""

    rms_tolerance_ratio: float = 0.005  # max RMS as fraction of bbox diagonal
    ransac_iterations: int = 1000
    ransac_inlier_threshold_ratio: float = 0.01  # inlier distance as fraction of bbox diagonal
    snap_tolerance: float = 0.05  # snap params to round values within this abs tolerance
    axis_snap_angle_deg: float = 2.0  # snap axis to principal axis if within this angle
    try_alternative_types: bool = True  # retry with other surface types if RMS too high

    def model_post_init(self, __context) -> None:
        logger.debug(
            "FitConfig loaded: rms_tol=%.4f, ransac_iter=%d, snap_tol=%.3f, alt_types=%s",
            self.rms_tolerance_ratio,
            self.ransac_iterations,
            self.snap_tolerance,
            self.try_alternative_types,
        )


class TopologyConfig(BaseModel):
    """Configuration for Stage 4: Topology & Edge Classification."""

    sharp_angle_threshold_deg: float = 160.0  # dihedral below this → SHARP
    smooth_angle_threshold_deg: float = 175.0  # dihedral above this → SMOOTH
    fillet_radius_ratio: float = 0.05  # max fillet radius as fraction of bbox diagonal
    chamfer_width_ratio: float = 0.05  # max chamfer extent as fraction of bbox diagonal

    def model_post_init(self, __context) -> None:
        logger.debug(
            "TopologyConfig loaded: sharp_angle=%.1f deg, smooth_angle=%.1f deg",
            self.sharp_angle_threshold_deg,
            self.smooth_angle_threshold_deg,
        )


class RecognizeConfig(BaseModel):
    """Configuration for Stage 5: Feature Recognition."""

    hole_max_radius_ratio: float = 0.15  # max hole radius as fraction of bbox diagonal
    pocket_max_depth_ratio: float = 0.8  # max pocket depth as fraction of bbox diagonal
    boss_max_height_ratio: float = 0.8  # max boss height as fraction of bbox diagonal
    slot_max_width_ratio: float = 0.3  # max slot width as fraction of bbox diagonal
    chamfer_width_ratio: float = 0.05  # max chamfer extent as fraction of bbox diagonal
    min_feature_faces: int = 3  # minimum faces for a feature

    def model_post_init(self, __context) -> None:
        logger.debug(
            "RecognizeConfig loaded: min_feature_faces=%d, hole_max_r=%.2f, pocket_max_d=%.2f",
            self.min_feature_faces,
            self.hole_max_radius_ratio,
            self.pocket_max_depth_ratio,
        )


class DecomposeConfig(BaseModel):
    """Configuration for Stage 1.5: Body Decomposition."""

    axis_angle_threshold_deg: float = 20.0  # face normal to axis tolerance
    min_body_faces: int = 20               # minimum faces for a sub-body
    min_body_volume_ratio: float = 0.05    # minimum volume fraction
    enabled: bool = True


class ExportConfig(BaseModel):
    """Configuration for Stage 6: Onshape Export."""

    dry_run: bool = True  # generate plan without calling API
    document_name: str = ""  # live export: Onshape document name ("" = auto)
    onshape_base_url: str = "https://cad.onshape.com"
    api_timeout: float = 30.0
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    volume_tolerance_pct: float = 5.0  # acceptable volume error %

    def model_post_init(self, __context) -> None:
        logger.debug(
            "ExportConfig loaded: dry_run=%s, base_url=%s, timeout=%.1fs, retries=%d",
            self.dry_run,
            self.onshape_base_url,
            self.api_timeout,
            self.max_retries,
        )
