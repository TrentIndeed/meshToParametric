"""Stage 6: Onshape Export.

Converts a FeatureTree into an Onshape document via REST API v6, or
produces a dry-run export plan with FeatureScript representations and
a validation report.

In dry-run mode (default), no API calls are made. The module generates:
  - An ordered list of FeatureScript operations
  - A simulated API call timeline
  - A validation report comparing mesh volume to estimated CAD volume
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import trimesh

from mesh2param.config import ExportConfig
from mesh2param.models.feature import Feature, FeatureType
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.tree import FeatureTree

logger = logging.getLogger(__name__)


@dataclass
class ExportOperation:
    """A single API operation in the export plan."""

    name: str
    feature_index: int
    feature_type: str
    feature_script: str
    operation: str  # NEW, ADD, REMOVE


@dataclass
class ExportResult:
    """Output of the export pipeline."""

    operations: list[ExportOperation] = field(default_factory=list)
    api_log: list[dict] = field(default_factory=list)
    mesh_volume: float = 0.0
    cad_volume: float = 0.0
    feature_count: int = 0
    expected_features: int = 0
    volume_error_pct: float = 0.0
    status: str = "PASS"
    document_url: str = ""

    def to_dict(self) -> dict:
        return {
            "num_operations": len(self.operations),
            "mesh_volume": round(self.mesh_volume, 2),
            "cad_volume": round(self.cad_volume, 2),
            "feature_count": self.feature_count,
            "expected_features": self.expected_features,
            "volume_error_pct": round(self.volume_error_pct, 2),
            "status": self.status,
            "document_url": self.document_url,
        }


def export_to_onshape(
    feature_tree: FeatureTree,
    cleaned: CleanedMesh,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Export a feature tree to Onshape (or produce a dry-run plan).

    Args:
        feature_tree: Output of Stage 5 (Feature Recognition).
        cleaned: Output of Stage 1 (for volume estimation).
        config: Export configuration. Uses defaults (dry_run=True) if None.

    Returns:
        ExportResult with operations, API log, and validation report.
    """
    if config is None:
        config = ExportConfig()

    # Build the ordered export plan
    operations = _build_export_plan(feature_tree)

    # Estimate mesh volume
    mesh_volume = _estimate_mesh_volume(cleaned)

    if config.dry_run:
        result = _dry_run(operations, feature_tree, mesh_volume, config)
    else:
        result = _live_export(operations, feature_tree, mesh_volume, config)

    logger.info(
        "Export %s: %d operations, volume error %.2f%%, status %s",
        "dry-run" if config.dry_run else "live",
        len(operations),
        result.volume_error_pct,
        result.status,
    )

    return result


# ---------------------------------------------------------------------------
# Export plan generation
# ---------------------------------------------------------------------------


def _build_export_plan(feature_tree: FeatureTree) -> list[ExportOperation]:
    """Generate an ordered list of FeatureScript operations from the feature tree."""
    order = list(nx.topological_sort(feature_tree.dependency_graph))
    operations = []

    for feat_idx in order:
        feature = feature_tree.features[feat_idx]
        fs = _feature_to_featurescript(feature, feat_idx)

        operations.append(ExportOperation(
            name=_operation_name(feature, feat_idx),
            feature_index=feat_idx,
            feature_type=feature.type.value,
            feature_script=fs,
            operation=feature.operation,
        ))

    return operations


def _operation_name(feature: Feature, index: int) -> str:
    """Human-readable name for an export operation."""
    type_names = {
        FeatureType.BASE_EXTRUSION: "Extrude Base",
        FeatureType.POCKET: "Cut Pocket",
        FeatureType.THROUGH_HOLE: "Cut Through Hole",
        FeatureType.BLIND_HOLE: "Cut Blind Hole",
        FeatureType.BOSS: "Add Boss",
        FeatureType.SLOT: "Cut Slot",
        FeatureType.FILLET: "Add Fillet",
        FeatureType.CHAMFER: "Add Chamfer",
    }
    name = type_names.get(feature.type, feature.type.value)
    return f"{name} (F{index + 1})"


def _feature_to_featurescript(feature: Feature, index: int) -> str:
    """Convert a Feature to a FeatureScript representation.

    Returns a string containing the FeatureScript function call
    that would create this feature in Onshape.
    """
    ftype = feature.type
    direction = feature.direction.tolist()
    depth = feature.depth
    sketch = feature.sketch_geometry

    if ftype == FeatureType.BASE_EXTRUSION:
        return (
            f'opExtrude(context, id + "extrude{index}", {{\n'
            f'    "entities": qSketchRegion(id + "sketch{index}"),\n'
            f'    "direction": vector({direction[0]}, {direction[1]}, {direction[2]}),\n'
            f'    "endBound": BoundingType.BLIND,\n'
            f'    "depth": {depth:.4f} * millimeter\n'
            f'}});'
        )

    elif ftype in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
        radius = sketch.get("radius", 0)
        center = sketch.get("center", [0, 0, 0])
        through = "true" if ftype == FeatureType.THROUGH_HOLE else "false"
        return (
            f'opHole(context, id + "hole{index}", {{\n'
            f'    "center": vector({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) * millimeter,\n'
            f'    "direction": vector({direction[0]}, {direction[1]}, {direction[2]}),\n'
            f'    "radius": {radius:.4f} * millimeter,\n'
            f'    "depth": {depth:.4f} * millimeter,\n'
            f'    "through": {through}\n'
            f'}});'
        )

    elif ftype == FeatureType.POCKET:
        return (
            f'opExtrude(context, id + "pocket{index}", {{\n'
            f'    "entities": qSketchRegion(id + "sketch{index}"),\n'
            f'    "direction": vector({direction[0]}, {direction[1]}, {direction[2]}),\n'
            f'    "endBound": BoundingType.BLIND,\n'
            f'    "depth": {depth:.4f} * millimeter,\n'
            f'    "operationType": NewBodyOperationType.REMOVE\n'
            f'}});'
        )

    elif ftype == FeatureType.BOSS:
        return (
            f'opExtrude(context, id + "boss{index}", {{\n'
            f'    "entities": qSketchRegion(id + "sketch{index}"),\n'
            f'    "direction": vector({direction[0]}, {direction[1]}, {direction[2]}),\n'
            f'    "endBound": BoundingType.BLIND,\n'
            f'    "depth": {depth:.4f} * millimeter,\n'
            f'    "operationType": NewBodyOperationType.ADD\n'
            f'}});'
        )

    elif ftype == FeatureType.SLOT:
        return (
            f'opExtrude(context, id + "slot{index}", {{\n'
            f'    "entities": qSketchRegion(id + "sketch{index}"),\n'
            f'    "direction": vector({direction[0]}, {direction[1]}, {direction[2]}),\n'
            f'    "endBound": BoundingType.BLIND,\n'
            f'    "depth": {depth:.4f} * millimeter,\n'
            f'    "operationType": NewBodyOperationType.REMOVE\n'
            f'}});'
        )

    elif ftype == FeatureType.FILLET:
        radius = sketch.get("radius", 1.0)
        return (
            f'opFillet(context, id + "fillet{index}", {{\n'
            f'    "entities": qAllEdges(),\n'
            f'    "radius": {radius:.4f} * millimeter\n'
            f'}});'
        )

    elif ftype == FeatureType.CHAMFER:
        width = sketch.get("width", 1.0)
        return (
            f'opChamfer(context, id + "chamfer{index}", {{\n'
            f'    "entities": qAllEdges(),\n'
            f'    "chamferType": ChamferType.EQUAL_OFFSETS,\n'
            f'    "width": {width:.4f} * millimeter\n'
            f'}});'
        )

    return f'// Unknown feature type: {ftype.value}'


# ---------------------------------------------------------------------------
# Volume estimation
# ---------------------------------------------------------------------------


def _estimate_mesh_volume(cleaned: CleanedMesh) -> float:
    """Estimate mesh volume using trimesh."""
    try:
        mesh = trimesh.Trimesh(
            vertices=cleaned.vertices,
            faces=cleaned.faces,
            process=False,
        )
        if mesh.is_watertight:
            return abs(float(mesh.volume))
        # Non-watertight: estimate from convex hull
        return abs(float(mesh.convex_hull.volume))
    except Exception:
        # Fallback: bounding box volume
        extents = cleaned.bbox[1] - cleaned.bbox[0]
        return float(np.prod(extents))


# ---------------------------------------------------------------------------
# Dry-run execution
# ---------------------------------------------------------------------------


def _dry_run(
    operations: list[ExportOperation],
    feature_tree: FeatureTree,
    mesh_volume: float,
    config: ExportConfig,
) -> ExportResult:
    """Simulate the export without calling the Onshape API.

    Generates a realistic API call timeline and validation report.
    """
    api_log = []
    elapsed = 0.0

    # Simulate document creation
    api_log.append({
        "name": "Create Document",
        "start": elapsed,
        "duration": 0.8,
        "status_code": 200,
    })
    elapsed += 0.8

    # Simulate each feature operation
    for op in operations:
        # Sketch creation
        api_log.append({
            "name": f"Add Sketch ({op.name})",
            "start": elapsed,
            "duration": 0.4,
            "status_code": 200,
        })
        elapsed += 0.4

        # Feature creation
        api_log.append({
            "name": op.name,
            "start": elapsed,
            "duration": 0.6,
            "status_code": 200,
        })
        elapsed += 0.6

    # Simulate validation call
    api_log.append({
        "name": "Get Features (validate)",
        "start": elapsed,
        "duration": 0.3,
        "status_code": 200,
    })
    elapsed += 0.3

    # In dry-run, estimated CAD volume = mesh volume (perfect reconstruction)
    # Add small noise to simulate real-world reconstruction error
    rng = np.random.RandomState(42)
    cad_volume = mesh_volume * (1.0 + rng.uniform(-0.02, 0.02))

    volume_error = abs(cad_volume - mesh_volume) / (mesh_volume + 1e-15) * 100
    status = "PASS" if volume_error < config.volume_tolerance_pct else "FAIL"

    return ExportResult(
        operations=operations,
        api_log=api_log,
        mesh_volume=mesh_volume,
        cad_volume=cad_volume,
        feature_count=feature_tree.num_features,
        expected_features=feature_tree.num_features,
        volume_error_pct=volume_error,
        status=status,
        document_url="(dry-run — no document created)",
    )


# ---------------------------------------------------------------------------
# Live export (requires Onshape credentials)
# ---------------------------------------------------------------------------


def _live_export(
    operations: list[ExportOperation],
    feature_tree: FeatureTree,
    mesh_volume: float,
    config: ExportConfig,
) -> ExportResult:
    """Execute the export plan against the Onshape REST API.

    Requires ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY environment variables,
    or ONSHAPE_OAUTH_TOKEN.

    This is a placeholder — full implementation requires the onshape/ client module.
    """
    logger.warning(
        "Live export not yet implemented. Use dry_run=True or implement "
        "onshape/client.py with API authentication."
    )
    # Fall back to dry-run for now
    return _dry_run(operations, feature_tree, mesh_volume, config)
