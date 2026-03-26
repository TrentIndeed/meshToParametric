"""Export mesh data as GLB or JSON for the browser-based Three.js viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


def export_colored_mesh_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors_rgba: np.ndarray,
    output_path: str | Path,
) -> str:
    """Export a mesh with per-face colors as GLB.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        face_colors_rgba: (F, 4) RGBA colors per face (0-255).
        output_path: Path for the .glb file.

    Returns:
        The output path as a string.
    """
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_colors=face_colors_rgba,
    )
    output_path = str(output_path)
    mesh.export(output_path, file_type="glb")
    return output_path


def export_edges_json(
    classified_edges: list,
    output_path: str | Path,
) -> str:
    """Export classified edges as JSON for Three.js line rendering.

    Each edge entry: {vertices: [[x,y,z],[x,y,z]], type: "SHARP", dihedral: 90.0}

    Args:
        classified_edges: List of SurfaceEdge objects.
        output_path: Path for the .json file.

    Returns:
        The output path as a string.
    """
    data = []
    for edge in classified_edges:
        entry = {
            "type": edge.transition_type.value if hasattr(edge.transition_type, "value") else str(edge.transition_type),
            "dihedral": float(edge.dihedral_angle),
        }
        if edge.edge_vertices is not None:
            entry["vertices"] = edge.edge_vertices.tolist()
        data.append(entry)

    output_path = str(output_path)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path


def export_feature_tree_json(
    feature_tree,
    output_path: str | Path,
) -> str:
    """Export feature tree for the browser feature explorer.

    Args:
        feature_tree: A FeatureTree object.
        output_path: Path for the .json file.

    Returns:
        The output path as a string.
    """
    data = {
        "features": [],
        "dependencies": [],
    }
    for i, feat in enumerate(feature_tree.features):
        data["features"].append({
            "id": i,
            "type": feat.type.value if hasattr(feat.type, "value") else str(feat.type),
            "params": feat.params_dict(),
            "face_indices": feat.face_indices.tolist(),
        })
    for edge in feature_tree.dependency_graph.edges():
        data["dependencies"].append({"from": edge[0], "to": edge[1]})

    output_path = str(output_path)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path
