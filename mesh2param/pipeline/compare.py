"""Cross-section slicer comparison: original mesh vs reconstructed CAD solid.

Slices both geometries at multiple heights along the extrude direction,
computes polygon overlap metrics, and generates an annotated matplotlib figure.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import trimesh

from mesh2param.models.mesh import CleanedMesh

logger = logging.getLogger(__name__)


def _make_trimesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Build a trimesh, ensuring correct dtypes."""
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _section_to_polygons(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    ax_u: int = 0,
    ax_v: int = 2,
) -> list:
    """Slice a mesh at a plane and return shapely polygons.

    Uses trimesh's to_planar() for robust polygon extraction, then
    re-maps the 2D coordinates to fixed world axes (ax_u, ax_v) so
    both the original mesh and CAD mesh use the same coordinate frame.

    Returns an empty list if the section is empty or fails.
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.validation import make_valid

    try:
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    except Exception:
        logger.debug("Mesh section failed at origin=%s, normal=%s", plane_origin, plane_normal)
        return []

    if section is None:
        return []

    # Get 3D vertices of the section
    verts_3d = np.array(section.vertices)
    if len(verts_3d) == 0:
        return []

    try:
        path2d, to_3d_matrix = section.to_planar()
    except Exception:
        return []

    if path2d is None or len(path2d.entities) == 0:
        return []

    # Build a mapping from 2D path vertices back to 3D vertices
    verts_2d = np.array(path2d.vertices)

    # Instead of using the arbitrary 2D coordinates from to_planar(),
    # project the original 3D section vertices onto fixed world axes.
    # This ensures both meshes share the same coordinate frame.
    fixed_2d = verts_3d[:, [ax_u, ax_v]]

    # Build index mapping: for each 2D path vertex, find the closest 3D vertex
    from scipy.spatial import cKDTree
    tree_3d = cKDTree(verts_3d[:, :3])

    # Reconstruct the 2D path vertices using to_3d_matrix to get back to 3D,
    # then project onto fixed axes
    ones = np.ones((len(verts_2d), 1))
    verts_2d_h = np.hstack([verts_2d, np.zeros((len(verts_2d), 1)), ones])
    verts_3d_reconstructed = (to_3d_matrix @ verts_2d_h.T).T[:, :3]
    fixed_coords = verts_3d_reconstructed[:, [ax_u, ax_v]]

    # Replace the path's vertex coordinates with fixed-axis projections
    path2d.vertices = fixed_coords

    # Now extract polygons — they'll be in the fixed coordinate frame
    polys = []
    try:
        for poly in path2d.polygons_full:
            p = make_valid(poly)
            if p.is_empty:
                continue
            if isinstance(p, MultiPolygon):
                polys.extend(g for g in p.geoms if g.area > 0)
            elif isinstance(p, Polygon) and p.area > 0:
                polys.append(p)
    except Exception:
        try:
            for poly in path2d.polygons_closed:
                p = make_valid(poly)
                if not p.is_empty and p.area > 0:
                    polys.append(p)
        except Exception:
            pass

    return polys


def _union_polygons(polys: list):
    """Union a list of shapely polygons into one geometry. Returns empty polygon on failure."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    if not polys:
        return Polygon()
    try:
        result = unary_union(polys)
        return make_valid(result)
    except Exception:
        return Polygon()


def compare_slices(
    cleaned: CleanedMesh,
    cad_vertices: np.ndarray,
    cad_faces: np.ndarray,
    extrude_dir: np.ndarray,
    n_slices: int = 7,
) -> dict:
    """Compare original mesh to CAD reconstruction via cross-section slicing.

    Args:
        cleaned: Original mesh from Stage 1.
        cad_vertices: Tessellated CAD solid vertices (V, 3).
        cad_faces: Tessellated CAD solid faces (F, 3).
        extrude_dir: Primary extrusion direction (3,).
        n_slices: Number of slices (default 7, clamped to 5-10).

    Returns:
        {
            "score": float 0-100 (average per-slice IoU),
            "per_slice": [{"height_pct": float, "iou": float, "orig_area": float,
                           "cad_area": float, "intersect_area": float}, ...],
            "figure_path": str (path to saved PNG),
        }
    """
    from shapely.geometry import MultiPolygon, Polygon

    logger.debug(
        "compare_slices called: orig_verts=%d, orig_faces=%d, cad_verts=%d, cad_faces=%d, n_slices=%d",
        len(cleaned.vertices), len(cleaned.faces), len(cad_vertices), len(cad_faces), n_slices,
    )

    n_slices = max(5, min(10, n_slices))
    extrude_dir = np.asarray(extrude_dir, dtype=np.float64)
    norm = np.linalg.norm(extrude_dir)
    if norm < 1e-12:
        extrude_dir = np.array([0.0, 0.0, 1.0])
    else:
        extrude_dir = extrude_dir / norm

    # Build trimesh objects
    try:
        orig_mesh = _make_trimesh(cleaned.vertices, cleaned.faces)
        cad_mesh = _make_trimesh(cad_vertices, cad_faces)
    except Exception:
        logger.exception(
            "Failed to build trimesh objects for comparison: "
            "orig_verts=%d, cad_verts=%d",
            len(cleaned.vertices), len(cad_vertices),
        )
        raise

    # Compute extent along the PRIMARY AXIS (not the signed extrude direction).
    # This avoids sign issues when the extrude dir is negative.
    primary_axis = int(np.argmax(np.abs(extrude_dir)))
    orig_proj = orig_mesh.vertices[:, primary_axis]
    cad_proj = cad_mesh.vertices[:, primary_axis]

    global_min = min(orig_proj.min(), cad_proj.min())
    global_max = max(orig_proj.max(), cad_proj.max())
    span = global_max - global_min

    if span < 1e-9:
        return {
            "score": 0.0,
            "per_slice": [],
            "figure_path": "",
        }

    # Slice at evenly spaced interior heights (avoid the very edges)
    margin = 0.05  # skip 5% on each end
    heights = np.linspace(
        global_min + margin * span,
        global_max - margin * span,
        n_slices,
    )

    per_slice = []
    orig_poly_slices = []
    cad_poly_slices = []

    # Determine the two non-extrude axes for consistent 2D projection
    axes_2d = [i for i in range(3) if i != primary_axis]
    ax_u, ax_v = axes_2d[0], axes_2d[1]

    for h in heights:
        # Slice plane: perpendicular to the primary axis at height h
        orig_center = orig_mesh.vertices.mean(axis=0)
        origin = orig_center.copy()
        origin[primary_axis] = h
        # Use axis-aligned normal (avoids sign issues)
        slice_normal = np.zeros(3)
        slice_normal[primary_axis] = 1.0
        orig_polys = _section_to_polygons(orig_mesh, origin, slice_normal, ax_u, ax_v)
        cad_polys = _section_to_polygons(cad_mesh, origin, slice_normal, ax_u, ax_v)

        orig_union = _union_polygons(orig_polys)
        cad_union = _union_polygons(cad_polys)

        orig_poly_slices.append(orig_union)
        cad_poly_slices.append(cad_union)

        orig_area = orig_union.area
        cad_area = cad_union.area

        # Compute IoU
        try:
            intersect = orig_union.intersection(cad_union)
            intersect_area = intersect.area
        except Exception:
            intersect_area = 0.0

        try:
            union_geom = orig_union.union(cad_union)
            union_area = union_geom.area
        except Exception:
            union_area = max(orig_area, cad_area)

        if union_area > 1e-12:
            iou = intersect_area / union_area
        elif orig_area < 1e-12 and cad_area < 1e-12:
            # Both empty at this height -- perfect match (skip from score)
            iou = 1.0
        else:
            iou = 0.0

        height_pct = (h - global_min) / span * 100.0

        per_slice.append({
            "height_pct": round(height_pct, 1),
            "iou": round(iou, 4),
            "orig_area": round(orig_area, 4),
            "cad_area": round(cad_area, 4),
            "intersect_area": round(intersect_area, 4),
        })

    # Overall score: average IoU * 100
    ious = [s["iou"] for s in per_slice]
    score = float(np.mean(ious) * 100) if ious else 0.0
    logger.debug("Slice comparison score: %.1f%% (%d slices)", score, len(per_slice))

    # Generate figure
    try:
        figure_path = _render_figure(
            per_slice, orig_poly_slices, cad_poly_slices, heights, score,
        )
    except Exception:
        logger.exception("Failed to render comparison figure")
        figure_path = ""

    return {
        "score": round(score, 1),
        "per_slice": per_slice,
        "figure_path": figure_path,
    }


def _render_figure(
    per_slice: list[dict],
    orig_polys: list,
    cad_polys: list,
    heights: np.ndarray,
    score: float,
) -> str:
    """Render a matplotlib figure showing all slice comparisons.

    Green = intersection (matching), Red = original only, Blue = CAD only.
    Returns path to saved PNG.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath
    from shapely.geometry import MultiPolygon, Polygon

    n = len(per_slice)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows + 0.8))
    fig.suptitle(
        f"Cross-Section Comparison  |  Score: {score:.1f}%",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )

    # Flatten axes for easy indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(n):
        ax = axes[i]
        info = per_slice[i]
        orig_g = orig_polys[i]
        cad_g = cad_polys[i]

        ax.set_aspect("equal")
        ax.set_title(
            f"h={info['height_pct']:.0f}%  IoU={info['iou']:.2f}",
            fontsize=10,
        )

        # Compute intersection and symmetric differences
        try:
            intersect = orig_g.intersection(cad_g)
        except Exception:
            intersect = Polygon()
        try:
            orig_only = orig_g.difference(cad_g)
        except Exception:
            orig_only = Polygon()
        try:
            cad_only = cad_g.difference(orig_g)
        except Exception:
            cad_only = Polygon()

        _plot_geometry(ax, intersect, color="#2ecc71", alpha=0.7)  # green
        _plot_geometry(ax, orig_only, color="#e74c3c", alpha=0.6)  # red
        _plot_geometry(ax, cad_only, color="#3498db", alpha=0.6)   # blue

        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Legend
    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.7, label="Match (intersection)"),
        Patch(facecolor="#e74c3c", alpha=0.6, label="Original only (missing)"),
        Patch(facecolor="#3498db", alpha=0.6, label="CAD only (extra)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    # Save to temp file
    out_dir = Path(tempfile.gettempdir()) / "paramesh_compare"
    out_dir.mkdir(exist_ok=True)

    import uuid
    filename = f"compare_{uuid.uuid4().hex[:8]}.png"
    path = out_dir / filename
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def _plot_geometry(ax, geom, color: str, alpha: float):
    """Plot a shapely geometry (Polygon or MultiPolygon) on a matplotlib axis."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath
    from shapely.geometry import MultiPolygon, Polygon

    if geom is None or geom.is_empty:
        return

    polygons = []
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        # GeometryCollection or other — extract polygons
        try:
            for g in geom.geoms:
                if isinstance(g, Polygon) and not g.is_empty:
                    polygons.append(g)
        except Exception:
            return

    for poly in polygons:
        if poly.is_empty or poly.area < 1e-15:
            continue

        # Build matplotlib path with exterior + holes
        codes = []
        verts = []

        def _add_ring(ring):
            coords = list(ring.coords)
            if len(coords) < 3:
                return
            verts.extend(coords)
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(coords) - 2))
            codes.append(MplPath.CLOSEPOLY)

        _add_ring(poly.exterior)
        for interior in poly.interiors:
            _add_ring(interior)

        if not verts:
            continue

        mpl_path = MplPath(verts, codes)
        patch = PathPatch(mpl_path, facecolor=color, edgecolor="none", alpha=alpha)
        ax.add_patch(patch)

    # Auto-scale
    ax.autoscale_view()
