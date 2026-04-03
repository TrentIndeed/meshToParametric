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
) -> list:
    """Slice a mesh at a plane and return a list of shapely polygons.

    Returns an empty list if the section is empty or fails.
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    try:
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    except Exception:
        return []

    if section is None:
        return []

    try:
        # to_planar projects the 3D section onto a 2D coordinate system
        path2d, to_3d = section.to_planar()
    except Exception:
        return []

    if path2d is None or len(path2d.entities) == 0:
        return []

    polys = []
    try:
        # path2d.polygons_full gives closed polygons (with holes handled)
        for poly in path2d.polygons_full:
            p = make_valid(poly)
            if p.is_empty:
                continue
            if isinstance(p, MultiPolygon):
                polys.extend(p.geoms)
            elif isinstance(p, Polygon) and p.area > 0:
                polys.append(p)
    except Exception:
        # Fallback: try polygons_closed (simpler, no hole detection)
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

    n_slices = max(5, min(10, n_slices))
    extrude_dir = np.asarray(extrude_dir, dtype=np.float64)
    norm = np.linalg.norm(extrude_dir)
    if norm < 1e-12:
        extrude_dir = np.array([0.0, 0.0, 1.0])
    else:
        extrude_dir = extrude_dir / norm

    # Build trimesh objects
    orig_mesh = _make_trimesh(cleaned.vertices, cleaned.faces)
    cad_mesh = _make_trimesh(cad_vertices, cad_faces)

    # Compute extent along extrude direction for both meshes
    orig_proj = orig_mesh.vertices @ extrude_dir
    cad_proj = cad_mesh.vertices @ extrude_dir

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

    for h in heights:
        origin = extrude_dir * h
        orig_polys = _section_to_polygons(orig_mesh, origin, extrude_dir)
        cad_polys = _section_to_polygons(cad_mesh, origin, extrude_dir)

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

    # Generate figure
    figure_path = _render_figure(
        per_slice, orig_poly_slices, cad_poly_slices, heights, score,
    )

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
