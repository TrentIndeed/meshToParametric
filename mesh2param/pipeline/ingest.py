"""Stage 1: Mesh Ingestion & Cleanup.

Loads STL/OBJ/PLY meshes, repairs topology, fixes normals,
removes degenerate faces, and optionally remeshes for downstream quality.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from mesh2param.config import IngestConfig
from mesh2param.models.mesh import CleanedMesh

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".stl", ".obj", ".ply", ".3mf"}


def ingest_mesh(
    file_path: str | Path,
    config: IngestConfig | None = None,
) -> CleanedMesh:
    """Load a mesh file and return a cleaned, validated CleanedMesh.

    Args:
        file_path: Path to an STL, OBJ, or PLY mesh file.
        config: Ingestion configuration. Uses defaults if None.

    Returns:
        A CleanedMesh with consistent winding, no degenerate triangles,
        and (ideally) watertight manifold topology.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported or the file
            cannot be parsed as a triangle mesh.
    """
    if config is None:
        config = IngestConfig()

    path = Path(file_path)
    logger.debug("ingest_mesh called: file_path=%s, config=%s", path, config)

    _validate_file(path)

    try:
        mesh = _load_mesh(path)
    except Exception:
        logger.exception("Stage 1 failed during mesh loading: file=%s", path)
        raise

    try:
        mesh = _remove_degenerate_faces(mesh, config.min_face_area)
        mesh = _merge_vertices(mesh, config.merge_tolerance)
        mesh = _fix_normals(mesh)
        mesh = _fill_holes(mesh, config.max_hole_edges)
    except Exception:
        logger.exception(
            "Stage 1 failed during mesh cleanup: file=%s, verts=%d, faces=%d",
            path, len(mesh.vertices), len(mesh.faces),
        )
        raise

    if config.remesh:
        mesh = _remesh(mesh, config.target_edge_ratio)

    # Re-fix normals after remeshing since it can introduce inconsistencies
    mesh = _fix_normals(mesh)

    result = _build_cleaned_mesh(mesh, path)
    logger.debug(
        "ingest_mesh completed: verts=%d, faces=%d, watertight=%s",
        len(result.vertices), len(result.faces), result.is_watertight,
    )
    return result


def clean_trimesh_piece(mesh: trimesh.Trimesh, config: IngestConfig | None = None) -> CleanedMesh:
    """Clean a trimesh piece (e.g. from a planar cut) into a CleanedMesh.

    Runs the same cleanup pipeline as ingest_mesh but on an in-memory mesh.
    """
    if config is None:
        config = IngestConfig()

    mesh = _remove_degenerate_faces(mesh, config.min_face_area)
    mesh = _merge_vertices(mesh, config.merge_tolerance)
    mesh = _fix_normals(mesh)
    mesh = _fill_holes(mesh, config.max_hole_edges)
    if config.remesh:
        mesh = _remesh(mesh, config.target_edge_ratio)
    mesh = _fix_normals(mesh)

    return CleanedMesh(
        vertices=np.array(mesh.vertices, dtype=np.float64),
        faces=np.array(mesh.faces, dtype=np.int64),
        normals=np.array(mesh.face_normals, dtype=np.float64),
        is_watertight=mesh.is_watertight,
        bbox=np.array(mesh.bounds, dtype=np.float64),
    )


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def _load_mesh(path: Path) -> trimesh.Trimesh:
    logger.debug("Loading mesh file: %s (%.1f KB)", path, path.stat().st_size / 1024)
    loaded = trimesh.load(str(path))

    # 3MF and some formats return a Scene; extract the first mesh geometry
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("Scene contains no triangle meshes.")
        loaded = meshes[0]
        if len(meshes) > 1:
            logger.warning(
                "Scene contains %d meshes; using the first one ('%s')",
                len(meshes),
                next(iter(loaded.metadata.get("name", "unnamed")), "unnamed"),
            )

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            f"File did not produce a single triangle mesh. "
            f"Got {type(loaded).__name__} instead."
        )
    logger.info(
        "Loaded mesh: %d vertices, %d faces", len(loaded.vertices), len(loaded.faces)
    )
    return loaded


def _remove_degenerate_faces(mesh: trimesh.Trimesh, min_area: float) -> trimesh.Trimesh:
    areas = mesh.area_faces
    degenerate_mask = areas < min_area
    n_degenerate = int(degenerate_mask.sum())
    if n_degenerate > 0:
        valid_mask = ~degenerate_mask
        mesh.update_faces(valid_mask)
        mesh.remove_unreferenced_vertices()
        logger.info("Removed %d degenerate faces (area < %e)", n_degenerate, min_area)
    return mesh


def _merge_vertices(mesh: trimesh.Trimesh, tolerance: float) -> trimesh.Trimesh:
    pre_count = len(mesh.vertices)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    post_count = len(mesh.vertices)
    if pre_count != post_count:
        logger.info("Merged vertices: %d -> %d", pre_count, post_count)
    return mesh


def _fix_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception:
        logger.exception("Normal repair failed: verts=%d, faces=%d", len(mesh.vertices), len(mesh.faces))
        raise
    return mesh


def _fill_holes(mesh: trimesh.Trimesh, max_hole_edges: int) -> trimesh.Trimesh:
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            logger.exception(
                "Hole filling raised an exception: verts=%d, faces=%d",
                len(mesh.vertices), len(mesh.faces),
            )
            raise
        if mesh.is_watertight:
            logger.info("Filled holes — mesh is now watertight")
        else:
            logger.warning("Attempted hole filling but mesh is still not watertight")
    return mesh


def _subdivide_long_edges(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Split triangles whose longest edge exceeds a fraction of the
    bounding-box diagonal.

    New vertices land exactly on parent edges (no smoothing / no
    reprojection), so each child triangle is coplanar with its
    parent — child normals are identical to the parent's, which
    keeps downstream axis-class segmentation stable.
    """
    try:
        bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        max_edge = bbox_diag * 0.12
        new_verts, new_faces = trimesh.remesh.subdivide_to_size(
            mesh.vertices, mesh.faces, max_edge=max_edge,
        )
        # process=False: do not merge / reorder vertices or recompute
        # normals. Preserves the coplanarity guarantee.
        subdivided = trimesh.Trimesh(
            vertices=new_verts, faces=new_faces, process=False,
        )
        logger.info(
            "Subdivided long edges: %d → %d faces (max edge %.3f)",
            len(mesh.faces), len(subdivided.faces), max_edge,
        )
        return subdivided
    except Exception as e:
        logger.warning("Edge subdivision failed (%s) — using original", e)
        return mesh


def _remesh(mesh: trimesh.Trimesh, target_edge_ratio: float) -> trimesh.Trimesh:
    """Topology repair (pymeshfix) + isotropic explicit remesh (pymeshlab).

    The isotropic pass is the key step: without it, the source mesh's
    triangle density varies across geometrically-identical features, and
    downstream region-growing + primitive-fitting thresholds produce
    different regions for features that should be symmetric.
    """
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    target_edge_length = bbox_diag * target_edge_ratio

    # Pass 1: topology repair (fills holes, fixes non-manifold).
    try:
        import pymeshfix
        fixer = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        fixer.repair()
        # pymeshfix exposes output arrays as .points / .faces (older
        # versions used .v / .f).
        verts = getattr(fixer, "points", None)
        if verts is None:
            verts = getattr(fixer, "v")
        faces = getattr(fixer, "faces", None)
        if faces is None:
            faces = getattr(fixer, "f")
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    except ImportError:
        logger.warning("pymeshfix not installed — skipping topology repair")
    except Exception as e:
        logger.warning("pymeshfix repair failed (%s) — continuing", e)

    # Pass 2: isotropic explicit remesh to uniform edge length.
    # This is what makes symmetric features produce symmetric regions
    # downstream. Source STL/3MF triangle layouts are rarely symmetric
    # even when the underlying geometry is.
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices),
            face_matrix=np.asarray(mesh.faces),
        ))
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3,
            targetlen=pymeshlab.PureValue(float(target_edge_length)),
            featuredeg=30.0,
            smoothflag=False,          # don't move vertices tangentially
            reprojectflag=True,        # snap new vertices onto original surface
            checksurfdist=True,        # enforce max deviation from surface
            maxsurfdist=pymeshlab.PercentageValue(0.5),  # 0.5% of bbox diag
        )
        out = ms.current_mesh()
        remeshed = trimesh.Trimesh(
            vertices=np.asarray(out.vertex_matrix()),
            faces=np.asarray(out.face_matrix()),
            process=True,
        )
        logger.info(
            "Isotropic remesh: %d → %d vertices, %d → %d faces "
            "(target edge %.4f)",
            len(mesh.vertices), len(remeshed.vertices),
            len(mesh.faces), len(remeshed.faces),
            target_edge_length,
        )
        return remeshed
    except ImportError:
        logger.warning(
            "pymeshlab not installed — skipping isotropic remesh "
            "(symmetric features may segment asymmetrically)"
        )
        return mesh
    except Exception as e:
        logger.warning("Isotropic remesh failed (%s) — using repaired mesh", e)
        return mesh


def _build_cleaned_mesh(mesh: trimesh.Trimesh, source_path: Path) -> CleanedMesh:
    logger.debug(
        "Building CleanedMesh: verts=%d, faces=%d, watertight=%s",
        len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
    )
    return CleanedMesh(
        vertices=np.array(mesh.vertices, dtype=np.float64),
        faces=np.array(mesh.faces, dtype=np.int64),
        normals=np.array(mesh.face_normals, dtype=np.float64),
        is_watertight=mesh.is_watertight,
        bbox=np.array(mesh.bounds, dtype=np.float64),
        metadata={
            "source_file": str(source_path),
            "original_format": source_path.suffix.lower(),
        },
    )
