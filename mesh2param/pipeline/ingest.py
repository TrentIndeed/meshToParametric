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

    # Scan detection: dense triangle soup (>> the decimation cap) is a
    # scan, not clean CAD. Clean test meshes are 1.6k-9.5k faces; scans
    # are 300k-2.5M. Watertightness does NOT separate them (the wrench
    # scan is watertight). With scan_auto on, this turns on the scan
    # front-end (denoise + canonical-frame alignment) automatically.
    raw_faces = len(mesh.faces)
    is_scan = bool(config.scan_auto and raw_faces > config.scan_face_threshold)
    do_denoise = config.denoise_scan or is_scan
    do_align = config.align_canonical or is_scan
    if is_scan:
        logger.info("Scan detected (%d raw faces > %d) — scan front-end on",
                    raw_faces, config.scan_face_threshold)

    try:
        mesh = _remove_degenerate_faces(mesh, config.min_face_area)
        mesh = _merge_vertices(mesh, config.merge_tolerance)
        mesh = _decimate(mesh, config.max_faces)
        mesh = _fix_normals(mesh)
        # Scan front-end: feature-preserving denoise flattens scanner
        # noise on the surfaces while keeping sharp edges, BEFORE
        # hole-fill / Poisson fit a cleaner surface.
        if do_denoise:
            mesh = _denoise_scan(mesh, config)
            mesh = _fix_normals(mesh)
        mesh = _fill_holes(mesh, config.max_hole_edges)
        # Ragged scans still have large open boundaries small-hole filling
        # can't close. Screened-Poisson reconstruction (scan input only)
        # closes them so the part has a real volume downstream.
        if config.poisson_reconstruct and not mesh.is_watertight:
            mesh = _poisson_reconstruct(mesh, config)
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

    # Scan front-end: rotate the part onto its own orthogonal frame so
    # downstream axis snapping / extrusion-engine silhouettes / prismatic
    # reasoning (all world-axis-relative) become meaningful.
    align_transform = None
    align_method = None
    if do_align:
        from mesh2param.pipeline.align import align_to_canonical_frame
        mesh, align_transform, align_method = align_to_canonical_frame(mesh)
        mesh = _fix_normals(mesh)

    mesh, unit_scale = _normalize_scale(mesh, config)

    result = _build_cleaned_mesh(mesh, path)
    if unit_scale != 1.0:
        result.metadata["unit_scale"] = unit_scale
    if align_transform is not None:
        result.metadata["align_transform"] = align_transform.tolist()
        result.metadata["align_method"] = align_method
    if is_scan:
        result.metadata["scan_mode"] = True
        result.metadata["raw_faces"] = raw_faces
        result.metadata["denoised"] = bool(do_denoise)
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
    mesh = _decimate(mesh, config.max_faces)
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


def _normalize_scale(
    mesh: trimesh.Trimesh, config: IngestConfig,
) -> tuple[trimesh.Trimesh, float]:
    """Scale wildly out-of-range parts to the canonical size.

    The pipeline's absolute thresholds assume mm-scale parts. A part
    whose bbox diagonal falls outside [normalize_min_diag,
    normalize_max_diag] (e.g. inch units read as raw numbers, or a 10x
    scan) is scaled so its diagonal equals canonical_diag. Returns the
    (possibly scaled) mesh and unit_scale, where
    mesh_units = original_units * unit_scale.

    Pieces cut from an already-ingested body must NOT be re-normalized
    (they're legitimately small), so this runs only in ingest_mesh, not
    clean_trimesh_piece.
    """
    if not config.normalize_scale:
        return mesh, 1.0
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if diag <= 0 or config.normalize_min_diag <= diag <= config.normalize_max_diag:
        return mesh, 1.0
    unit_scale = config.canonical_diag / diag
    mesh.apply_scale(unit_scale)
    logger.info(
        "Normalized part scale: bbox diagonal %.4g -> %.1f (unit_scale=%.6g); "
        "STEP export will scale back to original units",
        diag, config.canonical_diag, unit_scale,
    )
    return mesh, unit_scale


def _decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Quadric edge-collapse decimation down to max_faces.

    Scanned meshes arrive at 300K-2.5M triangles while carrying the same
    analytic surfaces as a 10K-face CAD export. Segmentation cost grows
    superlinearly with face count, so cap it here. planarquadric keeps
    flat areas well-tessellated; topology preservation keeps the mesh
    manifold for downstream watertightness checks.
    """
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh

    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices),
            face_matrix=np.asarray(mesh.faces),
        ))
        # Two-stage for very dense scans: quadric edge-collapse cost
        # scales with the INPUT face count, so a million-triangle scan
        # spends ~60s. A cheap vertex-clustering pre-pass knocks the
        # count down by ~10x in ~2s, then the quality quadric runs on
        # the much smaller mesh (~10s total instead of ~60s).
        if len(mesh.faces) > 5 * max_faces:
            ms.meshing_decimation_clustering(
                threshold=pymeshlab.PercentageValue(0.3)
            )
            logger.info(
                "Clustering pre-decimation: %d -> %d faces",
                len(mesh.faces), ms.current_mesh().face_number(),
            )
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(max_faces),
            preservenormal=True,
            preservetopology=True,
            planarquadric=True,
            qualitythr=0.3,
        )
        out = ms.current_mesh()
        decimated = trimesh.Trimesh(
            vertices=np.asarray(out.vertex_matrix()),
            faces=np.asarray(out.face_matrix()),
            process=True,
        )
        if len(decimated.faces) == 0:
            logger.warning("Decimation produced an empty mesh — using original")
            return mesh
        logger.info(
            "Decimated mesh: %d -> %d faces (target %d)",
            len(mesh.faces), len(decimated.faces), max_faces,
        )
        return decimated
    except ImportError:
        logger.warning(
            "pymeshlab not installed — cannot decimate %d-face mesh; "
            "downstream stages will be slow", len(mesh.faces),
        )
        return mesh
    except Exception as e:
        logger.warning("Decimation failed (%s) — using original mesh", e)
        return mesh


def _denoise_scan(mesh: trimesh.Trimesh, config: IngestConfig) -> trimesh.Trimesh:
    """Feature-preserving denoise for scans.

    Smooths scanner noise off the surfaces while KEEPING sharp edges —
    naive Laplacian smoothing would round every corner (the same reason
    Poisson is gated). Prefers two-step smoothing (smooth face normals
    over a neighbourhood, then refit vertices to them — feature
    preserving), falling back to surface-preserving Laplacian, then
    Taubin. Returns the input unchanged on any failure.
    """
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices),
            face_matrix=np.asarray(mesh.faces),
        ))
        it = int(max(1, config.denoise_iterations))
        applied = None
        for fname, kw in (
            ("apply_coord_two_steps_smoothing", {"stepsmoothnum": it}),
            ("apply_coord_laplacian_smoothing_surface_preserving", {}),
            ("apply_coord_taubin_smoothing", {"stepsmoothnum": it}),
        ):
            fn = getattr(ms, fname, None)
            if fn is None:
                continue
            try:
                fn(**kw)
                applied = fname
                break
            except Exception:
                try:
                    fn()
                    applied = fname
                    break
                except Exception:
                    continue
        if applied is None:
            logger.warning("No usable pymeshlab denoise filter found")
            return mesh
        out = ms.current_mesh()
        den = trimesh.Trimesh(
            vertices=np.asarray(out.vertex_matrix()),
            faces=np.asarray(out.face_matrix()),
            process=False,
        )
        if len(den.faces) == 0:
            return mesh
        logger.info("Feature-preserving denoise via %s (%d its): %d verts",
                    applied, it, len(den.vertices))
        return den
    except ImportError:
        logger.warning("pymeshlab not installed — skipping scan denoise")
        return mesh
    except Exception as e:
        logger.warning("Scan denoise failed (%s) — keeping original", e)
        return mesh


def _poisson_reconstruct(mesh: trimesh.Trimesh, config: IngestConfig) -> trimesh.Trimesh:
    """Screened-Poisson surface reconstruction to close a ragged scan.

    Runs only on non-watertight input (the caller gates this). Poisson
    fits a smooth closed surface to the mesh's points+normals, sealing
    large open boundaries that small-hole filling can't. The result is
    re-decimated back to the face cap. Returns the original mesh on any
    failure or if the result isn't actually watertight (don't trade a
    valid open mesh for a broken closed one).
    """
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices),
            face_matrix=np.asarray(mesh.faces),
        ))
        ms.generate_surface_reconstruction_screened_poisson(
            depth=int(config.poisson_depth), preclean=True,
        )
        if config.max_faces > 0:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=int(config.max_faces),
                preservenormal=True, planarquadric=True, qualitythr=0.3,
            )
        out = ms.current_mesh()
        recon = trimesh.Trimesh(
            vertices=np.asarray(out.vertex_matrix()),
            faces=np.asarray(out.face_matrix()),
            process=True,
        )
        if len(recon.faces) == 0 or not recon.is_watertight:
            logger.warning(
                "Poisson reconstruction did not yield a watertight mesh "
                "(%d faces) — keeping the original open mesh",
                len(recon.faces),
            )
            return mesh
        logger.info(
            "Poisson reconstruction: %d -> %d faces, now watertight",
            len(mesh.faces), len(recon.faces),
        )
        return recon
    except ImportError:
        logger.warning("pymeshlab not installed — cannot Poisson-reconstruct scan")
        return mesh
    except Exception as e:
        logger.warning("Poisson reconstruction failed (%s) — keeping original", e)
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
