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
    on_stage=None,
) -> CleanedMesh:
    """Load a mesh file and return a cleaned, validated CleanedMesh.

    Args:
        file_path: Path to an STL, OBJ, or PLY mesh file.
        config: Ingestion configuration. Uses defaults if None.
        on_stage: optional callback(name, trimesh) fired after the
            cleanup front-end's distinct steps ("denoise", "align") with a
            copy of the intermediate mesh — lets the UI show each cleaning
            step as its own visual stage. No-op for clean CAD (the front-end
            never runs).

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
    # Scan OR just much denser than the decimation cap: a clean CAD part of a
    # simple bracket is a few thousand faces; a mesh several times the cap is a
    # scan/AI export (e.g. AI8 RackmountEar at 72k faces slips under the 100k
    # scan threshold AND is too smooth-dense for the wobble test, but it IS
    # dirty). Clean test meshes (<=9.5k) stay well under 2x the cap.
    dense = config.scan_auto and raw_faces > 2 * config.max_faces
    is_scan = bool(config.scan_auto
                   and (raw_faces > config.scan_face_threshold or dense))
    # AI-generated / dirty low-poly meshes (Meshy, Tripo) are NOT big enough
    # to trip the face-count test, but their near-flat faces are wobbly and
    # angles approximate — so they still need the cleanup front-end. Detect
    # them by SURFACE EXACTNESS (clean CAD ~0 deg wobble, dirty ~5 deg);
    # cheap on the low-poly meshes this targets (huge scans already tripped).
    is_dirty = False
    if config.scan_auto and not is_scan:
        wobble = _surface_wobble(mesh)
        is_dirty = wobble > config.dirty_wobble_deg
        if is_dirty:
            logger.info("Dirty mesh detected (flat-face wobble %.2f deg > "
                        "%.2f) — cleanup front-end on", wobble,
                        config.dirty_wobble_deg)
    needs_frontend = is_scan or is_dirty
    do_denoise = config.denoise_scan or needs_frontend
    do_align = config.align_canonical or needs_frontend
    if is_scan:
        logger.info("Scan detected (%d raw faces > %d) — scan front-end on",
                    raw_faces, config.scan_face_threshold)

    try:
        mesh = _remove_degenerate_faces(mesh, config.min_face_area)
        mesh = _merge_vertices(mesh, config.merge_tolerance)
        # Decimation cap depends on the path:
        #   * Clean CAD: drop to max_faces (it's usually already below).
        #   * Dirty/scan: only a COST-GUARD cap here (align_max_faces, ~120k) so
        #     denoise/repair AND the canonical-frame alignment run on a
        #     high-resolution mesh — alignment derives its rotation from
        #     area-weighted face normals, which coarse triangles degrade. After
        #     align the mesh is capped to max_faces (see the OUTPUT CAP below);
        #     segmentation density is convert._decimate_to_density's job, NOT a
        #     12k pre-pass here. align_max_faces=0 disables the guard.
        if needs_frontend:
            _cap = config.align_max_faces or config.max_faces
        else:
            _cap = config.max_faces
        mesh = _decimate(mesh, _cap)
        mesh = _fix_normals(mesh)
        # Scan front-end: feature-preserving denoise flattens scanner
        # noise on the surfaces while keeping sharp edges, BEFORE
        # hole-fill / Poisson fit a cleaner surface.
        if do_denoise:
            mesh = _denoise_scan(mesh, config)
            mesh = _fix_normals(mesh)
            if on_stage is not None:
                try:
                    on_stage("denoise", mesh.copy())
                except Exception:
                    logger.debug("on_stage(denoise) failed", exc_info=True)
        mesh = _fill_holes(mesh, config.max_hole_edges)
        # Ragged scans still have large open boundaries small-hole filling
        # can't close. Screened-Poisson reconstruction (scan input only)
        # closes them so the part has a real volume downstream.
        if config.poisson_reconstruct and not mesh.is_watertight:
            mesh = _poisson_reconstruct(mesh, config)
        # Last resort: a non-manifold "soup" Poisson can't close (AI meshes).
        if config.voxel_repair and not mesh.is_watertight:
            mesh = _voxel_remesh_repair(mesh, config)
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
        if on_stage is not None:
            try:
                on_stage("align", mesh.copy())
            except Exception:
                logger.debug("on_stage(align) failed", exc_info=True)

    # OUTPUT CAP (dirty/scan only): bound the high-res working mesh back to
    # max_faces for downstream memory/speed. NOT the aggressive dirty_max_faces
    # (12k) cap — verified (scripts/diag_align_resolution.py) that decimating
    # the clean denoised+repaired surface that hard REINTRODUCES normal wobble
    # (alignQ collapsed 0.79->0.38 on AI4) AND shatters segmentation: it hands
    # convert.py a pre-coarsened mesh that its own _decimate_to_density then
    # degrades further (production region counts 11/15 AI parts worse). The old
    # 12k cap ran BEFORE Poisson/voxel repair, which re-inflated to ~max_faces,
    # so it was effectively a no-op — segmentation always saw ~max_faces. Keep
    # that: cap here at max_faces and let convert._decimate_to_density own the
    # segmentation-density reduction (it decimates to a reference faces/cm^2
    # from a HIGH-res mesh, which segments far cleaner than a 12k pre-pass).
    if needs_frontend and config.max_faces and len(mesh.faces) > config.max_faces:
        mesh = _decimate(mesh, config.max_faces)
        mesh = _fix_normals(mesh)

    # Final watertight GUARANTEE (dirty/scan only): if the repairs above (and the
    # output-cap decimation, which can re-open a voxel-repaired mesh) left it
    # non-watertight, alpha-wrap to a manifold. Without this, non-watertight AI
    # meshes give plan_extrusions 0 segments and the build collapses to a solid
    # rectangle (AI10). Gated on the alpha-wrap plugin (Linux); _alpha_wrap
    # returns None on Windows, so this is a no-op there and clean CAD never
    # reaches it (needs_frontend is False). fast-simplification keeps the wrap
    # watertight through the re-cap.
    if needs_frontend and config.alpha_wrap_repair and not mesh.is_watertight:
        try:
            from mesh2param.pipeline.iou_refine import _alpha_wrap
            wrapped = _alpha_wrap(mesh)
        except Exception:
            wrapped = None
        if wrapped is not None and len(wrapped.faces) > 0 and wrapped.is_watertight:
            logger.info("Alpha-wrap repair: non-watertight -> watertight "
                        "manifold (%d -> %d faces)",
                        len(mesh.faces), len(wrapped.faces))
            mesh = wrapped
            if config.max_faces and len(mesh.faces) > config.max_faces:
                mesh = _decimate(mesh, config.max_faces)
            mesh = _fix_normals(mesh)

    mesh, unit_scale = _normalize_scale(mesh, config)

    result = _build_cleaned_mesh(mesh, path)
    if unit_scale != 1.0:
        result.metadata["unit_scale"] = unit_scale
    if align_transform is not None:
        result.metadata["align_transform"] = align_transform.tolist()
        result.metadata["align_method"] = align_method
    if needs_frontend:
        # scan_mode drives the "Scan cleanup" pipeline-viz stage + UI copy;
        # source distinguishes a true scan from an AI/dirty low-poly mesh.
        result.metadata["scan_mode"] = True
        result.metadata["cleanup_source"] = "scan" if is_scan else "dirty_mesh"
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


def _decimate_deterministic(mesh: trimesh.Trimesh,
                            max_faces: int) -> trimesh.Trimesh | None:
    """Deterministic quadric decimation via trimesh's fast-simplification
    backend, or None if unavailable/failed.

    pymeshlab's quadric decimation parallelises the edge-collapse priority
    queue, so its tie-breaking — and thus the OUTPUT VERTEX POSITIONS — vary
    run-to-run even on identical input (same topology, jittered coords). That
    jitter was the root of the whole AI/scan pipeline's non-determinism (it
    cascades into segmentation region counts and final cad_vol_err). The
    fast-simplification library is single-threaded C++ and reproducible, and
    faster (522k->30k in ~0.6s), so it is the primary path.
    """
    try:
        import fast_simplification  # noqa: F401  (trimesh's backend)
    except Exception:
        return None
    try:
        out = mesh.simplify_quadric_decimation(face_count=int(max_faces))
        if out is None or len(out.faces) == 0:
            return None
        logger.info("Decimated (deterministic): %d -> %d faces (target %d)",
                    len(mesh.faces), len(out.faces), max_faces)
        return out
    except Exception as e:
        logger.warning("Deterministic decimation failed (%s) — "
                       "falling back to pymeshlab", e)
        return None


def _decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Quadric edge-collapse decimation down to max_faces.

    Scanned meshes arrive at 300K-2.5M triangles while carrying the same
    analytic surfaces as a 10K-face CAD export. Segmentation cost grows
    superlinearly with face count, so cap it here.

    Primary path is the DETERMINISTIC fast-simplification backend (see
    _decimate_deterministic — reproducible run-to-run, which the pymeshlab path
    is not). Falls back to pymeshlab's two-stage quadric (non-deterministic but
    always available) when fast-simplification isn't installed.
    """
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh

    det = _decimate_deterministic(mesh, max_faces)
    if det is not None:
        return det

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


def _surface_wobble(mesh: trimesh.Trimesh) -> float:
    """Median dihedral (deg) among 'same-surface' adjacent face pairs — how
    far near-flat faces deviate from EXACTLY flat. Clean CAD is ~0 (flat
    faces tessellate into exactly coplanar triangles); AI-generated / scanned
    meshes sit a few degrees off. A face-count-INDEPENDENT 'needs cleaning'
    signal, so a dirty low-poly mesh is caught where the size test misses it.
    """
    try:
        adj = mesh.face_adjacency
        if len(adj) == 0:
            return 0.0
        n = mesh.face_normals
        d = np.clip((n[adj[:, 0]] * n[adj[:, 1]]).sum(1), -1.0, 1.0)
        ang = np.degrees(np.arccos(d))
        same = ang < 10.0   # candidate 'same flat surface' pairs
        return float(np.median(ang[same])) if same.any() else 0.0
    except Exception:
        return 0.0


def _bilateral_denoise(mesh: trimesh.Trimesh,
                       config: IngestConfig) -> trimesh.Trimesh:
    """Bilateral normal-filtering denoise (Zheng et al.).

    Smooths scanner noise off surfaces while preserving TRUE sharp edges
    and, crucially, WITHOUT faceting smooth curves — the pymeshlab two-step
    flattens curved scan surfaces into patches that meet at false creases
    (validated: it turned 0.6% -> 5.6% spurious >30 deg edges on the
    spanner; this keeps it at ~0.5%). Face normals are averaged with weights
    that fall off in both space (sigma_c, from mean adjacent-centroid
    spacing) and normal difference (sigma_s, the edge-preservation knob:
    small => normals across an edge barely mix, so the edge survives while
    same-surface noise averages out). Vertices are then refit to the
    filtered normals. Pure-numpy + vectorized; returns input on any failure.
    """
    try:
        V = np.asarray(mesh.vertices, dtype=np.float64).copy()
        F = np.asarray(mesh.faces)
        if len(F) < 4:
            return mesh
        adj = mesh.face_adjacency
        if len(adj) == 0:
            return mesh

        # (F,K) padded neighbour-face index + mask from 1-ring edge adjacency
        nlists: list[list[int]] = [[] for _ in range(len(F))]
        for a, b in adj:
            nlists[a].append(b)
            nlists[b].append(a)
        K = max((len(x) for x in nlists), default=1) or 1
        nb = np.zeros((len(F), K), dtype=np.int64)
        nbm = np.zeros((len(F), K), dtype=bool)
        for i, x in enumerate(nlists):
            if x:
                nb[i, :len(x)] = x
                nbm[i, :len(x)] = True

        # (V,Kv) padded incident-face index + count for the vertex refit
        vlists: list[list[int]] = [[] for _ in range(len(V))]
        for fi, tri in enumerate(F):
            for v in tri:
                vlists[v].append(fi)
        Kv = max((len(x) for x in vlists), default=1) or 1
        vf = np.zeros((len(V), Kv), dtype=np.int64)
        vfm = np.zeros((len(V), Kv), dtype=bool)
        for i, x in enumerate(vlists):
            if x:
                vf[i, :len(x)] = x
                vfm[i, :len(x)] = True
        vf_cnt = vfm.sum(1).clip(min=1)[:, None]

        cur = trimesh.Trimesh(V, F, process=False)
        cents = cur.triangles_center
        n = cur.face_normals.copy()
        areas = cur.area_faces
        sc = float(np.mean(np.linalg.norm(
            cents[adj[:, 0]] - cents[adj[:, 1]], axis=1))) or 1.0
        ss = float(max(0.05, config.denoise_sigma_s))

        for _ in range(int(max(1, config.denoise_normal_iter))):
            cg, ng, ag = cents[nb], n[nb], areas[nb]
            dc = np.linalg.norm(cg - cents[:, None, :], axis=2)
            dn = np.linalg.norm(ng - n[:, None, :], axis=2)
            w = (ag * np.exp(-(dc ** 2) / (2 * sc ** 2))
                 * np.exp(-(dn ** 2) / (2 * ss ** 2))) * nbm
            acc = (w[..., None] * ng).sum(1) + n
            ln = np.linalg.norm(acc, axis=1, keepdims=True)
            n = np.where(ln > 1e-12, acc / ln, n)

        for _ in range(int(max(1, config.denoise_vertex_iter))):
            cents = trimesh.Trimesh(V, F, process=False).triangles_center
            cf, nf = cents[vf], n[vf]
            proj = (nf * (cf - V[:, None, :])).sum(2)
            V = V + (nf * proj[..., None] * vfm[..., None]).sum(1) / vf_cnt

        den = trimesh.Trimesh(V, F, process=False)
        logger.info("Bilateral denoise (sigma_s=%.2f, %d/%d its): %d verts",
                    ss, config.denoise_normal_iter,
                    config.denoise_vertex_iter, len(den.vertices))
        return den
    except Exception as e:
        logger.warning("Bilateral denoise failed (%s) — trying pymeshlab", e)
        return mesh


def _denoise_scan(mesh: trimesh.Trimesh, config: IngestConfig) -> trimesh.Trimesh:
    """Feature-preserving denoise for scans.

    Default path is the bilateral normal-filtering denoise (edge-preserving,
    no faceting). Set denoise_method="twostep" for the pymeshlab two-step
    (more aggressive surface smoothing, but facets smooth curves). Either
    way, naive Laplacian is avoided (it rounds every corner). Returns the
    input unchanged on any failure.
    """
    if (config.denoise_method or "bilateral").lower() == "bilateral":
        out = _bilateral_denoise(mesh, config)
        if out is not mesh and len(out.faces) > 0:
            return out
        # fall through to pymeshlab if the bilateral bailed out
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


def _voxel_remesh_repair(mesh: trimesh.Trimesh,
                         config: IngestConfig) -> trimesh.Trimesh:
    """Last-resort watertighting for a BADLY-BROKEN mesh — non-manifold "soup"
    (overlapping/self-intersecting surfaces, thousands of components) that
    Poisson can't close. Common in AI-generated meshes (e.g. AI13: 0 open
    edges but 23k non-manifold edges, ~18k components). Voxelizes the mesh and
    extracts a marching-cubes isosurface = a GUARANTEED watertight manifold, at
    the cost of sharp detail (the denoise/align/regularize front-end recovers
    flats/angles). Coarser resolutions are MORE robustly closed (thin features
    stay connected), so it tries finer first and falls back coarser. Returns
    the repaired mesh, or the original if nothing closes it. Needs
    scikit-image (trimesh's marching_cubes backend)."""
    try:
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) or 1.0
    except Exception:
        return mesh
    for res in (96, 72, 56, 44):
        try:
            rm = mesh.voxelized(diag / res).fill().marching_cubes
            rm.fix_normals()
            if rm.is_watertight and len(rm.faces) > 0:
                logger.info("Voxel-remesh repair: non-manifold soup -> "
                            "watertight manifold (res=%d, %d faces)",
                            res, len(rm.faces))
                return rm
        except ImportError:
            logger.warning("scikit-image not installed — cannot voxel-remesh")
            return mesh
        except Exception:
            continue
    logger.warning("Voxel-remesh repair could not close the mesh")
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
