"""Single source of truth for the mesh->feature-tree pipeline.

The web server (viz_server/job_queue.py) and standalone scripts/tests MUST run
the SAME computation, or a standalone reproduction won't match what a user sees
in the app. This module holds that shared computation so the two can't drift:
both call `recognize_body` for a single body, and `convert_mesh` for the whole
end-to-end (ingest -> decompose -> per-body recognize -> assemble tree).

The server wraps these with its own visualization + pause/step orchestration,
but the GEOMETRY/FEATURE results come from here, identically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass


from mesh2param.config import (
    DecomposeConfig, FitConfig, IngestConfig, RecognizeConfig, SegmentConfig,
    TopologyConfig,
)
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.tree import FeatureTree

logger = logging.getLogger(__name__)


@dataclass
class BodyResult:
    tree: FeatureTree
    regions: list
    fitted: list
    graph: object
    edges: list
    blends: list
    # The mesh the REGIONS live on (for segmentation viz). For dirty/AI parts
    # this is a DECIMATED copy whose flats segment cleanly; the tree's build
    # geometry comes from the full mesh. For clean CAD it is the full mesh.
    seg_mesh: CleanedMesh | None = None


def recognize_body(sub_cleaned: CleanedMesh, scan_mode: bool) -> BodyResult:
    """Single-body recognition core, shared by server and scripts.

    segment -> fit -> (if dirty: regularize -> re-segment -> re-fit) ->
    topology -> blend identification -> recognize_features. This is EXACTLY
    what viz_server/job_queue._process_single_body runs (minus the viz), so a
    standalone reproduction calling this matches the app's feature tree.

    DECOUPLING (dirty/AI only): the BUILD (component planner, base/join
    profiles, holes) uses the FULL-density mesh for accuracy, while
    SEGMENTATION + feature recognition run on a DECIMATED copy so noisy flats
    stay ONE region instead of shattering into hundreds (which spawned bogus
    chamfers/slots). The two share a coordinate frame, so the coordinate-based
    features line up with the full-density build.
    """
    from mesh2param.pipeline.fit import fit_surfaces
    from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
    from mesh2param.pipeline.recognize import recognize_features
    from mesh2param.pipeline.segment import segment_mesh
    from mesh2param.pipeline.topology import build_topology

    seg = SegmentConfig(noise_robust=scan_mode)

    # DECIMATE FIRST (scan/AI only). The expensive analysis — segment, fit,
    # regularize, topology — all runs on a mesh at the reference triangle
    # density, so it's FAST and the noisy flats stay ONE region instead of
    # shattering into hundreds. The full-density mesh is kept ONLY as
    # build_mesh, for the component planner's cross-section build. Previously
    # we segmented the full mesh first and threw that work away — slow, and it
    # left the viz showing the fragmented full-mesh segmentation.
    seg_mesh = sub_cleaned
    build_mesh = None
    if scan_mode:
        decimated = _decimate_to_density(sub_cleaned)
        if decimated is not None and len(decimated.faces) < len(sub_cleaned.faces):
            seg_mesh = decimated
            build_mesh = sub_cleaned

    regions = segment_mesh(seg_mesh, seg)
    fitted = fit_surfaces(regions, seg_mesh, FitConfig())
    if scan_mode:
        try:
            from mesh2param.pipeline.regularize import regularize_mesh
            regularize_mesh(seg_mesh, regions, fitted, snap_angle_deg=12.0)
            regions = segment_mesh(seg_mesh, seg)
            fitted = fit_surfaces(regions, seg_mesh, FitConfig())
        except Exception:
            logger.exception("Regularization failed — using raw fit")

    graph, edges = build_topology(regions, fitted, seg_mesh, TopologyConfig())
    try:
        blends = identify_blends_from_topology(regions, fitted, edges, seg_mesh)
    except Exception:
        logger.exception("Blend identification failed")
        blends = []
    # On dirty AI/scan parts the build is robust but the region-based detail
    # finders hallucinate fillets/chamfers/slots from surface noise — gate
    # them conservatively (build stays locked, weak detail dropped).
    rec_cfg = RecognizeConfig(conservative_detail=scan_mode)
    tree = recognize_features(
        regions, fitted, graph, edges, seg_mesh,
        rec_cfg, topo_blends=blends, build_mesh=build_mesh,
    )
    return BodyResult(tree, regions, fitted, graph, edges, blends, seg_mesh)


def _decimate_to_density(mesh: CleanedMesh, faces_per_cm2: float = 120.0,
                         min_faces: int = 4_000) -> CleanedMesh | None:
    """A decimated copy at the reference triangle density (faces per cm^2),
    scaled with surface area. Used ONLY for segmentation; the build keeps the
    full mesh. Returns None on failure."""
    try:
        import trimesh
        import numpy as np
        from mesh2param.pipeline.ingest import _decimate
        from mesh2param.models.mesh import CleanedMesh as _CM
        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                             process=False)
        area_cm2 = float(tm.area) / 100.0
        target = int(np.clip(faces_per_cm2 * area_cm2, min_faces,
                             len(mesh.faces)))
        if target >= len(mesh.faces):
            return None
        dm = _decimate(tm, target)
        dm.fix_normals()
        verts = np.asarray(dm.vertices, dtype=float)
        faces = np.asarray(dm.faces)
        return _CM(
            vertices=verts,
            faces=faces,
            normals=np.asarray(dm.face_normals, dtype=float),
            is_watertight=bool(dm.is_watertight),
            bbox=np.array([verts.min(0), verts.max(0)]),
            metadata=dict(mesh.metadata),
        )
    except Exception:
        logger.exception("Segmentation decimation failed")
        return None


def _ingest_and_decompose(mesh_path: str, ingest_config: IngestConfig | None):
    """ingest(scan_auto) -> connected-component decompose decision. Shared by
    convert_mesh and analyze_part so the ingest (expensive: alpha-wrap) and the
    decompose decision run exactly once and identically."""
    import trimesh

    from mesh2param.pipeline.decompose import decompose_bodies
    from mesh2param.pipeline.ingest import ingest_mesh

    if ingest_config is None:
        ingest_config = IngestConfig(remesh=False, scan_auto=True)
    cleaned = ingest_mesh(mesh_path, ingest_config)
    scan_mode = bool(cleaned.metadata.get("scan_mode"))

    conn = trimesh.Trimesh(
        vertices=cleaned.vertices, faces=cleaned.faces, process=False)
    try:
        n_components = len(conn.split(only_watertight=False))
    except Exception:
        n_components = 1
    decomposition = decompose_bodies(
        cleaned, DecomposeConfig(enabled=n_components > 1))
    return cleaned, scan_mode, decomposition


def _recognize_bodies(cleaned: CleanedMesh, scan_mode: bool, decomposition):
    """Per-body recognize_body over the decomposition. Returns the list of
    (body, full_density_sub, BodyResult) tuples."""
    from mesh2param.pipeline.decompose import extract_sub_mesh

    body_results: list[tuple] = []
    for body in decomposition.bodies:
        sub = extract_sub_mesh(cleaned, body.face_indices, body=body)
        br = recognize_body(sub, scan_mode)
        body_results.append((body, sub, br))
    return body_results


def convert_mesh(
    mesh_path: str, ingest_config: IngestConfig | None = None,
):
    """End-to-end headless conversion matching the server's computation.

    ingest(scan_auto) -> connected-component decompose decision -> per-body
    recognize_body -> assemble the feature tree (single body => that body's
    tree; multi-body => combined). Returns (cleaned, tree, body_results).
    The server's _run_pipeline does the same; standalone scripts/tests call
    this so they match. For a build (incl. stitched fillets) use analyze_part.
    """
    cleaned, scan_mode, decomposition = _ingest_and_decompose(
        mesh_path, ingest_config)
    body_results = _recognize_bodies(cleaned, scan_mode, decomposition)

    if not body_results:
        return cleaned, None, []
    # Single body => its tree IS the part tree (mirrors job_queue line ~2287).
    tree = body_results[0][2].tree
    return cleaned, tree, body_results


@dataclass
class AnalysisResult:
    """Everything export_cadquery needs to build the part, computed IDENTICALLY
    to the server's _run_pipeline. The cloud (modal_app) and scripts call
    analyze_part, then export_cadquery with these, so cloud == local geometry."""
    cleaned: CleanedMesh
    tree: FeatureTree | None
    body_results: list      # [(body, full_density_sub, body_tree)] for export
    stitched_clusters: list  # cross-body fillet/chamfer clusters for the apply pass
    multi_body: bool
    n_regions: int = 0
    decomposition: object = None  # the BodyDecomposition (for the editor bundle)


def analyze_part(mesh_path: str, ingest_config: IngestConfig | None = None) -> AnalysisResult:
    """Full analysis INCLUDING cross-body stitched fillets — the exact inputs the
    server hands its CAD subprocess (tree, cleaned, body_results, decomposition,
    stitched clusters). ingest -> decompose -> per-body recognize_body -> stitch.
    The cloud build calls this then export_cadquery(body_results,
    stitched_clusters) so its geometry matches what a local job produces, and
    pickles (tree, cleaned, body_results, decomposition) as the editor bundle."""
    cleaned, scan_mode, decomposition = _ingest_and_decompose(
        mesh_path, ingest_config)
    body_results = _recognize_bodies(cleaned, scan_mode, decomposition)
    if not body_results:
        return AnalysisResult(cleaned, None, [], [], False, 0, decomposition)
    tree = body_results[0][2].tree
    # Shape the per-body stage tuples the stitcher expects. Regions/blends live
    # on the seg_mesh (recognize_body's possibly-decimated analysis mesh).
    body_stage_data = [
        (body, br.seg_mesh, br.regions, br.fitted, br.graph, br.edges,
         br.tree, br.blends)
        for (body, _sub, br) in body_results
    ]
    try:
        clusters = stitch_fillet_clusters(body_stage_data).clusters
    except Exception:
        logger.exception("Stitched fillets failed — building without them")
        clusters = []
    # Export wants the FULL-density sub mesh per body, with that body's tree.
    export_body_results = [(body, sub, br.tree) for (body, sub, br) in body_results]
    n_regions = sum(len(br.regions) for (_b, _s, br) in body_results)
    return AnalysisResult(
        cleaned=cleaned,
        tree=tree,
        body_results=export_body_results,
        stitched_clusters=clusters,
        multi_body=len(body_results) > 1,
        n_regions=n_regions,
        decomposition=decomposition,
    )


@dataclass
class StitchedResult:
    """Cross-body stitched-fillet clusters + the data the server viz needs.

    `clusters` is the payload handed to export_cadquery (the CAD apply pass);
    `face_colors`/`uid_to_group` + the counts drive the server's stitched-stage
    visualization. Cloud callers use `clusters` and ignore the rest."""
    clusters: list
    uid_to_group: dict
    face_colors: list
    n_records: int
    n_groups: int
    n_multi: int
    n_singletons: int
    n_direct_unions: int
    n_axial_unions: int
    n_chamfers: int


def stitch_fillet_clusters(body_stage_data) -> StitchedResult:
    """Stitch per-body topology blend records into cross-body fillet/chamfer
    clusters for the CAD apply pass. SINGLE SOURCE OF TRUTH — the server
    (viz_server/job_queue) and the cloud (modal_app) both call this so their
    fillets can't drift. Pure geometry: consumes the per-body stage tuples
    `(body, seg_mesh, regions, fitted, graph, edges, tree, topo_blends)` and
    returns the cluster payload + viz data. Mutates each body's topo_blends
    in place (stamps stitch_uid, splits plate-wrap records)."""
    import numpy as np

    from scipy.spatial import cKDTree

    # Body decomposition shares vertex positions across
    # bodies, so the true cross-body seams sit at 0 mm
    # (vertices identical) or ≤ 1 mm (one mesh-edge apart).
    # Anything beyond ~2 mm is cross-plate-thickness, which
    # the user explicitly does NOT want merged. So a single
    # tight threshold serves both same-body touching pieces
    # and true cross-body seams.
    DIRECT_EPS = 2.0
    AXIS_DOT_MIN = 0.5

    # Pass 1: gather per-record info: face vertices + axis +
    # endpoints + radius. Records without a usable axis fall
    # back to the direct-contact rule only.
    from mesh2param.models.region import SurfaceType as _ST_pn
    records_info: list[dict] = []
    chamfer_records: list[dict] = []
    n_chamfers = 0
    for body_idx, _bst in enumerate(body_stage_data):
        _body, _sub, _regs, _fits, _g, _e, _bt, topo_blends = _bst
        # Per-body region/fit lookups so we can resolve a
        # fillet record's `plane_region_ids` into actual
        # outward face normals — the apply pass will use
        # those to reject candidate edges whose adjacent
        # face normals don't match the original fillet's
        # adjacent plate normals.
        regs_lookup = {r.region_id: r for r in _regs}
        fits_lookup = {f.region_id: f for f in _fits}

        def _plane_normal_for(rid: int) -> np.ndarray | None:
            f = fits_lookup.get(rid)
            if f is None:
                return None
            if f.surface_type == _ST_pn.PLANE:
                n = np.asarray(
                    f.params.get("normal", [0, 0, 0]), dtype=float,
                )
            else:
                reg = regs_lookup.get(rid)
                if reg is None or len(reg.face_indices) == 0:
                    return None
                n = _sub.normals[reg.face_indices].mean(axis=0)
            nrm = float(np.linalg.norm(n))
            if nrm < 0.5:
                return None
            return n / nrm

        _pending_blend_halves = []
        for rec in topo_blends:
            kind = rec.get("kind")
            if kind == "chamfer":
                n_chamfers += 1
                # Chamfers ride the SAME cluster->match->apply path
                # as fillets (apply uses OCC .chamfer instead of
                # .fillet). Collect the band's vertex cloud + a
                # setback estimate; build chamfer clusters after the
                # fillet payload below.
                fi_c = rec.get("face_indices")
                if fi_c is not None and len(fi_c) > 0:
                    cverts = _sub.vertices[
                        _sub.faces[np.asarray(fi_c, dtype=np.int64)]
                    ].reshape(-1, 3)
                    chamfer_records.append({
                        "verts": cverts,
                        "radius": float(rec.get("radius", 0.0) or 0.0),
                    })
                continue
            if kind != "fillet":
                continue
            fi = rec.get("face_indices")
            if fi is None or len(fi) == 0:
                continue
            fi_arr = np.asarray(fi, dtype=np.int64)
            verts = _sub.vertices[_sub.faces[fi_arr]].reshape(-1, 3)

            # PLATE-WRAP SPLIT: a band that wraps BOTH faces of a
            # thin plate (RackMount's side-plate corner/bottom
            # bands) physically touches each face's outline
            # fillet, so contact-union chained the left-face and
            # right-face outline fillets into ONE unappliable
            # group. Such a band's vertex cloud is bimodal across
            # the thickness with a void WIDER than the 2mm
            # contact radius — split it there and each half
            # joins its own face's group.
            halves = [(rec, fi_arr, verts)]
            _dims = verts.max(axis=0) - verts.min(axis=0)
            # Search EVERY plate-thickness-like axis for the
            # widest interior void — the record's own thinnest
            # bbox dim is the WRONG axis for the short-side
            # bands (5mm tall in Y but spanning the plate's X
            # thickness; testing Y found nothing and the bands
            # kept bridging the two faces).
            _thin, _best_gap = -1, 0.0
            _vt = _gaps = None
            _gi = 0
            _frac = 0.0
            # Only the single-body path needs record splitting:
            # whole-mesh topology emits face-spanning records.
            # The body-decompose path's per-piece records are
            # already face-separated, and splitting them creates
            # unmatched noise halves.
            if len(body_stage_data) == 1 and len(verts) >= 24:
                for _ax in range(3):
                    if float(_dims[_ax]) > 12.0:
                        continue
                    _vt_a = np.sort(verts[:, _ax])
                    _gaps_a = np.diff(_vt_a)
                    _gi_a = int(np.argmax(_gaps_a))
                    if float(_gaps_a[_gi_a]) > _best_gap:
                        _best_gap = float(_gaps_a[_gi_a])
                        _thin = _ax
                        _vt, _gaps, _gi = _vt_a, _gaps_a, _gi_a
                        _frac = (_gi_a + 1) / len(_vt_a)
            if _thin >= 0:
                # Two split triggers:
                # (a) a VOID in the distribution — balanced for
                #     moderate voids, unconditional for wide
                #     (>=2.5mm) ones (3 stray vertices on the
                #     opposite face bled color across the plate);
                # (b) a FUSED WRAP band with no void: continuous
                #     bridge vertices crossing the thickness. The
                #     user's rule: a fillet must not continue
                #     TANGENTIALLY out of its face plane — so
                #     excise the descending connector strip (a
                #     1.2mm dead zone each side of the midplane,
                #     orphaned unpainted) and keep the two face
                #     rims, now separated by >2.4mm so the
                #     contact union can never re-chain them
                #     through a neighboring band's corner.
                _rec_r = float(rec.get("radius", 0.0) or 0.0)
                _void_split = float(_gaps[_gi]) >= 1.5 and (
                    0.2 < _frac < 0.8 or float(_gaps[_gi]) >= 2.5
                )
                _fused_split = (
                    not _void_split
                    and _rec_r > 0
                    and float(_dims[_thin])
                    >= max(5.0, 1.6 * 1.5708 * _rec_r)
                )
                if _void_split or _fused_split:
                    _cent = _sub.vertices[
                        _sub.faces[fi_arr]
                    ].mean(axis=1)[:, _thin]
                    if _void_split:
                        # Split at the void's BOUNDARIES, not its
                        # midpoint: the void itself is the dead
                        # zone keeping the halves out of contact
                        # range of each other and of neighboring
                        # bands' opposite halves at the corners.
                        _lo_b = float(_vt[_gi]) + 1e-6
                        _hi_b = float(_vt[_gi + 1]) - 1e-6
                        _fA = fi_arr[_cent <= _lo_b]
                        _fB = fi_arr[_cent >= _hi_b]
                    else:
                        _mid = 0.5 * (
                            float(_vt[0]) + float(_vt[-1])
                        )
                        _fA = fi_arr[_cent < _mid - 1.2]
                        _fB = fi_arr[_cent > _mid + 1.2]
                    # A TINY side (a few stray faces) is not a
                    # fillet feature: keep only the substantial
                    # half and orphan the strays (unpainted gray
                    # beats a bleeding color or a noise group).
                    if len(_fA) >= 3 and len(_fB) < 3:
                        rec["face_indices"] = _fA.tolist()
                        _fB = _fB[:0]
                    elif len(_fB) >= 3 and len(_fA) < 3:
                        rec["face_indices"] = _fB.tolist()
                        _fA = _fA[:0]
                    if len(_fA) >= 3 and len(_fB) >= 3:
                        rec_b = dict(rec)
                        rec_b["face_indices"] = _fB.tolist()
                        rec["face_indices"] = _fA.tolist()
                        # Tag the halves so contact-union can't
                        # re-merge what we deliberately split (a
                        # 1.9mm void is inside the 2mm contact
                        # radius — the left plate's bore rims
                        # re-merged and bled across both faces).
                        _split_tag = id(rec)
                        rec["_split_parent"] = _split_tag
                        rec["_split_side"] = 0
                        rec_b["_split_parent"] = _split_tag
                        rec_b["_split_side"] = 1
                        _pending_blend_halves.append(rec_b)
                        _vA = _sub.vertices[
                            _sub.faces[_fA]
                        ].reshape(-1, 3)
                        _vB = _sub.vertices[
                            _sub.faces[_fB]
                        ].reshape(-1, 3)
                        halves = [
                            (rec, _fA, _vA), (rec_b, _fB, _vB),
                        ]
                        logger.info(
                            "Split plate-wrap blend record into "
                            "face-side halves (%d/%d faces, gap "
                            "%.1fmm)",
                            len(_fA), len(_fB), float(_gaps[_gi]),
                        )

            for rec_h, fi_h, verts_h in halves:
                ad_raw = rec_h.get("axis_dir")
                ap_raw = rec_h.get("axis_point")
                if ad_raw is not None:
                    axis_dir = np.asarray(ad_raw, dtype=float)
                    an = float(np.linalg.norm(axis_dir))
                else:
                    axis_dir = None
                    an = 0.0
                if axis_dir is not None and an >= 1e-9:
                    axis_dir = axis_dir / an
                    axis_point = (
                        np.asarray(ap_raw, dtype=float)
                        if ap_raw is not None
                        else verts_h.mean(axis=0)
                    )
                    t = (verts_h - axis_point) @ axis_dir
                    t_min, t_max = float(t.min()), float(t.max())
                    endpoints = np.array([
                        axis_point + t_min * axis_dir,
                        axis_point + t_max * axis_dir,
                    ])
                else:
                    axis_dir = None
                    endpoints = None
                # Collect the adjacent plate normals for this
                # record (one per plane_region_id).  Empty list
                # is fine — the matcher just won't have a plane
                # gate for that record's cluster.
                plane_normals_rec: list[list[float]] = []
                for prid in (rec_h.get("plane_region_ids") or []):
                    n = _plane_normal_for(int(prid))
                    if n is not None:
                        plane_normals_rec.append(n.tolist())
                records_info.append({
                    "body_idx": body_idx,
                    "rec": rec_h,
                    "verts": verts_h,
                    "axis_dir": axis_dir,
                    "endpoints": endpoints,
                    "radius": float(rec_h.get("radius", 0.0) or 0.0),
                    "plane_normals": plane_normals_rec,
                })

        if _pending_blend_halves:
            topo_blends.extend(_pending_blend_halves)

    # Per-record geometry classification. Four categories:
    #   hole_edge   — closed ring around an axis, tight axial span
    #                 (~radius). The rolled edge that opens a
    #                 through-hole. Pair-of-rings on opposite faces
    #                 of a thin plate share the same axis but
    #                 represent TWO distinct fillet features and
    #                 must NOT be axially merged.
    #   looped      — closed loop around an axis but with large
    #                 perp extent (perimeter wrap around a body).
    #   straight_edge — open arc, narrow angular span, long axial
    #                 extent vs radius (rolled edge along a single
    #                 straight CAD edge).
    #   curve_edge  — short axial vs radius, or no usable axis fit
    #                 (corner curls, sphere-fit residuals, arcs).
    for ri in records_info:
        axis_dir = ri["axis_dir"]
        endpoints = ri["endpoints"]
        radius = ri["radius"]
        verts = ri["verts"]
        if axis_dir is None or endpoints is None or radius <= 0:
            ri["category"] = "curve_edge"
            ri["is_closed_loop"] = False
            ri["axial_extent"] = 0.0
            continue
        axis_point = 0.5 * (endpoints[0] + endpoints[1])
        rel = verts - axis_point
        along = rel @ axis_dir
        axial_extent = float(along.max() - along.min())
        perp = rel - np.outer(along, axis_dir)
        perp_d = np.linalg.norm(perp, axis=1)
        # Project to plane perp to axis to measure angular wrap.
        arb = (
            np.array([1.0, 0.0, 0.0])
            if abs(float(axis_dir[0])) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = arb - axis_dir * float(arb @ axis_dir)
        un = float(np.linalg.norm(u))
        if un < 1e-9:
            ri["category"] = "curve_edge"
            ri["is_closed_loop"] = False
            ri["axial_extent"] = axial_extent
            continue
        u = u / un
        v = np.cross(axis_dir, u)
        pu = perp @ u
        pv = perp @ v
        angles = np.arctan2(pv, pu)
        # Closed-loop test: sort angles, compute the largest gap.
        # A wrap with no gap > 90° is treated as a full ring.
        if len(angles) >= 4:
            angles_sorted = np.sort(angles)
            diffs = np.diff(angles_sorted)
            wrap = (2.0 * np.pi) - (
                float(angles_sorted[-1]) - float(angles_sorted[0])
            )
            max_gap = float(max(diffs.max(), wrap))
        else:
            max_gap = 2.0 * np.pi
        is_closed_loop = max_gap < (np.pi / 2.0)
        ri["is_closed_loop"] = is_closed_loop
        ri["axial_extent"] = axial_extent
        if is_closed_loop:
            # Distinguish hole_edge (tight ring around a hole axis)
            # from looped (perimeter wrap around the body). A hole
            # edge has axial extent ≈ 2r (just the rolled surface);
            # a perimeter loop spans the body length, >> r.
            if axial_extent < 5.0 * radius:
                ri["category"] = "hole_edge"
            else:
                ri["category"] = "looped"
        else:
            if axial_extent > 3.0 * radius:
                ri["category"] = "straight_edge"
            else:
                ri["category"] = "curve_edge"

    for uid, ri in enumerate(records_info):
        ri["uid"] = uid
        ri["rec"]["stitch_uid"] = uid

    n_records = len(records_info)

    # Per-record diagnostic so we can see exactly what the
    # union-find sees: does each topology fillet record have
    # a usable axis? does the classifier see it as a closed
    # ring or an open arc? This is what determines whether
    # the two through-hole-face fillets get fused or kept
    # apart, so logging it makes the union-find behaviour
    # legible from the subprocess log.
    for ri in records_info:
        verts = ri["verts"]
        bmin = verts.min(axis=0)
        bmax = verts.max(axis=0)
        dims = bmax - bmin
        axis_dir = ri["axis_dir"]
        ax_str = (
            f"({axis_dir[0]:.2f},{axis_dir[1]:.2f},{axis_dir[2]:.2f})"
            if axis_dir is not None else "NONE"
        )
        logger.info(
            "  record uid=%d body=%d nverts=%d r=%.2f "
            "axis=%s axial_extent=%.1f closed_loop=%s "
            "category=%s bbox dims=(%.1f,%.1f,%.1f)",
            ri["uid"], ri["body_idx"], len(verts),
            ri["radius"], ax_str,
            ri.get("axial_extent", 0.0),
            bool(ri.get("is_closed_loop", False)),
            ri.get("category", "?"),
            dims[0], dims[1], dims[2],
        )

    parent: dict[int, int] = {ri["uid"]: ri["uid"] for ri in records_info}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # Pass 2a: direct-contact via a single global KDTree over
    # every fillet vertex tagged with its record uid. The
    # DIRECT_EPS contact threshold itself is the cross-body
    # stitching gate — two records that share a vertex
    # within 2 mm fuse, two that don't, stay separate. For
    # opposite-face through-hole rings (same-body, axially
    # offset by plate thickness), the proto-cluster bimodal
    # split below splits them apart geometrically, without
    # any axial-gap threshold.
    n_direct_unions = 0
    if records_info:
        all_verts_arr = np.vstack([ri["verts"] for ri in records_info])
        all_uids_arr = np.concatenate([
            np.full(len(ri["verts"]), ri["uid"], dtype=np.int64)
            for ri in records_info
        ])
        tree = cKDTree(all_verts_arr)
        _uid_to_split = {
            ri["uid"]: (
                ri["rec"].get("_split_parent"),
                ri["rec"].get("_split_side"),
            )
            for ri in records_info
        }
        for i, j in tree.query_pairs(r=DIRECT_EPS):
            ua = int(all_uids_arr[i])
            ub = int(all_uids_arr[j])
            if ua == ub or _find(ua) == _find(ub):
                continue
            # Never re-merge the two sides of a deliberately
            # split plate-wrap record.
            pa, sa = _uid_to_split.get(ua, (None, None))
            pb, sb = _uid_to_split.get(ub, (None, None))
            if pa is not None and pa == pb and sa != sb:
                continue
            _union(ua, ub)
            n_direct_unions += 1

    # Proto-cluster closed-loop check. The per-record
    # classifier above looks at a SINGLE topology record
    # and asks "do its vertices wrap a full 360° around its
    # axis?" — but in practice topology often splits a
    # rolled hole edge into multiple arc records that each
    # cover < 180°. After the direct-contact pass merges
    # those arcs into one proto-cluster, the cluster
    # together wraps the full loop and we can detect it.
    # Stamping `is_closed_loop` on every member record means
    # the axial-continuation guard below will fire even when
    # individual records were classified as open arcs.
    def _proto_cluster_axis_and_loop(verts: np.ndarray) -> tuple[np.ndarray | None, bool]:
        if len(verts) < 8:
            return None, False
        centered = verts - verts.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except Exception:
            return None, False
        # For a ring, the axis is the SHORTEST principal axis
        # (the tube's central axis — vertices spread broadly
        # in the perpendicular plane and narrowly along the
        # axis).
        axis = vh[2]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        rel = centered
        along = rel @ axis
        perp = rel - np.outer(along, axis)
        arb = (
            np.array([1.0, 0.0, 0.0])
            if abs(float(axis[0])) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = arb - axis * float(arb @ axis)
        un = float(np.linalg.norm(u))
        if un < 1e-9:
            return axis, False
        u = u / un
        v_ = np.cross(axis, u)
        pu = perp @ u
        pv = perp @ v_
        angles = np.arctan2(pv, pu)
        if len(angles) < 8:
            return axis, False
        angles_sorted = np.sort(angles)
        diffs = np.diff(angles_sorted)
        wrap = (2.0 * np.pi) - (
            float(angles_sorted[-1]) - float(angles_sorted[0])
        )
        max_gap = float(max(diffs.max(), wrap))
        return axis, (max_gap < (np.pi / 2.0))

    # Group records by current direct-contact root, then
    # classify each proto-cluster as closed-loop or not.
    # When a proto-cluster IS a closed loop AND its records
    # split into two axial layers (the two faces of a thin
    # plate hosting a through-hole), assign each record a
    # `bimodal_side` of 0 or 1 so the final group labelling
    # treats the two halves as separate clusters even
    # though direct-contact glued their records together.
    proto_members: dict[int, list[dict]] = {}
    for ri in records_info:
        proto_members.setdefault(_find(ri["uid"]), []).append(ri)
        ri["bimodal_side"] = 0
    # Map root -> aggregated verts of the WHOLE proto
    # cluster (pre-split). Each member record carries the
    # root reference so the cluster builder downstream can
    # compute fillet length from the full ring even after
    # a through-hole proto cluster gets split into two
    # halves — both halves should report the same hole
    # circumference, not just their partial slice.
    proto_full_verts_by_root: dict[int, np.ndarray] = {}
    for root, members_ in proto_members.items():
        proto_verts = np.vstack([m["verts"] for m in members_])
        proto_full_verts_by_root[root] = proto_verts
        proto_axis, proto_is_loop = _proto_cluster_axis_and_loop(
            proto_verts
        )
        if proto_is_loop and proto_axis is not None:
            for m in members_:
                m["is_closed_loop"] = True
                if m.get("category") == "curve_edge":
                    m["category"] = "hole_edge"
        # Opposite-plate-face split. The rule the user
        # asked for: fillets on opposite sides of a plate
        # must NEVER share a cluster, no matter how close
        # the plate faces are. Concretely, if this proto
        # cluster's combined vertex cloud has a small
        # smallest-dim (≤ 30 mm — anything plate-thickness-
        # like) AND its records' centroids sit on both
        # sides of the median plane along that smallest
        # dim, then the cluster spans two faces and
        # splits unconditionally.
        #
        # No axial-gap threshold, no closed-loop test on
        # each half, no radii-similarity check. The
        # geometric fact "smallest bbox dim spans a plate"
        # combined with "records lie on both sides of the
        # midplane" is enough — the merge only happened
        # because direct-contact glued them through a
        # bridging record (e.g. a hole-tube), and that's
        # exactly the case the user wants undone.
        if len(members_) < 2:
            continue
        # A plate is by definition one body. Cross-body
        # proto clusters live in the same group only
        # because direct-contact stitched them at a body-
        # decomp seam — that's a different problem from
        # opposite-side-of-a-plate and shouldn't get split
        # by this rule.
        if len({m["body_idx"] for m in members_}) > 1:
            continue
        proto_min = proto_verts.min(axis=0)
        proto_max = proto_verts.max(axis=0)
        proto_dims = proto_max - proto_min
        thin_axis = int(np.argmin(proto_dims))
        if float(proto_dims[thin_axis]) > 30.0:
            continue
        rec_thin = np.array([
            float(m["verts"].mean(axis=0)[thin_axis])
            for m in members_
        ])
        midplane = 0.5 * (
            float(proto_min[thin_axis])
            + float(proto_max[thin_axis])
        )
        if proto_is_loop:
            split_at = midplane
        else:
            # NON-loop proto clusters split too, but only on a
            # clear BIMODAL gap in the vertex distribution along
            # the thin axis: RackMount's +X plate rim creases
            # (inner face, outer face, bridged by the base-
            # junction band) merged into one 13-record group
            # because the rim ring's angular coverage gap was
            # 91 deg — one degree past the loop test. Two
            # parallel bands separated by plate thickness show a
            # >=1.5mm void in the histogram; an L-band that
            # legitimately spans the thin dim does not.
            vt = np.sort(proto_verts[:, thin_axis])
            gaps = np.diff(vt)
            gi = int(np.argmax(gaps))
            frac_low = (gi + 1) / len(vt)
            if gaps[gi] < 1.5 or not (0.15 < frac_low < 0.85):
                continue
            split_at = 0.5 * (float(vt[gi]) + float(vt[gi + 1]))
        side_a = {
            i for i, v in enumerate(rec_thin) if v < split_at
        }
        side_b = set(range(len(members_))) - side_a
        if not side_a or not side_b:
            continue
        for i, m in enumerate(members_):
            m["bimodal_side"] = 0 if i in side_a else 1

    # Pass 2b: collinear-continuation. O(N²) but N is small.
    n_axial_unions = 0
    for i_idx in range(len(records_info)):
        A = records_info[i_idx]
        if A["axis_dir"] is None or A["endpoints"] is None:
            continue
        for j_idx in range(i_idx + 1, len(records_info)):
            B = records_info[j_idx]
            if B["axis_dir"] is None or B["endpoints"] is None:
                continue
            if _find(A["uid"]) == _find(B["uid"]):
                continue
            # Closed-loop guard: a ring fillet (one face of a
            # through-hole) and another ring on the OPPOSITE
            # face of the same plate share the hole's axis but
            # are TWO distinct fillet features. The axial gap
            # between them equals the plate thickness, which
            # the eps_axial heuristic below would happily
            # accept. Block ring-to-ring axial merges outright;
            # they should stay as separate clusters even though
            # they're collinear. The flag is stamped per-record
            # AFTER direct-contact stitching (see above), so it
            # covers cases where the topology stage gave us
            # multiple arcs instead of one ring.
            if A.get("is_closed_loop") and B.get("is_closed_loop"):
                continue
            align = abs(float(A["axis_dir"] @ B["axis_dir"]))
            if align < AXIS_DOT_MIN:
                continue
            r_min = min(A["radius"], B["radius"])
            # Keep eps_perp tight — opposite-face parallel
            # fillets on a thin plate sit ~plate-thickness
            # apart and must NOT merge here. We tolerate
            # only small axis-fit slop (~r).
            eps_perp = max(1.5, 1.5 * r_min)
            eps_axial = max(4.0, 3.0 * r_min)
            # Continuation means the VERTEX CLOUDS line up along
            # the shared axis. The endpoint test below can't see
            # this on its own: both bands' endpoints are computed
            # from each record's FITTED axis line, and when
            # topology fits both short-side bands of a plate to
            # the same line between the two faces, the endpoint
            # perp distance reads ~0 while the actual clouds sit
            # a plate thickness apart (RackMount +X plate: the
            # inner and outer outline rings fused into one group
            # this way, leaving one face's ring unmatched).
            # Gate on the cloud-centroid offset perpendicular to
            # BOTH axes — parallel offset bands fail, genuine
            # along-axis continuations pass.
            d_c = B["verts"].mean(axis=0) - A["verts"].mean(axis=0)
            perp_ok = True
            for ax_chk in (A["axis_dir"], B["axis_dir"]):
                perp_c = d_c - float(d_c @ ax_chk) * ax_chk
                if float(np.linalg.norm(perp_c)) >= eps_perp:
                    perp_ok = False
                    break
            if not perp_ok:
                continue
            stitched = False
            for EA in A["endpoints"]:
                for EB in B["endpoints"]:
                    d = EB - EA
                    axial = float(d @ A["axis_dir"])
                    perp_vec = d - axial * A["axis_dir"]
                    perp_d = float(np.linalg.norm(perp_vec))
                    if perp_d < eps_perp and abs(axial) < eps_axial:
                        _union(A["uid"], B["uid"])
                        n_axial_unions += 1
                        stitched = True
                        break
                if stitched:
                    break

    # Pass 3: connected-component labelling. Records share
    # a group ID iff they share a (root, bimodal_side) key.
    # bimodal_side splits a closed-loop proto-cluster whose
    # records sit on opposite plate faces (through-hole on
    # both sides) into two separate clusters even though
    # the direct-contact pass merged them into one root.
    subkey_to_group: dict[tuple, int] = {}
    uid_to_group: dict[int, int] = {}
    for ri in records_info:
        root = _find(ri["uid"])
        sk = (root, int(ri.get("bimodal_side", 0)))
        if sk not in subkey_to_group:
            subkey_to_group[sk] = len(subkey_to_group)
        uid_to_group[ri["uid"]] = subkey_to_group[sk]

    n_groups = len(subkey_to_group)
    group_counts: dict[int, int] = {}
    for gi in uid_to_group.values():
        group_counts[gi] = group_counts.get(gi, 0) + 1
    n_multi = sum(1 for c in group_counts.values() if c > 1)
    n_singletons = n_groups - n_multi

    # Pass 4: paint faces per group.
    stitch_palette = np.array([
        [1.00, 0.20, 0.20], [0.20, 1.00, 0.40], [0.30, 0.55, 1.00],
        [1.00, 0.70, 0.20], [0.90, 0.20, 0.95], [0.20, 0.95, 0.95],
        [1.00, 0.45, 0.80], [0.60, 0.95, 0.20], [0.50, 0.30, 1.00],
        [0.10, 0.50, 0.10], [0.30, 0.85, 0.65], [0.95, 0.95, 0.95],
        [0.65, 0.20, 0.55], [0.25, 0.65, 0.40], [0.95, 0.30, 0.55],
        [0.40, 0.40, 0.95], [0.85, 0.60, 0.95], [0.20, 0.75, 0.95],
    ])
    BASE_GRAY = np.array([0.55, 0.55, 0.55])
    CHAMFER_COLOR = np.array([1.00, 0.55, 0.00])

    stitched_face_colors: list[tuple] = []
    for body_idx, _bst in enumerate(body_stage_data):
        body, sub_cleaned, _regs, _fits, _g, _e, _bt, topo_blends = _bst
        fc = np.tile(BASE_GRAY, (len(sub_cleaned.faces), 1))
        for rec in topo_blends:
            kind = rec.get("kind")
            idx = rec.get("face_indices")
            if idx is None or len(idx) == 0:
                continue
            if kind == "chamfer":
                fc[idx] = CHAMFER_COLOR
                continue
            if kind != "fillet":
                continue
            uid = int(rec.get("stitch_uid", -1))
            gi = uid_to_group.get(uid)
            if gi is not None:
                fc[idx] = stitch_palette[gi % len(stitch_palette)]
        stitched_face_colors.append((body, sub_cleaned, fc))

    # Build the cluster payload for the CAD apply pass. Each
    # stitched group is one physical fillet feature with a
    # vertex cloud + a single representative radius. The
    # apply pass uses the vertex BBOX directly: every
    # concave edge whose midpoint falls inside the bbox
    # (with a few mm of margin) gets filleted at the
    # group's radius. No axes, no per-record corners, no
    # cylinder/sphere distinction — the matcher just asks
    # "what edges are in the general area this fillet
    # lived in", which is what `.fillet()` in any CAD app
    # actually does.
    stitched_clusters_payload: list[dict] = []
    for gi, members in sorted(
        {gi_: [u for u, g in uid_to_group.items() if g == gi_]
         for gi_ in set(uid_to_group.values())}.items()
    ):
        # Per-record radius: median perp distance from the
        # record's vertices to its fitted axis line. Skip
        # records without a clean axis (sphere/cone fits
        # whose "axis" is a plane normal); they still
        # contribute vertices to the group's bbox, just no
        # radius vote.
        per_record_radii: list[float] = []
        topology_radii: list[float] = []
        verts_per_record: list[np.ndarray] = []
        all_plane_normals: list[np.ndarray] = []
        cat_votes: dict[str, int] = {}
        any_closed_loop = False
        for u in members:
            ri = next(r for r in records_info if r["uid"] == u)
            verts_per_record.append(ri["verts"])
            cat = ri.get("category", "curve_edge")
            cat_votes[cat] = cat_votes.get(cat, 0) + 1
            if ri.get("is_closed_loop"):
                any_closed_loop = True
            for pn in ri.get("plane_normals", []):
                all_plane_normals.append(np.asarray(pn, dtype=float))
            # Topology-assigned fillet radius — the small
            # TUBE radius. The hole-edge classifier uses
            # this. Computing perp distance from each
            # record's own fitted axis (below) is unsafe
            # because topology sometimes fits the wrong
            # axis on toroidal surfaces and the resulting
            # perp distance becomes the MAJOR radius
            # (e.g. 18 mm on a 3 mm fillet), pushing real
            # hole-edge clusters out of the filter.
            if ri.get("radius", 0) > 0:
                topology_radii.append(float(ri["radius"]))
            if ri["axis_dir"] is None or ri["endpoints"] is None:
                continue
            axis_dir = ri["axis_dir"]
            axis_point = 0.5 * (ri["endpoints"][0] + ri["endpoints"][1])
            rel = ri["verts"] - axis_point
            along = rel @ axis_dir
            perp = rel - np.outer(along, axis_dir)
            perp_d = np.linalg.norm(perp, axis=1)
            if len(perp_d) >= 3:
                per_record_radii.append(float(np.median(perp_d)))
        verts_for_group = np.vstack(verts_per_record)
        # Group radius: prefer the topology-assigned tube
        # radius (median). Falls back to perp-distance
        # median if topology had no radii, then to bbox.
        if topology_radii:
            group_radius = float(np.median(topology_radii))
        elif per_record_radii:
            group_radius = float(np.median(per_record_radii))
        else:
            bbox_dims = verts_for_group.max(axis=0) - verts_for_group.min(axis=0)
            group_radius = float(np.min(bbox_dims)) * 0.5
        # Deduplicate plane normals: two normals within ~10°
        # are considered the same plate orientation.
        dedup_normals: list[list[float]] = []
        for n in all_plane_normals:
            is_dup = False
            for kept in dedup_normals:
                if abs(float(np.dot(n, np.asarray(kept)))) > 0.985:
                    is_dup = True
                    break
            if not is_dup:
                dedup_normals.append(n.tolist())
        # Cluster category by ANGULAR COVERAGE of the
        # cluster's aggregated vertex cloud — measured
        # around whichever principal axis maximises wrap.
        # This reflects what the cluster IS after split,
        # not what its pre-split records were labelled.
        #
        #   coverage > 270°       → hole_edge  (closed loop;
        #                            tight when the loop's
        #                            in-plane extent is
        #                            small, otherwise
        #                            "looped" perimeter wrap)
        #   135° < coverage ≤ 270° → curve_edge  (rolled
        #                            edge that bends ≈ 180°
        #                            from start to end —
        #                            U-bend / half-loop)
        #   coverage ≤ 135°       → straight_edge
        #
        # Hole-edge vs looped distinction: a hole-edge's
        # in-plane diameter (max perpendicular extent) is
        # bounded by typical hole radii on this part class
        # (≤ ~40 mm); long perimeter wraps run to hundreds
        # of mm.
        def _coverage_and_inplane(verts: np.ndarray) -> tuple[float, float]:
            if len(verts) < 4:
                return 0.0, 0.0
            centered = verts - verts.mean(axis=0)
            try:
                _, _, vh = np.linalg.svd(
                    centered, full_matrices=False,
                )
            except Exception:
                return 0.0, 0.0
            best_cov = 0.0
            best_inplane = 0.0
            for axis_candidate in (vh[2], vh[0]):
                axis = axis_candidate / (
                    np.linalg.norm(axis_candidate) + 1e-12
                )
                along = centered @ axis
                perp = centered - np.outer(along, axis)
                arb = (
                    np.array([1.0, 0.0, 0.0])
                    if abs(float(axis[0])) < 0.9
                    else np.array([0.0, 1.0, 0.0])
                )
                u = arb - axis * float(arb @ axis)
                un = float(np.linalg.norm(u))
                if un < 1e-9:
                    continue
                u = u / un
                v_ = np.cross(axis, u)
                angles = np.arctan2(perp @ v_, perp @ u)
                if len(angles) < 4:
                    continue
                s = np.sort(angles)
                diffs = np.diff(s)
                wrap = (2.0 * np.pi) - (
                    float(s[-1]) - float(s[0])
                )
                max_gap = float(max(diffs.max(), wrap))
                coverage = (2.0 * np.pi) - max_gap
                if coverage > best_cov:
                    best_cov = coverage
                    pu = perp @ u
                    pv = perp @ v_
                    extent = max(
                        float(pu.max() - pu.min()),
                        float(pv.max() - pv.min()),
                    )
                    best_inplane = extent
            return best_cov, best_inplane

        # For coverage / in-plane / length computation, use
        # the FULL proto-cluster vertices (pre-split), not
        # just this half's subset. Both halves of a
        # through-hole share the same fillet circumference;
        # using only the split half would report two
        # different lengths for the same physical fillet.
        first_uid = members[0]
        root_for_cluster = _find(first_uid)
        full_verts = proto_full_verts_by_root.get(
            root_for_cluster, verts_for_group,
        )
        cov_rad, inplane = _coverage_and_inplane(full_verts)
        cov_deg = cov_rad * 180.0 / float(np.pi)
        # Three categories only:
        #   hole_edge — tight ring around a real hole.
        #               Requires closed-loop coverage AND
        #               small fillet radius AND in-plane
        #               diameter consistent with hole-edge
        #               geometry. The max-r-among-records
        #               check filters out tab-perimeter
        #               wraps whose topology fit gave one
        #               record a large "path-cylinder"
        #               radius alongside small actual
        #               fillet radii.
        #   curve_edge — anything that bends but isn't a
        #               tight hole-edge ring. Covers both
        #               half-loop / 180° U-bends and long
        #               perimeter wraps.
        #   straight_edge — almost no angular wrap.
        # hole_edge = a NEAR-COMPLETE ring (coverage > 320°)
        # of hole-scale diameter (in-plane extent ≤ 40 mm).
        # We do NOT gate on the radius field: for a rolled
        # hole edge, topology measures the HOLE radius
        # (~15 mm on the rackmount), not the small fillet
        # tube radius, so a radius cap wrongly rejects the
        # real hole rings. Coverage cleanly separates them:
        # the four through-hole rings wrap ~353°, while
        # perimeter wraps and tab-corner rounds top out at
        # ~287°. The in-plane cap keeps full-body perimeter
        # loops (in-plane ≫ 100 mm) out of the hole bucket.
        if cov_deg > 320.0 and inplane <= 40.0:
            group_category = "hole_edge"
        elif cov_deg > 135.0:
            group_category = "curve_edge"
        else:
            group_category = "straight_edge"
        # Fillet path length:
        #   bent path → arc length = coverage_rad * (inplane/2)
        #   straight  → principal-axis extent
        if cov_deg > 90.0 and inplane > 0:
            group_length = float(cov_rad * inplane * 0.5)
        else:
            centered = full_verts - full_verts.mean(axis=0)
            try:
                _, sv, _ = np.linalg.svd(centered, full_matrices=False)
                group_length = float(2.0 * sv[0] / np.sqrt(len(full_verts)))
            except Exception:
                group_length = 0.0
        logger.info(
            "  cluster gi=%d: coverage=%.0f°, inplane=%.1fmm, "
            "r=%.2f, length=%.1fmm -> %s",
            gi, cov_deg, inplane, group_radius,
            group_length, group_category,
        )
        stitched_clusters_payload.append({
            "group_id": gi,
            "vertices": verts_for_group.astype(np.float64),
            "n_records": len(members),
            "radius": group_radius,
            "plane_normals": dedup_normals,
            "category": group_category,
            "is_closed_loop": any_closed_loop,
            "length": group_length,
            "coverage_deg": cov_deg,
            "inplane_mm": inplane,
        })
    # --- Chamfer clusters: one per chamfer band, kind="chamfer" ---
    # Merge chamfer bands whose vertex clouds touch (a chamfer that
    # got split into several records) so each physical chamfer is
    # one cluster, then hand them to the same apply path as fillets.
    if chamfer_records:
        from scipy.spatial import cKDTree as _cKDTree_ch
        n_ch = len(chamfer_records)
        ch_parent = list(range(n_ch))

        def _ch_find(x):
            while ch_parent[x] != x:
                ch_parent[x] = ch_parent[ch_parent[x]]
                x = ch_parent[x]
            return x

        allv = np.vstack([r["verts"] for r in chamfer_records])
        allu = np.concatenate([
            np.full(len(r["verts"]), i) for i, r in enumerate(chamfer_records)
        ])
        for i, j in _cKDTree_ch(allv).query_pairs(r=2.0):
            a, b = int(allu[i]), int(allu[j])
            ra, rb = _ch_find(a), _ch_find(b)
            if ra != rb:
                ch_parent[rb] = ra
        ch_groups: dict[int, list[int]] = {}
        for i in range(n_ch):
            ch_groups.setdefault(_ch_find(i), []).append(i)

        gid0 = len(stitched_clusters_payload)
        for off, members_ch in enumerate(ch_groups.values()):
            cv = np.vstack([chamfer_records[i]["verts"] for i in members_ch])
            radii = [chamfer_records[i]["radius"] for i in members_ch
                     if chamfer_records[i]["radius"] > 0]
            # Setback estimate: a flat chamfer band of width W spans
            # W = setback*sqrt(2) for a 45 deg cut; the record radius
            # already tracks the band half-extent, so use it directly,
            # floored at 1mm.
            setback = max(1.0, float(np.median(radii)) if radii else 1.0)
            stitched_clusters_payload.append({
                "group_id": gid0 + off,
                "vertices": cv.astype(np.float64),
                "n_records": len(members_ch),
                "radius": setback,
                "category": "curve_edge",
                "kind": "chamfer",
            })
        logger.info("Stitched %d chamfer record(s) into %d chamfer cluster(s)",
                    n_ch, len(ch_groups))

    return StitchedResult(
        clusters=stitched_clusters_payload,
        uid_to_group=uid_to_group,
        face_colors=stitched_face_colors,
        n_records=n_records,
        n_groups=n_groups,
        n_multi=n_multi,
        n_singletons=n_singletons,
        n_direct_unions=n_direct_unions,
        n_axial_unions=n_axial_unions,
        n_chamfers=n_chamfers,
    )