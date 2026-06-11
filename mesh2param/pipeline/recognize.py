"""Stage 5: Feature Recognition.

Identifies manufacturing features (base extrusion, holes, pockets, bosses,
slots, fillets, chamfers) from the fitted surfaces, topology graph, and
edge classifications produced by the earlier pipeline stages.

The recognition strategy for prismatic parts:
1. Identify the base extrusion (largest planar region group).
2. Detect holes (full cylinders penetrating the body).
3. Detect pockets / slots (groups of inward-facing planes).
4. Detect bosses (groups of outward-facing planes above the base).
5. Detect fillets and chamfers (already hinted by Stage 4 edge types).
6. Build a dependency DAG and topologically sort.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx
import numpy as np

from mesh2param.config import RecognizeConfig
from mesh2param.models.feature import Feature, FeatureType
from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.region import FaceRegion, SurfaceType
from mesh2param.models.surface import FittedSurface, SurfaceEdge, TransitionType
from mesh2param.models.tree import FeatureTree

logger = logging.getLogger(__name__)


def recognize_features(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    edges: list[SurfaceEdge],
    cleaned: CleanedMesh,
    config: RecognizeConfig | None = None,
    topo_blends: list[dict] | None = None,
) -> FeatureTree:
    """Recognize manufacturing features from surface data and topology.

    Args:
        regions: Output of Stage 2 (Surface Segmentation).
        fitted: Output of Stage 3 (Primitive Fitting).
        graph: Surface adjacency graph from Stage 4.
        edges: Classified surface edges from Stage 4.
        cleaned: Output of Stage 1 (for mesh geometry).
        config: Recognition configuration. Uses defaults if None.
        topo_blends: Optional Stage 4.5 blend records. Any blend whose
            region isn't already absorbed into a feature here gets a
            FILLET (or CHAMFER) feature emitted from its record's
            radius/axis/length parameters. Without this, blend
            regions that fail Stage 5's stricter strip-cylinder
            check (Strategy 3) silently get absorbed into the base
            extrusion, hiding them from the feature tree.

    Returns:
        FeatureTree with ordered features and dependency graph.
    """
    if config is None:
        config = RecognizeConfig()

    logger.debug(
        "recognize_features called: %d regions, %d fitted, %d graph nodes, %d edges",
        len(regions), len(fitted), graph.number_of_nodes(), len(edges),
    )


    region_lookup = {r.region_id: r for r in regions}
    fitted_lookup = {f.region_id: f for f in fitted}

    # Track which regions have been assigned to features
    assigned: set[int] = set()
    features: list[Feature] = []

    # --- 1. Base extrusion: the largest group of connected planar faces ---
    try:
        base_feature, base_rids = _find_base_extrusion(
            regions, fitted, graph, cleaned,
        )
    except Exception:
        logger.exception("Stage 5 failed finding base extrusion: regions=%d", len(regions))
        raise
    features.append(base_feature)
    assigned.update(base_rids)
    base_idx = 0
    logger.debug("Base extrusion found: %d region IDs, depth=%.2f", len(base_rids), base_feature.depth)

    # --- 2. Holes: full cylinders ---
    try:
        hole_features = _find_holes(
            regions, fitted, graph, region_lookup, cleaned, config, assigned,
        )
    except Exception:
        logger.exception("Stage 5 failed finding holes")
        raise
    # Group symmetric holes (same radius, same axis, same entry plane)
    # into a single multi-position feature so the construction sequence
    # emits ONE "Sketch + Cut" step per group instead of one per hole.
    # Catches the common case of N mounting holes drilled through the
    # same face — they share a sketch in CAD intent, not N separate
    # operations.
    hole_features = _group_symmetric_holes(hole_features)
    for hf, rids in hole_features:
        hf.dependencies = [base_idx]
        features.append(hf)
        assigned.update(rids)
    logger.debug("Found %d holes (after symmetry grouping)", len(hole_features))

    # --- 3. Pockets: inward-facing plane groups connected together ---
    try:
        pocket_features = _find_pockets(
            regions, fitted, graph, region_lookup, fitted_lookup,
            cleaned, config, assigned, base_rids,
        )
    except Exception:
        logger.exception("Stage 5 failed finding pockets")
        raise
    for pf, rids in pocket_features:
        pf.dependencies = [base_idx]
        features.append(pf)
        assigned.update(rids)
    logger.debug("Found %d pockets", len(pocket_features))

    # --- 4. Bosses: outward-facing plane groups above the base ---
    try:
        boss_features = _find_bosses(
            regions, fitted, graph, region_lookup, fitted_lookup,
            cleaned, config, assigned, base_rids,
        )
    except Exception:
        logger.exception("Stage 5 failed finding bosses")
        raise
    for bf, rids in boss_features:
        bf.dependencies = [base_idx]
        features.append(bf)
        assigned.update(rids)
    logger.debug("Found %d bosses", len(boss_features))

    # --- 5. Fillets and chamfers (from Stage 4 edge classification) ---
    try:
        fillet_chamfer = _find_fillets_and_chamfers(
            regions, fitted, edges, region_lookup, fitted_lookup,
            config, assigned, graph, cleaned, topo_blends,
        )
    except Exception:
        logger.exception("Stage 5 failed finding fillets/chamfers")
        raise
    for fc, rids in fillet_chamfer:
        # Fillets/chamfers depend on whichever features own their neighbor surfaces
        fc.dependencies = _resolve_dependencies(
            rids, features, region_lookup,
        )
        if not fc.dependencies:
            fc.dependencies = [base_idx]
        features.append(fc)
        assigned.update(rids)

    # --- 6. Remaining unassigned regions → absorb into base ---
    remaining = set(r.region_id for r in regions) - assigned
    if remaining:
        remaining_faces = np.concatenate(
            [region_lookup[rid].face_indices for rid in remaining]
        )
        base_feature.face_indices = np.concatenate(
            [base_feature.face_indices, remaining_faces]
        )
        assigned.update(remaining)

    # Build dependency DAG
    dep_graph = nx.DiGraph()
    for i, f in enumerate(features):
        dep_graph.add_node(i, type=f.type.value)
        for dep in f.dependencies:
            if dep < i:
                dep_graph.add_edge(dep, i)

    tree = FeatureTree(
        features=features,
        dependency_graph=dep_graph,
        metadata={
            "num_regions": len(regions),
            "num_assigned": len(assigned),
        },
    )

    logger.info(
        "Recognized %d features: %s",
        tree.num_features,
        ", ".join(f"{f.type.value}" for f in features),
    )

    return tree


# ---------------------------------------------------------------------------
# Base extrusion
# ---------------------------------------------------------------------------


def _cross_section_cov(verts: np.ndarray, bbox: np.ndarray, axis: int, n_bins: int = 10) -> float:
    """Coefficient of variation of the cross-section bbox-area along `axis`.

    Low = the part is prismatic along this axis (constant cross-section);
    high = it tapers/steps. Used to pick the extrude axis that keeps the
    cross-section most constant.
    """
    u, v = [x for x in range(3) if x != axis]
    lo, hi = float(bbox[0][axis]), float(bbox[1][axis])
    if hi - lo < 1e-9:
        return float("inf")
    coord = verts[:, axis]
    edges = np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), n_bins + 1)
    areas = []
    for k in range(n_bins):
        m = (coord >= edges[k]) & (coord < edges[k + 1])
        if int(m.sum()) < 3:
            continue
        pu, pv = verts[m][:, u], verts[m][:, v]
        areas.append((pu.max() - pu.min()) * (pv.max() - pv.min()))
    if len(areas) < 3:
        return float("inf")
    areas = np.asarray(areas, dtype=float)
    mean = float(areas.mean())
    return float(areas.std() / mean) if mean > 1e-9 else float("inf")


def _thin_axis_tapered(verts: np.ndarray, bbox: np.ndarray, axis: int,
                       band_frac: float = 0.15, ratio_thresh: float = 0.9) -> bool:
    """True if the cross-section at the two ENDS of `axis` differ — a draft
    /tapered plate (one face wider than the other), not a flat plate."""
    u, v = [x for x in range(3) if x != axis]
    lo, hi = float(bbox[0][axis]), float(bbox[1][axis])
    if hi - lo < 1e-9:
        return False
    coord = verts[:, axis]
    band = band_frac * (hi - lo)

    def _area(mask: np.ndarray) -> float:
        if int(mask.sum()) < 3:
            return 0.0
        pu, pv = verts[mask][:, u], verts[mask][:, v]
        return float((pu.max() - pu.min()) * (pv.max() - pv.min()))

    a_min = _area(coord < lo + band)
    a_max = _area(coord > hi - band)
    if max(a_min, a_max) < 1e-9:
        return False
    return (min(a_min, a_max) / max(a_min, a_max)) < ratio_thresh


def _find_base_extrusion(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    cleaned: CleanedMesh,
) -> tuple[Feature, set[int]]:
    """Find the base extrusion — the largest connected group of planes.

    For a prismatic part, the base is the dominant planar body that
    everything else attaches to or subtracts from.
    """
    fitted_lookup = {f.region_id: f for f in fitted}

    # Score each planar region by face count
    plane_regions = [
        r for r in regions
        if fitted_lookup.get(r.region_id) is not None
        and fitted_lookup[r.region_id].surface_type == SurfaceType.PLANE
    ]

    if not plane_regions:
        # Fallback: use the single largest region
        largest = max(regions, key=lambda r: r.num_faces)
        return (
            Feature(
                type=FeatureType.BASE_EXTRUSION,
                face_indices=largest.face_indices.copy(),
                depth=0.0,
                direction=np.array([0.0, 0.0, 1.0]),
                operation="NEW",
            ),
            {largest.region_id},
        )

    # Find connected components of planar regions in the adjacency graph
    plane_ids = {r.region_id for r in plane_regions}
    plane_subgraph = graph.subgraph(plane_ids)

    best_component: set[int] = set()
    best_face_count = 0
    for component in nx.connected_components(plane_subgraph):
        face_count = sum(
            r.num_faces for r in plane_regions if r.region_id in component
        )
        if face_count > best_face_count:
            best_face_count = face_count
            best_component = set(component)

    # If the best planar component is very small, include all planes
    total_faces = sum(r.num_faces for r in regions)
    if best_face_count < total_faces * 0.1:
        best_component = plane_ids

    # Collect face indices
    base_faces = np.concatenate(
        [r.face_indices for r in plane_regions if r.region_id in best_component]
    )

    # Estimate extrusion direction from the dominant plane normal,
    # weighted by AREA (not triangle count). A finely-tessellated 45°
    # chamfer can out-count the main plate face; weighting by area
    # keeps the true dominant face winning. On the re-exported
    # BottomMount this stopped the base from extruding along a diagonal
    # chamfer normal ([0.7,0,-0.72]) instead of the plate's thickness.
    def _plane_area(r: FaceRegion) -> float:
        tris = cleaned.vertices[cleaned.faces[r.face_indices]]
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())

    largest_plane = max(
        (r for r in plane_regions if r.region_id in best_component),
        key=_plane_area,
    )
    f = fitted_lookup[largest_plane.region_id]
    direction = np.array(f.params.get("normal", [0, 0, 1]), dtype=float)
    direction = direction / (np.linalg.norm(direction) + 1e-15)

    # Snap to the nearest principal axis — prismatic base faces are
    # axis-aligned, and a raw fitted normal a few degrees off produces a
    # skewed extrude and a thin, wrong profile.
    principal = int(np.argmax(np.abs(direction)))
    snapped = np.zeros(3)
    snapped[principal] = float(np.sign(direction[principal]) or 1.0)
    direction = snapped

    # Thin-plate override: a plate is always extruded through its
    # thickness. If one bbox dimension is much smaller than the other
    # two, force the extrude axis to that thin direction regardless of
    # which face has the most area — this is the canonical prismatic
    # interpretation and is robust to chamfers/bosses dominating area.
    size = cleaned.bbox[1] - cleaned.bbox[0]
    order = np.argsort(size)
    if size[order[0]] > 0 and size[order[1]] >= 3.0 * size[order[0]]:
        thin_axis = int(order[0])
        ax = thin_axis
        # A TAPERED/drafted plate (one face wider than the other) is NOT
        # prismatic through its thickness — extruding the thin-axis profile
        # gives a constant cross-section that's wrong end-to-end. When the
        # thin axis tapers and a perpendicular axis has a more-constant
        # cross-section, extrude that perpendicular SIDE profile (the
        # trapezoid silhouette) instead. Flat plates keep the thin axis.
        if _thin_axis_tapered(cleaned.vertices, cleaned.bbox, ax):
            perp = [a for a in range(3) if a != ax]
            covs = {a: _cross_section_cov(cleaned.vertices, cleaned.bbox, a) for a in perp}
            best_perp = min(perp, key=lambda a: covs[a])
            if covs[best_perp] < _cross_section_cov(cleaned.vertices, cleaned.bbox, ax):
                logger.info(
                    "Base extrusion: tapered plate, extruding SIDE profile "
                    "along axis %d instead of thin axis %d", best_perp, thin_axis,
                )
                ax = best_perp
        if ax != principal:
            logger.info(
                "Base extrusion: plate override, axis %d -> %d (bbox=%s)",
                principal, ax, np.round(size, 1).tolist(),
            )
        direction = np.zeros(3)
        direction[ax] = 1.0

    # Estimate depth from bbox extent along the extrusion direction
    verts = cleaned.vertices
    proj = verts @ direction
    depth = float(proj.max() - proj.min())

    return (
        Feature(
            type=FeatureType.BASE_EXTRUSION,
            face_indices=base_faces,
            depth=depth,
            direction=direction,
            operation="NEW",
        ),
        best_component,
    )


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------


def _group_symmetric_holes(
    hole_features: list[tuple[Feature, set[int]]],
    radius_tol_pct: float = 0.05,
    plane_dist_tol: float = 1.0,
    through_hole_radius_tol_pct: float = 0.6,
) -> list[tuple[Feature, set[int]]]:
    """Collapse symmetric hole patterns into single multi-center features.

    Two through-holes drilled by the same operation share:
        • radius (within ~5 %)
        • axis direction (parallel)
        • entry-plane position (their centers project to the same point
          along the shared axis, within ~1 mm)

    Such holes get merged into one Feature whose `sketch_geometry` carries
    a `centers` list. The CAD export then emits ONE Sketch step (multiple
    circles) followed by ONE Cut Extrude — matching how a CAD operator
    would actually model the part. The single-center `center` field is
    kept too so any consumer that only expects one center (older render
    paths) still works.
    """
    if not hole_features:
        return hole_features

    # Pairwise union-find by relative radius / axis / entry-plane match.
    # Bucketed binning misfires near bin boundaries (a 4.00/4.10 pair
    # gets split if 4.05 is the boundary), but pairwise comparison
    # is correct by construction.
    standalone: list[int] = []
    candidates: list[tuple[int, dict]] = []  # (idx, normalized_meta)
    for idx, (feat, _rids) in enumerate(hole_features):
        sg = feat.sketch_geometry or {}
        if sg.get("type") != "circle":
            standalone.append(idx)
            continue
        center = np.asarray(sg.get("center"), dtype=float)
        radius = float(sg.get("radius", 0.0))
        axis = np.asarray(feat.direction, dtype=float)
        axn = float(np.linalg.norm(axis))
        if axn < 1e-12 or radius <= 0:
            standalone.append(idx)
            continue
        axis = axis / axn
        ax_idx = int(np.argmax(np.abs(axis)))
        candidates.append((idx, {
            "type": feat.type,
            "radius": radius,
            "ax_idx": ax_idx,
            "axis": axis,
            "center": center,
        }))

    # Dedupe coaxial fragments GLOBALLY, before pattern grouping. A single
    # bore split during segmentation (a wall fragmented by a curvature
    # seam, a barrel + coaxial stub) yields two candidates whose centers
    # project to nearly the same point in the plane perpendicular to the
    # axis. Their fitted radii can differ by ~10% (r=2.94 vs r=2.67 on
    # Body1Mesh's edge bore), which lands them in DIFFERENT pattern
    # groups — so a per-group dedup never compares them and the export
    # drills the same hole twice. Compare every same-axis candidate pair
    # here and keep the lowest-RMS fragment. Deliberately-split tangent
    # pairs overlap in-plane by design and are exempt.
    def _axial_span(idx: int, meta: dict) -> tuple[float, float]:
        feat = hole_features[idx][0]
        mid = float(np.dot(meta["center"], meta["axis"]))
        half = max(float(feat.depth), 0.0) / 2.0
        return (mid - half, mid + half)

    deduped: list[tuple[int, dict]] = []
    for idx, meta in candidates:
        sg = hole_features[idx][0].sketch_geometry or {}
        if sg.get("tangent_pair"):
            deduped.append((idx, meta))
            continue
        ax = meta["axis"]
        in_plane = meta["center"] - np.dot(meta["center"], ax) * ax
        lo, hi = _axial_span(idx, meta)
        rms = float(sg.get("fit_rms", float("inf")))
        tol = max(meta["radius"], 2.0)
        hit = None
        for k, (kidx, kmeta) in enumerate(deduped):
            ksg = hole_features[kidx][0].sketch_geometry or {}
            if ksg.get("tangent_pair"):
                continue
            if kmeta["ax_idx"] != meta["ax_idx"]:
                continue
            k_in_plane = kmeta["center"] - np.dot(kmeta["center"], ax) * ax
            if float(np.linalg.norm(in_plane - k_in_plane)) >= tol:
                continue
            # Same axis line is necessary but NOT sufficient: two plates
            # of an assembly carry coaxial holes by design (a rod hole
            # through both side plates of RackMount). Fragments of ONE
            # bore also overlap along the axis; separate stacked holes
            # don't. Require axial overlap (with 1mm slack for caps).
            klo, khi = _axial_span(kidx, kmeta)
            if min(hi, khi) - max(lo, klo) < -1.0:
                continue
            hit = k
            break
        if hit is None:
            deduped.append((idx, meta))
        else:
            k_rms = float(
                (hole_features[deduped[hit][0]][0].sketch_geometry or {})
                .get("fit_rms", float("inf"))
            )
            if rms < k_rms:
                deduped[hit] = (idx, meta)
            logger.info(
                "Deduped coaxial hole fragments at in-plane %s (r=%.2f/%.2f)",
                np.round(in_plane, 1).tolist(),
                meta["radius"], deduped[hit][1]["radius"],
            )
    candidates = deduped

    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def _same_pattern(a: dict, b: dict) -> bool:
        if a["type"] != b["type"]:
            return False
        if a["ax_idx"] != b["ax_idx"]:
            return False
        # Radius tolerance: looser for THROUGH holes because the
        # cylinder fitter is unstable for short stubby cylinders
        # (a 4-tab through-hole pattern can come back as one tab
        # at r=4 and another at r=6 just because their bbox shapes
        # differ slightly). Cap a 2-tab pattern at 60 % radius
        # spread; tighter for blind holes where the depth is part
        # of the signature.
        if a["type"] == FeatureType.THROUGH_HOLE:
            tol = through_hole_radius_tol_pct
        else:
            tol = radius_tol_pct
        rmax = max(a["radius"], b["radius"])
        if abs(a["radius"] - b["radius"]) / rmax > tol:
            return False
        # For BLIND holes, entry-plane positions must match (signed
        # along axis). For THROUGH holes, both faces are entry
        # candidates — skip this check.
        if a["type"] != FeatureType.THROUGH_HOLE:
            ax = a["axis"]
            if abs(np.dot(b["center"] - a["center"], ax)) > plane_dist_tol:
                return False
        return True

    for i in range(n):
        for j in range(i + 1, n):
            if _same_pattern(candidates[i][1], candidates[j][1]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(candidates[k][0])

    merged: list[tuple[Feature, set[int]]] = []
    # Standalone holes (no sketch_geometry) pass through.
    for idx in standalone:
        merged.append(hole_features[idx])

    def _partition_by_radius(indices: list[int], rel_tol: float = 0.2) -> list[list[int]]:
        """Split a pattern group into sub-patterns of like radius.

        The loose through-hole tolerance (60%) is needed so unstable
        short-cylinder fits land in ONE group, but unifying the radius
        across the whole group then drills genuinely different sizes the
        same: BottomMount has 9× r=2.5 and 4× r=4.0 bolt holes, and one
        unified group drilled all of them at r=4.0. Greedy-cluster the
        sorted radii (20% relative gap starts a new sub-pattern) and emit
        one merged feature per cluster.
        """
        order = sorted(indices, key=lambda i: float(
            hole_features[i][0].sketch_geometry.get("radius", 0.0)))
        parts: list[list[int]] = [[order[0]]]
        for i in order[1:]:
            r_i = float(hole_features[i][0].sketch_geometry.get("radius", 0.0))
            r_prev = float(
                hole_features[parts[-1][-1]][0].sketch_geometry.get("radius", 0.0))
            if r_prev > 0 and (r_i - r_prev) / r_prev > rel_tol:
                parts.append([i])
            else:
                parts[-1].append(i)
        return parts

    flat_groups: list[list[int]] = []
    for indices in groups.values():
        if len(indices) == 1:
            flat_groups.append(indices)
        else:
            flat_groups.extend(_partition_by_radius(indices))

    for indices in flat_groups:
        if len(indices) == 1:
            merged.append(hole_features[indices[0]])
            continue
        # Merge group: keep the first member as the seed, attach all
        # centers, union all face_indices, union all region IDs.
        seed_feat, seed_rids = hole_features[indices[0]]
        all_rids: set[int] = set()
        face_chunks: list[np.ndarray] = []
        centers: list[list[float]] = []
        radii: list[float] = []
        rms_vals: list[float] = []
        for idx in indices:
            f, rids = hole_features[idx]
            all_rids.update(rids)
            face_chunks.append(np.asarray(f.face_indices, dtype=np.int64))
            c = f.sketch_geometry.get("center")
            if c is not None:
                centers.append([float(x) for x in c])
            radii.append(float(f.sketch_geometry.get("radius", 0.0)))
            rms_vals.append(float(f.sketch_geometry.get("fit_rms", float("inf"))))
        merged_faces = np.concatenate(face_chunks) if face_chunks else seed_feat.face_indices
        merged_sketch = dict(seed_feat.sketch_geometry)
        merged_sketch["centers"] = centers
        merged_sketch["count"] = len(centers)
        # Unify radius across the group: take the radius of the member
        # with the LOWEST fit RMS. Both known failure modes produce a
        # high-RMS fit — short-arc patches that under-estimate radius,
        # and over-merged regions (bore fused with a neighboring
        # cylinder) that inflate it — while a clean 360-degree bore
        # fits with near-zero RMS. The previous max(radii) rule
        # rewarded the over-merged fit. Fall back to max() when no
        # member carries an RMS (older callers).
        if radii:
            if rms_vals and np.isfinite(min(rms_vals)):
                merged_sketch["radius"] = radii[int(np.argmin(rms_vals))]
            else:
                merged_sketch["radius"] = max(radii)
        merged_feat = Feature(
            type=seed_feat.type,
            face_indices=merged_faces,
            sketch_geometry=merged_sketch,
            depth=seed_feat.depth,
            direction=seed_feat.direction.copy() if hasattr(seed_feat.direction, 'copy') else seed_feat.direction,
            operation=seed_feat.operation,
        )
        merged.append((merged_feat, all_rids))
        logger.info(
            "Grouped %d symmetric %s holes into one feature (r=%.2f, "
            "from %s)",
            len(indices), seed_feat.type.value,
            float(merged_sketch.get("radius", 0.0)),
            sorted(radii),
        )
    return merged


def _find_holes(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    region_lookup: dict[int, FaceRegion],
    cleaned: CleanedMesh,
    config: RecognizeConfig,
    assigned: set[int],
) -> list[tuple[Feature, set[int]]]:
    """Find cylindrical holes (through-holes and blind holes)."""
    bbox_diag = cleaned.bbox_diagonal
    results = []

    for f in fitted:
        if f.region_id in assigned:
            continue
        if f.surface_type not in (SurfaceType.CYLINDER, SurfaceType.CONE):
            continue

        radius = f.params.get("radius", 0)
        # For cones with small half_angle (<5°), treat as cylinders.
        # Compute radius from the region's vertex distances to the axis.
        if f.surface_type == SurfaceType.CONE:
            half_angle = f.params.get("half_angle", 0)
            if half_angle > np.radians(45):
                continue  # wide cone, not a hole
            region = region_lookup.get(f.region_id)
            if region is None:
                continue
            verts = cleaned.vertices[cleaned.faces[region.face_indices].flatten()]
            # Estimate radius from bbox: smallest two dims averaged / 2
            bbox_size = verts.max(axis=0) - verts.min(axis=0)
            sorted_dims = sorted(bbox_size)
            radius = (sorted_dims[0] + sorted_dims[1]) / 4
        if radius <= 0 or radius > config.hole_max_radius_ratio * bbox_diag:
            continue
        # Reject tiny cylinders — these are edge radii/fillets, not holes.
        # Minimum 1.5mm radius for a real hole.
        if radius < 1.5:
            continue

        region = region_lookup.get(f.region_id)
        if region is None:
            continue

        # Check angular span to confirm it's a full cylinder (hole), not a partial arc
        span = _cylinder_angular_span(f, region)
        if span < 270.0:
            continue  # partial arc, not a hole

        # Reject full-length cylinders whose faces point OUTWARD (away
        # from the axis) — rounded ends / edge fillets / rod surfaces.
        # The full-length test alone is wrong for through-holes: a hole
        # drilled through a plate always spans the plate's full extent
        # along its axis (Body1Mesh's 4 corner holes are h=12 in a
        # 12mm-thick plate). Hole walls point INWARD toward the axis,
        # which cleanly separates the two cases.
        height_pre = f.params.get("height", 0)
        if height_pre > 0:
            axis_pre = np.array(f.params.get("axis_dir", [0, 0, 1]))
            part_proj_pre = cleaned.vertices @ axis_pre
            part_extent_pre = float(part_proj_pre.max() - part_proj_pre.min())
            if part_extent_pre > 0 and height_pre > part_extent_pre * 0.9:
                if not _faces_point_inward(f, region, cleaned):
                    # Outward-facing full-length cylinder → not a hole
                    continue

        height = f.params.get("height", 0)
        axis_dir = np.array(f.params.get("axis_dir", [0, 0, 1]))

        # For cones, compute height from region vertex extent along axis
        if height <= 0 and region is not None:
            verts = cleaned.vertices[cleaned.faces[region.face_indices].flatten()]
            proj = verts @ axis_dir
            height = float(proj.max() - proj.min())

        # Thin-plate override: when the body is much thinner in one
        # axis than the others (a plate), all through-holes drilled
        # through the plate face must run in that thin direction. The
        # cylinder fitter is unstable for short cylinders (bbox like
        # 8×8×6 — three nearly-equally-good axes), so it can return
        # axis Y for one hole and axis Z for its mirror. Force the
        # axis to the thin direction so a plate's mounting holes get
        # consistent axes regardless of which axis the fitter picked.
        body_size = cleaned.bbox[1] - cleaned.bbox[0]
        sorted_dims = np.sort(body_size)
        is_thin_plate = (
            sorted_dims[1] >= 3.0 * sorted_dims[0]  # smallest dim ≪ middle dim
            and sorted_dims[0] > 0
        )
        thin_axis_idx = int(np.argmin(body_size))
        thin_axis = np.zeros(3)
        thin_axis[thin_axis_idx] = 1.0

        # Snap hole axis to nearest principal direction. Reject holes whose
        # axis is more than 15° from any principal direction — these are
        # artifacts from cut-piece cap geometry, not real holes.
        principal_axes = [
            np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
            np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1]),
        ]
        best_dot = 0
        best_pa = None
        for pa in principal_axes:
            d = abs(np.dot(axis_dir, pa))
            if d > best_dot:
                best_dot = d
                best_pa = pa
        if best_pa is None or best_dot < np.cos(np.radians(15)):
            continue  # axis too far from any principal direction
        axis_dir = best_pa * np.sign(np.dot(axis_dir, best_pa))

        # Override snapped axis with the body's thin direction when the
        # body is plate-shaped — but only when the geometry doesn't
        # clearly contradict it. Plates can carry SIDE holes too (e.g.
        # Body1Mesh's two Z-axis tab holes), and blindly forcing the
        # thin axis cut them along the wrong direction. The radial
        # spread of the region's vertices about an axis is near-zero
        # for the true axis on a >=270-degree wrap, so compare the
        # fitted axis against the thin-axis hypothesis and only
        # override when the thin axis explains the geometry comparably.
        if is_thin_plate and abs(np.dot(axis_dir, thin_axis)) < 0.7:
            hole_verts = cleaned.vertices[cleaned.faces[region.face_indices].flatten()]
            res_fitted = _axis_radial_spread(hole_verts, axis_dir)
            res_thin = _axis_radial_spread(hole_verts, thin_axis)
            # Physical plausibility: a real hole spans the full part extent
            # only along its DRILL direction (a through-hole runs the plate's
            # thickness). When the fitted axis is one of the part's LONG axes
            # AND the cylinder height spans nearly that whole long extent, the
            # short-cylinder fitter has latched onto the part's long axis
            # rather than a real bore — BottomMount's two spurious r=5.6 "X
            # holes" measure 140mm along the 152mm-wide X axis (span ≈ 0.9),
            # so force them onto the thin axis to land on the large plane.
            # Body1Mesh's legitimate Z tab holes are drilled only partway into
            # a tab (height ≈ 27–47 of a 156mm Z extent, span ≈ 0.2) and so
            # stay on their fitted axis. Both cases share low res_fitted/high
            # res_thin, so radial spread alone can't separate them — the
            # fraction of the axis the cylinder spans does.
            part_proj_axis = cleaned.vertices @ axis_dir
            part_extent_axis = float(part_proj_axis.max() - part_proj_axis.min())
            spans_full_axis = (
                part_extent_axis > 0 and height > 0.7 * part_extent_axis
            )
            if spans_full_axis or res_thin <= res_fitted * 1.5:
                axis_dir = thin_axis * np.sign(np.dot(axis_dir, thin_axis) or 1.0)
            else:
                logger.debug(
                    "Keeping fitted hole axis %s over thin-plate axis %s "
                    "(radial spread %.3f vs %.3f)",
                    axis_dir.tolist(), thin_axis.tolist(), res_fitted, res_thin,
                )
        elif is_thin_plate:
            axis_dir = thin_axis * np.sign(np.dot(axis_dir, thin_axis) or 1.0)

        # If the axis was overridden, the fitter's height (measured along
        # the OLD axis) no longer describes the hole — recompute the depth
        # as the region's extent along the final axis so the drilled cut
        # doesn't punch 140mm through a 13mm plate.
        wall_proj = (
            cleaned.vertices[cleaned.faces[region.face_indices].flatten()]
            @ axis_dir
        )
        final_height = float(np.ptp(wall_proj))
        if final_height > 0:
            height = final_height
        # Re-center the hole's axial position at the WALL midpoint. The
        # fitter's axis_point can sit anywhere along the axis line; a
        # through-cut spans the whole part so it never mattered, but a
        # blind cut starts at center - depth/2 and an off-mid center
        # shifts the drilled hole off the real bore.
        hole_center = np.asarray(
            f.params.get("axis_point")
            or region.vertices.mean(axis=0),
            dtype=float,
        )
        axial_mid = float((wall_proj.min() + wall_proj.max()) / 2.0)
        hole_center = (
            hole_center
            + (axial_mid - float(hole_center @ axis_dir)) * axis_dir
        )

        # Determine through vs blind. Radius alone is NOT enough:
        # Body1Mesh's small-radius edge holes are drilled only 27-47mm
        # into the plate edge (Z extent 138mm), and classifying them as
        # through made the export punch 138mm cuts out the far side
        # (+6% volume error). The geometric signature of a blind hole is
        # its BOTTOM CAP: an adjacent near-planar region perpendicular
        # to the axis whose vertices all sit inside the bore. (A part
        # extent test fails on multi-body assemblies, where the extent
        # along the axis is the whole assembly, not the plate the hole
        # pierces.)
        has_cap = _hole_has_bottom_cap(
            f.region_id, axis_dir, hole_center, radius, height,
            graph, region_lookup, fitted, cleaned,
        )
        is_through = radius < bbox_diag * 0.1 and not has_cap

        hole_type = FeatureType.THROUGH_HOLE if is_through else FeatureType.BLIND_HOLE

        # Two TANGENT holes whose walls overlap get segmented as ONE fused
        # region (a "peanut" bore). The single-cylinder fit averages them
        # (Body1Mesh: two r=2.5 bores 1.4mm apart fit as one r=2.67 with
        # high residual). When a two-circle decomposition of the wall's
        # cross-section explains the points dramatically better, emit BOTH
        # holes instead of the averaged one.
        split = _split_tangent_bores(
            cleaned.vertices[np.unique(cleaned.faces[region.face_indices].flatten())],
            axis_dir, radius,
        )
        if split is not None:
            for (center3d, sub_radius) in split:
                results.append((
                    Feature(
                        type=hole_type,
                        face_indices=region.face_indices.copy(),
                        sketch_geometry={
                            "type": "circle",
                            "center": center3d,
                            "radius": sub_radius,
                            "fit_rms": float(f.rms_error),
                            # Deliberate tangent pair — exempt from
                            # coaxial-fragment dedup, which would re-merge
                            # the two centers (they overlap by design).
                            "tangent_pair": True,
                        },
                        depth=height,
                        direction=-axis_dir,
                        operation="REMOVE",
                    ),
                    {f.region_id},
                ))
            continue

        results.append((
            Feature(
                type=hole_type,
                face_indices=region.face_indices.copy(),
                sketch_geometry={
                    "type": "circle",
                    "center": hole_center.tolist(),
                    "radius": radius,
                    # Carried so symmetric-hole grouping can prefer the
                    # cleanest fit's radius over the worst one.
                    "fit_rms": float(f.rms_error),
                },
                depth=height,
                direction=-axis_dir,  # holes cut inward
                operation="REMOVE",
            ),
            {f.region_id},
        ))

    return results


def _split_tangent_bores(
    verts: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
) -> list[tuple[list[float], float]] | None:
    """Detect a fused pair of tangent bores and return their two circles.

    Projects the wall vertices onto the plane perpendicular to the hole
    axis and compares a one-circle fit against a two-circle alternating
    fit (k-means-style point assignment, seeded from several split
    directions). Returns [(center3d, radius), (center3d, radius)] only
    when ALL of:
      • the one-circle fit is poor (rms > 4 % of radius — clean bores fit
        at tessellation noise, ~0)
      • the two-circle fit cuts the rms by >60 %
      • the sub-circles have similar radii (within 30 %)
      • the centers are separated between 0.3r and 2.2r (overlapping or
        tangent — farther apart and the walls couldn't have fused into
        one region)
      • each sub-circle owns enough points (≥ 8) to be a real wall
    Otherwise returns None and the caller keeps the single-hole result.
    """
    n = np.asarray(axis_dir, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-15)
    # Orthonormal in-plane basis
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref); u /= np.linalg.norm(u) + 1e-15
    w = np.cross(n, u)
    pts = np.c_[verts @ u, verts @ w]
    if len(pts) < 24:
        return None

    def circle_fit(p: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x, y = p[:, 0], p[:, 1]
        A = np.c_[2 * x, 2 * y, np.ones(len(x))]
        sol, *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r = float(np.sqrt(max(sol[2] + cx**2 + cy**2, 0.0)))
        res = np.hypot(x - cx, y - cy) - r
        return np.array([cx, cy]), r, res

    c1, _r1, res1 = circle_fit(pts)
    rms1 = float(res1.std())
    if rms1 < 0.04 * radius:
        return None  # already a clean single bore

    best: tuple | None = None
    for ang in np.linspace(0.0, np.pi, 8, endpoint=False):
        d = np.array([np.cos(ang), np.sin(ang)])
        lab = (pts - c1) @ d > 0
        for _ in range(20):
            if lab.sum() < 8 or (~lab).sum() < 8:
                break
            ca, ra, _ = circle_fit(pts[lab])
            cb, rb, _ = circle_fit(pts[~lab])
            da = np.abs(np.hypot(*(pts - ca).T) - ra)
            db = np.abs(np.hypot(*(pts - cb).T) - rb)
            new_lab = da < db
            if (new_lab == lab).all():
                break
            lab = new_lab
        if lab.sum() < 8 or (~lab).sum() < 8:
            continue
        ca, ra, resa = circle_fit(pts[lab])
        cb, rb, resb = circle_fit(pts[~lab])
        rms2 = float(np.sqrt(np.mean(np.r_[resa, resb] ** 2)))
        if best is None or rms2 < best[0]:
            best = (rms2, ca, ra, cb, rb)
    if best is None:
        return None
    rms2, ca, ra, cb, rb = best

    rmean = 0.5 * (ra + rb)
    sep = float(np.linalg.norm(ca - cb))
    if (
        rms2 > 0.4 * rms1
        or min(ra, rb) <= 0
        or abs(ra - rb) / max(ra, rb) > 0.3
        or not (0.3 * rmean < sep < 2.2 * rmean)
    ):
        return None

    axial = float(np.mean(verts @ n))
    out: list[tuple[list[float], float]] = []
    for c2d, r2d in ((ca, ra), (cb, rb)):
        p3 = c2d[0] * u + c2d[1] * w + axial * n
        out.append(([float(x) for x in p3], float(r2d)))
    logger.info(
        "Split fused tangent bore: r=%.2f rms=%.3f -> two r=%.2f/%.2f sep=%.2f rms=%.3f",
        radius, rms1, ra, rb, sep, rms2,
    )
    return out


def _hole_has_bottom_cap(
    region_id: int,
    axis_dir: np.ndarray,
    hole_center: np.ndarray,
    radius: float,
    height: float,
    graph: nx.Graph,
    region_lookup: dict[int, FaceRegion],
    fitted: list[FittedSurface],
    cleaned: CleanedMesh,
) -> bool:
    """True if the hole wall has an adjacent bottom cap (blind hole).

    The cap is a near-planar neighbor region perpendicular to the hole
    axis whose vertices ALL lie inside the bore (within ~1.6r of the
    axis line). The part faces the hole pierces are also planar and
    perpendicular, but they surround the bore and extend far beyond it,
    so the radial-containment test cleanly separates the two.
    """
    if graph is None or not graph.has_node(region_id):
        return False
    fitted_by_id = {s.region_id: s for s in fitted}
    # The bore wall may be fragmented into several coaxial cylinder
    # regions, with the cap adjacent to a DIFFERENT fragment than the one
    # that survived dedup. Search the planar neighbors of the wall AND of
    # any adjacent cylinder fragment (one hop through the bore).
    to_scan = set(graph.neighbors(region_id))
    for nb in list(to_scan):
        nf = fitted_by_id.get(nb)
        if nf is not None and nf.surface_type in (
            SurfaceType.CYLINDER, SurfaceType.CONE,
        ):
            to_scan.update(graph.neighbors(nb))
    to_scan.discard(region_id)
    for nb in to_scan:
        nf = fitted_by_id.get(nb)
        nreg = region_lookup.get(nb)
        if nf is None or nreg is None:
            continue
        if nf.surface_type != SurfaceType.PLANE:
            continue
        normal = np.asarray(nf.params.get("normal", [0, 0, 1]), dtype=float)
        nn = float(np.linalg.norm(normal))
        if nn < 1e-12 or abs(float(normal @ axis_dir)) / nn < 0.8:
            continue
        verts = cleaned.vertices[np.unique(cleaned.faces[nreg.face_indices].flatten())]
        rel = verts - hole_center
        radial = rel - np.outer(rel @ axis_dir, axis_dir)
        if float(np.linalg.norm(radial, axis=1).max()) <= 1.6 * radius:
            return True
    return False


def _faces_point_inward(
    fitted: FittedSurface, region: FaceRegion, cleaned: CleanedMesh,
) -> bool:
    """True if the region's face normals point toward the cylinder axis.

    Hole walls face inward; boss/rod/fillet barrels face outward.
    """
    if region is None:
        return False
    axis_point = np.array(fitted.params.get("axis_point", fitted.params.get("apex", [0, 0, 0])))
    axis_dir = np.array(fitted.params.get("axis_dir", [0, 0, 1]))
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)
    centroids = cleaned.vertices[cleaned.faces[region.face_indices]].mean(axis=1)
    v = centroids - axis_point
    radial = v - np.outer(v @ axis_dir, axis_dir)
    radial /= np.linalg.norm(radial, axis=1)[:, None] + 1e-15
    dots = np.einsum("ij,ij->i", region.normals, radial)
    return float(np.mean(dots)) < 0.0


def _axis_radial_spread(verts: np.ndarray, axis: np.ndarray) -> float:
    """Std of radial distances about the axis line through the centroid.

    For a full cylindrical wrap the centroid lies on the true axis and
    every vertex sits at radius r, so the spread is ~tessellation noise;
    for a wrong axis the distances vary by the cylinder height.
    """
    v = verts - verts.mean(axis=0)
    t = v @ axis
    perp = v - np.outer(t, axis)
    return float(np.std(np.linalg.norm(perp, axis=1)))


def _cylinder_angular_span(fitted: FittedSurface, region: FaceRegion) -> float:
    """Compute angular span of a cylinder/cone region (degrees)."""
    axis_point = np.array(fitted.params.get("axis_point", fitted.params.get("apex", [0, 0, 0])))
    axis_dir = np.array(fitted.params.get("axis_dir", [0, 0, 1]))
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-15)

    vecs = region.vertices - axis_point
    proj_along = np.outer(vecs @ axis_dir, axis_dir)
    proj_perp = vecs - proj_along

    norms = np.linalg.norm(proj_perp, axis=1)
    valid = norms > 1e-10
    if valid.sum() < 3:
        return 360.0

    proj_perp = proj_perp[valid]
    ref = proj_perp[0] / np.linalg.norm(proj_perp[0])
    ref2 = np.cross(axis_dir, ref)

    angles = np.arctan2(proj_perp @ ref2, proj_perp @ ref)
    sorted_angles = np.sort(angles)
    gaps = np.diff(sorted_angles)
    wraparound_gap = (2 * np.pi) - (sorted_angles[-1] - sorted_angles[0])
    max_gap = max(float(np.max(gaps)) if len(gaps) > 0 else 0.0, wraparound_gap)

    return float(np.degrees((2 * np.pi) - max_gap))


# ---------------------------------------------------------------------------
# Pockets
# ---------------------------------------------------------------------------


def _find_pockets(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    region_lookup: dict[int, FaceRegion],
    fitted_lookup: dict[int, FittedSurface],
    cleaned: CleanedMesh,
    config: RecognizeConfig,
    assigned: set[int],
    base_rids: set[int],
) -> list[tuple[Feature, set[int]]]:
    """Find pockets — groups of unassigned planes that face inward.

    A pocket is a concavity: its dominant normal points opposite to
    the base surface normal (or perpendicular, for side walls).
    """
    bbox_diag = cleaned.bbox_diagonal

    # Get the base extrusion direction
    base_normals = []
    for rid in base_rids:
        f = fitted_lookup.get(rid)
        if f and f.surface_type == SurfaceType.PLANE:
            r = region_lookup.get(rid)
            if r:
                base_normals.extend([f.params["normal"]] * r.num_faces)
    if not base_normals:
        return []
    base_dir = np.mean(base_normals, axis=0)
    base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-15)

    # Find unassigned plane regions whose normal is roughly opposite to base
    inward_rids: set[int] = set()
    for r in regions:
        if r.region_id in assigned:
            continue
        f = fitted_lookup.get(r.region_id)
        if f is None or f.surface_type != SurfaceType.PLANE:
            continue
        normal = np.array(f.params.get("normal", [0, 0, 0]))
        dot = np.dot(normal, base_dir)
        # Inward-facing: normal opposes base dir or is perpendicular (side wall)
        if dot < 0.3:
            inward_rids.add(r.region_id)

    if not inward_rids:
        return []

    # Group connected inward-facing planes
    sub = graph.subgraph(inward_rids)
    results = []
    for component in nx.connected_components(sub):
        face_count = sum(region_lookup[rid].num_faces for rid in component)
        if face_count < config.min_feature_faces:
            continue

        faces = np.concatenate([region_lookup[rid].face_indices for rid in component])

        # Estimate pocket depth
        pocket_verts = np.concatenate([region_lookup[rid].vertices for rid in component])
        proj = pocket_verts @ base_dir
        depth = float(proj.max() - proj.min())

        if depth > config.pocket_max_depth_ratio * bbox_diag:
            continue

        # Narrow elongated pockets → SLOT
        pocket_all_verts = pocket_verts
        centered = pocket_all_verts - pocket_all_verts.mean(axis=0)
        if len(centered) >= 3:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            aspect = s[0] / (s[1] + 1e-15) if len(s) >= 2 else 1.0
        else:
            aspect = 1.0

        feat_type = FeatureType.SLOT if aspect > 4.0 else FeatureType.POCKET

        results.append((
            Feature(
                type=feat_type,
                face_indices=faces,
                depth=depth,
                direction=-base_dir,
                operation="REMOVE",
            ),
            set(component),
        ))

    return results


# ---------------------------------------------------------------------------
# Bosses
# ---------------------------------------------------------------------------


def _find_bosses(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    graph: nx.Graph,
    region_lookup: dict[int, FaceRegion],
    fitted_lookup: dict[int, FittedSurface],
    cleaned: CleanedMesh,
    config: RecognizeConfig,
    assigned: set[int],
    base_rids: set[int],
) -> list[tuple[Feature, set[int]]]:
    """Find bosses — groups of unassigned planes that protrude outward."""
    bbox_diag = cleaned.bbox_diagonal

    base_normals = []
    for rid in base_rids:
        f = fitted_lookup.get(rid)
        if f and f.surface_type == SurfaceType.PLANE:
            r = region_lookup.get(rid)
            if r:
                base_normals.extend([f.params["normal"]] * r.num_faces)
    if not base_normals:
        return []
    base_dir = np.mean(base_normals, axis=0)
    base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-15)

    # Find unassigned planes with normals roughly aligned with base (outward)
    outward_rids: set[int] = set()
    for r in regions:
        if r.region_id in assigned:
            continue
        f = fitted_lookup.get(r.region_id)
        if f is None or f.surface_type != SurfaceType.PLANE:
            continue
        normal = np.array(f.params.get("normal", [0, 0, 0]))
        dot = np.dot(normal, base_dir)
        if dot > 0.7:
            outward_rids.add(r.region_id)

    if not outward_rids:
        return []

    sub = graph.subgraph(outward_rids)
    results = []
    for component in nx.connected_components(sub):
        # Bosses should NOT be the largest faces (those are the base)
        face_count = sum(region_lookup[rid].num_faces for rid in component)
        if face_count < config.min_feature_faces:
            continue

        faces = np.concatenate([region_lookup[rid].face_indices for rid in component])

        boss_verts = np.concatenate([region_lookup[rid].vertices for rid in component])
        proj = boss_verts @ base_dir
        height = float(proj.max() - proj.min())

        if height > config.boss_max_height_ratio * bbox_diag:
            continue

        results.append((
            Feature(
                type=FeatureType.BOSS,
                face_indices=faces,
                depth=height,
                direction=base_dir,
                operation="ADD",
            ),
            set(component),
        ))

    return results


# ---------------------------------------------------------------------------
# Fillets and chamfers
# ---------------------------------------------------------------------------


def _find_fillets_and_chamfers(
    regions: list[FaceRegion],
    fitted: list[FittedSurface],
    edges: list[SurfaceEdge],
    region_lookup: dict[int, FaceRegion],
    fitted_lookup: dict[int, FittedSurface],
    config: RecognizeConfig,
    assigned: set[int],
    graph: nx.Graph | None = None,
    cleaned: CleanedMesh | None = None,
    topo_blends: list[dict] | None = None,
) -> list[tuple[Feature, set[int]]]:
    """Find fillet and chamfer features.

    Detection strategies:
    1. Cone surfaces → CHAMFER — only if the cone is small relative to the
       part and actually sits between two principal (plane/cylinder) faces
       in the adjacency graph (not a standalone conical feature).
    2. Small narrow plane strips at ~135 deg to neighbors → CHAMFER (flat bevels)
    3. Small partial-arc cylinders → FILLET (rounded edges)
    4. Surfaces adjacent to chamfer/fillet/smooth edges → check type
    """
    results = []
    fillet_rids: set[int] = set()
    chamfer_rids: set[int] = set()

    # Part-scale reference for size checks
    bbox_diag = cleaned.bbox_diagonal if cleaned is not None else 100.0

    # Helper: check whether *rid* is adjacent (in the surface graph) to at
    # least two principal faces (PLANE or CYLINDER that are either assigned
    # to a feature or large enough to be principal).
    def _adjacent_to_two_principal(rid: int) -> bool:
        if graph is None or not graph.has_node(rid):
            return False
        principal_count = 0
        for nbr in graph.neighbors(rid):
            nbr_fit = fitted_lookup.get(nbr)
            if nbr_fit is None:
                continue
            if nbr_fit.surface_type in (SurfaceType.PLANE, SurfaceType.CYLINDER):
                principal_count += 1
                if principal_count >= 2:
                    return True
        return False

    # Strategy 1: Unassigned CONE surfaces → CHAMFER, but only if:
    #   a) The cone's bounding box is small relative to the part (< 10% of
    #      bbox diagonal), which rules out large conical features.
    #   b) The cone sits between two principal faces in the adjacency graph,
    #      which rules out standalone cone geometry (e.g. a conical boss).
    for f in fitted:
        if f.region_id in assigned:
            continue
        if f.surface_type != SurfaceType.CONE:
            continue
        r = region_lookup.get(f.region_id)
        if r is None:
            continue

        # (a) Size check: cone must be small relative to part
        cone_bbox_size = r.vertices.max(axis=0) - r.vertices.min(axis=0)
        cone_diag = float(np.linalg.norm(cone_bbox_size))
        if cone_diag > bbox_diag * 0.10:
            logger.debug(
                "Strategy 1: cone rid=%d too large (diag=%.2f vs part=%.2f), skipping",
                f.region_id, cone_diag, bbox_diag,
            )
            continue

        # (b) Adjacency check: must border at least 2 principal faces
        if not _adjacent_to_two_principal(f.region_id):
            logger.debug(
                "Strategy 1: cone rid=%d not adjacent to 2 principal faces, skipping",
                f.region_id,
            )
            continue

        logger.debug("Strategy 1: cone rid=%d accepted as chamfer (diag=%.2f)", f.region_id, cone_diag)
        chamfer_rids.add(f.region_id)

    # Strategy 2: Small narrow PLANE strips adjacent to chamfer edges
    # (flat 45 deg bevels between two larger faces)
    for e in edges:
        if e.transition_type == TransitionType.CHAMFER:
            for rid in (e.surface_a_id, e.surface_b_id):
                if rid in assigned or rid in chamfer_rids:
                    continue
                f = fitted_lookup.get(rid)
                r = region_lookup.get(rid)
                if f is None or r is None:
                    continue
                # Small plane strips are chamfer faces
                if f.surface_type == SurfaceType.PLANE and r.num_faces <= 50:
                    extent = f.params.get("extent", float("inf"))
                    if extent < config.chamfer_width_ratio * float(r.vertices.max() - r.vertices.min()) * 2:
                        chamfer_rids.add(rid)

    # Strategy 3: Topology-driven fillet blend detection.
    #
    # A fillet is a rolling-ball blend sitting tangent to two or more
    # principal faces. Geometrically that means: a non-planar region whose
    # boundary to neighboring regions is near-tangent (the stage-4 dihedral
    # classifier calls those SMOOTH or FILLET edges). It does not matter
    # whether the segmenter/fitter happened to fit the blend as a cylinder,
    # cone, or sphere — what matters is the topological role it plays.
    #
    # For each candidate region we reconstruct the cylinder axis from a
    # closed-form 2D circle fit on the vertex cross-section (project the
    # vertices perpendicular to the SVD length direction and fit a circle
    # by least squares). A closed-form 2D fit is far more stable than a
    # 6-parameter 3D cylinder fit on the handful of faces a fillet strip
    # contains, which is where the previous approach broke down.
    fillet_fit_lookup: dict[int, dict] = {}  # rid -> fitted cylinder params

    def _fit_strip_as_cylinder(pts: np.ndarray, face_normals: np.ndarray | None):
        """Fit a cylinder to a curved strip via normal-covariance + 2D circle.

        The cylinder axis is recovered from the *face normals* of the
        region, not from the vertex positions: on a cylinder all surface
        normals are perpendicular to the axis, so the axis is the
        direction along which the normal covariance is smallest. Using
        vertex SVD instead (as a previous version did) can pick the
        wrong axis when the region spans the length of the part but is
        actually a blend around a shorter direction.

        Returns (axis_point, axis_dir, radius, rms) or None if the fit
        is degenerate.
        """
        if pts is None or len(pts) < 4:
            return None
        if face_normals is None or len(face_normals) < 2:
            return None
        if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(face_normals)):
            return None
        mean_p = pts.mean(axis=0)
        rel = pts - mean_p
        if not np.all(np.isfinite(rel)):
            return None

        # Axis direction from face normals: for a cylinder with axis d,
        # every surface normal is perpendicular to d. So d is the
        # eigenvector of N^T N with the smallest eigenvalue.
        n_norms = np.linalg.norm(face_normals, axis=1)
        valid = n_norms > 1e-8
        if valid.sum() < 2:
            return None
        n_unit = face_normals[valid] / n_norms[valid, None]
        M = n_unit.T @ n_unit  # (3, 3)
        try:
            eig_vals, eig_vecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            return None
        # eigenvalues in ascending order
        axis_dir = eig_vecs[:, 0]
        # The smallest eigenvalue measures how non-perpendicular the
        # normals are to the axis. For a genuine cylindrical region it
        # should be near zero; for a saddle or corner cap it is large.
        # Rank the three eigenvalues — for a cylinder we expect roughly
        # one tiny eigenvalue and two comparable larger ones.
        lam = np.sort(np.abs(eig_vals))
        if lam[2] < 1e-10:
            return None
        # The smallest eigenvalue must be ~0 — i.e. all normals lie
        # in a plane. For a cylinder, that plane is perpendicular to
        # the axis.
        if lam[0] / lam[2] > 0.2:
            return None
        # The middle eigenvalue must be smaller than the largest. For
        # a partial-arc cylinder the normal spread within the plane is
        # anisotropic (middle ≪ max). For two perpendicular planes
        # meeting at a sharp interior edge the two directions are
        # evenly populated and middle/max ≈ 1 — that is the signature
        # of a phantom that we need to reject.
        if lam[1] / lam[2] > 0.85:
            return None
        n = float(np.linalg.norm(axis_dir))
        if n < 1e-12:
            return None
        axis_dir = axis_dir / n

        # Orthonormal 2D basis in the plane perpendicular to axis_dir.
        tmp = np.array([0.0, 0.0, 1.0]) if abs(axis_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u_vec = np.cross(axis_dir, tmp)
        nu = float(np.linalg.norm(u_vec))
        if nu < 1e-12:
            return None
        u_vec = u_vec / nu
        v_vec = np.cross(axis_dir, u_vec)

        x2 = rel @ u_vec
        y2 = rel @ v_vec
        # A cylindrical strip has a genuinely two-dimensional cross-section
        # in the perpendicular plane: a quarter-arc spans comparable width
        # in both directions. A strip that is effectively one-dimensional
        # in cross-section (all vertices on a line) is a flat band, not a
        # cylinder, and the Kasa circle fit below is rank-deficient on it.
        try:
            _, sv2, _ = np.linalg.svd(np.column_stack([x2, y2]), full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if sv2[0] < 1e-6:
            return None
        if sv2[1] / sv2[0] < 0.1:
            return None

        # Kasa algebraic circle fit gives a closed-form cheap estimate.
        # We then follow with a Levenberg-Marquardt geometric refinement
        # of the residuals sqrt((x-a)^2 + (y-b)^2) - r, which is what we
        # actually care about. If the geometric fit blows up (e.g. on a
        # near-line cross-section), we fall back to the algebraic answer
        # — both are reported in the RMS check below.
        A = np.column_stack([2 * x2, 2 * y2, np.ones_like(x2)])
        rhs = x2 ** 2 + y2 ** 2
        try:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        except Exception:
            return None
        a2d, b2d, c_val = sol
        r_sq = c_val + a2d * a2d + b2d * b2d
        if r_sq <= 0:
            return None
        radius_seed = float(np.sqrt(r_sq))

        from scipy.optimize import least_squares

        def _circle_residuals(params):
            ax_, bx_, rr = params
            return np.sqrt((x2 - ax_) ** 2 + (y2 - bx_) ** 2) - rr

        radius = radius_seed
        try:
            lsq = least_squares(
                _circle_residuals,
                np.array([a2d, b2d, radius_seed]),
                method="lm",
                max_nfev=200,
            )
            # Only accept the refined fit if it stays close to the
            # algebraic seed — a 10× explosion means LM wandered off.
            r_lm = float(abs(lsq.x[2]))
            if 0.25 * radius_seed <= r_lm <= 4.0 * radius_seed:
                a2d = float(lsq.x[0])
                b2d = float(lsq.x[1])
                radius = r_lm
        except Exception:
            pass
        if radius <= 0:
            return None

        # Recover the 3D axis point: lift the 2D circle center back through
        # the (u_vec, v_vec) basis relative to mean_p.
        axis_point = mean_p + a2d * u_vec + b2d * v_vec

        # RMS of perpendicular distance residuals.
        vecs = pts - axis_point
        proj_along = vecs @ axis_dir
        perp = vecs - np.outer(proj_along, axis_dir)
        perp_d = np.linalg.norm(perp, axis=1)
        rms = float(np.sqrt(np.mean((perp_d - radius) ** 2)))
        return axis_point, axis_dir, radius, rms

    for f in fitted:
        rid = f.region_id
        if rid in assigned or rid in chamfer_rids:
            continue
        if f.surface_type == SurfaceType.PLANE:
            continue  # planes are principal faces, not blends
        region = region_lookup.get(rid)
        if region is None or graph is None or not graph.has_node(rid):
            continue

        # Tangent-blend adjacency: the region must share at least two
        # near-tangent (SMOOTH or FILLET) boundaries with its neighbors.
        # Any neighbor counts — plane, cylinder, cone, whatever — because
        # a fillet's defining trait is that its boundary to the rest of
        # the part is smooth, not that those neighbors are planar.
        tangent_neighbors = 0
        for nbr in graph.neighbors(rid):
            edge_data = graph.get_edge_data(rid, nbr) or {}
            ttype = str(edge_data.get("transition_type", "")).upper()
            if ttype in ("SMOOTH", "FILLET"):
                tangent_neighbors += 1
        if tangent_neighbors < 2:
            continue

        pts = np.asarray(region.vertices, dtype=float)
        # Per-face normals for the region (defined on the full mesh).
        if cleaned is not None:
            face_normals = cleaned.normals[region.face_indices]
        else:
            face_normals = None

        # Connectivity check: a segmented region ought to be a single
        # connected patch (triangles linked via shared edges). When the
        # segmenter merges disjoint pieces of the mesh into one region
        # — e.g. the two front/back corners of a pocket — fitting a
        # single cylinder to the union is nonsense, and the resulting
        # "phantom fillet" ends up applied as a stripe that does not
        # exist in the reference mesh. We reject non-connected regions.
        if cleaned is not None and len(region.face_indices) > 1:
            edge_to_local: dict[tuple[int, int], list[int]] = defaultdict(list)
            region_faces = cleaned.faces[region.face_indices]
            for local_i, tri in enumerate(region_faces):
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
                    key = (min(int(a), int(b)), max(int(a), int(b)))
                    edge_to_local[key].append(local_i)
            adj: dict[int, set[int]] = defaultdict(set)
            for locs in edge_to_local.values():
                if len(locs) < 2:
                    continue
                for i in range(len(locs)):
                    for j in range(i + 1, len(locs)):
                        adj[locs[i]].add(locs[j])
                        adj[locs[j]].add(locs[i])
            seen: set[int] = set()
            stack = [0]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(adj[cur] - seen)
            if len(seen) != len(region.face_indices):
                logger.debug(
                    "Strategy 3: rid=%d is %d/%d connected — region "
                    "is split across multiple pieces, skipping",
                    rid, len(seen), len(region.face_indices),
                )
                continue

        fit_res = _fit_strip_as_cylinder(pts, face_normals)
        if fit_res is None:
            continue
        axis_point, axis_dir, radius, rms = fit_res
        if radius <= 0.2 or radius > bbox_diag * 0.15:
            continue
        if rms > 0.35 * radius:
            logger.debug(
                "Strategy 3: rid=%d cylinder fit rms=%.2f vs radius=%.2f — "
                "not a clean cylindrical blend, skipping",
                rid, rms, radius,
            )
            continue

        # Snap axis direction so the stored axis_point sits at the
        # midpoint of the strip along its length — makes downstream edge
        # matching more predictable.
        vecs = pts - axis_point
        along = vecs @ axis_dir
        mid = 0.5 * (float(along.max()) + float(along.min()))
        axis_point = axis_point + mid * axis_dir
        strip_length = float(along.max() - along.min())

        fillet_rids.add(rid)
        fillet_fit_lookup[rid] = {
            "radius": float(radius),
            "axis_dir": [float(x) for x in axis_dir],
            "axis_point": [float(x) for x in axis_point],
            "length": strip_length,
            "rms": rms,
        }
        logger.debug(
            "Strategy 3: rid=%d accepted as fillet (r=%.2f, rms=%.2f, "
            "len=%.1f, %d faces, %d tangent neighbors)",
            rid, radius, rms, strip_length, region.num_faces,
            tangent_neighbors,
        )

    # Build features — use the fitted cylinder params from the topology
    # pass above, which reflect the actual rolling-ball blend geometry.
    for rid in fillet_rids:
        if rid in assigned:
            continue
        region = region_lookup.get(rid)
        if region is None:
            continue
        fit = fillet_fit_lookup.get(rid)
        if fit is None:
            # Fell through without a cylinder fit — skip silently.
            continue
        results.append((
            Feature(
                type=FeatureType.FILLET,
                face_indices=region.face_indices.copy(),
                sketch_geometry={
                    "type": "arc",
                    "radius": round(fit["radius"], 3),
                    "axis_dir": [round(float(x), 4) for x in fit["axis_dir"]],
                    "axis_point": [round(float(x), 4) for x in fit["axis_point"]],
                    "length": round(fit["length"], 3),
                },
                depth=0.0,
                operation="REMOVE",
            ),
            {rid},
        ))

    for rid in chamfer_rids:
        if rid in assigned:
            continue
        region = region_lookup.get(rid)
        f = fitted_lookup.get(rid)
        if region is None or f is None:
            continue
        # Compute chamfer width from the region's bounding box
        # For countersink cones, the smallest bbox dimension is the chamfer depth
        bbox_size = region.vertices.max(axis=0) - region.vertices.min(axis=0)
        width = float(sorted(bbox_size)[0])  # smallest dimension = chamfer depth
        if width < 0.1:
            width = float(sorted(bbox_size)[1])  # fallback to second smallest
        results.append((
            Feature(
                type=FeatureType.CHAMFER,
                face_indices=region.face_indices.copy(),
                sketch_geometry={
                    "type": "bevel",
                    "width": round(width, 2),
                },
                depth=round(width, 2),
                operation="REMOVE",
            ),
            {rid},
        ))

    # Strategy 4: Stage 4.5 blend-record fallback. Any blend region that
    # passed the topology blend graph but failed Strategy 3's stricter
    # strip-cylinder fit (e.g. large 3D corner caps fit as SPHERE,
    # perimeter rolled-edge "frame" regions that wrap the whole plate)
    # would otherwise be silently absorbed into BASE_EXTRUSION at the
    # catch-all in recognize_features. Emit a FILLET / CHAMFER feature
    # using the blend record's own fitted radius/axis so the feature
    # tree matches what the user saw cyan in the blend graph.
    if topo_blends:
        for rec in topo_blends:
            rid = rec.get("region_id")
            if rid is None:
                continue
            if rid in assigned or rid in fillet_rids or rid in chamfer_rids:
                continue
            region = region_lookup.get(rid)
            if region is None:
                continue
            kind = rec.get("kind", "fillet")
            radius = float(rec.get("radius", 0.0) or 0.0)
            axis_dir = rec.get("axis_dir")
            axis_point = rec.get("axis_point")
            length = float(rec.get("length", 0.0) or 0.0)
            if kind == "fillet":
                sketch = {"type": "arc", "radius": round(radius, 3)}
                if axis_dir is not None:
                    sketch["axis_dir"] = [round(float(x), 4) for x in axis_dir]
                if axis_point is not None:
                    sketch["axis_point"] = [round(float(x), 4) for x in axis_point]
                if length > 0:
                    sketch["length"] = round(length, 3)
                results.append((
                    Feature(
                        type=FeatureType.FILLET,
                        face_indices=region.face_indices.copy(),
                        sketch_geometry=sketch,
                        depth=0.0,
                        operation="REMOVE",
                    ),
                    {rid},
                ))
            else:
                bbox_size = region.vertices.max(axis=0) - region.vertices.min(axis=0)
                width = float(sorted(bbox_size)[0])
                if width < 0.1:
                    width = float(sorted(bbox_size)[1])
                results.append((
                    Feature(
                        type=FeatureType.CHAMFER,
                        face_indices=region.face_indices.copy(),
                        sketch_geometry={
                            "type": "bevel",
                            "width": round(width, 2),
                        },
                        depth=round(width, 2),
                        operation="REMOVE",
                    ),
                    {rid},
                ))

    return results


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


def _resolve_dependencies(
    feature_rids: set[int],
    existing_features: list[Feature],
    region_lookup: dict[int, FaceRegion],
) -> list[int]:
    """Find which existing features contain faces adjacent to this feature's regions."""
    deps: set[int] = set()
    for i, feat in enumerate(existing_features):
        feat_faces = set(feat.face_indices.tolist())
        for rid in feature_rids:
            region = region_lookup.get(rid)
            if region is None:
                continue
            region_faces = set(region.face_indices.tolist())
            # Simple heuristic: if any faces are nearby in index space,
            # it's likely a dependency. More robust: use mesh adjacency.
            # For now, depend on the base and any feature that shares
            # the same face index range.
            if feat_faces & region_faces:
                deps.add(i)
    # Always depend on at least the base (index 0)
    if not deps:
        deps.add(0)
    return sorted(deps)
