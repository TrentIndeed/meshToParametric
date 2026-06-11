"""Iterative extrusion engine: prismatic-segment decomposition.

Decides how a part decomposes into a sequence of extrusions by looking at
cross-sections — never by matching volume, which can be gamed by an
averaged profile (extruding the mean cross-section reproduces the volume
of a tapered part while getting every section wrong).

Algorithm:
1. **Segment extraction** — slice the mesh at N stations along each
   principal axis. Each station yields the OUTER outlines of its
   cross-section components (interior rings are holes — features, not
   profile). Components are tracked across stations into runs: a run
   extends while each new station's component stays similar (IoU) to the
   run's FIRST outline, so a gradual taper breaks the run instead of
   averaging through it. Every maximal run is a candidate extrusion.
2. **Greedy residual filling** — voxelize the part, then repeatedly pick
   the candidate that explains the most still-unexplained volume,
   preferring the larger profile area on ties (CAD intent: replicate the
   largest single extrude area first). Stop when the best remaining gain
   falls below the noise floor (min_gain_frac of the part volume) — a
   marginal-gain criterion, not a fixed segment count.

The first segment returned is the base extrusion; the rest are joins in
descending order of explained volume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import trimesh

from mesh2param.models.mesh import CleanedMesh

logger = logging.getLogger(__name__)


@dataclass
class ExtrusionSegment:
    """One prismatic segment: profile swept along a principal axis."""

    axis_idx: int                 # 0/1/2 — principal sweep axis
    direction: np.ndarray         # unit vector along axis_idx
    start: float                  # span along the axis (world coords)
    end: float
    profile: list[tuple[float, float]]   # outer outline in (u, v) plane coords
    area: float                   # profile area
    gain_frac: float = 0.0        # fraction of part volume this segment explained
    coverage_frac: float = 0.0    # cumulative coverage after this segment

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def axes_uv(self) -> tuple[int, int]:
        """The two in-plane principal axes (u, v) for `profile` coords."""
        return tuple(i for i in range(3) if i != self.axis_idx)


@dataclass
class ExtrusionPlan:
    segments: list[ExtrusionSegment] = field(default_factory=list)
    coverage_frac: float = 0.0    # total fraction of voxels explained
    voxel_pitch: float = 0.0

    @property
    def is_single(self) -> bool:
        return len(self.segments) == 1


def plan_extrusions(
    cleaned: CleanedMesh,
    n_slices: int = 24,
    iou_thresh: float = 0.90,
    min_gain_frac: float = 0.02,
    max_segments: int = 20,
) -> ExtrusionPlan:
    """Decompose a part into a sequence of prismatic extrusions."""
    mesh = trimesh.Trimesh(
        vertices=cleaned.vertices, faces=cleaned.faces, process=False,
    )
    if not mesh.is_watertight:
        # Sections of an open shell trace wall bands, not cross-sections,
        # and the voxel fill leaks — both pillars of the analysis are
        # meaningless. Decomposed multibody pieces are UNCAPPED virtual
        # cuts and land here; the caller falls back to face-normal logic.
        logger.info("Extrusion engine: mesh not watertight, declining")
        return ExtrusionPlan()
    candidates = []
    for axis in range(3):
        candidates.extend(_candidate_runs(mesh, axis, n_slices, iou_thresh))
    candidates = _merge_step_runs(candidates)
    if not candidates:
        return ExtrusionPlan()

    occupied, pitch = _voxel_occupancy(mesh)
    if occupied is None or len(occupied) == 0:
        # Voxelization failed — fall back to the single largest-volume run.
        best = max(candidates, key=lambda c: c["area"] * (c["end"] - c["start"]))
        return ExtrusionPlan(segments=[_to_segment(best, 1.0, 1.0)])

    total = len(occupied)
    unclaimed = np.ones(total, dtype=bool)
    segments: list[ExtrusionSegment] = []
    # Precompute per-candidate voxel membership lazily (cache by id).
    membership: dict[int, np.ndarray] = {}

    while len(segments) < max_segments:
        gains = np.zeros(len(candidates), dtype=int)
        for ci, cand in enumerate(candidates):
            mem = membership.get(ci)
            if mem is None:
                mem = _voxels_in_prism(occupied, cand, pitch)
                membership[ci] = mem
            gains[ci] = int(np.count_nonzero(mem & unclaimed))
        top_gain = int(gains.max()) if len(gains) else 0
        if top_gain < min_gain_frac * total:
            break
        # Selection (user's "largest single extrude area" principle, made
        # safe by a DOMINATION filter):
        #   1. window — candidates explaining at least 40% of the best
        #      gain are contenders.
        #   2. domination — drop any contender whose residual claim is a
        #      subset of another contender's (BottomMount's inscribed
        #      rectangle is dominated by the trapezoid; the side-plate
        #      X-faces are NOT dominated by the base, they cover
        #      different material).
        #   3. area — among the survivors, extrude the largest profile
        #      first (RackMount's plus-shaped base with its four tabs
        #      beats the U-channel section).
        window = [
            ci for ci in range(len(candidates))
            if gains[ci] >= max(0.4 * top_gain, min_gain_frac * total)
        ]
        survivors = []
        for ci in window:
            mem_i = membership[ci] & unclaimed
            dominated = False
            for cj in window:
                if cj == ci:
                    continue
                if gains[cj] < gains[ci]:
                    continue
                outside = int(np.count_nonzero(mem_i & ~membership[cj]))
                if outside < 0.05 * max(int(gains[ci]), 1) and (
                    gains[cj] > gains[ci]
                    or candidates[cj]["area"] > candidates[ci]["area"]
                ):
                    dominated = True
                    break
            if not dominated:
                survivors.append(ci)
        if not survivors:
            survivors = window
        best_idx = max(survivors, key=lambda ci: candidates[ci]["area"])
        best_gain = int(gains[best_idx])
        cand = candidates[best_idx]
        # Claim removal uses a GENEROUS band so half-cell boundary crumbs
        # don't surface as phantom follow-up segments.
        unclaimed &= ~_voxels_in_prism(occupied, cand, pitch, tol_cells=1.5)
        covered = 1.0 - float(np.count_nonzero(unclaimed)) / total
        segments.append(_to_segment(cand, best_gain / total, covered))
        logger.info(
            "Extrusion segment %d: axis %d span %.1f..%.1f area=%.0f "
            "gain=%.1f%% cumulative=%.1f%%",
            len(segments), cand["axis"], cand["start"], cand["end"],
            cand["area"], 100.0 * best_gain / total, 100.0 * covered,
        )

    coverage = 1.0 - float(np.count_nonzero(unclaimed)) / total
    return ExtrusionPlan(
        segments=segments, coverage_frac=coverage, voxel_pitch=pitch,
    )


def _to_segment(cand: dict, gain_frac: float, coverage: float) -> ExtrusionSegment:
    direction = np.zeros(3)
    direction[cand["axis"]] = 1.0
    return ExtrusionSegment(
        axis_idx=cand["axis"],
        direction=direction,
        start=cand["start"],
        end=cand["end"],
        profile=[(float(x), float(y)) for x, y in cand["profile"]],
        area=float(cand["area"]),
        gain_frac=float(gain_frac),
        coverage_frac=float(coverage),
    )


def _slice_components(mesh: trimesh.Trimesh, axis: int, pos: float):
    """Outer-outline polygons of the cross-section at `pos` along `axis`.

    Interior rings (holes) are dropped — they're features, not profile,
    and letting them fragment the similarity tracking was a known failure
    of the first prototype.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    normal = np.zeros(3); normal[axis] = 1.0
    origin = np.zeros(3); origin[axis] = pos
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return []
    ax_u, ax_v = (i for i in range(3) if i != axis)
    filled = []
    try:
        discrete = section.discrete  # list of (n, 3) closed loops
    except Exception:
        return []
    for ring3d in discrete:
        p = Polygon(np.column_stack([ring3d[:, ax_u], ring3d[:, ax_v]]))
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            continue
        # Heal degenerate SLIT loops: trimesh can trace an annular
        # section (plate ring around a window) as ONE self-touching loop
        # — outer boundary plus window boundary connected through a
        # zero-width slit. Taken at face value its area is just the ring
        # band and the outline looks like a different cross-section
        # (Front Plate misread as a 3-segment stepped part). A small
        # buffer round-trip closes the slit, turning the window into a
        # proper interior ring that the exterior-fill below erases.
        p = p.buffer(0.25, join_style=2).buffer(-0.25, join_style=2)
        parts = p.geoms if p.geom_type == "MultiPolygon" else [p]
        for part in parts:
            if part.is_empty or part.area <= 1.0:
                continue
            # FILL each loop (drop interior rings) — holes are features,
            # not profile. Hole loops become disks that the union below
            # swallows into their surrounding outline.
            filled.append(Polygon(part.exterior))
    if not filled:
        return []
    merged = unary_union(filled)
    geoms = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
    return [
        Polygon(g.exterior) for g in geoms
        if not g.is_empty and g.area > 1.0
    ]


def _candidate_runs(
    mesh: trimesh.Trimesh, axis: int, n_slices: int, iou_thresh: float,
) -> list[dict]:
    """Maximal runs of similar cross-section components along one axis."""
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    extent = hi - lo
    if extent <= 0:
        return []
    # Station density must resolve the THINNEST feature, not divide the
    # extent uniformly: RackMount's 6.7mm side plates get at most one
    # station from 24 over a 140mm axis, and a run needs two. Aim for
    # ~3mm steps, clamped so small parts don't over-slice.
    n_slices = int(np.clip(extent / 3.0, n_slices, 72))
    stations = np.linspace(lo + 0.02 * extent, hi - 0.02 * extent, n_slices)
    half_step = (stations[1] - stations[0]) / 2.0 if n_slices > 1 else extent / 2

    def iou(a, b) -> float:
        try:
            inter = a.intersection(b).area
            union = a.union(b).area
            return inter / union if union > 1e-9 else 0.0
        except Exception:
            return 0.0

    max_gap_mm = max(0.08 * extent, 2.0 * 2 * half_step)
    runs: list[dict] = []          # finished candidates
    active: list[dict] = []        # {ref, best, start, last, gap}
    for pos in stations:
        comps = _slice_components(mesh, axis, float(pos))
        next_active = []
        used = [False] * len(comps)
        for run in active:
            matched = False
            for k, comp in enumerate(comps):
                # Compare against the run's FIRST outline (reference-IoU):
                # a taper drifts away from the reference and breaks the
                # run; consecutive-IoU would let it average through.
                if not used[k] and iou(run["ref"], comp) > iou_thresh:
                    used[k] = True
                    run["last"] = float(pos)
                    run["gap"] = 0
                    # Keep the LARGEST matched outline as the profile: the
                    # first station can land on a chamfered/filleted end
                    # whose inset outline under-claims the prism (a thin
                    # tube wall loses most of its voxels to a 1mm inset).
                    if comp.area > run["best"].area:
                        run["best"] = comp
                    next_active.append(run)
                    matched = True
                    break
            if not matched:
                # Bridge stations that slice through a hole or a small
                # local feature — those split/shrink the outline at a few
                # stations without ending the extrude. The budget is in
                # MILLIMETERS (a station count broke when density rose):
                # small enough that a real profile change (RackMount's
                # 62mm window region) still terminates the run, large
                # enough for bolt-hole bands. A taper never recovers
                # within the gap, so it still terminates too.
                run["gap"] += 1
                if run["gap"] * 2 * half_step <= max_gap_mm:
                    next_active.append(run)
                else:
                    runs.append(run)
        for k, comp in enumerate(comps):
            if not used[k]:
                next_active.append({
                    "ref": comp, "best": comp,
                    "start": float(pos), "last": float(pos), "gap": 0,
                })
        active = next_active
    runs.extend(active)

    out = []
    first_station = float(stations[0])
    last_station = float(stations[-1])
    for run in runs:
        if run["last"] <= run["start"]:
            continue  # single-station blip, not a prismatic span
        poly = run["best"]
        # A run reaching the terminal stations extends to the part FACE,
        # not just half a station beyond: the 2% station inset left every
        # boundary-touching segment short (RackMount's base 1.3mm thin,
        # each side plate 0.6mm — 30k mm³ missing across the part).
        start = run["start"] - half_step
        end = run["last"] + half_step
        if run["start"] <= first_station + 1e-9:
            start = lo
        if run["last"] >= last_station - 1e-9:
            end = hi
        out.append({
            "axis": axis,
            "start": start,
            "end": end,
            "profile": list(poly.exterior.coords)[:-1],
            "area": float(poly.area),
            "shapely": poly,
        })
    return out


def segment_inner_rings(
    cleaned: CleanedMesh,
    segment: ExtrusionSegment,
    min_frac: float = 0.01,
) -> list[list[tuple[float, float]]]:
    """Interior rings (windows/bores) of a segment's cross-section.

    The segment's profile is the OUTER outline only — extruding it fills
    any window in the plate (RackMount's side plates carry a 660mm²
    window each). Slice at the segment's mid-station and return interior
    rings larger than min_frac of the outer area, in the segment's (u, v)
    plane coordinates, so the export can draw the join profile with its
    holes in one sketch.
    """
    from shapely.geometry import Polygon

    mesh = trimesh.Trimesh(
        vertices=cleaned.vertices, faces=cleaned.faces, process=False,
    )
    ax_u, ax_v = segment.axes_uv
    outer = Polygon(segment.profile)
    if outer.is_empty or not outer.is_valid:
        outer = outer.buffer(0)

    def _station_rings(frac: float) -> list:
        pos = segment.start + frac * (segment.end - segment.start)
        n = np.zeros(3); n[segment.axis_idx] = 1.0
        o = np.zeros(3); o[segment.axis_idx] = pos
        section = mesh.section(plane_origin=o, plane_normal=n)
        if section is None:
            return []
        loops = []
        try:
            discrete = section.discrete
        except Exception:
            return []
        for ring3d in discrete:
            p = Polygon(np.column_stack([ring3d[:, ax_u], ring3d[:, ax_v]]))
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty and p.area > 1.0:
                loops.append(p)
        # A hole is a loop with ODD containment depth among the
        # SECTION's own loops — merely lying inside the segment outline
        # is not enough: separate pieces (a station through other
        # features) sit inside its footprint but are MATERIAL.
        rings = []
        for i, p in enumerate(loops):
            depth = sum(
                1 for j, q in enumerate(loops)
                if j != i and q.contains(p.representative_point())
                and q.area > p.area
            )
            if depth % 2 != 1:
                continue
            if p.area < min_frac * max(outer.area, 1.0):
                continue
            try:
                if outer.contains(p.representative_point()):
                    rings.append(p)
            except Exception:
                continue
        return rings

    # A baked-in window must run THROUGH the segment: require the ring
    # at three stations (it would otherwise swallow partial-depth voids —
    # RackMount's tab reliefs cut 40% out of the base when a single
    # mid-station was trusted). Partial-depth voids are pocket features,
    # not profile holes.
    station_rings = [_station_rings(f) for f in (0.25, 0.5, 0.75)]
    out = []
    for p in station_rings[1]:
        present_everywhere = all(
            any(
                p.intersection(q).area / max(p.union(q).area, 1e-9) > 0.7
                for q in others
            )
            for others in (station_rings[0], station_rings[2])
        )
        if present_everywhere:
            out.append([
                (float(x), float(y)) for x, y in p.exterior.coords[:-1]
            ])
    return out


def _merge_step_runs(cands: list[dict]) -> list[dict]:
    """Merge coaxial adjacent runs separated by a FEATURE STEP, not a taper.

    Front Plate's window opens to the bottom edge only through the middle
    of the thickness: end stations see a closed ring (a hole, dropped),
    mid stations see a concave outline — three coaxial runs for what a
    CAD operator models as ONE extrude plus cut features. The signature
    of such a step (vs a genuine taper, which must NOT merge):
      • the smaller profile is a SUBSET of the larger (the difference is
        removed material — a cut), and
      • the transition is ABRUPT (profile IoU across the boundary is low;
        a taper drifts smoothly, with boundary IoU near the run threshold).
    Merged runs take the LARGER profile over the union span; the missing
    material is the downstream cut features' job, exactly like Mesh2's
    baked corner cuts.
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
                a, b = cands[i], cands[j]
                if a["axis"] != b["axis"]:
                    continue
                # Adjacent spans (a ends where b starts, within a station)
                gap = b["start"] - a["end"]
                step = max(a["end"] - a["start"], b["end"] - b["start"])
                if not (-0.25 * step < gap < 0.25 * step):
                    continue
                try:
                    inter = a["shapely"].intersection(b["shapely"]).area
                    union = a["shapely"].union(b["shapely"]).area
                except Exception:
                    continue
                if union <= 1e-9:
                    continue
                containment = inter / min(a["area"], b["area"])
                boundary_iou = inter / union
                # Rule 1: abrupt subset step — removed material is a cut
                # feature, the extrude continues with the larger profile.
                # Rule 2: same profile resuming — two runs of the SAME
                # outline separated only because a feature interrupted
                # the track (Front Plate's outline runs re-join across
                # the merged mid span). A taper never satisfies either:
                # its halves are neither identical nor an abrupt subset.
                # The cut-step merge also needs the continuation to be a
                # substantial FRACTION of the larger profile: Front
                # Plate's mid-section is 49% of the plate (cuts), but
                # RackMount's side-plate sections are ~6% of the base
                # outline — that's a JOIN landscape, and merging would
                # model the side plates as a 60mm base extrude with 90%
                # carved away.
                area_ratio = min(a["area"], b["area"]) / max(a["area"], b["area"])
                step_cut = (
                    containment > 0.85
                    and boundary_iou < 0.85
                    and area_ratio > 0.25
                )
                same_profile = boundary_iou > 0.93
                if step_cut or same_profile:
                    bigger = a if a["area"] >= b["area"] else b
                    merged = dict(bigger)
                    merged["start"] = min(a["start"], b["start"])
                    merged["end"] = max(a["end"], b["end"])
                    if step_cut:
                        merged["step_cuts"] = True
                    cands = [
                        c for k, c in enumerate(cands) if k not in (i, j)
                    ] + [merged]
                    changed = True
                    break
            if changed:
                break
    return cands


def _voxel_occupancy(mesh: trimesh.Trimesh) -> tuple[np.ndarray | None, float]:
    """Filled-interior voxel centers of the part."""
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    pitch = diag / 90.0
    # Thin-walled parts need the pitch to resolve the WALL, not the
    # bbox: at 2.9mm pitch a 6mm web carries barely one voxel layer and
    # boundary tolerance dwarfs the interior, so wrong candidates claim
    # whole features (RackMount's X webs out-claimed the true Y side
    # profiles). Mean wall thickness ~ 2V/A.
    try:
        if mesh.area > 1e-9:
            t_wall = 2.0 * abs(float(mesh.volume)) / float(mesh.area)
            pitch = min(pitch, t_wall / 3.0)
    except Exception:
        pass
    pitch = float(np.clip(pitch, 0.8, 3.0))
    try:
        vox = mesh.voxelized(pitch).fill()
        points = np.asarray(vox.points, dtype=float)
        # Drop the SHELL: surface voxelization marks every cell touching
        # a face as occupied, so a ring of centers OUTSIDE the solid
        # (below the bottom face, beside the walls) reads as material.
        # Big-profile candidates claim that phantom ring and poison the
        # domination comparison. One binary erosion removes exactly the
        # outermost layer; thin webs (3+ cells after the wall-thickness
        # pitch rule) keep their interior layer.
        try:
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(vox.matrix)
            if eroded.any():
                idx = np.argwhere(eroded)
                points = vox.indices_to_points(idx).astype(float)
        except Exception:
            pass
        return points, pitch
    except Exception as exc:
        logger.warning("Voxelization failed (%s); falling back", exc)
        return None, pitch


def _voxels_in_prism(
    points: np.ndarray, cand: dict, pitch: float, tol_cells: float = 0.5,
) -> np.ndarray:
    """Boolean mask of voxel centers inside the candidate's swept prism.

    tol_cells scales the boundary tolerance in cell units. Gains are
    SCORED with the conservative half-cell band (a center within pitch/2
    of the wall genuinely intersects material; a full-pitch band let
    wrong candidates claim entire thin features). Claims are REMOVED
    with a generous band (1.5 cells) so the boundary crumbs a strict
    score can't reach don't surface as phantom follow-up segments.
    """
    from matplotlib.path import Path

    axis = cand["axis"]
    ax_u, ax_v = (i for i in range(3) if i != axis)
    axial = points[:, axis]
    tol = tol_cells * pitch
    # Symmetric tolerance axially and in-plane — an asymmetric band
    # systematically favored short-axial candidates with big profiles
    # over long-axial candidates with thin profiles.
    mask = (axial >= cand["start"] - tol) & (axial <= cand["end"] + tol)
    if not mask.any():
        return mask
    path = Path(np.asarray(cand["profile"], dtype=float))
    pts2 = np.column_stack([points[mask, ax_u], points[mask, ax_v]])
    # matplotlib's `radius` is SIGNED by ring winding (CW shrinks where
    # CCW grows), and the profile's winding is whatever the section
    # tracing produced. Test both signs and OR them.
    near = path.contains_points(pts2, radius=tol) | path.contains_points(
        pts2, radius=-tol,
    )
    inside = np.zeros(len(points), dtype=bool)
    inside[mask] = near
    return inside
