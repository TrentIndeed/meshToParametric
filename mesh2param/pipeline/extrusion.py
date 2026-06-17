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


@dataclass
class ExtrusionGroup:
    """A set of identical extrusions modelled as ONE feature.

    Two extrudes belong together when they share an axis, profile, and depth:
    either co-planar on the SAME span (RackMount's two side plates, AI1's four
    top screws — one sketch, several profile loops) or a mirror/pattern of the
    same wall. The group carries the representative profile once and the per-
    instance (start, end) span + in-plane centre, so the export emits a single
    sketch+extrude feature instead of N copies.
    """

    axis_idx: int
    direction: np.ndarray
    profile: list[tuple[float, float]]
    area: float
    members: list[ExtrusionSegment] = field(default_factory=list)
    same_plane: bool = False      # all members share the start plane


def group_extrusion_segments(
    plan: ExtrusionPlan, area_tol: float = 0.06, len_tol: float = 0.2,
) -> list[ExtrusionGroup]:
    """Collapse identical extrusions into one feature each.

    Two-stage, matching how a CAD operator sketches:
    1. CO-PLANAR sets — same axis + profile + depth AND the same start plane
       become one sketch with several loops (AI1's four TOP screws share a
       slice; its four BOTTOM screws share a different slice -> two features,
       not eight).
    2. MIRROR pairs — two SINGLE walls with the same profile/depth on parallel
       planes are a symmetric pair modelled as one feature (RackMount's two
       side plates). Restricting this to singletons keeps the four-loop screw
       sketches on their own planes as separate features.
    """
    def _akey(s):
        return (s.axis_idx, round(s.area, 1), round(s.length, 1))

    # Stage 1: co-planar grouping (start plane included in the key).
    groups: list[ExtrusionGroup] = []
    for seg in sorted(plan.segments, key=lambda s: -s.area):
        placed = False
        for g in groups:
            if g.axis_idx != seg.axis_idx:
                continue
            if abs(g.area - seg.area) / max(g.area, seg.area, 1e-9) > area_tol:
                continue
            gl = g.members[0].length
            if abs(gl - seg.length) / max(gl, seg.length, 1e-9) > len_tol:
                continue
            if abs(g.members[0].start - seg.start) > len_tol * max(gl, 1e-9):
                continue  # different sketch slice -> different feature
            g.members.append(seg)
            placed = True
            break
        if not placed:
            groups.append(ExtrusionGroup(
                axis_idx=seg.axis_idx, direction=seg.direction,
                profile=seg.profile, area=seg.area, members=[seg],
            ))

    # Stage 2: merge SINGLETON mirror pairs (identical lone walls on parallel
    # planes). Multi-loop co-planar sets (the 4-screw sketches) are left alone.
    merged: list[ExtrusionGroup] = []
    used = [False] * len(groups)
    for i, g in enumerate(groups):
        if used[i]:
            continue
        if len(g.members) == 1:
            for j in range(i + 1, len(groups)):
                if used[j] or len(groups[j].members) != 1:
                    continue
                h = groups[j]
                if (h.axis_idx == g.axis_idx
                        and abs(h.area - g.area) / max(h.area, g.area, 1e-9) <= area_tol
                        and abs(h.members[0].length - g.members[0].length)
                        / max(g.members[0].length, 1e-9) <= len_tol):
                    g.members.append(h.members[0])
                    used[j] = True
        used[i] = True
        merged.append(g)

    for g in merged:
        starts = [m.start for m in g.members]
        g.same_plane = (max(starts) - min(starts)) <= len_tol * g.members[0].length
    return merged


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
        # Claim removal uses a GENEROUS in-plane band so lateral wall
        # crumbs don't surface as phantom follow-up segments, but a TIGHT
        # axial band so a thin level's claim doesn't reach into the stacked
        # neighbour above/below it (see _voxels_in_prism).
        unclaimed &= ~_voxels_in_prism(
            occupied, cand, pitch, tol_cells=1.5, axial_tol_cells=0.5,
        )
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


def _reaxis_to_base(seg: ExtrusionSegment, base_axis: int) -> ExtrusionSegment:
    """Re-express a body that sweeps PERPENDICULAR to the base plate as a
    BASE-AXIS extrude: sketch its footprint on the base plane and extrude along
    the base normal. A wall sketched on its own side face (extruded sideways)
    unions badly with the base — its outer face fuses perpendicular to the base
    plate and can't be filleted. Re-extruded along the base axis, the same wall
    is a clean box whose outer edge is a 2D sketch corner (rounds in-profile).

    The footprint is the body's axis-aligned extent projected onto the base
    plane — exact for a rectangular wall; for a J/L wall it's the bounding box
    (the small flange notch is the trade for a clean, filletable outer edge).
    Interior rings (the wall's through-holes) are dropped here and re-emitted as
    tangential cuts downstream. No-op when the body already sweeps the base
    axis.
    """
    if seg.axis_idx == base_axis:
        return seg
    P = np.asarray(seg.profile, dtype=float)
    if len(P) < 3:
        return seg
    u, v = seg.axes_uv
    ext = {
        seg.axis_idx: (float(seg.start), float(seg.end)),
        u: (float(P[:, 0].min()), float(P[:, 0].max())),
        v: (float(P[:, 1].min()), float(P[:, 1].max())),
    }
    nonbase = [i for i in range(3) if i != base_axis]   # ascending = (u, v)
    nu, nv = nonbase[0], nonbase[1]
    (u0, u1), (v0, v1) = ext[nu], ext[nv]
    s0, s1 = ext[base_axis]
    nd = np.zeros(3)
    nd[base_axis] = 1.0
    w, h = u1 - u0, v1 - v0
    # ROUND the footprint corners. Re-axised, the footprint corners become the
    # wall's VERTICAL edges; a human sketches them rounded so they fillet
    # in-profile. A sharp footprint leaves those edges sharp and the post-build
    # fillet can't roll across them (the long cap edge fillets, the short
    # vertical edges stay sharp). Radius = the part's whole-mm fillet, capped
    # below half the thin wall so the round never self-intersects.
    r = min(2.0, 0.45 * min(w, h))
    if r >= 0.3:
        prof = _rounded_rect_profile(u0, u1, v0, v1, r)
    else:
        prof = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    return ExtrusionSegment(
        axis_idx=base_axis,
        direction=nd,
        start=s0,
        end=s1,
        profile=prof,
        area=abs(w * h),
        gain_frac=seg.gain_frac,
        coverage_frac=seg.coverage_frac,
    )


def _rounded_rect_profile(u0, u1, v0, v1, r, n=8):
    """Outline of an axis-aligned rectangle [u0,u1]x[v0,v1] with all four
    corners rounded to radius r, sampled n points per corner arc. The arc
    samples let `_regularize_outline` recover the exact tangent fillets so the
    export draws true corner arcs (not chords)."""
    r = float(min(r, 0.5 * (u1 - u0), 0.5 * (v1 - v0)))
    # corner centres: (cu, cv), arc start-angle (CCW), in CCW outline order
    corners = [
        (u1 - r, v0 + r, -np.pi / 2),  # bottom-right
        (u1 - r, v1 - r, 0.0),         # top-right
        (u0 + r, v1 - r, np.pi / 2),   # top-left
        (u0 + r, v0 + r, np.pi),       # bottom-left
    ]
    pts: list[tuple[float, float]] = []
    for cu, cv, a0 in corners:
        for k in range(n + 1):
            a = a0 + (np.pi / 2) * (k / n)
            pts.append((float(cu + r * np.cos(a)), float(cv + r * np.sin(a))))
    return pts


def _slice_components(mesh: trimesh.Trimesh, axis: int, pos: float):
    """Outer-outline polygons of the cross-section at `pos` along `axis`.

    Interior rings (holes) are dropped — they're features, not profile,
    and letting them fragment the similarity tracking was a known failure
    of the first prototype.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    normal = np.zeros(3)
    normal[axis] = 1.0
    origin = np.zeros(3)
    origin[axis] = pos
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
        n = np.zeros(3)
        n[segment.axis_idx] = 1.0
        o = np.zeros(3)
        o[segment.axis_idx] = pos
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
            if not p.is_empty and p.area > 0.1:
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
            # Absolute floor (not relative to the outer): small mounting bores
            # — AI meshes make them ~0.4 mm radius — are a tiny fraction of a
            # big plate but are real through-holes. The 3-station persistence
            # check below is what actually rejects noise voids.
            if p.area < 0.12:
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
    # Per-axis part extent, taken from the runs themselves: a run reaching
    # a part FACE was snapped to the mesh bound in _candidate_runs, so the
    # min start / max end per axis is the true extent. Used to tell a
    # terminal STEP (an exposed thinner top/bottom face = added material,
    # a boss) from an interior CUT (the reduced section is bracketed by the
    # full profile on both sides — a window/pocket). Computed once: merges
    # only ever UNION spans, so the extent never grows.
    axis_lo: dict[int, float] = {}
    axis_hi: dict[int, float] = {}
    for c in cands:
        ax = c["axis"]
        axis_lo[ax] = min(axis_lo.get(ax, c["start"]), c["start"])
        axis_hi[ax] = max(axis_hi.get(ax, c["end"]), c["end"])

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
                # Terminal-step guard: a is below b (adjacent, a.end~b.start).
                # If the SMALLER run's outer face (the one away from the
                # larger run) is an exposed part boundary, the reduction is
                # added material — a thinner second plate / boss — not a cut.
                # CAD intent is base extrude + boss extrude, so keep them
                # separate (the smaller run survives to become a JOIN). Only
                # an INTERIOR reduction (Front Plate's window, bracketed by
                # full profile on both axial sides) is a true step-cut.
                ax = a["axis"]
                ext = max(axis_hi[ax] - axis_lo[ax], 1e-9)
                term_tol = 0.02 * ext
                smaller = a if a["area"] < b["area"] else b
                larger = b if smaller is a else a
                if smaller["start"] >= larger["end"] - term_tol:
                    terminal_step = abs(smaller["end"] - axis_hi[ax]) <= term_tol
                elif smaller["end"] <= larger["start"] + term_tol:
                    terminal_step = abs(smaller["start"] - axis_lo[ax]) <= term_tol
                else:
                    terminal_step = False
                step_cut = (
                    containment > 0.85
                    and boundary_iou < 0.85
                    and area_ratio > 0.25
                    and not terminal_step
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
    return _occupancy_at_pitch(mesh, pitch), pitch


def _occupancy_at_pitch(mesh: trimesh.Trimesh, pitch: float) -> np.ndarray | None:
    """Filled-interior voxel centers at an explicit pitch (shell removed)."""
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
        return points
    except Exception as exc:
        logger.warning("Voxelization failed (%s); falling back", exc)
        return None


def _voxels_in_prism(
    points: np.ndarray, cand: dict, pitch: float, tol_cells: float = 0.5,
    axial_tol_cells: float | None = None,
) -> np.ndarray:
    """Boolean mask of voxel centers inside the candidate's swept prism.

    tol_cells scales the IN-PLANE boundary tolerance in cell units. Gains
    are SCORED with the conservative half-cell band (a center within
    pitch/2 of the wall genuinely intersects material; a full-pitch band
    let wrong candidates claim entire thin features). Claims are REMOVED
    with a generous in-plane band (1.5 cells) so the lateral wall crumbs a
    strict score can't reach don't surface as phantom follow-up segments.

    axial_tol_cells overrides the band ALONG THE EXTRUSION AXIS (defaults
    to tol_cells). The claim path keeps this TIGHT even while the in-plane
    band is generous: a generous AXIAL band reaches past a thin level into
    the stacked neighbour above/below it (AI1's three ~2mm Y-levels — a
    base plate, a second plate, four screws — collapsed to one segment
    when claiming the middle plate erased both neighbours). The lateral
    crumb-clearing that motivated 1.5 cells is purely in-plane, so
    tightening only the axial band keeps it without the bleed.
    """
    from matplotlib.path import Path

    axis = cand["axis"]
    ax_u, ax_v = (i for i in range(3) if i != axis)
    axial = points[:, axis]
    tol = tol_cells * pitch
    axial_tol = (tol_cells if axial_tol_cells is None else axial_tol_cells) * pitch
    mask = (axial >= cand["start"] - axial_tol) & (axial <= cand["end"] + axial_tol)
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


# ---------------------------------------------------------------------------
# Component-tracking planner (v2)
# ---------------------------------------------------------------------------
# A different decomposition that finds the extrusion COUNT from cross-section
# SHAPE, never from volume-similarity. It tracks each connected component of
# the cross-section across densely-spaced slices and segments each track at
# AREA PLATEAUS — so a part like AI1 (a plate, a smaller plate, and eight
# corner screw heads) resolves into its 10 real extrusions instead of one
# over-extruded slab. Volume is used ONLY at the end, to dedup objects that
# the same material produces on different axes (a slab is "extrudable" on all
# three; coverage picks the one that owns the material). See the module
# docstring for the original volume-greedy engine, still the default.


def _comp_sig(poly) -> dict:
    """Translation-invariant component descriptor (NO volume): centroid,
    area, in-plane extents, and the outline coords (the candidate profile)."""
    c = np.array(poly.centroid.coords[0], dtype=float)
    xy = np.array(poly.exterior.coords)[:-1]
    ext = xy.max(0) - xy.min(0) if len(xy) else np.zeros(2)
    return {
        "centroid": c,
        "area": float(poly.area),
        "ext": ext.astype(float),
        "coords": [(float(x), float(y)) for x, y in poly.exterior.coords[:-1]],
    }


def _same_track(a: dict, b: dict, scale: float) -> bool:
    """Is component b the continuation of track a, one slice on? Purely
    SPATIAL — a component's centroid barely moves along its extrusion axis
    (a corner screw stays in its corner), so the lock tolerates a domed
    screw whose area triples without ever binding it to the centre plate.
    Area discontinuities are handled by the plateau segmentation, not here."""
    rad = 0.5 * (float(np.mean(a["ext"])) + float(np.mean(b["ext"])))
    cen_tol = max(0.6 * rad, 0.03 * scale)
    return bool(np.linalg.norm(a["centroid"] - b["centroid"]) <= cen_tol)


def _segment_track_plateaus(
    track: list, tol: float = 0.15, min_plateau: float = 0.6,
) -> list:
    """Segment one raw track into objects by AREA PLATEAUS.

    An extrusion has a sustained near-constant cross-section. A screw HEAD
    tapers continuously (no clean step to cut on) but still has ONE sustained
    plateau then a taper; two STACKED plates show TWO plateaus separated by a
    jump. So: find maximal plateaus (area within +/-tol of the running median
    for at least min_plateau of axial length), one object per plateau, and a
    track with NO plateau (a pure cone) stays one object. The ramps (tapered
    tip, the step between plateaus) are absorbed by assigning every slice to
    its nearest plateau centre. Returns a list of sub-tracks.
    """
    if len(track) < 2:
        return [track]
    areas = [e["sig"]["area"] for e in track]
    pos = [e["pos"] for e in track]
    plateaus = []  # (start_idx, end_idx) inclusive
    i, n = 0, len(track)
    while i < n:
        j = i
        ref = areas[i]
        while j + 1 < n:
            a2 = areas[j + 1]
            if abs(a2 - ref) / max(ref, a2, 1e-9) > tol:
                break
            j += 1
            ref = float(np.median(areas[i:j + 1]))
        if pos[j] - pos[i] >= min_plateau:
            plateaus.append((i, j))
        i = j + 1 if j > i else i + 1
    if not plateaus:
        return [track]
    # Merge an interior-CUT plateau: a strict local minimum in area bracketed
    # by larger plateaus of similar area is removed material (Front Plate's
    # window — full / reduced-middle / full), not a stacked level. Absorb it
    # into one object with the larger profile (the window becomes a downstream
    # cut). A terminal step (AI1's bottom plate -> bigger main plate) is NOT
    # bracketed, so it still splits.
    pa = [float(np.median([areas[i] for i in range(a, b + 1)])) for a, b in plateaus]
    parent = list(range(len(plateaus)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x, y):
        parent[_find(x)] = _find(y)

    for k in range(1, len(plateaus) - 1):
        sim = abs(pa[k - 1] - pa[k + 1]) / max(pa[k - 1], pa[k + 1], 1e-9) < 0.1
        if pa[k] < pa[k - 1] and pa[k] < pa[k + 1] and sim:
            _union(k - 1, k)
            _union(k, k + 1)
    # Group plateaus by merged root, then assign every slice to its nearest
    # plateau and bucket by that plateau's group.
    centers = [0.5 * (pos[a] + pos[b]) for a, b in plateaus]
    roots = [_find(k) for k in range(len(plateaus))]
    order, seen = [], {}
    for r in roots:
        if r not in seen:
            seen[r] = len(order)
            order.append(r)
    groups = [[] for _ in order]
    for k in range(n):
        p = int(np.argmin([abs(pos[k] - cc) for cc in centers]))
        groups[seen[roots[p]]].append(track[k])
    return [g for g in groups if g]


def _track_axis_components(
    mesh: trimesh.Trimesh, axis: int, max_slices: int = 400,
    min_step: float = 0.1, min_span_frac: float = 0.02,
) -> list[dict]:
    """All candidate extrusion objects along one axis, by component tracking.

    Slices at sub-millimetre resolution (capped at max_slices stations),
    tracks each connected component into a raw track, then plateau-segments
    each track. Returns candidate dicts: {axis, start, end, profile, area,
    centroid}. Volume is NOT consulted here.
    """
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    extent = hi - lo
    if extent <= 0:
        return []
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    # Sub-mm slicing so small features (AI1's ~1.5mm screws, ~0.6mm plateaus)
    # are resolved; capped so a long axis stays tractable.
    n = int(np.clip(extent / min_step, 30, max_slices))
    real_step = extent / n
    min_span = max(min_span_frac * extent, 2.0 * real_step)
    min_plateau = max(min_span, 0.5)
    # Bridge a one-slice section dropout (a hole band) but break the big
    # empty middle between a bottom screw and the top screw above it.
    max_gap = max(2.5 * real_step, 0.4)
    stations = np.linspace(lo + 0.01 * extent, hi - 0.01 * extent, n)

    tracks: list[list] = []
    active: list[dict] = []
    for pos in stations:
        comps = [_comp_sig(p) for p in _slice_components(mesh, axis, float(pos))]
        used = [False] * len(comps)
        nxt = []
        for tr in active:
            last = tr["seq"][-1]["sig"]
            m = -1
            for k, s in enumerate(comps):
                if not used[k] and _same_track(last, s, diag):
                    m = k
                    break
            if m >= 0:
                used[m] = True
                tr["seq"].append({"pos": float(pos), "sig": comps[m]})
                tr["gap"] = 0.0
                nxt.append(tr)
            else:
                tr["gap"] += real_step
                if tr["gap"] <= max_gap:
                    nxt.append(tr)
                else:
                    tracks.append(tr["seq"])
        for k, s in enumerate(comps):
            if not used[k]:
                nxt.append({"seq": [{"pos": float(pos), "sig": s}], "gap": 0.0})
        active = nxt
    tracks.extend(tr["seq"] for tr in active)

    out = []
    half = 0.5 * real_step
    for tr in tracks:
        for sub in _segment_track_plateaus(tr, min_plateau=min_plateau):
            widest = max(sub, key=lambda e: e["sig"]["area"])["sig"]
            # Span = the PLATEAU CORE (slices within half the plateau area),
            # not every assigned slice. A wall's X-section drops SHARPLY to the
            # base-only strip past its faces; with no competing plateau there
            # those base slices get assigned to the wall and over-extend its
            # extrude into the base (RackMount: 7.2mm vs 6mm -> 17% over-build).
            # Clipping to the core stops the wall at the junction. It only
            # trims a screw's thin tapering TIP (tiny, and the cylinder
            # approximation of a dome is actually closer without it). A window
            # dip sits in the MIDDLE, so the full-area end slices still bound
            # the span.
            wa = float(widest["area"])
            core = [e["pos"] for e in sub if e["sig"]["area"] >= 0.5 * wa]
            start, end = (min(core), max(core)) if core else (
                sub[0]["pos"], sub[-1]["pos"])
            if (end - start) < min_span:
                continue
            # Drift: the change in cross-section across the span's INTERIOR
            # (the 25% and 75% stations), NOT its end faces. ~0 for a true
            # prism; high only for a monotonic draft/taper that changes through
            # the whole body (BottomMount's big face). Sampling the interior
            # makes it immune to FILLETS, which only round the cross-section
            # over a short ramp at an END (a bottom-edge fillet, or the seam
            # between two stacked plates) — those ramps are absorbed into the
            # plateau and must NOT read as a draft that flips the axis. A real
            # draft still shows between the quartiles; a fillet does not.
            sub_areas = [e["sig"]["area"] for e in sub]
            n_sa = len(sub_areas)
            w = max(1, n_sa // 10)
            lo_i, hi_i = n_sa // 4, (3 * n_sa) // 4
            a_lo = float(np.median(sub_areas[max(0, lo_i - w):lo_i + w + 1]))
            a_hi = float(np.median(sub_areas[max(0, hi_i - w):hi_i + w + 1]))
            drift = abs(a_hi - a_lo) / max(a_hi, a_lo, 1e-9)
            # Extend the span to the part face at the ends (the slice inset
            # and plateau-centre assignment leave it a half-step short).
            s0 = lo if start <= stations[0] + 1e-9 else start - half
            e0 = hi if end >= stations[-1] - 1e-9 else end + half
            out.append({
                "axis": axis,
                "start": float(s0),
                "end": float(e0),
                "profile": _simplify_profile(widest["coords"], diag),
                "area": float(widest["area"]),
                "centroid": widest["centroid"],
                "drift": float(drift),
            })
    return out


def _symmetrize_rect_corners(coords: list, arc_samples: int = 18) -> list:
    """Rebuild an axis-aligned rounded rectangle with ALL corners at the
    measured radius.

    An AI/scan mesh samples some corners densely (the arc survives
    simplification) and others sparsely (the arc collapses to a sharp 90°), so
    the base profile comes out a rectangle with, say, two rounded corners and
    two sharp ones — the user's "one side is sharp for no reason". When the
    outline is clearly a rectangle with a MAJORITY of corners rounded at a
    consistent radius and at least one left sharp, rebuild it cleanly with every
    corner rounded. Leaves non-rectangles, fully-sharp rectangles, and
    already-symmetric rounded rectangles unchanged.
    """
    import numpy as np
    P = np.asarray(coords, dtype=float)
    if len(P) < 8:
        return coords
    umin, umax = float(P[:, 0].min()), float(P[:, 0].max())
    vmin, vmax = float(P[:, 1].min()), float(P[:, 1].max())
    W, H = umax - umin, vmax - vmin
    if W < 2.0 or H < 2.0:
        return coords
    # Rectangularity: most of the perimeter must run along an axis.
    seg = np.roll(P, -1, axis=0) - P
    L = np.linalg.norm(seg, axis=1)
    d = seg / (L[:, None] + 1e-9)
    near_axis = np.maximum(np.abs(d[:, 0]), np.abs(d[:, 1])) > np.cos(np.radians(12.0))
    if L.sum() <= 0 or float(L[near_axis].sum() / L.sum()) < 0.7:
        return coords
    # CRITICAL GUARD: only rebuild a TRUE rectangle — the profile must nearly
    # FILL its bounding box. A plate with U-tabs, mounting ears, notches or
    # slots fills much less of its bbox, and rebuilding it as a plain rounded
    # rectangle would ERASE those features (this regressed Mesh5's U-tabs).
    # A rounded rectangle only loses the four little corner triangles, so it
    # stays ≥ ~95% of the bbox.
    try:
        from shapely.geometry import Polygon as _SPoly
        prof_area = abs(_SPoly([(float(a), float(b)) for a, b in P]).buffer(0).area)
    except Exception:
        return coords
    if W * H <= 0 or prof_area / (W * H) < 0.93:
        return coords
    # Window tight enough to capture a corner's arc + its two tangent points
    # but exclude the far ends of the straight sides.
    win = 0.3 * min(W, H)
    corners = [(umax, vmin), (umax, vmax), (umin, vmax), (umin, vmin)]
    radii, sharp_count = [], 0
    for cu, cv in corners:
        m = (np.abs(P[:, 0] - cu) < win) & (np.abs(P[:, 1] - cv) < win)
        pts = P[m]
        if len(pts) < 3:
            sharp_count += 1
            continue
        # Fillet radius ~ how far the straight sides END from the bbox corner
        # (the tangent insets), i.e. the MAX inset along each axis within the
        # corner window — not the nearest arc sample.
        du = np.abs(pts[:, 0] - cu)
        dv = np.abs(pts[:, 1] - cv)
        r_est = min(float(du.max()), float(dv.max()))
        if r_est > 0.3:
            radii.append(r_est)
        else:
            sharp_count += 1
    rounded = [r for r in radii if r > 0.3]
    if len(rounded) < 2 or sharp_count == 0:
        return coords  # fully sharp, fully rounded, or ambiguous — leave alone
    r = min(float(np.median(rounded)), 0.45 * min(W, H))
    if r < 0.3:
        return coords
    out: list = []

    def _arc(cx, cy, a0, a1):
        for s in range(arc_samples + 1):
            a = a0 + (a1 - a0) * s / arc_samples
            out.append((float(cx + r * np.cos(a)), float(cy + r * np.sin(a))))

    out.append((umin + r, vmin))
    out.append((umax - r, vmin))
    _arc(umax - r, vmin + r, -np.pi / 2, 0.0)       # bottom-right
    out.append((umax, vmax - r))
    _arc(umax - r, vmax - r, 0.0, np.pi / 2)         # top-right
    out.append((umin + r, vmax))
    _arc(umin + r, vmax - r, np.pi / 2, np.pi)       # top-left
    out.append((umin, vmin + r))
    _arc(umin + r, vmin + r, np.pi, 1.5 * np.pi)     # bottom-left
    clean: list = []
    for pt in out:
        if not clean or abs(pt[0] - clean[-1][0]) > 1e-6 or abs(pt[1] - clean[-1][1]) > 1e-6:
            clean.append(pt)
    logger.info("Rect-corner symmetrize: rebuilt rounded rectangle r=%.2f "
                "(%d rounded, %d sharp corners)", r, len(rounded), sharp_count)
    return clean


def _simplify_profile(coords: list, diag: float) -> list[tuple[float, float]]:
    """Clean a raw slice outline into a sketchable profile.

    The widest-slice polygon of a dirty AI/scan mesh carries hundreds of jagged
    vertices and can self-intersect — extruding it produces an invalid face and
    a near-empty solid. buffer(0) heals self-intersections; simplify drops the
    noise while preserving corners and curve sampling (a screw circle keeps
    enough points to stay round)."""
    from shapely.geometry import Polygon
    p = Polygon(coords)
    if not p.is_valid:
        p = p.buffer(0)
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
    p = p.simplify(max(0.003 * diag, 0.15), preserve_topology=True)
    if p.is_empty or p.geom_type != "Polygon":
        return [(float(x), float(y)) for x, y in coords]
    simplified = [(float(x), float(y)) for x, y in p.exterior.coords[:-1]]
    # Regularize into a true CAD sketch: axis-aligned straight sides meeting at
    # exact 90-deg corners, with a tangent arc at each rounded corner. Self-
    # gating — returns None for a round/organic outline, so a screw circle
    # stays a circle while a wavy plate outline becomes a clean rectangle.
    try:
        from mesh2param.pipeline.regularize import regularize_profile_2d
        reg = regularize_profile_2d(simplified)
        if reg and len(reg) >= 3:
            return _symmetrize_rect_corners(
                [(float(x), float(y)) for x, y in reg])
    except Exception:
        pass
    # Even when regularize bails (it rejects densely-sampled rounded outlines),
    # symmetrize the corners so a half-sharp rectangle becomes a clean rounded
    # one instead of extruding with one square side.
    return _symmetrize_rect_corners(simplified)


def plan_extrusions_components(
    cleaned: CleanedMesh, max_slices: int = 400, min_step: float = 0.1,
    keep_self_frac: float = 0.5,
) -> ExtrusionPlan:
    """Component-tracking + plateau decomposition with coverage-based
    cross-axis dedup. Separate from `plan_extrusions` (the default); switch
    over only once validated against the reference meshes.

    Validated against every reference mesh, matching the default engine's
    decomposition while finding AI1's stack (no volume used to match shapes):
    AI1 -> 10 (2 plates + 8 screws), Mesh1/3/4 -> 1, Mesh5 -> 3, and
    BottomMount (a DRAFTED plate) -> 1 on its constant-section long axis (the
    full-prism rule below picks the clean trapezoid axis over the drifting big
    face). Identical extrudes can be collapsed into one feature each with
    `group_extrusion_segments` (RackMount's two side plates, AI1's screws)."""
    mesh = trimesh.Trimesh(
        vertices=cleaned.vertices, faces=cleaned.faces, process=False,
    )
    if not mesh.is_watertight:
        logger.info("Component planner: mesh not watertight, declining")
        return ExtrusionPlan()

    candidates: list[dict] = []
    for axis in range(3):
        candidates.extend(
            _track_axis_components(mesh, axis, max_slices, min_step)
        )
    if not candidates:
        return ExtrusionPlan()

    # Cross-axis dedup needs voxels FINE enough to separate a thin boss/screw
    # from the plate it sits on (the default diag/90 pitch shares one voxel
    # layer between them). Resolve the THINNEST candidate object with ~3
    # voxels, but cap the grid by a memory budget so a big part — whose
    # features are correspondingly bigger — uses a coarser-but-adequate pitch.
    spans = [c["end"] - c["start"] for c in candidates if c["end"] > c["start"]]
    thin = min(spans) if spans else float(np.min(mesh.extents))
    bbox_vol = float(np.prod(mesh.extents)) or 1.0
    budget_pitch = (bbox_vol / 3_000_000.0) ** (1.0 / 3.0)
    dedup_pitch = float(np.clip(max(thin / 3.0, budget_pitch), 0.2, 3.0))
    occupied = _occupancy_at_pitch(mesh, dedup_pitch)
    pitch = dedup_pitch
    if occupied is None or len(occupied) == 0:
        # No volume to dedup against — return the candidates as-is, biggest
        # profile first (degenerate fallback; coverage is the normal path).
        candidates.sort(key=lambda c: -c["area"])
        segs = [_to_segment(c, 0.0, 0.0) for c in candidates]
        return ExtrusionPlan(segments=segs)

    total = len(occupied)
    # Precompute each candidate's own prism membership once. The axial band is
    # TIGHT: a screw abutting a plate's top face must not have its prism bleed
    # DOWN into the plate (those plate voxels are claimed and would deflate the
    # screw's self-coverage below the keep threshold). In-plane stays at the
    # half-cell scoring band so the thin circle is captured.
    for c in candidates:
        c["_mem"] = _voxels_in_prism(
            occupied, c, pitch, tol_cells=0.5, axial_tol_cells=0.1,
        )
    counts = [int(np.count_nonzero(c["_mem"])) or 1 for c in candidates]
    cover_frac = [n / total for n in counts]

    # FULL-PRISM preference (drafted/single-block parts). When ONE candidate
    # explains nearly the whole part by itself, it is the single correct
    # extrude and every other candidate is a redundant view of the same
    # material. A drafted plate (BottomMount) is cleanly prismatic only along
    # its constant-section axis (the trapezoid swept the 213mm length covers
    # ~100% as one object); its drifting big face FRAGMENTS, so no single
    # candidate there reaches the bar — the clean axis wins for free, lowest
    # drift breaking ties. This is INERT for genuinely multi-body parts (AI1's
    # stack, RackMount's base+plates) where NO single prism covers the part,
    # so they fall through to the per-object pass below unchanged.
    FULL = 0.92
    full = [i for i, f in enumerate(cover_frac) if f >= FULL]
    if full:
        # Lowest drift wins (the constant-section axis over a drafted face);
        # ties go to the LARGEST profile (sketch the biggest face, extrude the
        # short way), then highest coverage. So an edge-filleted plate — zero
        # interior drift on every axis — picks its big flat face, the fillet
        # handled downstream.
        best = min(full, key=lambda i: (round(candidates[i].get("drift", 0.0), 3),
                                        -candidates[i]["area"], -cover_frac[i]))
        bc = candidates[best]
        kept = [bc]
        for i, c in enumerate(candidates):
            if i == best:
                continue
            inter = int(np.count_nonzero(c["_mem"] & bc["_mem"]))
            if inter / counts[i] < 0.8:   # material the full prism doesn't own
                kept.append(c)
        chosen = [(c, cover_frac[candidates.index(c)], 0.0) for c in kept]
        segments = [_to_segment(c, gf, 1.0) for c, gf, _ in chosen]
        return ExtrusionPlan(segments=segments, coverage_frac=1.0,
                             voxel_pitch=pitch)

    # Cross-axis dedup by SELF-COVERAGE, largest profile first (sketch the
    # biggest face, extrude the short way — the base, then bosses, then
    # screws). A candidate is KEPT iff most of ITS OWN prism is still
    # unclaimed: that drops the REDUNDANT phantom slices (a slab is extrudable
    # on all three axes; its edge-on views are already claimed by the chosen
    # face) WITHOUT dropping SMALL real objects (a screw owns its own column —
    # nothing else covers it — so it survives no matter how little of the part
    # it is). Volume is consulted ONLY for this redundancy test, never to
    # match shapes or to gate on a part-fraction threshold.
    # Selection by self-coverage, largest profile first (sketch the biggest
    # face, extrude the short way). We run the SAME claim loop with two orders
    # and keep the LESS-FRAGMENTED result:
    #   - area-first  : the original order.
    #   - base-axis-first : PREFER the base-plate's axis. A human sketches on
    #     (parallel to) the base plate and extrudes along its normal; a body
    #     whose sketch plane is TANGENTIAL/perpendicular to the base — a wall
    #     sketched on its own side face and extruded sideways — unions badly
    #     with the base (the fused outer face can't be filleted) and is NOT how
    #     it's modelled. Processing base-axis candidates first lets a
    #     base-parallel body claim before a perpendicular VIEW of the same wall.
    # Keeping whichever yields FEWER segments honours the base-axis preference
    # WHENEVER it's clean, yet falls back to the perpendicular extrude when the
    # base-axis view FRAGMENTS a wall — a wall's through-holes disconnect its
    # base-axis slices into pieces, so its clean perpendicular run must not be
    # traded for several base-axis fragments. base axis = largest-profile axis.
    base_axis = max(candidates, key=lambda c: c["area"])["axis"]

    def _select(order_key):
        unclaimed = np.ones(total, dtype=bool)
        chosen = []
        for c in sorted(candidates, key=order_key):
            own = int(np.count_nonzero(c["_mem"]))
            if own == 0:
                continue
            new = int(np.count_nonzero(c["_mem"] & unclaimed))
            if new / own < keep_self_frac:
                continue  # redundant — its material already belongs to another
            # Claim PRECISELY the material this object explains: tight axial
            # band so a plate's claim does NOT bleed up into the domed bases of
            # the screws abutting its top face.
            unclaimed &= ~_voxels_in_prism(
                occupied, c, pitch, tol_cells=0.5, axial_tol_cells=0.1,
            )
            cov = 1.0 - float(np.count_nonzero(unclaimed)) / total
            chosen.append((c, new / total, cov))
        return chosen, 1.0 - float(np.count_nonzero(unclaimed)) / total

    area_chosen, area_cov = _select(lambda c: -c["area"])
    base_chosen, base_cov = _select(
        lambda c: (c["axis"] != base_axis, -c["area"]))
    # Prefer base-axis only when it covers comparably AND is no more fragmented.
    if base_cov >= area_cov - 1e-6 and len(base_chosen) <= len(area_chosen):
        chosen, coverage = base_chosen, base_cov
    else:
        chosen, coverage = area_chosen, area_cov

    segments = [_to_segment(c, gf, cov) for c, gf, cov in chosen]
    # Re-express any body that still sweeps PERPENDICULAR to the base plate as a
    # base-axis box (footprint on the base plane). The base (segment 0, the
    # largest profile) defines the base axis and is untouched; a wall the
    # selector kept as a sideways extrude becomes a clean base-axis box so its
    # outer edge is a fillet-able 2D sketch corner rather than a fused
    # perpendicular face. Its through-holes are re-emitted as tangential cuts
    # downstream (recognize).
    if segments:
        base_ax = segments[0].axis_idx
        segments = [segments[0]] + [
            _reaxis_to_base(s, base_ax) for s in segments[1:]
        ]
    return ExtrusionPlan(
        segments=segments, coverage_frac=coverage, voxel_pitch=pitch,
    )
