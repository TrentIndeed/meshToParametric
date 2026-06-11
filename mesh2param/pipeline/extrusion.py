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
    candidates = []
    for axis in range(3):
        candidates.extend(_candidate_runs(mesh, axis, n_slices, iou_thresh))
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
        best_idx, best_gain = -1, 0
        for ci, cand in enumerate(candidates):
            mem = membership.get(ci)
            if mem is None:
                mem = _voxels_in_prism(occupied, cand, pitch)
                membership[ci] = mem
            gain = int(np.count_nonzero(mem & unclaimed))
            if gain > best_gain or (
                gain == best_gain and best_idx >= 0
                and cand["area"] > candidates[best_idx]["area"]
            ):
                best_gain, best_idx = gain, ci
        if best_idx < 0 or best_gain < min_gain_frac * total:
            break
        cand = candidates[best_idx]
        unclaimed &= ~membership[best_idx]
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

    normal = np.zeros(3); normal[axis] = 1.0
    origin = np.zeros(3); origin[axis] = pos
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return []
    ax_u, ax_v = (i for i in range(3) if i != axis)
    loops = []
    try:
        discrete = section.discrete  # list of (n, 3) closed loops
    except Exception:
        return []
    for ring3d in discrete:
        p = Polygon(np.column_stack([ring3d[:, ax_u], ring3d[:, ax_v]]))
        if p.is_valid and p.area > 1.0:
            loops.append(p)
    # Outer outlines have even containment depth (0 = outline, 1 = hole,
    # 2 = island inside a hole, ...). trimesh's own classification needs
    # rtree; loop counts here are tiny so O(n²) shapely containment is fine.
    out = []
    for i, p in enumerate(loops):
        depth = sum(
            1 for j, q in enumerate(loops)
            if j != i and q.contains(p.representative_point()) and q.area > p.area
        )
        if depth % 2 == 0:
            out.append(p)
    return out


def _candidate_runs(
    mesh: trimesh.Trimesh, axis: int, n_slices: int, iou_thresh: float,
) -> list[dict]:
    """Maximal runs of similar cross-section components along one axis."""
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    extent = hi - lo
    if extent <= 0:
        return []
    stations = np.linspace(lo + 0.02 * extent, hi - 0.02 * extent, n_slices)
    half_step = (stations[1] - stations[0]) / 2.0 if n_slices > 1 else extent / 2

    def iou(a, b) -> float:
        try:
            inter = a.intersection(b).area
            union = a.union(b).area
            return inter / union if union > 1e-9 else 0.0
        except Exception:
            return 0.0

    max_gap = 2  # stations a run may bridge without a match
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
                # Bridge stations that slice straight through a hole —
                # the bore splits the outline into halves that fail IoU,
                # but holes are features, not profile breaks. A taper
                # never recovers within the gap, so it still terminates.
                run["gap"] += 1
                if run["gap"] <= max_gap:
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
    for run in runs:
        if run["last"] <= run["start"]:
            continue  # single-station blip, not a prismatic span
        poly = run["best"]
        out.append({
            "axis": axis,
            "start": run["start"] - half_step,
            "end": run["last"] + half_step,
            "profile": list(poly.exterior.coords)[:-1],
            "area": float(poly.area),
            "shapely": poly,
        })
    return out


def _voxel_occupancy(mesh: trimesh.Trimesh) -> tuple[np.ndarray | None, float]:
    """Filled-interior voxel centers of the part."""
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    pitch = float(np.clip(diag / 90.0, 0.8, 3.0))
    try:
        vox = mesh.voxelized(pitch).fill()
        return np.asarray(vox.points, dtype=float), pitch
    except Exception as exc:
        logger.warning("Voxelization failed (%s); falling back", exc)
        return None, pitch


def _voxels_in_prism(points: np.ndarray, cand: dict, pitch: float) -> np.ndarray:
    """Boolean mask of voxel centers inside the candidate's swept prism."""
    from matplotlib.path import Path

    axis = cand["axis"]
    ax_u, ax_v = (i for i in range(3) if i != axis)
    axial = points[:, axis]
    mask = (axial >= cand["start"] - pitch) & (axial <= cand["end"] + pitch)
    if not mask.any():
        return mask
    path = Path(np.asarray(cand["profile"], dtype=float))
    inside = np.zeros(len(points), dtype=bool)
    inside[mask] = path.contains_points(
        np.column_stack([points[mask, ax_u], points[mask, ax_v]]),
        radius=pitch,  # tolerance: voxel centers within half a cell of the wall count
    )
    return inside
