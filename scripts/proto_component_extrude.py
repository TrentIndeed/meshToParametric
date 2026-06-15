"""PROTOTYPE: connected-component shape-tracking extrusion decomposition.

Validates the redesign before touching pipeline/extrusion.py:
  - slice every ~1mm along an axis
  - connected components (separate polygons) per slice
  - track each component across slices by SHAPE SIMILARITY (Hu-moment +
    area + centroid), NOT whole-section IoU and NOT volume
  - every persistent component-run = one extrusion (count them)

Target: AI1 -> 10 objects (main plate + top plate + 4 top screws + 4 bottom
screws). No volume metric anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh2param.config import IngestConfig  # noqa: E402
from mesh2param.pipeline.ingest import ingest_mesh  # noqa: E402


def slice_components(mesh, axis, pos):
    """Outer-outline polygons of the cross-section at pos along axis."""
    from shapely.ops import unary_union
    n = np.zeros(3); n[axis] = 1.0
    o = np.zeros(3); o[axis] = pos
    section = mesh.section(plane_origin=o, plane_normal=n)
    if section is None:
        return []
    ax_u, ax_v = (i for i in range(3) if i != axis)
    filled = []
    try:
        discrete = section.discrete
    except Exception:
        return []
    for ring3d in discrete:
        p = Polygon(np.column_stack([ring3d[:, ax_u], ring3d[:, ax_v]]))
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            continue
        p = p.buffer(0.25, join_style=2).buffer(-0.25, join_style=2)
        for part in (p.geoms if p.geom_type == "MultiPolygon" else [p]):
            if not part.is_empty and part.area > 1.0:
                filled.append(Polygon(part.exterior))
    if not filled:
        return []
    merged = unary_union(filled)
    geoms = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
    return [Polygon(g.exterior) for g in geoms if not g.is_empty and g.area > 1.0]


def shape_signature(poly: Polygon) -> dict:
    """Translation-invariant shape descriptor (NO volume): centroid, area,
    normalized side-extents, and a coarse Hu-like log-moment vector."""
    c = np.array(poly.centroid.coords[0])
    xy = np.array(poly.exterior.coords)[:-1] - c
    area = poly.area
    ext = xy.max(0) - xy.min(0)
    return {"centroid": c, "area": area, "ext": ext}


def same_track(a: dict, b: dict, scale: float) -> bool:
    """PHASE 1 continuity: is component b the same TRACK as a, one slice on?
    Purely SPATIAL (centroid lock) — NO area gate, so a domed screw whose area
    triples stays one track, while a corner screw never binds to the centre
    plate (different centroid). Area discontinuities are handled later, in
    phase 2, so the bottom-plate->main-plate STEP is not split here."""
    rad = 0.5 * (np.mean(a["ext"]) + np.mean(b["ext"]))
    cen_tol = max(0.6 * rad, 0.03 * scale)
    return bool(np.linalg.norm(a["centroid"] - b["centroid"]) <= cen_tol)


def split_track_at_steps(track: list, tol: float = 0.15,
                         min_plateau: float = 0.6) -> list:
    """PHASE 2: segment a raw track into objects by AREA PLATEAUS.

    An extrusion has a sustained near-constant cross-section. A screw HEAD
    tapers continuously (no clean step to cut on), so step-splitting shatters
    it; but it still has ONE sustained plateau (its shank/widest band) then a
    taper. Two STACKED plates show TWO plateaus separated by a jump. So:
      - find maximal plateaus (area within +/-tol of the running median for at
        least min_plateau of axial length),
      - one object per plateau (its representative = widest slice),
      - a track with NO plateau (a pure cone) is still one object.
    Ramps (the tapered tip, the step between plateaus) are absorbed into the
    nearest plateau via the span midpoints. Returns list of sub-tracks.
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
    # Assign every slice to the nearest plateau; object span = those slices.
    centers = [0.5 * (pos[a] + pos[b]) for a, b in plateaus]
    groups = [[] for _ in plateaus]
    for k, e in enumerate(track):
        g = int(np.argmin([abs(pos[k] - cc) for cc in centers]))
        groups[g].append(e)
    return [g for g in groups if g]


def track_axis(mesh, axis, step=0.5, min_span_frac=0.04):
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    extent = hi - lo
    # Dense uniform slicing, but resolve the THINNEST stacked feature: a 6mm
    # axis with 10 stacked objects needs sub-mm steps. Cap station count.
    n = int(np.clip(extent / step, 20, 600))
    real_step = (hi - lo) / n
    min_span = max(min_span_frac * extent, 1.0)
    # Bridge a small axial gap so a one-slice section miss (a hole band, a
    # numerical dropout) doesn't end a track; large gaps (the empty middle of
    # a corner between bottom and top screws) still break it into two tracks.
    max_gap = max(2.5 * real_step, 0.4)
    stations = np.linspace(lo + 0.01 * extent, hi - 0.01 * extent, n)
    tracks = []   # finished raw tracks: list of {pos, sig}
    active = []   # {seq: [{pos,sig}], gap: float}
    for pos in stations:
        comps = [shape_signature(p) for p in slice_components(mesh, axis, float(pos))]
        used = [False] * len(comps)
        nxt = []
        for tr in active:
            last = tr["seq"][-1]["sig"]
            m = -1
            for k, s in enumerate(comps):
                if not used[k] and same_track(last, s, diag):
                    m = k
                    break
            if m >= 0:
                used[m] = True
                tr["seq"].append({"pos": pos, "sig": comps[m]})
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
                nxt.append({"seq": [{"pos": pos, "sig": s}], "gap": 0.0})
        active = nxt
    tracks.extend(tr["seq"] for tr in active)

    # Phase 2: split each raw track at area discontinuities, emit objects.
    objs = []
    for tr in tracks:
        for sub in split_track_at_steps(tr):
            start, last = sub[0]["pos"], sub[-1]["pos"]
            if (last - start) < min_span:
                continue
            biggest = max(sub, key=lambda e: e["sig"]["area"])["sig"]
            objs.append({"start": start, "last": last, "sig": biggest})
    return objs


def main(paths):
    for path in paths:
        print(f"\n=== {path} ===")
        cleaned = ingest_mesh(path, IngestConfig(remesh=False))
        mesh = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces,
                               process=False)
        bb = mesh.bounds[1] - mesh.bounds[0]
        print(f"  bbox={bb.round(1)} watertight={mesh.is_watertight}")
        for axis in range(3):
            objs = track_axis(mesh, axis)
            print(f"  axis={axis}: {len(objs)} persistent objects")
            for o in sorted(objs, key=lambda r: r["start"]):
                print(f"     span={o['start']:7.2f}..{o['last']:7.2f} "
                      f"area={o['sig']['area']:8.1f} "
                      f"ext={o['sig']['ext'].round(1)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["models/AI1/MountingPlate.3mf"])
