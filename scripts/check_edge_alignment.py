"""Edge-alignment check: do the rebuilt solid's edges sit on the mesh?

Before matching fillets to the rebuilt object, we need confidence that
the rebuilt solid's edges (the extrusion boundaries) actually line up
with the original mesh's surface and sharp creases. This script:

  1. Builds the rackmount CAD solid (multi-body, no fillets applied).
  2. Enumerates every B-rep edge of the rebuilt solid.
  3. For each edge, samples points along it and measures the distance
     to the nearest mesh surface point — an edge that lies ON the mesh
     has all its sample points within a small tolerance.
  4. Reports per-edge alignment and an overall summary.

Also runs the reverse check: detect the mesh's sharp edges (creases
between segmented regions) and confirm each has a rebuilt edge nearby,
so we catch mesh edges the rebuild FAILED to reproduce.

Usage:
    python scripts/check_edge_alignment.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.disable(logging.CRITICAL)

from scipy.spatial import cKDTree

from mesh2param.config import (
    DecomposeConfig,
    FitConfig,
    IngestConfig,
    SegmentConfig,
    TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.export_cadquery import export_cadquery
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.topology import build_topology

ON_MESH_TOL = 1.0   # mm: a sample point this close to the mesh counts as "on"
N_SAMPLES = 11      # points sampled along each edge


def _sample_surface(vertices: np.ndarray, faces: np.ndarray, per_face: int = 3) -> np.ndarray:
    """Dense point cloud on the mesh surface: vertices + face-interior
    samples, so the nearest-point query measures distance to the
    surface, not just to the nearest vertex."""
    pts = [vertices]
    tri = vertices[faces]  # (F, 3, 3)
    # barycentric samples
    for _ in range(per_face):
        w = np.random.default_rng(0).dirichlet([1, 1, 1], size=len(faces))
        pts.append((tri * w[:, :, None]).sum(axis=1))
    # edge midpoints
    pts.append((tri[:, 0] + tri[:, 1]) * 0.5)
    pts.append((tri[:, 1] + tri[:, 2]) * 0.5)
    pts.append((tri[:, 2] + tri[:, 0]) * 0.5)
    return np.vstack(pts)


def run() -> int:
    mesh_path = ROOT / "models" / "Mesh5" / "RackMount.3mf"
    cleaned = ingest_mesh(str(mesh_path), IngestConfig(remesh=False))
    decomp = decompose_bodies(cleaned, DecomposeConfig())
    body_results = []
    for body in decomp.bodies:
        sub = extract_sub_mesh(cleaned, body.face_indices, body=body)
        regions = segment_mesh(sub, SegmentConfig())
        fitted = fit_surfaces(regions, sub, FitConfig())
        graph, edges = build_topology(regions, fitted, sub, TopologyConfig())
        tree = recognize_features(regions, fitted, graph, edges, sub)
        body_results.append((body, sub, tree))

    tree0 = body_results[0][2]
    res = export_cadquery(
        tree0, cleaned, use_llm=False, num_variants=1,
        body_results=body_results if decomp.is_multi_body else None,
    )
    if res.solid is None:
        print("No solid produced — cannot check edges.")
        return 1

    # Mesh surface point cloud for distance queries.
    surf = _sample_surface(cleaned.vertices, cleaned.faces)
    surf_tree = cKDTree(surf)

    # --- Forward: every rebuilt edge, how well does it lie on the mesh? ---
    import cadquery as cq  # noqa: F401

    edges = res.solid.edges().vals()
    print("=" * 78)
    print(f"REBUILT SOLID has {len(edges)} edges")
    print("=" * 78)

    on, partial, off = 0, 0, 0
    rows = []
    for ei, edge in enumerate(edges):
        try:
            gtype = edge.geomType()
        except Exception:
            gtype = "?"
        try:
            sp = edge.startPoint(); ep = edge.endPoint()
            start = np.array([sp.x, sp.y, sp.z])
            end = np.array([ep.x, ep.y, ep.z])
        except Exception:
            continue
        length = float(np.linalg.norm(end - start))
        ts = np.linspace(0, 1, N_SAMPLES)
        samples = start[None, :] * (1 - ts)[:, None] + end[None, :] * ts[:, None]
        d, _ = surf_tree.query(samples)
        max_d = float(d.max())
        mean_d = float(d.mean())
        frac_on = float(np.mean(d < ON_MESH_TOL))
        if frac_on >= 0.99:
            on += 1
            status = "ON"
        elif frac_on >= 0.5:
            partial += 1
            status = "PARTIAL"
        else:
            off += 1
            status = "OFF"
        rows.append({
            "ei": ei, "type": gtype, "len": length,
            "max_d": max_d, "mean_d": mean_d, "frac_on": frac_on,
            "status": status, "mid": 0.5 * (start + end),
        })

    print(f"  ON mesh   (>=99% samples within {ON_MESH_TOL}mm): {on}")
    print(f"  PARTIAL   (50-99%): {partial}")
    print(f"  OFF mesh  (<50%): {off}")

    # --- Recognizability: are the imperfect edges still CLOSE enough
    # that a tolerant matcher would snap them to the right mesh edge? ---
    # Two views: absolute mean deviation, and deviation as a % of the
    # edge's own length. The part diagonal sets the "10% of scale" bar.
    diag = float(np.linalg.norm(
        cleaned.bbox[1] - cleaned.bbox[0]
    ))
    pct10_scale = 0.10 * diag
    print()
    print(f"  RECOGNIZABILITY (part diagonal={diag:.0f}mm, 10% of scale={pct10_scale:.1f}mm)")
    abs_buckets = [0.5, 1.0, 2.0, 3.0, 5.0, 1e9]
    abs_labels = ["<=0.5mm", "<=1mm", "<=2mm", "<=3mm", "<=5mm", ">5mm"]
    counts = [0] * len(abs_buckets)
    rel_within_10 = 0
    rel_within_10_meaningful = 0
    n_meaningful = 0
    worst = []
    for r in rows:
        md = r["mean_d"]
        for bi, thr in enumerate(abs_buckets):
            if md <= thr:
                counts[bi] += 1
                break
        if r["len"] > 2.0:
            n_meaningful += 1
            if md <= 0.10 * r["len"]:
                rel_within_10_meaningful += 1
        if md <= 0.10 * max(r["len"], 1e-6):
            rel_within_10 += 1
        worst.append((md, r))
    total = len(rows)
    print("    by absolute mean deviation:")
    cum = 0
    for lbl, c in zip(abs_labels, counts):
        cum += c
        print(f"      {lbl:>7}: {c:>4}  (cumulative {cum:>4}/{total} = {100*cum/total:.0f}%)")
    print(f"    within 10% of own edge length (len>2mm edges): "
          f"{rel_within_10_meaningful}/{n_meaningful} "
          f"({100*rel_within_10_meaningful/max(n_meaningful,1):.0f}%)")
    within3 = sum(1 for r in rows if r["mean_d"] <= 3.0)
    print(f"    >>> within 3mm of the mesh (snap-recognizable): "
          f"{within3}/{total} ({100*within3/total:.0f}%)")
    print()
    print("  OFF / PARTIAL edges (these don't lie on the mesh):")
    print(f"    {'edge':>4} {'type':<8} {'len':>7} {'maxd':>7} {'meand':>7} {'on%':>5}  midpoint")
    for r in rows:
        if r["status"] == "ON":
            continue
        m = r["mid"]
        print(
            f"    {r['ei']:>4} {r['type']:<8} {r['len']:>6.1f} "
            f"{r['max_d']:>6.1f} {r['mean_d']:>6.1f} {r['frac_on']*100:>4.0f}  "
            f"({m[0]:>6.1f},{m[1]:>6.1f},{m[2]:>6.1f})"
        )

    # --- Reverse: mesh sharp edges, does the rebuild reproduce them? ---
    # Mesh sharp edges = triangle edges whose two adjacent faces have a
    # large dihedral angle. Build the rebuilt-edge point cloud and check
    # each mesh sharp-edge sample for a nearby rebuilt edge.
    rebuilt_pts = []
    for edge in edges:
        try:
            sp = edge.startPoint(); ep = edge.endPoint()
            s = np.array([sp.x, sp.y, sp.z]); e = np.array([ep.x, ep.y, ep.z])
            ts = np.linspace(0, 1, 9)
            rebuilt_pts.append(s[None] * (1 - ts)[:, None] + e[None] * ts[:, None])
        except Exception:
            continue
    rebuilt_cloud = np.vstack(rebuilt_pts) if rebuilt_pts else np.zeros((1, 3))
    rebuilt_tree = cKDTree(rebuilt_cloud)

    sharp_mid, sharp_cov = _mesh_sharp_edge_coverage(
        cleaned.vertices, cleaned.faces, rebuilt_tree,
    )
    print()
    print("=" * 78)
    print("REVERSE CHECK: mesh sharp edges (creases) reproduced by rebuild?")
    print("=" * 78)
    print(f"  {len(sharp_mid)} sharp mesh edges sampled")
    covered = int(np.sum(sharp_cov < 2.0))
    print(f"  covered by a rebuilt edge within 2mm: {covered}/{len(sharp_mid)} "
          f"({100.0*covered/max(len(sharp_mid),1):.0f}%)")
    uncovered = [(m, c) for m, c in zip(sharp_mid, sharp_cov) if c >= 2.0]
    if uncovered:
        print(f"  UNCOVERED sharp edges (mesh crease with NO nearby rebuilt edge):")
        for m, c in sorted(uncovered, key=lambda x: -x[1])[:25]:
            print(f"    dist={c:>5.1f}mm  midpoint=({m[0]:>6.1f},{m[1]:>6.1f},{m[2]:>6.1f})")
    print("=" * 78)
    print(f"mesh_vol={res.mesh_volume:.0f} cad_vol={res.cad_volume:.0f} "
          f"err={res.volume_error_pct:.2f}%")
    return 0


def _mesh_sharp_edge_coverage(vertices, faces, rebuilt_tree, dihedral_deg=35.0):
    """Find mesh edges with a sharp dihedral angle, return their
    midpoints and the distance from each to the nearest rebuilt edge."""
    # Build edge -> adjacent face map.
    from collections import defaultdict
    face_normals = _face_normals(vertices, faces)
    edge_faces: dict[tuple, list[int]] = defaultdict(list)
    for fi, f in enumerate(faces):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_faces[(min(a, b), max(a, b))].append(fi)
    cos_thresh = np.cos(np.deg2rad(180.0 - dihedral_deg))
    mids = []
    for (a, b), fl in edge_faces.items():
        if len(fl) != 2:
            continue
        n0, n1 = face_normals[fl[0]], face_normals[fl[1]]
        c = float(np.dot(n0, n1))
        # sharp if normals diverge a lot (dihedral far from flat)
        if c < np.cos(np.deg2rad(dihedral_deg)):
            mids.append(0.5 * (vertices[a] + vertices[b]))
    if not mids:
        return np.zeros((0, 3)), np.zeros((0,))
    mids = np.array(mids)
    d, _ = rebuilt_tree.query(mids)
    return mids, d


def _face_normals(vertices, faces):
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    return n / ln


if __name__ == "__main__":
    sys.exit(run())
