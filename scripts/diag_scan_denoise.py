"""Develop + validate an edge-preserving scan denoise.

Objective validation WITHOUT a scan ground truth: take a finely-tessellated
box (known flat faces, sharp edges, corners), add Gaussian vertex noise to
fake a scan, denoise, and measure how close the result returns to the clean
original. Topology is unchanged (noise moves vertices only), so the error is
a direct per-vertex distance, split by vertex class:
  - FLAT-interior verts  -> did we remove surface noise?
  - EDGE verts           -> did we ROUND the sharp edges? (the failure mode)
  - CORNER verts         -> did we keep the corners?

A good edge-preserving denoise drops flat error a lot while keeping edge /
corner error low; naive Laplacian drops flat error but rounds edges/corners.

Usage: python scripts/diag_scan_denoise.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh


# --------------------------------------------------------------------------
# Vectorized bilateral normal-filtering denoise (Zheng et al.)
# --------------------------------------------------------------------------
def _face_neighbors(mesh: trimesh.Trimesh):
    """(F,K) padded neighbor-face index + bool mask, from 1-ring edge adj."""
    F = len(mesh.faces)
    adj = mesh.face_adjacency
    lists: list[list[int]] = [[] for _ in range(F)]
    for a, b in adj:
        lists[a].append(b)
        lists[b].append(a)
    K = max((len(x) for x in lists), default=1) or 1
    idx = np.zeros((F, K), dtype=np.int64)
    msk = np.zeros((F, K), dtype=bool)
    for i, x in enumerate(lists):
        if x:
            idx[i, :len(x)] = x
            msk[i, :len(x)] = True
    return idx, msk


def _vertex_faces(mesh: trimesh.Trimesh):
    """(V,K) padded incident-face index + count, for the vertex update."""
    V = len(mesh.vertices)
    lists: list[list[int]] = [[] for _ in range(V)]
    for fi, tri in enumerate(mesh.faces):
        for v in tri:
            lists[v].append(fi)
    K = max((len(x) for x in lists), default=1) or 1
    idx = np.zeros((V, K), dtype=np.int64)
    msk = np.zeros((V, K), dtype=bool)
    for i, x in enumerate(lists):
        if x:
            idx[i, :len(x)] = x
            msk[i, :len(x)] = True
    return idx, msk


def bilateral_denoise(mesh: trimesh.Trimesh, normal_iter: int = 6,
                      vertex_iter: int = 12, sigma_s: float = 0.3,
                      sigma_c: float | None = None) -> trimesh.Trimesh:
    """Bilateral normal filtering then vertex refit. sigma_s (on the normal
    difference, ~radians) is the edge-preservation knob: small -> normals
    across a sharp edge barely mix, so the edge survives while same-surface
    noise averages out (no faceting — the soft range weight replaces the hard
    feature-angle threshold)."""
    V = np.asarray(mesh.vertices, float).copy()
    F = np.asarray(mesh.faces)
    nb, nbm = _face_neighbors(mesh)
    vf, vfm = _vertex_faces(mesh)
    vf_cnt = vfm.sum(1).clip(min=1)[:, None]

    cur = trimesh.Trimesh(V, F, process=False)
    cents = cur.triangles_center
    n = cur.face_normals.copy()
    areas = cur.area_faces
    if sigma_c is None:
        el = np.linalg.norm(cents[mesh.face_adjacency[:, 0]]
                            - cents[mesh.face_adjacency[:, 1]], axis=1)
        sigma_c = float(np.mean(el)) or 1.0

    # 1) bilateral normal filtering (iterate 1-ring to widen influence)
    for _ in range(normal_iter):
        cg, ng, ag = cents[nb], n[nb], areas[nb]
        dc = np.linalg.norm(cg - cents[:, None, :], axis=2)
        dn = np.linalg.norm(ng - n[:, None, :], axis=2)
        w = (ag * np.exp(-(dc ** 2) / (2 * sigma_c ** 2))
             * np.exp(-(dn ** 2) / (2 * sigma_s ** 2))) * nbm
        acc = (w[..., None] * ng).sum(1) + n
        ln = np.linalg.norm(acc, axis=1, keepdims=True)
        n = np.where(ln > 1e-12, acc / ln, n)

    # 2) vertex update to satisfy the filtered normals
    for _ in range(vertex_iter):
        cents = trimesh.Trimesh(V, F, process=False).triangles_center
        cf, nf = cents[vf], n[vf]
        proj = (nf * (cf - V[:, None, :])).sum(2)
        disp = (nf * (proj[..., None]) * vfm[..., None]).sum(1) / vf_cnt
        V = V + disp
    return trimesh.Trimesh(V, F, process=False)


def laplacian_denoise(mesh: trimesh.Trimesh, iters: int = 12) -> trimesh.Trimesh:
    V = np.asarray(mesh.vertices, float).copy()
    F = np.asarray(mesh.faces)
    vf, vfm = _vertex_faces(mesh)  # reuse for a simple umbrella smooth
    # uniform Laplacian via vertex 1-ring (edges)
    edges = trimesh.Trimesh(V, F, process=False).edges_unique
    from collections import defaultdict
    nb = defaultdict(list)
    for a, b in edges:
        nb[a].append(b)
        nb[b].append(a)
    K = max(len(v) for v in nb.values())
    idx = np.zeros((len(V), K), np.int64)
    msk = np.zeros((len(V), K), bool)
    for i in range(len(V)):
        x = nb.get(i, [])
        idx[i, :len(x)] = x
        msk[i, :len(x)] = True
    cnt = msk.sum(1).clip(min=1)[:, None]
    for _ in range(iters):
        avg = (V[idx] * msk[..., None]).sum(1) / cnt
        V = V + 0.5 * (avg - V)
    return trimesh.Trimesh(V, F, process=False)


# --------------------------------------------------------------------------
# Synthetic testbed: finely-tessellated box with analytic vertex classes
# --------------------------------------------------------------------------
def make_box(extent=(60.0, 40.0, 20.0), n=24) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=extent)
    for _ in range(int(np.log2(n))):
        b = b.subdivide()
    return b


def classify_box_vertices(mesh, extent):
    """How many of a vertex's coords sit on the box boundary -> 1 flat, 2
    edge, 3 corner."""
    half = np.asarray(extent) / 2.0
    onb = np.abs(np.abs(mesh.vertices) - half) < 1e-6
    k = onb.sum(1)
    return k == 1, k == 2, k == 3   # flat, edge, corner


def evaluate(noise_frac=0.01):
    extent = (60.0, 40.0, 20.0)
    clean = make_box(extent, n=24)
    flat, edge, corner = classify_box_vertices(clean, extent)
    diag = float(np.linalg.norm(clean.bounds[1] - clean.bounds[0]))
    sigma = noise_frac * float(min(extent))
    # deterministic pseudo-Gaussian noise (no Date/random dep)
    t = np.arange(len(clean.vertices) * 3) * 0.123457
    g = (np.sin(t) + np.sin(2.7 * t + 1.1) + np.sin(6.1 * t + 0.3)) / np.sqrt(3)
    noisy = trimesh.Trimesh(clean.vertices + g.reshape(-1, 3) * sigma,
                            clean.faces, process=False)

    print(f"  box {extent} verts={len(clean.vertices)} faces={len(clean.faces)}"
          f"  flat={flat.sum()} edge={edge.sum()} corner={corner.sum()}"
          f"  noise sigma={sigma:.3f}mm")

    def err(m):
        d = np.linalg.norm(m.vertices - clean.vertices, axis=1)
        return d[flat].mean(), d[edge].mean(), d[corner].mean(), d.mean()

    print(f"  {'method':<22}{'flat':>9}{'edge':>9}{'corner':>9}{'overall':>9}")
    fe = err(noisy)
    print(f"  {'noisy (input)':<22}{fe[0]:9.4f}{fe[1]:9.4f}{fe[2]:9.4f}{fe[3]:9.4f}")
    for name, m in (("laplacian", laplacian_denoise(noisy)),
                    ("bilateral ss=0.20", bilateral_denoise(noisy, sigma_s=0.20)),
                    ("bilateral ss=0.30", bilateral_denoise(noisy, sigma_s=0.30)),
                    ("bilateral ss=0.50", bilateral_denoise(noisy, sigma_s=0.50))):
        e = err(m)
        print(f"  {name:<22}{e[0]:9.4f}{e[1]:9.4f}{e[2]:9.4f}{e[3]:9.4f}")


def main() -> int:
    for nf in (0.01, 0.02):
        print(f"\n=== noise_frac={nf} ===")
        t = time.time()
        evaluate(nf)
        print(f"  ({time.time() - t:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
