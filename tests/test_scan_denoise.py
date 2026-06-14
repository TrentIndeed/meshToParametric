"""Edge-preserving scan denoise regression.

Validated with ground truth (no scan reference needed): a finely-tessellated
box with known flat / edge / corner vertices is noised, then denoised. The
bilateral normal-filtering denoise must remove surface noise while KEEPING
edges and corners — where a naive Laplacian rounds them — and must NOT facet
a smooth curved surface (the pymeshlab two-step's failure mode).
"""
from __future__ import annotations

import numpy as np
import trimesh

from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import _bilateral_denoise


def _det_noise(n_vertices: int, sigma: float) -> np.ndarray:
    """Deterministic pseudo-Gaussian vertex noise (no RNG dependency)."""
    t = np.arange(n_vertices * 3) * 0.123457
    g = (np.sin(t) + np.sin(2.7 * t + 1.1) + np.sin(6.1 * t + 0.3)) / np.sqrt(3)
    return g.reshape(-1, 3) * sigma


def _box(extent=(60.0, 40.0, 20.0), subdiv=4):
    b = trimesh.creation.box(extents=extent)
    for _ in range(subdiv):
        b = b.subdivide()
    half = np.asarray(extent) / 2.0
    on_boundary = np.abs(np.abs(b.vertices) - half) < 1e-6
    k = on_boundary.sum(1)
    return b, (k == 1), (k == 2), (k == 3)  # mesh, flat, edge, corner


def _uniform_laplacian(mesh, iters=12):
    V = np.asarray(mesh.vertices, float).copy()
    edges = mesh.edges_unique
    nb = [[] for _ in range(len(V))]
    for a, b in edges:
        nb[a].append(b)
        nb[b].append(a)
    for _ in range(iters):
        nxt = V.copy()
        for i, ns in enumerate(nb):
            if ns:
                nxt[i] = V[i] + 0.5 * (V[ns].mean(0) - V[i])
        V = nxt
    return trimesh.Trimesh(V, mesh.faces, process=False)


def _err(m, clean, mask):
    return float(np.linalg.norm(m.vertices - clean.vertices, axis=1)[mask].mean())


def test_bilateral_reduces_flat_noise():
    clean, flat, edge, corner = _box()
    noisy = trimesh.Trimesh(clean.vertices + _det_noise(len(clean.vertices), 0.2),
                            clean.faces, process=False)
    out = _bilateral_denoise(noisy, IngestConfig(denoise_sigma_s=0.3))
    assert _err(out, clean, flat) < _err(noisy, clean, flat)


def test_bilateral_preserves_edges_and_corners():
    clean, flat, edge, corner = _box()
    noisy = trimesh.Trimesh(clean.vertices + _det_noise(len(clean.vertices), 0.2),
                            clean.faces, process=False)
    out = _bilateral_denoise(noisy, IngestConfig(denoise_sigma_s=0.3))
    # Edges/corners must not be ROUNDED: error stays near the input noise,
    # never blown up the way smoothing does.
    assert _err(out, clean, edge) < _err(noisy, clean, edge) * 1.2
    assert _err(out, clean, corner) < _err(noisy, clean, corner) * 1.2


def test_bilateral_beats_laplacian_on_edges():
    clean, flat, edge, corner = _box()
    noisy = trimesh.Trimesh(clean.vertices + _det_noise(len(clean.vertices), 0.2),
                            clean.faces, process=False)
    bil = _bilateral_denoise(noisy, IngestConfig(denoise_sigma_s=0.3))
    lap = _uniform_laplacian(noisy)
    # Laplacian rounds edges/corners hard; bilateral keeps them — expect a
    # large margin (Laplacian edge error was ~25x bilateral in validation).
    assert _err(bil, clean, edge) < _err(lap, clean, edge) * 0.5
    assert _err(bil, clean, corner) < _err(lap, clean, corner) * 0.5


def test_bilateral_does_not_facet_curves():
    """On a smooth cylinder, denoise must not invent sharp creases."""
    cyl = trimesh.creation.cylinder(radius=10.0, height=40.0, sections=72)
    noisy = trimesh.Trimesh(cyl.vertices + _det_noise(len(cyl.vertices), 0.15),
                            cyl.faces, process=False)

    def frac_sharp(m):
        adj = m.face_adjacency
        n = m.face_normals
        d = np.clip((n[adj[:, 0]] * n[adj[:, 1]]).sum(1), -1, 1)
        return float((np.degrees(np.arccos(d)) > 30.0).mean())

    out = _bilateral_denoise(noisy, IngestConfig(denoise_sigma_s=0.3))
    # No faceting: the spurious-sharp-edge fraction must not balloon.
    assert frac_sharp(out) < frac_sharp(noisy) + 0.02
