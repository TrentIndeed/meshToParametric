"""Global plane regularization regression.

A scan / AI mesh fits the faces of a box a few degrees off, so its corners
come out at 87/92 deg. regularize_planes must recover EXACT mutual
parallel/perpendicular (snap onto the part's own orthogonal frame), leave a
genuinely oblique face (chamfer/draft) alone, and merge near-coplanar
parallel faces to a shared offset.
"""
from __future__ import annotations

import numpy as np

import trimesh

from mesh2param.models.mesh import CleanedMesh
from mesh2param.models.region import FaceRegion
from mesh2param.models.surface import FittedSurface, SurfaceType
from mesh2param.pipeline.regularize import regularize_mesh, regularize_planes


def _rot(v, axis, ang):
    """Rodrigues rotation of v about unit axis by ang radians."""
    v = np.asarray(v, float)
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    return (v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c))


def _plane(rid, point, normal):
    return FittedSurface(region_id=rid, surface_type=SurfaceType.PLANE,
                         params={"point": list(map(float, point)),
                                 "normal": list(map(float, normal))},
                         rms_error=0.01)


def _perturbed_box(perturb_deg=4.0):
    base = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    planes = []
    for i, n in enumerate(base):
        n = np.asarray(n, float)
        ax = np.cross(n, [0.3, 0.5, 0.7])
        nn = _rot(n, ax, np.radians(perturb_deg * (((i % 3) - 1) + 0.5)))
        planes.append(_plane(i, n * 10.0, nn / np.linalg.norm(nn)))
    return planes


def test_planes_snap_to_exact_orthogonal_frame():
    planes = _perturbed_box(4.0)
    out, rep = regularize_planes(planes, snap_angle_deg=15.0)
    assert all(r["snapped"] for r in rep)
    norms = [np.asarray(p.snapped_params["normal"], float) for p in out]
    # Every pair of snapped normals is EXACTLY parallel or perpendicular.
    for a in norms:
        for b in norms:
            d = abs(float(a @ b))
            assert d < 1e-6 or d > 1 - 1e-6


def test_oblique_face_is_left_alone():
    planes = _perturbed_box(2.0)
    chamf = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)   # 45 deg from X and Y
    planes.append(_plane(99, [0, 0, 0], chamf))
    out, rep = regularize_planes(planes, snap_angle_deg=15.0)
    r = next(x for x in rep if x["region_id"] == 99)
    assert not r["snapped"]


def test_exact_input_is_unchanged():
    planes = [_plane(0, [0, 0, 5], [0, 0, 1]),
              _plane(1, [10, 0, 0], [1, 0, 0]),
              _plane(2, [0, 10, 0], [0, 1, 0])]
    out, rep = regularize_planes(planes, snap_angle_deg=15.0)
    for p in out:
        n0 = np.asarray(p.params["normal"], float)
        n1 = np.asarray(p.snapped_params["normal"], float)
        assert abs(abs(float(n0 @ n1)) - 1.0) < 1e-9   # direction unchanged


def test_coplanar_faces_merge_offset():
    planes = [_plane(0, [0, 0, 5.0], [0, 0, 1]),
              _plane(1, [3, 4, 5.2], [0, 0, 1]),   # ~coplanar, 0.2 off
              _plane(2, [10, 0, 0], [1, 0, 0]),
              _plane(3, [0, 10, 0], [0, 1, 0])]
    out, rep = regularize_planes(planes, snap_angle_deg=15.0, offset_merge=0.5)
    n = np.asarray(out[0].snapped_params["normal"], float)
    o0 = np.asarray(out[0].snapped_params["point"], float) @ n
    o1 = np.asarray(out[1].snapped_params["point"], float) @ n
    assert abs(o0 - o1) < 1e-6


# --------------------------------------------------------------------------
# End-to-end mesh regularization: a noisy box's faces become exactly flat
# and axis-aligned (so the existing recognition then sees a crisp part).
# --------------------------------------------------------------------------
def _det_noise(n, sigma):
    t = np.arange(n * 3) * 0.123457
    g = (np.sin(t) + np.sin(2.7 * t + 1.1) + np.sin(6.1 * t + 0.3)) / np.sqrt(3)
    return g.reshape(-1, 3) * sigma


def _axis_dev_deg(mesh):
    n = mesh.face_normals
    return np.degrees(np.arccos(np.clip(np.abs(n).max(1), -1, 1)))


def test_regularize_mesh_flattens_noisy_box():
    extent = (60.0, 40.0, 20.0)
    box = trimesh.creation.box(extents=extent)
    for _ in range(4):
        box = box.subdivide()
    clean_n = box.face_normals.copy()
    noisy_v = box.vertices + _det_noise(len(box.vertices), 0.003 * float(np.linalg.norm(box.extents)))
    noisy = trimesh.Trimesh(noisy_v, box.faces, process=False)

    before = float(np.median(_axis_dev_deg(noisy)))
    assert before > 1.0   # genuinely dirty

    # Group faces into 6 regions by the CLEAN dominant axis+sign.
    axis = np.argmax(np.abs(clean_n), 1)
    sign = np.sign(clean_n[np.arange(len(clean_n)), axis])
    key = axis * 2 + (sign > 0).astype(int)
    regions, fitted = [], []
    for rid in np.unique(key):
        fmask = key == rid
        fidx = np.where(fmask)[0]
        vidx = np.unique(box.faces[fidx])
        regions.append(FaceRegion(region_id=int(rid), face_indices=fidx,
                                  surface_type=SurfaceType.PLANE,
                                  vertices=noisy.vertices[vidx],
                                  normals=noisy.face_normals[fidx]))
        nrm = noisy.face_normals[fidx].mean(0)
        nrm /= np.linalg.norm(nrm)
        fitted.append(FittedSurface(region_id=int(rid),
                                    surface_type=SurfaceType.PLANE,
                                    params={"normal": nrm.tolist(),
                                            "point": noisy.vertices[vidx].mean(0).tolist()},
                                    rms_error=0.01))

    cleaned = CleanedMesh(vertices=noisy.vertices.copy(), faces=box.faces.copy(),
                          normals=noisy.face_normals.copy(), is_watertight=True,
                          bbox=np.array([noisy.vertices.min(0), noisy.vertices.max(0)]))
    out, report = regularize_mesh(cleaned, regions, fitted, snap_angle_deg=15.0)

    reg = trimesh.Trimesh(out.vertices, out.faces, process=False)
    after = float(np.median(_axis_dev_deg(reg)))
    assert all(r["snapped"] for r in report)     # all 6 box faces snapped
    assert after < 0.01                          # faces now exactly axis-aligned
    assert after < before / 10.0                 # and a big improvement


# --------------------------------------------------------------------------
# Stage F: 2D sketch-profile regularization (axis-snap sides + tangent arcs)
# --------------------------------------------------------------------------
def _rounded_rect(W, H, r, ns=10):
    pts, hw, hh = [], W / 2, H / 2
    corners = [(hw - r, -hh + r, -np.pi / 2, 0), (hw - r, hh - r, 0, np.pi / 2),
               (-hw + r, hh - r, np.pi / 2, np.pi),
               (-hw + r, -hh + r, np.pi, 3 * np.pi / 2)]
    for cx, cy, a0, a1 in corners:
        for k in range(ns + 1):
            a = a0 + (a1 - a0) * k / ns
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return pts


def _straight_dev_deg(pts):
    from mesh2param.pipeline.regularize import regularize_profile_2d  # noqa
    Q = np.asarray(pts, float)
    seg = np.roll(Q, -1, 0) - Q
    L = np.linalg.norm(seg, axis=1)
    d = seg / (L[:, None] + 1e-9)
    ang = np.degrees(np.arccos(np.clip(np.abs(d).max(1), -1, 1)))
    long = L > 3.0
    return float(np.median(ang[long])) if long.any() else 0.0


def test_profile_regularizer_snaps_sides_and_keeps_area():
    from shapely.geometry import Polygon
    from mesh2param.pipeline.regularize import regularize_profile_2d
    clean = _rounded_rect(100, 60, 8)
    P = np.asarray(clean)
    t = np.arange(len(P) * 2) * 0.37
    g = (np.sin(t) + np.sin(2.7 * t + 1)) / np.sqrt(2)
    noisy = (P + g.reshape(-1, 2) * 0.4).tolist()
    reg = regularize_profile_2d(noisy)
    assert reg is not None and len(reg) >= 3
    assert _straight_dev_deg(reg) < 0.05          # sides exactly axis-aligned
    assert _straight_dev_deg(reg) < _straight_dev_deg(noisy)
    a_clean, a_reg = Polygon(clean).area, Polygon(reg).area
    assert abs(a_reg - a_clean) / a_clean < 0.02  # shape preserved


def test_profile_regularizer_rejects_non_prismatic():
    from mesh2param.pipeline.regularize import regularize_profile_2d
    # a circle is not an axis-aligned prismatic outline -> None
    circ = [(20 * np.cos(a), 20 * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 40, endpoint=False)]
    assert regularize_profile_2d(circ) is None
