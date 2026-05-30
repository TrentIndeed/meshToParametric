"""Show why each corner does/doesn't pass the bridge test."""
import numpy as np
from mesh2param.config import (
    DecomposeConfig, IngestConfig, SegmentConfig, FitConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.models.region import SurfaceType

cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
entries = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    size = sub.bbox[1] - sub.bbox[0]
    entries.append((size, b, sub))
entries.sort(key=lambda e: -sorted(e[0])[-1] * sorted(e[0])[-2])
_, body, sub = entries[0]
regs = segment_mesh(sub, SegmentConfig())
fits = fit_surfaces(regs, sub, FitConfig())
_, surface_edges = build_topology(regs, fits, sub, TopologyConfig())

normals = np.asarray(sub.normals)
verts = np.asarray(sub.vertices)
region_lookup = {r.region_id: r for r in regs}
fitted_lookup = {f.region_id: f for f in fits}

from collections import defaultdict
adj_by_region = defaultdict(list)
for se in surface_edges:
    adj_by_region[se.surface_a_id].append(se)
    adj_by_region[se.surface_b_id].append(se)

corner_rids = (34, 35, 39, 40)
for rid in corner_rids:
    region = region_lookup[rid]
    f = fitted_lookup[rid]
    face_norms = normals[region.face_indices]
    mean_n = face_norms.mean(axis=0)
    mag = float(np.linalg.norm(mean_n))
    print(f"\n=== rid {rid} ({region.num_faces}f, fit={f.surface_type.name}, mag={mag:.2f}) ===")
    print(f"   mean_n={mean_n.round(2)}")

    neighbors = adj_by_region.get(rid, [])
    print(f"   neighbors ({len(neighbors)}):")

    plane_nbrs = []
    for e in neighbors:
        other = e.surface_b_id if e.surface_a_id == rid else e.surface_a_id
        of = fitted_lookup.get(other)
        or_ = region_lookup.get(other)
        if of is None or or_ is None:
            continue
        other_mean_n = normals[or_.face_indices].mean(axis=0)
        other_mag = float(np.linalg.norm(other_mean_n))
        is_flat = of.surface_type == SurfaceType.PLANE or other_mag > 0.95
        verts_in = verts[np.unique(sub.faces[or_.face_indices].ravel())]
        bb_size = verts_in.max(axis=0) - verts_in.min(axis=0)
        marker = "  FLAT" if is_flat else " curved"
        print(f"      rid {other:3d} ({or_.num_faces:4d}f) {of.surface_type.name:10s} mag={other_mag:.2f} bbox={bb_size.round(1)} {marker} edge={e.transition_type.name}")
        if is_flat:
            plane_nbrs.append((other, other_mean_n / max(other_mag, 1e-6)))

    print(f"   {len(plane_nbrs)} flat plane neighbors")

    # Bridge test against each
    print(f"   Bridge test:")
    for other_rid, n_p in plane_nbrs:
        dots = face_norms @ n_p
        spread = float(dots.max() - dots.min())
        any_aligned = bool((dots > 0.866).any())
        print(f"      vs rid {other_rid}: dots range=[{dots.min():.2f},{dots.max():.2f}] spread={spread:.2f} (>1.0 fail) any>0.866={any_aligned}")
        bridge_ok = (spread <= 1.0) and any_aligned
        print(f"         bridge_ok={bridge_ok}")
