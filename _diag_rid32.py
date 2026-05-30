"""Compare rid 32's plane-neighbor count before/after my changes."""
from collections import defaultdict
from mesh2param.config import (
    DecomposeConfig, IngestConfig, SegmentConfig, FitConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.models.region import SurfaceType
import numpy as np

cleaned = ingest_mesh('models/Mesh5/RackMount.3mf', IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
entries = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    sz = sub.bbox[1] - sub.bbox[0]
    entries.append((sz, b, sub))
entries.sort(key=lambda e: -sorted(e[0])[-1] * sorted(e[0])[-2])
_, body, sub = entries[0]
regs = segment_mesh(sub, SegmentConfig())
fits = fit_surfaces(regs, sub, FitConfig())
_, edges = build_topology(regs, fits, sub, TopologyConfig())

verts = np.asarray(sub.vertices)
normals = np.asarray(sub.normals)

# Find rid 32 by position (center near [58, 123, -95], 127 faces, CYLIND)
target = None
for r in regs:
    rverts = verts[np.unique(sub.faces[r.face_indices].ravel())]
    bb_mid = (rverts.max(axis=0) + rverts.min(axis=0)) / 2
    if (abs(bb_mid[0] - 58) < 5 and abs(bb_mid[1] - 123) < 10
            and abs(bb_mid[2] - (-95)) < 5 and 100 <= r.num_faces <= 150):
        target = r
        break
if target is None:
    print("rid 32 not found by position")
else:
    print(f"Found rid {target.region_id} ({target.num_faces}f)")
    region_lookup = {r.region_id: r for r in regs}
    fitted_lookup = {f.region_id: f for f in fits}
    adj = defaultdict(list)
    for e in edges:
        adj[e.surface_a_id].append(e)
        adj[e.surface_b_id].append(e)
    neighbors = adj[target.region_id]
    plane_count = 0
    print(f"  {len(neighbors)} edge neighbors:")
    for e in neighbors:
        other = e.surface_b_id if e.surface_a_id == target.region_id else e.surface_a_id
        of = fitted_lookup.get(other)
        or_ = region_lookup.get(other)
        if not of or not or_:
            continue
        other_mean_n = normals[or_.face_indices].mean(axis=0)
        other_mag = float(np.linalg.norm(other_mean_n))
        is_flat = of.surface_type == SurfaceType.PLANE or other_mag > 0.95
        if is_flat:
            plane_count += 1
        rverts = verts[np.unique(sub.faces[or_.face_indices].ravel())]
        bb_size = rverts.max(axis=0) - rverts.min(axis=0)
        bb_mid = (rverts.max(axis=0) + rverts.min(axis=0)) / 2
        marker = "FLAT" if is_flat else "    "
        print(f"    rid {other:3d} {marker} {of.surface_type.name:8s} {or_.num_faces:4d}f mag={other_mag:.2f} bbox={bb_size.round(1)} center={bb_mid.round(0)}")
    print(f"  -> {plane_count} flat plane neighbors")
