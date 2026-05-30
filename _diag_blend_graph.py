"""Verify the 4 corner regions pass through Stage 3 fit and Stage 4.5 blend graph."""
import numpy as np
from mesh2param.config import (
    DecomposeConfig, IngestConfig, SegmentConfig, FitConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology

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
blend_records = identify_blends_from_topology(regs, fits, surface_edges, sub)

verts = np.asarray(sub.vertices)

# Find corner rids by their geometric signature: small (<30 faces),
# bbox near 2x2-3x10, center at one of the 4 corner positions
expected_corners = [(-42, 51), (-42, 195), (44, 51), (44, 195)]
print("=== Corner region detection ===")
corner_rids = []
for cx, cy in expected_corners:
    found = None
    for r in regs:
        rverts = verts[np.unique(sub.faces[r.face_indices].ravel())]
        bb_size = rverts.max(axis=0) - rverts.min(axis=0)
        bb_mid = (rverts.max(axis=0) + rverts.min(axis=0)) / 2
        if (bb_size[0] < 4 and bb_size[1] < 4 and bb_size[2] > 8
                and abs(bb_mid[0] - cx) < 4 and abs(bb_mid[1] - cy) < 4):
            found = r.region_id
            break
    print(f"  corner ({cx},{cy}): rid={found}")
    if found is not None:
        corner_rids.append(found)

# Stage 3: fit results
print(f"\n=== Stage 3 (Fit) ===")
for rid in corner_rids:
    fit = next((f for f in fits if f.region_id == rid), None)
    region = next((r for r in regs if r.region_id == rid), None)
    if fit and region:
        print(f"  rid {rid}: type={fit.surface_type.name} rms={fit.rms_error:.3f} ({region.num_faces}f)")

# Stage 4.5: blend graph
print(f"\n=== Stage 4.5 (Blend graph) ===")
blend_rids = {rec.get("region_id") for rec in blend_records}
print(f"Total blends: {len(blend_records)}, region ids: {sorted(rid for rid in blend_rids if rid is not None)}")
for rid in corner_rids:
    in_blends = rid in blend_rids
    print(f"  rid {rid}: {'YES blend' if in_blends else 'NOT a blend'}")
