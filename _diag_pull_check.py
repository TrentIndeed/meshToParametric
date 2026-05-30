"""Check edge transition types for the small bevel regions at side plate corners."""
from collections import defaultdict
from mesh2param.config import (
    DecomposeConfig, IngestConfig, SegmentConfig, FitConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
import numpy as np

cleaned = ingest_mesh('models/Mesh5/RackMount.3mf', IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
entries = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    sz = sub.bbox[1] - sub.bbox[0]
    entries.append((sz, b, sub))
entries.sort(key=lambda e: -sorted(e[0])[-1] * sorted(e[0])[-2])

# Identify left side plate (X<-40)
for i, (sz, body, sub) in enumerate(entries):
    if sub.bbox[0][0] < -40 and sub.bbox[1][0] < 0:
        print(f"=== Left side plate (entry {i}) ===")
        regs = segment_mesh(sub, SegmentConfig())
        fits = fit_surfaces(regs, sub, FitConfig())
        _, edges = build_topology(regs, fits, sub, TopologyConfig())
        adj = defaultdict(list)
        for e in edges:
            adj[e.surface_a_id].append(e)
            adj[e.surface_b_id].append(e)
        verts = np.asarray(sub.vertices)
        for target_rid in (12, 13, 15, 16):
            r = next((rr for rr in regs if rr.region_id == target_rid), None)
            if r is None:
                continue
            rverts = verts[np.unique(sub.faces[r.face_indices].ravel())]
            bbmid = (rverts.max(axis=0) + rverts.min(axis=0)) / 2
            bbsz = rverts.max(axis=0) - rverts.min(axis=0)
            print(f"\nrid {target_rid} ({r.num_faces}f) bbox={bbsz.round(1)} center={bbmid.round(1)}:")
            for e in adj[target_rid]:
                other = e.surface_b_id if e.surface_a_id == target_rid else e.surface_a_id
                or_ = next((rr for rr in regs if rr.region_id == other), None)
                of = next((ff for ff in fits if ff.region_id == other), None)
                if or_ and of:
                    rverts2 = verts[np.unique(sub.faces[or_.face_indices].ravel())]
                    bbsz2 = rverts2.max(axis=0) - rverts2.min(axis=0)
                    print(f"  -> rid {other:3d} {of.surface_type.name:8s} {or_.num_faces:4d}f bbox={bbsz2.round(1)} edge={e.transition_type.name}")
        break
