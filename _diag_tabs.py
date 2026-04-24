"""Show the bottom plate's tab regions spatially to see the splits."""
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict

from mesh2param.config import DecomposeConfig, IngestConfig, SegmentConfig
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh


cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
entries = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    size = sub.bbox[1] - sub.bbox[0]
    entries.append((size, b, sub))
entries.sort(key=lambda e: -sorted(e[0])[-1] * sorted(e[0])[-2])
_, body, sub = entries[0]  # bottom plate

verts = np.asarray(sub.vertices)
normals = np.asarray(sub.normals)
centroids = np.array([verts[sub.faces[fi]].mean(axis=0) for fi in range(len(sub.faces))])

regs = segment_mesh(sub, SegmentConfig())
print(f"Body {body.body_id}: bbox={(sub.bbox[1]-sub.bbox[0]).round(1)}, {len(regs)} regions\n")

# Where is each region spatially?
for r in sorted(regs, key=lambda r: -r.num_faces):
    rc = centroids[r.face_indices]
    mean_n = normals[r.face_indices].mean(axis=0)
    mag = float(np.linalg.norm(mean_n))
    bb = rc.max(axis=0) - rc.min(axis=0)
    center = rc.mean(axis=0)
    kind = "PLATE" if mag > 0.95 else ("CYL " if mag < 0.3 else "curve")
    print(f"rid {r.region_id:3d}  {r.num_faces:4d}f  {kind}  bbox={bb.round(1)}  center=[{center[0]:+6.1f},{center[1]:+6.1f},{center[2]:+6.1f}]  mean_n=[{mean_n[0]:+.2f},{mean_n[1]:+.2f},{mean_n[2]:+.2f}]")
