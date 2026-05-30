"""Inspect rid 12 (right side wall + strips) to understand why it didn't split."""
import numpy as np
from collections import Counter
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
_, body, sub = entries[0]
regs = segment_mesh(sub, SegmentConfig())
verts = np.asarray(sub.vertices)
normals = np.asarray(sub.normals)

target = next(r for r in regs if r.region_id == 12)
faces = np.asarray(target.face_indices)
face_norms = normals[faces]
centroids = np.array([verts[sub.faces[fi]].mean(axis=0) for fi in faces])

print(f"rid 12: {len(faces)} faces")
bb_min = centroids.min(axis=0)
bb_max = centroids.max(axis=0)
print(f"centroids bbox: {bb_min.round(2)} to {bb_max.round(2)}")
print(f"size: {(bb_max - bb_min).round(2)}")

# Histogram of normal_x values
nx = face_norms[:, 0]
ny = face_norms[:, 1]
nz = face_norms[:, 2]
print(f"\nnormal_x: min={nx.min():.2f} max={nx.max():.2f} mean={nx.mean():.3f} sum={nx.sum():.1f}")
print(f"normal_y: min={ny.min():.2f} max={ny.max():.2f} mean={ny.mean():.3f} sum={ny.sum():.1f}")
print(f"normal_z: min={nz.min():.2f} max={nz.max():.2f} mean={nz.mean():.3f} sum={nz.sum():.1f}")

# Bucket by direction
buckets = {"+X": 0, "-X": 0, "+Y": 0, "-Y": 0, "+Z": 0, "-Z": 0, "tilted": 0}
for n in face_norms:
    if abs(n[0]) > 0.7:
        buckets["+X" if n[0] > 0 else "-X"] += 1
    elif abs(n[1]) > 0.7:
        buckets["+Y" if n[1] > 0 else "-Y"] += 1
    elif abs(n[2]) > 0.7:
        buckets["+Z" if n[2] > 0 else "-Z"] += 1
    else:
        buckets["tilted"] += 1
print(f"\nBuckets: {buckets}")

# Distribution of X position
print(f"\nX position deciles (of centroids):")
xs = sorted(centroids[:, 0])
for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    idx = min(int(q * (len(xs)-1)), len(xs)-1)
    print(f"  q{q:.2f}: x={xs[idx]:.2f}")
