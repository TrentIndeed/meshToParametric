"""Find the bore wall faces and show how they're segmented.

Bore faces have normals pointing RADIALLY (perpendicular to bore axis
= perpendicular to plate face normal). For a side plate with face
normal ±X, bore wall face normals are in Y-Z plane (Nx ≈ 0).
"""
import numpy as np

from mesh2param.config import DecomposeConfig, IngestConfig, SegmentConfig
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh

cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
foots = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    size = sorted(sub.bbox[1] - sub.bbox[0])
    foots.append((size[-1] * size[-2], b, sub))
foots.sort(key=lambda x: -x[0])
_, body, sub = foots[1]  # side plate body 0

# Side plate face normal is ±X
# Bore wall face normals: Nx ≈ 0 (radial in Y-Z)
verts = np.asarray(sub.vertices)
normals = np.asarray(sub.normals)
centroids = np.array([verts[sub.faces[fi]].mean(axis=0) for fi in range(len(sub.faces))])

# Faces with |Nx| < 0.3 (strongly radial)
bore_candidates = np.where(np.abs(normals[:, 0]) < 0.3)[0]
print(f"Body {body.body_id}: total {len(sub.faces)} faces, bore candidates (|Nx|<0.3): {len(bore_candidates)}")

if len(bore_candidates) == 0:
    print("No bore candidates — check orientation")
else:
    # Where are they spatially? Plate bbox:
    print(f"Plate bbox: {(sub.bbox[1] - sub.bbox[0]).round(2)}")
    c = centroids[bore_candidates]
    print(f"Bore candidate centroid ranges:")
    print(f"  X: {c[:,0].min():.1f} .. {c[:,0].max():.1f}")
    print(f"  Y: {c[:,1].min():.1f} .. {c[:,1].max():.1f}")
    print(f"  Z: {c[:,2].min():.1f} .. {c[:,2].max():.1f}")

# Now segment and see which regions the bore candidates land in
regs = segment_mesh(sub, SegmentConfig())
lbl_of = np.full(len(sub.faces), -1, dtype=int)
for r in regs:
    lbl_of[r.face_indices] = r.region_id

bore_labels = lbl_of[bore_candidates]
from collections import Counter
cnt = Counter(bore_labels.tolist())
print(f"\nBore candidate faces distributed across {len(cnt)} regions:")
for lbl, n in cnt.most_common():
    r = next(r for r in regs if r.region_id == lbl)
    mean_n = sub.normals[r.face_indices].mean(axis=0)
    region_verts = verts[np.unique(sub.faces[r.face_indices].ravel())]
    bb = region_verts.max(axis=0) - region_verts.min(axis=0)
    print(f"  rid {lbl:3d}  {n:4d} bore-faces (region total {r.num_faces})  mean_n={mean_n.round(2)}  bbox={bb.round(2)}")
