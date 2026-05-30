"""Inspect FULL face composition at one user-specified corner."""
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
centroids = np.array([verts[sub.faces[fi]].mean(axis=0) for fi in range(len(sub.faces))])
lbl_of = np.full(len(sub.faces), -1, dtype=int)
for r in regs:
    lbl_of[r.face_indices] = r.region_id

bb_min, bb_max = sub.bbox
print(f"Body bbox: min={bb_min.round(1)} max={bb_max.round(1)}")
print(f"Body size: {(bb_max - bb_min).round(1)}")

# Investigate front-left corner (-43, -72, 35) centered → translated
center = (cleaned.bbox[0] + cleaned.bbox[1]) / 2
cx, cy, cz = -43, -72, 35
ax, ay, az = center[0] + cx, center[1] + cy, center[2] + cz
print(f"\nTarget corner (centered {cx},{cy},{cz}) -> abs ({ax:.1f},{ay:.1f},{az:.1f})")

# Find faces within a CYLINDER of radius 5mm around the corner X,Y axis,
# spanning the FULL Z height of side wall
near = []
for fi in range(len(sub.faces)):
    c = centroids[fi]
    d_xy = np.sqrt((c[0]-ax)**2 + (c[1]-ay)**2)
    if d_xy < 5 and bb_min[2] < c[2] < bb_max[2]:
        near.append(fi)
print(f"\n{len(near)} faces within 5mm XY radius of corner, full Z range")

# Show region composition
cnt = Counter(lbl_of[near].tolist())
print(f"\nRegion composition at corner:")
for lbl, n in cnt.most_common(20):
    rid_faces = [fi for fi in near if lbl_of[fi]==lbl]
    cs_in = centroids[rid_faces]
    bb_in = cs_in.max(axis=0) - cs_in.min(axis=0) if len(cs_in) else [0,0,0]
    z_min = cs_in[:, 2].min() if len(cs_in) else 0
    z_max = cs_in[:, 2].max() if len(cs_in) else 0
    # Get full region info
    for r in regs:
        if r.region_id == lbl:
            mn = normals[r.face_indices].mean(axis=0)
            mag = float(np.linalg.norm(mn))
            full_verts = verts[np.unique(sub.faces[r.face_indices].ravel())]
            full_bb = full_verts.max(axis=0) - full_verts.min(axis=0)
            print(f"  rid {lbl:3d} ({r.num_faces:4d}f total, {n} here)")
            print(f"      mean_normal={mn.round(2)} mag={mag:.2f}")
            print(f"      full_bbox={full_bb.round(1)}")
            print(f"      cluster_at_corner: bbox={np.array(bb_in).round(2)} z=[{z_min:.1f},{z_max:.1f}]")
            break

# Look at ONLY faces in a tight 2.5mm XY radius, see if there is a clean
# 10mm-Z column
print(f"\n=== Tight 2.5mm cylinder analysis ===")
tight = []
for fi in range(len(sub.faces)):
    c = centroids[fi]
    d_xy = np.sqrt((c[0]-ax)**2 + (c[1]-ay)**2)
    if d_xy < 2.5:
        tight.append(fi)
tight_zs = sorted(centroids[tight, 2].tolist())
print(f"  {len(tight)} faces; Z values:")
for z in tight_zs:
    print(f"    z={z:.2f}")
