"""Find faces at user's exact corner positions, look for 10mm-Z strips."""
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

center = (cleaned.bbox[0] + cleaned.bbox[1]) / 2
# User said (-43, 72, 35) etc are CENTERED coords.
# Translate to absolute for body 2.
for cx, cy, cz in [(-43, -72, 35), (-43, 72, 35), (43, -72, 35), (43, 72, 35)]:
    ax, ay, az = center[0] + cx, center[1] + cy, center[2] + cz
    print(f"\n=== User corner ({cx}, {cy}, {cz}) -> abs ({ax:.1f}, {ay:.1f}, {az:.1f}) ===")
    # Look at ALL faces within a TIGHT region (3mm radius in XY, full Z extent of plate)
    near = []
    for fi in range(len(sub.faces)):
        c = centroids[fi]
        d_xy = np.sqrt((c[0]-ax)**2 + (c[1]-ay)**2)
        if d_xy < 3 and -97 < c[2] < -85:
            near.append(fi)
    z_min = centroids[near, 2].min() if near else 0
    z_max = centroids[near, 2].max() if near else 0
    print(f"  {len(near)} faces within 3mm XY of corner; Z range [{z_min:.1f}, {z_max:.1f}]")
    cnt = Counter(lbl_of[near].tolist())
    for lbl, n in cnt.most_common(8):
        for r in regs:
            if r.region_id == lbl:
                cs_in = centroids[[fi for fi in near if lbl_of[fi]==lbl]]
                if len(cs_in) > 0:
                    bb_in = cs_in.max(axis=0) - cs_in.min(axis=0)
                else:
                    bb_in = [0,0,0]
                mn = normals[r.face_indices].mean(axis=0)
                mag = float(np.linalg.norm(mn))
                kind = "PLATE" if mag > 0.95 else ("CYL" if mag < 0.3 else "curve")
                print(f"    rid {lbl:3d} ({r.num_faces:4d}f total, {n} here) {kind} mag={mag:.2f} cluster_bbox={np.array(bb_in).round(1)}")
                break
