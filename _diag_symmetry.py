"""Compare side1 vs side2 segmentation to spot asymmetries."""
import numpy as np
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
# [0] = bottom plate; [1] = side plate 1, [2] = side plate 2
_, body1, sub1 = entries[1]
_, body2, sub2 = entries[2]

print(f"Side plate 1 (body {body1.body_id}): {len(sub1.faces)} faces, bbox={(sub1.bbox[1]-sub1.bbox[0]).round(2)}")
print(f"Side plate 2 (body {body2.body_id}): {len(sub2.faces)} faces, bbox={(sub2.bbox[1]-sub2.bbox[0]).round(2)}")


def summary(sub, label):
    print(f"\n=== {label} ===")
    regs = segment_mesh(sub, SegmentConfig())
    print(f"  {len(regs)} regions")
    normals = np.asarray(sub.normals)
    verts = np.asarray(sub.vertices)
    centroids = np.array([
        verts[sub.faces[fi]].mean(axis=0) for fi in range(len(sub.faces))
    ])

    # Print regions sorted by (role, size)
    def role(r):
        mean_n = normals[r.face_indices].mean(axis=0)
        mag = float(np.linalg.norm(mean_n))
        if mag >= 0.95:
            return "PLATE"
        if mag < 0.3:
            return "CYL"
        return "curve"

    def normalize_center(c, bbox):
        # Normalize to [0,1] so left/right plates compare
        return (c - bbox[0]) / (bbox[1] - bbox[0] + 1e-9)

    sorted_regs = sorted(regs, key=lambda r: (role(r), -r.num_faces))
    for r in sorted_regs:
        mean_n = normals[r.face_indices].mean(axis=0)
        mag = float(np.linalg.norm(mean_n))
        rc = centroids[r.face_indices]
        rcen = rc.mean(axis=0)
        rbb = rc.max(axis=0) - rc.min(axis=0)
        nc = normalize_center(rcen, sub.bbox)
        print(f"  {role(r)} rid {r.region_id:3d} {r.num_faces:4d}f mag={mag:.2f} "
              f"normal=[{mean_n[0]:+.2f},{mean_n[1]:+.2f},{mean_n[2]:+.2f}] "
              f"ncenter=[{nc[0]:.2f},{nc[1]:.2f},{nc[2]:.2f}] bbox={rbb.round(1)}")


summary(sub1, "SIDE PLATE 1")
summary(sub2, "SIDE PLATE 2")
