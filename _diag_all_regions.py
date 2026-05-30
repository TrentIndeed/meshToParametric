"""List all regions on body 2 with their full geometry and where they are."""
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
_, body, sub = entries[0]
regs = segment_mesh(sub, SegmentConfig())
verts = np.asarray(sub.vertices)
normals = np.asarray(sub.normals)

print(f"Body bbox: {sub.bbox[0].round(1)} to {sub.bbox[1].round(1)}")
print(f"Body size: {(sub.bbox[1] - sub.bbox[0]).round(1)}\n")

print(f"{'rid':>4} {'nf':>5} {'mean_n':>17} {'mag':>5} {'bbox(xyz)':>20} {'cx':>7} {'cy':>7} {'cz':>7}")
print("-" * 90)
for r in sorted(regs, key=lambda r: -r.num_faces):
    region_verts = verts[np.unique(sub.faces[r.face_indices].ravel())]
    bb_min = region_verts.min(axis=0)
    bb_max = region_verts.max(axis=0)
    bb_size = bb_max - bb_min
    bb_mid = (bb_min + bb_max) / 2
    mn = normals[r.face_indices].mean(axis=0)
    mag = float(np.linalg.norm(mn))
    print(
        f"{r.region_id:>4} {r.num_faces:>5} "
        f"[{mn[0]:>4.1f},{mn[1]:>4.1f},{mn[2]:>4.1f}] "
        f"{mag:>5.2f} "
        f"[{bb_size[0]:>5.1f},{bb_size[1]:>5.1f},{bb_size[2]:>5.1f}] "
        f"{bb_mid[0]:>7.1f} {bb_mid[1]:>7.1f} {bb_mid[2]:>7.1f}"
    )
