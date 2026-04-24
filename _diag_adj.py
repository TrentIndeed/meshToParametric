"""Check region adjacency — is the fragment merger finding groups?"""
from collections import defaultdict
import numpy as np

from mesh2param.config import DecomposeConfig, IngestConfig, SegmentConfig
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import (
    _build_face_adjacency,
    _region_growing_by_axis_class,
    _merge_small_regions,
    _detect_sharp_edges,
)

cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
foots = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    size = sorted(sub.bbox[1] - sub.bbox[0])
    foots.append((size[-1] * size[-2], b, sub))
foots.sort(key=lambda x: -x[0])
_, body, sub = foots[1]

config = SegmentConfig()
adj = _build_face_adjacency(sub.faces)
sharp = _detect_sharp_edges(sub.normals, adj, config, face_curvatures=None, bbox_diag=sub.bbox_diagonal)

labels = _region_growing_by_axis_class(sub.normals, adj, sharp)
print(f"After region growing: {len(np.unique(labels))} regions")
labels = _merge_small_regions(labels, adj, sub.normals, config.min_region_faces)
print(f"After small-region merge: {len(np.unique(labels))} regions")

# Build region-level adjacency and report per-region stats
region_faces = {int(l): np.where(labels == l)[0] for l in np.unique(labels)}
region_adj = defaultdict(set)
for fi, nbrs in adj.items():
    li = int(labels[fi])
    for nb in nbrs:
        ln = int(labels[nb])
        if li != ln:
            region_adj[li].add(ln)
            region_adj[ln].add(li)

fragments = {l for l, f in region_faces.items() if 2 <= len(f) <= 400}
print(f"Fragments (2 <= faces <= 400): {len(fragments)}")

# For each region, show faces + mean normal + mesh-adjacent fragment neighbors
print("\n{:>4} {:>6} {:>6} {:>6} {:>6}  {:<30} fragment_nbrs".format(
    "rid", "faces", "mean_x", "mean_y", "mean_z", "is_fragment"))
for lbl in sorted(region_faces):
    faces = region_faces[lbl]
    mean_n = sub.normals[faces].mean(axis=0)
    mag = float(np.linalg.norm(mean_n))
    frag_nbrs = [n for n in region_adj[lbl] if n in fragments]
    is_frag = lbl in fragments
    print(f"{lbl:>4} {len(faces):>6} {mean_n[0]:>6.2f} {mean_n[1]:>6.2f} {mean_n[2]:>6.2f}  "
          f"{'FRAG' if is_frag else 'big ':<30} {frag_nbrs}")
