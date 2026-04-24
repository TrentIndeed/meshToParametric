"""Trace segmentation stages to find where rid 28 (the spiky span-the-plate
region on body 1) first appears."""
from collections import defaultdict
import numpy as np
import math

from mesh2param.config import DecomposeConfig, IngestConfig, SegmentConfig
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import (
    _build_face_adjacency, _detect_sharp_edges,
    _region_growing_by_axis_class, _merge_small_regions,
    _merge_adjacent_cylinder_fragments, _smooth_region_boundaries,
)


def compactness(sub, face_indices):
    verts = np.asarray(sub.vertices)
    face_tri = verts[sub.faces[face_indices]]
    e1 = face_tri[:, 1] - face_tri[:, 0]
    e2 = face_tri[:, 2] - face_tri[:, 0]
    area = 0.5 * float(np.linalg.norm(np.cross(e1, e2), axis=1).sum())
    edge_count = defaultdict(int)
    for fi in face_indices:
        face = sub.faces[fi]
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i+1)%3])
            edge_count[(min(v0,v1), max(v0,v1))] += 1
    perimeter = 0.0
    for (a,b), cnt in edge_count.items():
        if cnt == 1:
            perimeter += float(np.linalg.norm(verts[a] - verts[b]))
    if perimeter < 1e-9 or area < 1e-9:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def show_spikes(stage, labels, sub):
    lbls = np.unique(labels)
    bad = []
    for lbl in lbls:
        face_idx = np.where(labels == lbl)[0]
        if len(face_idx) < 20:
            continue
        c = compactness(sub, face_idx)
        if c < 0.05:
            mean_n = sub.normals[face_idx].mean(axis=0)
            bad.append((int(lbl), len(face_idx), c, mean_n))
    print(f"\n[{stage}] {len(lbls)} regions | {len(bad)} spiky (c<0.05):")
    for lbl, n, c, mn in bad:
        print(f"   rid {lbl:3d} | {n} faces | c={c:.3f} | mean_n={mn.round(2)}")


cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
foots = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    size = sorted(sub.bbox[1] - sub.bbox[0])
    foots.append((size[-1] * size[-2], b, sub))
foots.sort(key=lambda x: -x[0])
# Body 1 (footprint #2) — the one with the spike
_, body, sub = foots[2]
print(f"Body {body.body_id}: {len(sub.faces)} faces")

config = SegmentConfig()
adj = _build_face_adjacency(sub.faces)
sharp = _detect_sharp_edges(sub.normals, adj, config, face_curvatures=None, bbox_diag=sub.bbox_diagonal)
labels = _region_growing_by_axis_class(sub.normals, adj, sharp)
show_spikes("after region_growing", labels, sub)

labels = _merge_small_regions(labels, adj, sub.normals, config.min_region_faces)
show_spikes("after merge_small_regions", labels, sub)

labels = _merge_adjacent_cylinder_fragments(labels, sub, adj)
show_spikes("after merge_cylinder_fragments", labels, sub)

labels = _smooth_region_boundaries(labels, adj)
show_spikes("after smooth_region_boundaries", labels, sub)
