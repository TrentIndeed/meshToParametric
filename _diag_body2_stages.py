"""Trace each stage's region count on bottom plate (body 2)."""
import numpy as np
from mesh2param.config import DecomposeConfig, IngestConfig, SegmentConfig
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import (
    _build_face_adjacency, _detect_sharp_edges,
    _region_growing, _merge_small_regions, _split_flat_from_curved,
    _split_disconnected_components, _curvature_split,
    _split_cylinders_from_planes, _merge_adjacent_cylinder_fragments,
    _unify_cylinder_features,
)


cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
entries = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    size = sub.bbox[1] - sub.bbox[0]
    entries.append((size, b, sub))
entries.sort(key=lambda e: -sorted(e[0])[-1] * sorted(e[0])[-2])
_, body, sub = entries[0]  # bottom plate

config = SegmentConfig()
adj = _build_face_adjacency(sub.faces)
sharp = _detect_sharp_edges(sub.normals, adj, config)

def show(label, labels):
    n = len(np.unique(labels))
    sizes = sorted([int((labels == l).sum()) for l in np.unique(labels)], reverse=True)
    print(f"  {label}: {n} regions. Top sizes: {sizes[:8]}. Tiny (<10): {sum(1 for s in sizes if s < 10)}")

labels = _region_growing(sub.normals, adj, config, sharp)
show("region_growing             ", labels)
labels = _merge_small_regions(labels, adj, sub.normals, config.min_region_faces)
show("merge_small_regions        ", labels)
labels = _split_flat_from_curved(labels, sub.normals, adj)
show("split_flat_from_curved     ", labels)
labels = _split_disconnected_components(labels, adj)
show("split_disconnected_components", labels)
labels = _merge_small_regions(labels, adj, sub.normals, min(config.min_region_faces, 2))
show("merge_small_regions (2nd)  ", labels)
labels = _curvature_split(labels, sub, adj, config)
show("curvature_split            ", labels)
labels = _split_cylinders_from_planes(labels, sub, adj)
show("split_cylinders_from_planes", labels)
labels = _merge_adjacent_cylinder_fragments(labels, sub, adj)
show("merge_cylinder_fragments   ", labels)
labels = _unify_cylinder_features(labels, sub, adj)
show("unify_cylinder_features    ", labels)
