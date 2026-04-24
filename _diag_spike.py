"""Diagnose the spiky region in body 0 of RackMount.

Find the largest low-compactness region on a side plate, print:
- mean normal magnitude (curved vs flat)
- bbox extent
- per-face normal-class breakdown (how many face normals align with each axis)
- mesh-adjacency to other regions (boundary length per neighbor)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from mesh2param.config import (
    DecomposeConfig, FitConfig, IngestConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh

RACKMOUNT = "models/Mesh5/RackMount.3mf"

cleaned = ingest_mesh(RACKMOUNT, IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())

# Pick side plate body 0
foots = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    size = sorted(sub.bbox[1] - sub.bbox[0])
    foots.append((size[-1] * size[-2], b, sub))
foots.sort(key=lambda x: -x[0])
# Side plates are foots[1] and foots[2]
for footprint_idx, (_foot, body, sub) in enumerate(foots[1:], start=1):
    print(f"\n=== Side plate body {body.body_id} (footprint #{footprint_idx}) ===")
    regs = segment_mesh(sub, SegmentConfig())

    # Compute per-region compactness + stats
    verts = np.asarray(sub.vertices)
    normals = np.asarray(sub.normals)
    for r in regs:
        if r.num_faces < 20:
            continue
        face_tri = verts[sub.faces[r.face_indices]]
        e1 = face_tri[:, 1] - face_tri[:, 0]
        e2 = face_tri[:, 2] - face_tri[:, 0]
        area = 0.5 * float(np.linalg.norm(np.cross(e1, e2), axis=1).sum())
        edge_count: dict[tuple[int, int], int] = defaultdict(int)
        edge_verts: dict[tuple[int, int], tuple[int, int]] = {}
        for fi in r.face_indices:
            face = sub.faces[fi]
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                key = (min(v0, v1), max(v0, v1))
                edge_count[key] += 1
                edge_verts[key] = (v0, v1)
        perimeter = 0.0
        for key, cnt in edge_count.items():
            if cnt == 1:
                a, b = edge_verts[key]
                perimeter += float(np.linalg.norm(verts[a] - verts[b]))
        if perimeter < 1e-9 or area < 1e-9:
            c = 0.0
        else:
            c = 4.0 * math.pi * area / (perimeter * perimeter)

        if c < 0.15 or r.num_faces > 2000:
            mean_n = normals[r.face_indices].mean(axis=0)
            mag = float(np.linalg.norm(mean_n))
            unit = mean_n / max(mag, 1e-12)
            # Per-face axis breakdown
            abs_n = np.abs(normals[r.face_indices])
            axis_idx = np.argmax(abs_n, axis=1)
            axis_val = abs_n[np.arange(len(r.face_indices)), axis_idx]
            sign = np.sign(normals[r.face_indices][np.arange(len(r.face_indices)), axis_idx])
            axis_class = 2 * axis_idx + np.where(sign < 0, 1, 0)
            classified = (axis_val >= np.cos(np.radians(5.0)))
            class_counter = Counter(int(c) for c, ok in zip(axis_class, classified) if ok)
            n_curved = int((~classified).sum())
            class_names = {0:"+X",1:"-X",2:"+Y",3:"-Y",4:"+Z",5:"-Z"}
            class_str = ", ".join(f"{class_names[c]}:{n}" for c, n in class_counter.most_common())

            # Sub-mesh region bbox
            r_verts = verts[np.unique(sub.faces[r.face_indices].ravel())]
            bb = r_verts.max(axis=0) - r_verts.min(axis=0)

            print(f"  rid {r.region_id:3d}  faces={r.num_faces:5d}  c={c:.4f}  |mean_n|={mag:.3f}  unit={unit.round(3)}  bbox={bb.round(2)}")
            print(f"          curved-class: {n_curved}/{r.num_faces}, axis-classified: {class_str}")
