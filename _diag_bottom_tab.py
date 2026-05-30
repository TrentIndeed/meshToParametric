"""Inspect the bottom-back corners of the side plates."""
from collections import Counter, defaultdict
from mesh2param.config import (
    DecomposeConfig, IngestConfig, SegmentConfig, FitConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
from mesh2param.models.region import SurfaceType
import numpy as np

cleaned = ingest_mesh('models/Mesh5/RackMount.3mf', IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
center = (cleaned.bbox[0] + cleaned.bbox[1]) / 2
print(f"cleaned bbox: {cleaned.bbox[0].round(1)} to {cleaned.bbox[1].round(1)}")
print(f"cleaned center: {center.round(1)}")

# User's centered coords
targets_centered = [(43, 71, -37), (-39, 71, -40)]
for cx, cy, cz in targets_centered:
    ax, ay, az = center[0]+cx, center[1]+cy, center[2]+cz
    print(f"  centered ({cx},{cy},{cz}) -> abs ({ax:.1f},{ay:.1f},{az:.1f})")

for i, b in enumerate(decomp.bodies):
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    regs = segment_mesh(sub, SegmentConfig())
    fits = fit_surfaces(regs, sub, FitConfig())
    _, edges = build_topology(regs, fits, sub, TopologyConfig())
    blends = identify_blends_from_topology(regs, fits, edges, sub)
    blend_set = {r.get('region_id') for r in blends}
    verts = np.asarray(sub.vertices)
    centroids = np.array([verts[sub.faces[fi]].mean(axis=0) for fi in range(len(sub.faces))])
    lbl_of = np.full(len(sub.faces), -1, dtype=int)
    for r in regs:
        lbl_of[r.face_indices] = r.region_id

    for cx, cy, cz in targets_centered:
        ax, ay, az = center[0]+cx, center[1]+cy, center[2]+cz
        if not (sub.bbox[0][0] - 5 < ax < sub.bbox[1][0] + 5
                and sub.bbox[0][1] - 5 < ay < sub.bbox[1][1] + 5
                and sub.bbox[0][2] - 5 < az < sub.bbox[1][2] + 5):
            continue
        print(f"\n=== Body {i} corner at abs ({ax:.1f},{ay:.1f},{az:.1f}) ===")
        near = []
        for fi in range(len(sub.faces)):
            c = centroids[fi]
            d = np.sqrt((c[0]-ax)**2 + (c[1]-ay)**2 + (c[2]-az)**2)
            if d < 5:
                near.append(fi)
        print(f"  {len(near)} faces within 5mm of target")
        cnt = Counter(lbl_of[near].tolist())
        for lbl, n in cnt.most_common(8):
            for r in regs:
                if r.region_id == lbl:
                    rverts = verts[np.unique(sub.faces[r.face_indices].ravel())]
                    bbsz = rverts.max(axis=0) - rverts.min(axis=0)
                    bbmid = (rverts.max(axis=0) + rverts.min(axis=0)) / 2
                    fit = next((f for f in fits if f.region_id==lbl), None)
                    ftype = fit.surface_type.name if fit else "??"
                    norms = np.asarray(sub.normals)[r.face_indices]
                    mag = float(np.linalg.norm(norms.mean(axis=0)))
                    is_blend = lbl in blend_set
                    marker = "BLEND" if is_blend else "    "
                    print(f"    rid {lbl:3d} {marker} {ftype:8s} {r.num_faces:4d}f mag={mag:.2f} bbox={bbsz.round(1)} center={bbmid.round(0)} ({n} here)")
                    break
