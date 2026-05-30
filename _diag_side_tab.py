"""For each body's blend rids, show their geometric position so we can
identify which physical features they correspond to."""
from mesh2param.config import (
    DecomposeConfig, IngestConfig, SegmentConfig, FitConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
import numpy as np

cleaned = ingest_mesh('models/Mesh5/RackMount.3mf', IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())

for i, b in enumerate(decomp.bodies):
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    regs = segment_mesh(sub, SegmentConfig())
    fits = fit_surfaces(regs, sub, FitConfig())
    _, edges = build_topology(regs, fits, sub, TopologyConfig())
    blends = identify_blends_from_topology(regs, fits, edges, sub)
    verts = np.asarray(sub.vertices)
    print(f"\n=== Body {i} (bbox {sub.bbox[0].round(0)} to {sub.bbox[1].round(0)}) ===")
    blend_set = {r.get('region_id') for r in blends}
    for r in sorted(regs, key=lambda r: -r.num_faces):
        is_blend = r.region_id in blend_set
        if not is_blend and r.num_faces < 30:
            continue
        rverts = verts[np.unique(sub.faces[r.face_indices].ravel())]
        bb_size = rverts.max(axis=0) - rverts.min(axis=0)
        bb_mid = (rverts.max(axis=0) + rverts.min(axis=0)) / 2
        fit = next((f for f in fits if f.region_id == r.region_id), None)
        ftype = fit.surface_type.name[:6] if fit else "??"
        marker = "BLEND" if is_blend else "      "
        norms = np.asarray(sub.normals)[r.face_indices]
        mn = norms.mean(axis=0)
        mag = float(np.linalg.norm(mn))
        print(f"  rid {r.region_id:3d} {marker} {ftype:6s} {r.num_faces:4d}f mag={mag:.2f} bbox={bb_size.round(1)} center={bb_mid.round(0)}")
