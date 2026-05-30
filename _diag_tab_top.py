"""Show which fillet records the CAD matcher accepts/rejects."""
import numpy as np
from mesh2param.config import (
    DecomposeConfig, IngestConfig, SegmentConfig, FitConfig, TopologyConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology

cleaned = ingest_mesh('models/Mesh5/RackMount.3mf', IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())

all_blends = []
for i, b in enumerate(decomp.bodies):
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    regs = segment_mesh(sub, SegmentConfig())
    fits = fit_surfaces(regs, sub, FitConfig())
    _, edges = build_topology(regs, fits, sub, TopologyConfig())
    blends = identify_blends_from_topology(regs, fits, edges, sub)
    print(f"\n=== Body {i}: {len(blends)} blends ===")
    print(f"{'rid':>4} {'kind':>7} {'radius':>7} {'len':>7} {'nfaces':>6} {'axis_dir':>20} {'axis_point':>20}")
    for rec in blends:
        rid = rec.get('region_id')
        kind = rec.get('kind')
        radius = rec.get('radius', 0)
        length = rec.get('length', 0)
        nf = rec.get('num_faces', 0)
        ad = np.asarray(rec.get('axis_dir', [0,0,0]))
        ap = np.asarray(rec.get('axis_point', [0,0,0]))
        print(f"{rid:>4} {kind:>7} {radius:>7.2f} {length:>7.1f} {nf:>6} "
              f"[{ad[0]:>5.2f},{ad[1]:>5.2f},{ad[2]:>5.2f}] "
              f"[{ap[0]:>5.0f},{ap[1]:>5.0f},{ap[2]:>5.0f}]")
        all_blends.append(rec)

print(f"\n=== TOTAL: {len(all_blends)} blends fed to CAD export ===")
