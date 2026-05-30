"""Group blend records by their underlying physical edge."""
import numpy as np
from collections import defaultdict
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
bbox_diag = float(np.linalg.norm(cleaned.bbox[1] - cleaned.bbox[0]))
max_r = bbox_diag * 0.25

all_recs = []
for i, b in enumerate(decomp.bodies):
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    regs = segment_mesh(sub, SegmentConfig())
    fits = fit_surfaces(regs, sub, FitConfig())
    _, edges = build_topology(regs, fits, sub, TopologyConfig())
    blends = identify_blends_from_topology(regs, fits, edges, sub)
    fits_by_rid = {f.region_id: f for f in fits}
    for rec in blends:
        r = float(rec.get('radius', 0))
        if not (0.2 <= r <= max_r):
            continue
        rec_full = dict(rec)
        rec_full['body'] = i
        rec_full['fit_type'] = fits_by_rid[rec['region_id']].surface_type.value
        all_recs.append(rec_full)

print(f"22 kept records:")
for r in all_recs:
    ad = r['axis_dir']
    ap = r['axis_point']
    print(f"  body{r['body']} rid{r['region_id']:3d} fit={r['fit_type']:8s} r={r['radius']:6.2f} L={r['length']:5.1f} "
          f"ax=[{ad[0]:>5.2f},{ad[1]:>5.2f},{ad[2]:>5.2f}] pt=[{ap[0]:>4.0f},{ap[1]:>4.0f},{ap[2]:>4.0f}]")

def canonical_axis(ad):
    ad = np.asarray(ad)
    ad = ad / np.linalg.norm(ad)
    idx = int(np.argmax(np.abs(ad)))
    if ad[idx] < 0:
        ad = -ad
    return ad

clusters = defaultdict(list)
for r in all_recs:
    ad = canonical_axis(r['axis_dir'])
    ap = np.asarray(r['axis_point'])
    perp = ap - (ap @ ad) * ad
    key = (tuple((ad * 10).round().astype(int)), tuple((perp * 0.5).round().astype(int)))
    clusters[key].append(r)

print(f"\n{len(clusters)} unique axis lines (loose binning):")
for key, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    print(f"  axis dir~{key[0]} line~{key[1]}: {len(members)} records")
    for r in members:
        print(f"    body{r['body']} rid{r['region_id']:3d} fit={r['fit_type']:8s} r={r['radius']:6.2f}")
