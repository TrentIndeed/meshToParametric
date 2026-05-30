"""Debug the gap between the fillet BLEND stage and the fillet APPLY stage.

The blend stage (Stage 4.5) shows every blend region detected on the mesh.
The apply stage only shows fillets the CAD matcher could attach to an edge.
This script traces the dropoff at each step:

    topology blend regions  ->  radius filter  ->  collinear merge
                            ->  edge match in the CAD solid

and prints, per surviving record, whether it matched a CAD edge and — if
not — exactly why (no parallel edge / perp distance out of band), plus the
radius and axis so you can spot bad cylinder fits (wrong angle, tiny radius).

Run:  PYTHONIOENCODING=utf-8 python _diag_fillet_match.py [path-to-mesh]
Default mesh: models/Mesh5/RackMount.3mf
"""
import sys
import logging

import numpy as np

logging.basicConfig(level=logging.ERROR)

from mesh2param.config import (
    DecomposeConfig, IngestConfig, SegmentConfig, FitConfig,
    TopologyConfig, RecognizeConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.reconcile_fillets import merge_blend_records_for_export
from mesh2param.pipeline.export_cadquery import (
    export_cadquery, _apply_recognized_fillets_to_solid, _build_stepwise,
    _extract_profile_and_holes,
)

MESH = sys.argv[1] if len(sys.argv) > 1 else "models/Mesh5/RackMount.3mf"

print(f"=== Fillet blend-vs-apply diagnostic: {MESH} ===\n")
cleaned = ingest_mesh(MESH, IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())

all_blends = []
body_results = []
n_fillet_regions = 0
for i, b in enumerate(decomp.bodies):
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    regs = segment_mesh(sub, SegmentConfig())
    fits = fit_surfaces(regs, sub, FitConfig())
    graph, edges = build_topology(regs, fits, sub, TopologyConfig())
    blends = identify_blends_from_topology(regs, fits, edges, sub)
    fillet_blends = [x for x in blends if x.get("kind") == "fillet"]
    n_fillet_regions += len(fillet_blends)
    for bl in blends:
        bl["body_index"] = i
    all_blends.extend(blends)
    tree = recognize_features(
        regs, fits, graph, edges, sub, RecognizeConfig(), topo_blends=blends,
    )
    body_results.append((b, sub, tree))
    print(f"Body {i}: {len(fillet_blends)} fillet blend region(s)")

bbox_diag = float(np.linalg.norm(cleaned.bbox[1] - cleaned.bbox[0]))
max_r = bbox_diag * 0.25
filtered = [
    p for p in all_blends
    if 0.2 <= float(p.get("radius", 0)) <= max_r
]
n_fillet_filtered = sum(1 for p in filtered if p.get("kind") == "fillet")
merged = merge_blend_records_for_export(filtered)

print(f"\nBLEND stage:   {n_fillet_regions} fillet region(s) across all bodies")
print(f"radius filter: {len(all_blends)} -> {len(filtered)} records "
      f"(keep 0.2..{max_r:.1f} mm); {n_fillet_filtered} fillets")
print(f"collinear merge: {len(filtered)} -> {len(merged)} records fed to APPLY")

# Run the full export — the matcher records per-record diagnostics into
# result.fillet_match_diag during the real apply pass.
result = export_cadquery(
    body_results[0][2], body_results[0][1], use_llm=False, num_variants=1,
    body_results=body_results, peeled_fillets=merged,
)
beo = result.blend_edge_overlay or []
out_diag = result.fillet_match_diag or []

print("\n=== APPLY stage — per-record match outcome ===")
print(f"{'#':>3} {'matched':>7} {'r':>6} {'len':>6} {'#edge':>5} "
      f"{'axis_dir':>22} {'reason'}")
if not out_diag:
    print("(no diagnostics captured)")
else:
    out_diag.sort(key=lambda d: (not d["matched"], d["record_idx"]))
    n_match = sum(1 for d in out_diag if d["matched"])
    for d in out_diag:
        ad = ",".join(f"{x:+.2f}" for x in d["axis_dir"])
        print(f"{d['record_idx']:>3} {('YES' if d['matched'] else 'no'):>7} "
              f"{d['radius']:>6.2f} {d['length']:>6.1f} {d['n_edges']:>5} "
              f"[{ad}] {d['reason']}")
    print(f"\nMATCHED {n_match}/{len(out_diag)} records "
          f"-> {len(beo)} CAD edges filleted")

print("\nHow to read this:")
print("  - 'no parallel CAD edge' = the record's axis_dir doesn't line up")
print("    with any edge on the base extrusion (the feature isn't modelled,")
print("    e.g. a cantilever tab or corner cut the profile didn't capture).")
print("  - 'perp dist outside band' = there IS a parallel edge but it sits")
print("    too far/near for this radius -> bad cylinder fit (wrong r) OR the")
print("    axis_point is off. Compare r here vs the blend-stage radius.")
print("  - tiny r (<1) or odd axis_dir = noisy fit; that's why some rendered")
print("    fillets look small or angled wrong.")
