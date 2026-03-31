"""Debug: trace where mesh triangles come from."""
import sys, numpy as np
sys.path.insert(0, ".")
import cadquery as cq
from pathlib import Path
from mesh2param.config import *
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.export_cadquery import _extract_profile_and_holes, _tessellate_solid

cleaned = ingest_mesh(Path('models/Body1Mesh.3mf'), IngestConfig(remesh=False))
regions = segment_mesh(cleaned, SegmentConfig())
fitted = fit_surfaces(regions, cleaned, FitConfig())
graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())

extrude_dir = np.array(tree.features[0].direction)
depth = tree.features[0].depth
profiles, start, holes = _extract_profile_and_holes(tree, cleaned, extrude_dir)
print("Main profile:", len(profiles[0]), "pts")
print("Additional profiles:", len(profiles) - 1, "with sizes", [len(p) for p in profiles[1:]])
print()

# Build JUST the main profile extrusion
wp = cq.Workplane("XZ", origin=(0, start, 0))
wp = wp.polyline(profiles[0]).close().extrude(depth)
v, f = _tessellate_solid(wp)
print("Base extrude (main profile only):", len(v), "verts,", len(f), "faces")

# Now union each additional profile
for idx in range(1, len(profiles)):
    p = profiles[idx]
    v_before, f_before = _tessellate_solid(wp)
    extra = cq.Workplane("XZ", origin=(0, start, 0)).polyline(p).close().extrude(depth)
    wp = wp.union(extra)
    v_after, f_after = _tessellate_solid(wp)
    print("After union profile", idx, "(", len(p), "pts):",
          len(v_before), "->", len(v_after), "verts,",
          len(f_before), "->", len(f_after), "faces")

v_final, f_final = _tessellate_solid(wp)
print()
print("FINAL:", len(v_final), "verts,", len(f_final), "faces")
print("Volume:", round(wp.val().Volume(), 1))
