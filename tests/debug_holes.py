"""Debug: check solid bbox and blind hole cuts."""
import sys
import numpy as np
sys.path.insert(0, ".")
import cadquery as cq
from pathlib import Path
from mesh2param.config import *
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.export_cadquery import _extract_profile_and_holes
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

cleaned = ingest_mesh(Path('models/Body1Mesh.3mf'), IngestConfig(remesh=False))
regions = segment_mesh(cleaned, SegmentConfig())
fitted = fit_surfaces(regions, cleaned, FitConfig())
graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())

extrude_dir = np.array(tree.features[0].direction)
depth = tree.features[0].depth
profiles, start, holes = _extract_profile_and_holes(tree, cleaned, extrude_dir)

abs_dir = np.abs(extrude_dir)
primary_axis = int(np.argmax(abs_dir))
plane_map = {0: "YZ", 1: "XZ", 2: "XY"}
plane_name = plane_map[primary_axis]
origin_3d = [0.0, 0.0, 0.0]
origin_3d[primary_axis] = start

print(f"Extrude dir: {extrude_dir}, primary_axis: {primary_axis} ({plane_name})")
print(f"Origin: {origin_3d}, depth: {depth}")
print(f"Expected: Y from {start:.1f} to {start + depth:.1f}")
print()

# Try positive extrude
wp_pos = cq.Workplane(plane_name, origin=tuple(origin_3d))
wp_pos = wp_pos.polyline(profiles[0]).close().extrude(depth)
bb = wp_pos.val().BoundingBox()
print(f"extrude(+{depth:.1f}): Y=[{bb.ymin:.1f}, {bb.ymax:.1f}]")

# Try negative extrude
wp_neg = cq.Workplane(plane_name, origin=tuple(origin_3d))
wp_neg = wp_neg.polyline(profiles[0]).close().extrude(-depth)
bb2 = wp_neg.val().BoundingBox()
print(f"extrude(-{depth:.1f}): Y=[{bb2.ymin:.1f}, {bb2.ymax:.1f}]")

# Use the correct one
use_neg = abs(bb2.ymax - (start + depth)) < abs(bb.ymax - (start + depth))
wp = wp_neg if use_neg else wp_pos
bb_final = wp.val().BoundingBox()
print(f"Using {'negative' if use_neg else 'positive'}: Y=[{bb_final.ymin:.1f}, {bb_final.ymax:.1f}]")
print()

# Now test blind hole
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_ON

solid_shape = wp.val().wrapped
for label, pt in [("blind hole center", (70.8, 24.4, -54.8)),
                   ("part center", (0, 26, 0)),
                   ("blind hole start", (70.8, 24.4, -45.0))]:
    c = BRepClass3d_SolidClassifier(solid_shape, gp_Pnt(*pt), 0.01)
    state = "IN" if c.State() == TopAbs_IN else "ON" if c.State() == TopAbs_ON else "OUT"
    print(f"  {label} {pt}: {state}")
