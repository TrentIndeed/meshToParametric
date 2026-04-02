"""Debug: try large chamfer on clean extrusion (before holes)."""
import sys
sys.path.insert(0, ".")
import cadquery as cq
import numpy as np
from pathlib import Path
from mesh2param.config import *
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.export_cadquery import _extract_profile_and_holes

cleaned = ingest_mesh(Path('models/Mesh2/BottomMount.3mf'), IngestConfig(remesh=False))
regions = segment_mesh(cleaned, SegmentConfig())
fitted = fit_surfaces(regions, cleaned, FitConfig())
graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())

extrude_dir = np.array(tree.features[0].direction)
depth = tree.features[0].depth
profiles, start, holes, chamfers, fillets = _extract_profile_and_holes(tree, cleaned, extrude_dir)

print("Profile:", len(profiles[0]), "points")

# Build JUST the extrusion, no holes
wp = cq.Workplane("XY", origin=(0, 0, start))
wp = wp.polyline(profiles[0]).close().extrude(-depth)
print("Extrusion vol:", round(wp.val().Volume()))

# Count edges
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE
exp = TopExp_Explorer(wp.val().wrapped, TopAbs_EDGE)
n = 0
while exp.More():
    n += 1
    exp.Next()
print("Edges:", n)

# Try chamfer on clean extrusion
for sel in ["|Z", "|X", "|Y"]:
    for size in [4.9, 4.0, 3.0, 2.0, 1.0]:
        try:
            test = wp.edges(sel).chamfer(size)
            print("CLEAN edges('" + sel + "').chamfer(" + str(size) + "): OK vol=" + str(round(test.val().Volume())))
            break
        except Exception as e:
            print("CLEAN edges('" + sel + "').chamfer(" + str(size) + "): FAIL - " + str(e)[:60])
