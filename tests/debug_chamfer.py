"""Debug large chamfers on Mesh2."""
import pickle, sys
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

# Build solid with holes
wp = cq.Workplane("XY", origin=(0, 0, start))
wp = wp.polyline(profiles[0]).close().extrude(-depth)

for h in holes:
    s = h.sketch_geometry
    if s.get("type") != "circle":
        continue
    c = np.array(s["center"], dtype=float)
    r = float(s["radius"])
    hd = np.array(h.direction, dtype=float)
    hd = hd / (np.linalg.norm(hd) + 1e-15)
    bb2 = wp.val().BoundingBox()
    corners = np.array([[bb2.xmin, bb2.ymin, bb2.zmin], [bb2.xmax, bb2.ymax, bb2.zmax]])
    proj = corners @ hd
    cs = c + hd * (float(proj.min()) - float(c @ hd) - 1)
    ce = c + hd * (float(proj.max()) - float(c @ hd) + 1)
    cl = float(np.linalg.norm(ce - cs))
    try:
        wp = wp.cut(cq.Workplane(cq.Plane(origin=cq.Vector(*cs), normal=cq.Vector(*hd))).circle(r).extrude(cl))
    except Exception:
        pass

print("After holes:", round(wp.val().Volume()))

# Try large chamfer on different edge selectors
for size in [4.9, 4.0, 3.0, 2.0]:
    for sel in ["|Z", "|X", "|Y"]:
        try:
            test = wp.edges(sel).chamfer(size)
            print("edges('" + sel + "').chamfer(" + str(size) + "): OK vol=" + str(round(test.val().Volume())))
        except Exception:
            print("edges('" + sel + "').chamfer(" + str(size) + "): FAIL")
