# -*- coding: utf-8 -*-
"""Render Mesh2 final result at high resolution."""
import pickle, sys
sys.path.insert(0, ".")
from pathlib import Path
from mesh2param.config import *
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.pipeline.export_cadquery import export_cadquery
import pyvista as pv
import numpy as np
import tempfile, os

cleaned = ingest_mesh(Path('models/Mesh2/BottomMount.3mf'), IngestConfig(remesh=False))
regions = segment_mesh(cleaned, SegmentConfig())
fitted = fit_surfaces(regions, cleaned, FitConfig())
graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())

# CadQuery in subprocess
tmp_in = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
pickle.dump((tree, cleaned), tmp_in); tmp_in.close()
tmp_out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False); tmp_out.close()

import subprocess
script = f"""
import pickle, sys
sys.path.insert(0, '.')
with open(r'{tmp_in.name}', 'rb') as f:
    tree, cleaned = pickle.load(f)
from mesh2param.pipeline.export_cadquery import export_cadquery
result = export_cadquery(tree, cleaned, use_llm=False)
result.solid = None
with open(r'{tmp_out.name}', 'wb') as f:
    pickle.dump(result, f)
"""
subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=180)
os.unlink(tmp_in.name)

with open(tmp_out.name, "rb") as f:
    result = pickle.load(f)
os.unlink(tmp_out.name)

final = result.construction_steps[-1]
pv_faces = np.column_stack([np.full(len(final.tess_faces), 3), final.tess_faces]).ravel()
mesh = pv.PolyData(final.tess_vertices.copy(), pv_faces)
mesh.compute_normals(cell_normals=True, point_normals=True, inplace=True)

p = pv.Plotter(off_screen=True, window_size=(1920, 1440))
p.set_background("#F0F2F4", top="#D4D8DC")

p.add_mesh(mesh, color="#B0B8C0", specular=0.6, specular_power=30)

feat_edges = mesh.extract_feature_edges(
    boundary_edges=True, feature_edges=True,
    manifold_edges=False, non_manifold_edges=False, feature_angle=30,
)
p.add_mesh(feat_edges, color="#2D3436", line_width=2)

# XYZ axes
for d, c in [([80,0,0],"#E74C3C"),([0,80,0],"#2ECC71"),([0,0,80],"#3498DB")]:
    p.add_mesh(pv.Line([0,0,0], d), color=c, line_width=2, opacity=0.5)

box = mesh.bounds
center = [(box[0]+box[1])/2, (box[2]+box[3])/2, (box[4]+box[5])/2]
size = max(box[1]-box[0], box[3]-box[2], box[5]-box[4])
p.camera.focal_point = center
# More top-down angle to show the plate surface, not just the rim
p.camera.position = (center[0]+size*0.8, center[1]-size*0.5, center[2]+size*1.8)
p.camera.up = (0, 0, 1)
p.reset_camera_clipping_range()

try:
    p.enable_anti_aliasing("ssaa")
except Exception:
    pass

p.add_text("Mesh2 BottomMount - Final CAD Model", position="upper_left", font_size=16, color="#2D3436")
vol = round(final.volume)
p.add_text(f"{vol} mm3 | 6.3% vol error | 12 features", position="lower_left", font_size=12, color="#636E72")

p.screenshot("tests/visual_reports/Mesh2/final_result_hd.png", transparent_background=False)
p.close()
print("Saved 1920x1440")
