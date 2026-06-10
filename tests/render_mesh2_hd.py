# -*- coding: utf-8 -*-
"""Render Mesh2 final result at high resolution."""
import pickle
import sys
sys.path.insert(0, ".")
from pathlib import Path
from mesh2param.config import *
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.recognize import recognize_features
import pyvista as pv
import numpy as np
import tempfile
import os
import subprocess

cleaned = ingest_mesh(Path('models/Mesh2/BottomMount.3mf'), IngestConfig(remesh=False))
regions = segment_mesh(cleaned, SegmentConfig())
fitted = fit_surfaces(regions, cleaned, FitConfig())
graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())
tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())

# CadQuery in subprocess
tmp_in = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
pickle.dump((tree, cleaned), tmp_in); tmp_in.close()
tmp_out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False); tmp_out.close()

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
print(f"Tessellation: {len(final.tess_vertices)} verts, {len(final.tess_faces)} faces")
print(f"Volume: {round(final.volume)} mm3")

pv_faces = np.column_stack([np.full(len(final.tess_faces), 3), final.tess_faces]).ravel()
mesh = pv.PolyData(final.tess_vertices.copy(), pv_faces)

p = pv.Plotter(off_screen=True, window_size=(1920, 1440))
p.set_background("#1A1A2E", top="#16213E")

light1 = pv.Light(position=(0, 0, 300), focal_point=(0, 0, 0), intensity=1.2)
light2 = pv.Light(position=(200, -100, 100), focal_point=(0, 0, 0), intensity=0.6)
light3 = pv.Light(position=(-100, 200, 50), focal_point=(0, 0, 0), intensity=0.4)
p.add_light(light1)
p.add_light(light2)
p.add_light(light3)

p.add_mesh(mesh, color="#8899AA", specular=0.5, specular_power=20,
           show_edges=False, opacity=1.0, smooth_shading=True)

box = mesh.bounds
center = [(box[0]+box[1])/2, (box[2]+box[3])/2, (box[4]+box[5])/2]
size = max(box[1]-box[0], box[3]-box[2], box[5]-box[4])

p.camera.focal_point = center
p.camera.position = (center[0]+size*0.6, center[1]-size*0.4, center[2]+size*1.5)
p.camera.up = (0, 0, 1)
p.reset_camera_clipping_range()

try:
    p.enable_anti_aliasing("ssaa")
except Exception:
    pass

vol = round(final.volume)
mesh_vol = round(result.mesh_volume)
err = round(abs(final.volume - result.mesh_volume) / (result.mesh_volume + 1e-15) * 100, 1)
p.add_text("Mesh2 BottomMount - Final CAD Model", position="upper_left", font_size=16, color="#E2E8F0")
p.add_text(f"{vol} mm3 | {err}% vol error | {len(result.construction_steps)} steps", position="lower_left", font_size=12, color="#94A3B8")

os.makedirs("tests/visual_reports/Mesh2", exist_ok=True)
p.screenshot("tests/visual_reports/Mesh2/final_result_hd.png", transparent_background=False)
p.close()
print("Saved 1920x1440")
