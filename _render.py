from pathlib import Path
import numpy as np
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.config import IngestConfig, SegmentConfig, DecomposeConfig
import pyvista as pv
pv.set_plot_theme("document")
OUT = Path("_render_out"); OUT.mkdir(exist_ok=True)
PALETTE = np.array([
    [220,60,60],[60,200,60],[60,100,230],[230,200,50],[180,60,200],[60,220,230],
    [240,140,40],[140,230,60],[70,170,230],[230,120,180],[170,200,100],[130,170,230],
], dtype=float)/255.0
cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())

for bi in [0, 1, 2]:
    body = decomp.bodies[bi]
    sub = extract_sub_mesh(cleaned, body.face_indices, body=body)
    regs = segment_mesh(sub, SegmentConfig())
    fc = np.full((sub.num_faces, 3), 0.55)
    for ri, r in enumerate(regs):
        fc[r.face_indices] = PALETTE[ri % len(PALETTE)]
    verts = np.asarray(sub.vertices); faces = np.asarray(sub.faces)
    pv_faces = np.hstack([np.full((len(faces),1),3), faces]).ravel()
    mesh = pv.PolyData(verts, pv_faces)
    mesh.cell_data["c"] = (fc*255).astype(np.uint8)
    bbox_min = verts.min(0); bbox_max = verts.max(0); size = bbox_max - bbox_min

    # 4 corners
    for name, ymin, ymax, zmin, zmax in [
        ("bl", bbox_min[1], bbox_min[1]+size[1]*0.22, bbox_min[2], bbox_min[2]+size[2]*0.25),
        ("br", bbox_max[1]-size[1]*0.22, bbox_max[1], bbox_min[2], bbox_min[2]+size[2]*0.25),
        ("tl", bbox_min[1], bbox_min[1]+size[1]*0.22, bbox_max[2]-size[2]*0.25, bbox_max[2]),
        ("tr", bbox_max[1]-size[1]*0.22, bbox_max[1], bbox_max[2]-size[2]*0.25, bbox_max[2]),
    ]:
        clipped = mesh.clip_box([bbox_min[0]-1, bbox_max[0]+1, ymin, ymax, zmin, zmax], invert=False)
        if clipped.n_cells == 0:
            continue
        pl = pv.Plotter(off_screen=True, window_size=(900, 900))
        pl.add_mesh(clipped, scalars="c", rgb=True, show_edges=True, edge_color="black", line_width=0.5)
        c = clipped.center
        pl.set_focus(c); pl.set_position([c[0] - size[0]*5, c[1], c[2]]); pl.set_viewup([0, 0, 1])
        pl.screenshot(str(OUT / f"body{bi}_{name}.png")); pl.close()
    print(f"body{bi}: {len(regs)} regions")
print(f"-> {OUT}/body*_*.png")
