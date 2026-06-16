import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cadquery as cq
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.gp import gp_Pln, gp_Pnt, gp_Dir

shp = cq.importers.importStep(str(ROOT/"models"/"Mesh5"/"RackMountStep.step"))
solid = shp.val()
bb = solid.BoundingBox()
print("STEP bbox: x", round(bb.xmin,1), round(bb.xmax,1),
      "y", round(bb.ymin,1), round(bb.ymax,1), "z", round(bb.zmin,1), round(bb.zmax,1))
dims = {"x":bb.xlen,"y":bb.ylen,"z":bb.zlen}
thin = min(dims, key=dims.get)
print("thin axis:", thin, "len", round(dims[thin],2))
mid = {"x":(bb.xmin+bb.xmax)/2,"y":(bb.ymin+bb.ymax)/2,"z":(bb.zmin+bb.zmax)/2}[thin]
nrm = {"x":(1,0,0),"y":(0,1,0),"z":(0,0,1)}[thin]
org = {"x":(mid,0,0),"y":(0,mid,0),"z":(0,0,mid)}[thin]
pln = gp_Pln(gp_Pnt(*org), gp_Dir(*nrm))
sec = BRepAlgoAPI_Section(solid.wrapped, pln); sec.Build()
edges = cq.Shape(sec.Shape()).Edges()
ui,vi = [i for i,a in enumerate("xyz") if a!=thin]
segs=[]
for e in edges:
    try:
        vs=e.Vertices()
        if len(vs)>=2:
            a=vs[0].toTuple(); b=vs[-1].toTuple()
            segs.append(((a[ui],a[vi]),(b[ui],b[vi])))
    except Exception: pass
fig,ax=plt.subplots(figsize=(9,11))
for (a,b) in segs:
    ax.plot([a[0],b[0]],[a[1],b[1]],'-',color='#06c',lw=0.7)
ax.set_aspect('equal'); ax.set_title("REFERENCE RackMount STEP — mid cross-section outline")
fig.savefig(str(ROOT/"data"/"mesh5_ref_section.png"),dpi=110,bbox_inches='tight')
print("edges in section:", len(segs), "-> saved data/mesh5_ref_section.png")
