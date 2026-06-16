import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cadquery as cq
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.gp import gp_Pln, gp_Pnt, gp_Dir

shp = cq.importers.importStep(str(ROOT/"models"/"Mesh5"/"RackMountStep.step"))
solid = shp.val(); bb=solid.BoundingBox()
fig,axs=plt.subplots(1,3,figsize=(20,9))
for ax,frac in zip(axs,[0.02,0.10,0.5]):
    zc = bb.zmin + frac*(bb.zmax-bb.zmin)
    pln=gp_Pln(gp_Pnt(0,0,zc),gp_Dir(0,0,1))
    sec=BRepAlgoAPI_Section(solid.wrapped,pln); sec.Build()
    edges=cq.Shape(sec.Shape()).Edges(); n=0
    for e in edges:
        vs=e.Vertices()
        if len(vs)>=2:
            a=vs[0].toTuple(); b=vs[-1].toTuple()
            ax.plot([a[0],b[0]],[a[1],b[1]],'-',color='#06c',lw=0.8); n+=1
    ax.set_aspect('equal'); ax.set_title(f"z={zc:.0f} (frac {frac}) edges={n}")
fig.savefig(str(ROOT/"data"/"mesh5_ref_plate.png"),dpi=95,bbox_inches='tight')
print("saved data/mesh5_ref_plate.png")
