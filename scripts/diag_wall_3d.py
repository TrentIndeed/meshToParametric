import sys, glob, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesh2param.viz_server.job_queue import JobQueue, Job
f = [c for c in glob.glob(str(ROOT/"models/Mesh5/*")) if c.lower().endswith((".3mf",".stl",".obj"))][0]
q = JobQueue(); job = Job(id="w3d", filename=Path(f).name, file_path=f, step_through=False, pause_at_stage=None)
job.status="queued"; q._run_pipeline(job)
s = job.steps[-1]
V = np.asarray(s.tess_vertices); F = np.asarray(s.tess_faces)
# Right wall faces: all 3 verts have x in [37,48]
fc = V[F]  # (nf,3,3)
mask = (fc[:,:,0] >= 37) & (fc[:,:,0] <= 48)
keep = mask.all(axis=1)
tris = fc[keep]
fig = plt.figure(figsize=(18,9))
for i,(el,az,ttl) in enumerate([(20,-60,"iso outer"),(20,120,"iso inner"),(8,90,"end-on")]):
    ax = fig.add_subplot(1,3,i+1,projection='3d')
    pc = Poly3DCollection(tris, facecolor='#9fb4c8', edgecolor='#2c3e50', linewidths=0.25, alpha=1.0)
    ax.add_collection3d(pc)
    ax.set_xlim(37,48); ax.set_ylim(45,200); ax.set_zlim(-175,-82)
    ax.view_init(elev=el, azim=az); ax.set_title(ttl); ax.set_box_aspect((11,155,93))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
fig.savefig(str(ROOT/"data"/"mesh5_rightwall_3d.png"), dpi=90, bbox_inches='tight')
print("saved data/mesh5_rightwall_3d.png  tris:", len(tris))
