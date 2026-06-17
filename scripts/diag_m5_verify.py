import sys, glob, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import trimesh
from mesh2param.viz_server.job_queue import JobQueue, Job
f = [c for c in glob.glob(str(ROOT/"models/Mesh5/*")) if c.lower().endswith((".3mf",".stl",".obj"))][0]
q = JobQueue(); job = Job(id="m5v", filename=Path(f).name, file_path=f, step_through=False, pause_at_stage=None)
job.status="queued"; q._run_pipeline(job)
cv=getattr(job,'cad_volume',0) or 0; mv=getattr(job,'mesh_volume',0) or 0
nf=sum(1 for s in job.steps if s.name.startswith(("Fillet","Chamfer")))
print(f"VERIFY: status={job.status} steps={len(job.steps)} volerr={100*(cv-mv)/mv:+.1f}% fillets={nf}")
s=job.steps[-1]; V=np.asarray(s.tess_vertices); F=np.asarray(s.tess_faces)
tm=trimesh.Trimesh(vertices=V,faces=F,process=False)
fig,ax=plt.subplots(figsize=(8,10))
sec=tm.section(plane_origin=[0,0,-160],plane_normal=[0,0,1])
if sec:
    for e in sec.discrete: ax.plot(e[:,0],e[:,1],'-',color='#06c',lw=1.2)
ax.set_aspect('equal'); ax.set_title("Mesh5 z=-160 (solid wall) - corners rounded?")
fig.savefig(str(ROOT/"data"/"mesh5_wallcorners.png"),dpi=110,bbox_inches='tight')
print("saved data/mesh5_wallcorners.png")
