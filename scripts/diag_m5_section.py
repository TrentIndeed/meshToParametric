import sys, glob, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mesh2param.viz_server.job_queue import JobQueue, Job
import trimesh
f = [c for c in glob.glob(str(ROOT/"models/Mesh5/*")) if c.lower().endswith((".3mf",".stl",".obj"))][0]
q = JobQueue(); job = Job(id="m5s", filename=Path(f).name, file_path=f, step_through=False, pause_at_stage=None)
job.status="queued"; q._run_pipeline(job)
s = job.steps[-1]
V = np.asarray(s.tess_vertices); F = np.asarray(s.tess_faces)
tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
fig, axs = plt.subplots(1, 2, figsize=(16, 9))
# X-Y section at z=-150 (through walls, below base) -> wall outer corners
for ax, zc, ttl in [(axs[0], -130.0, "z=-130 (wall hole level)"), (axs[1], -90.0, "z=-90 (base + wall tops)")]:
    sec = tm.section(plane_origin=[0,0,zc], plane_normal=[0,0,1])
    if sec is not None:
        for e in sec.discrete:
            ax.plot(e[:,0], e[:,1], '-', color='#06c', lw=1.0)
    ax.set_aspect('equal'); ax.set_title(f"Mesh5 final build {ttl}")
fig.savefig(str(ROOT/"data"/"mesh5_holecheck.png"), dpi=100, bbox_inches='tight')
print("saved data/mesh5_holecheck.png  final-step:", s.name)
