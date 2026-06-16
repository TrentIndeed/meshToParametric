import sys, logging, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mesh2param.viz_server.job_queue import JobQueue, Job

cand = glob.glob(str(ROOT/"models"/"Mesh5"/"*"))
mesh = [c for c in cand if c.lower().endswith((".stl",".obj",".ply",".3mf"))][0]
q = JobQueue()
job = Job(id="diag_m5p", filename=Path(mesh).name, file_path=mesh, step_through=False, pause_at_stage=None)
job.status="queued"
q._run_pipeline(job)

f0 = job.features.features[0]
sg = f0.sketch_geometry or {}
pts = np.asarray(sg.get("points"))
corners = np.asarray(sg.get("corners"))
fillets = sg.get("fillets") or []

fig, ax = plt.subplots(figsize=(9,11))
# raw outline
ax.plot(np.r_[pts[:,0],pts[0,0]], np.r_[pts[:,1],pts[0,1]], '-', color='#bbb', lw=0.8, label='raw outline pts')
ax.plot(pts[:,0], pts[:,1], '.', color='#888', ms=2)
# regularized sharp corners
ax.plot(corners[:,0], corners[:,1], 'r+', ms=12, mew=2, label='reg. corner X')
for i,(c,r) in enumerate(zip(corners, list(fillets)+[None]*len(corners))):
    txt = f"{i}"
    if r and r>0.05: txt += f"\nR{r:.1f}"
    ax.annotate(txt, (c[0],c[1]), fontsize=7, color='blue')
ax.set_aspect('equal'); ax.legend(); ax.set_title("Mesh5 base outline: raw pts vs regularized corners")
fig.savefig(str(ROOT/"data"/"mesh5_tabs_diag.png"), dpi=110, bbox_inches='tight')
print("saved data/mesh5_tabs_diag.png; n_pts", len(pts), "n_corners", len(corners))
print("raw bbox x", pts[:,0].min(), pts[:,0].max(), "y", pts[:,1].min(), pts[:,1].max())
