import sys, logging, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
import numpy as np
from mesh2param.viz_server.job_queue import JobQueue, Job

cand = glob.glob(str(ROOT/"models"/"Mesh5"/"*"))
mesh = [c for c in cand if c.lower().endswith((".stl",".obj",".ply",".3mf"))][0]
q = JobQueue()
job = Job(id="diag_m5", filename=Path(mesh).name, file_path=mesh, step_through=False, pause_at_stage=None)
job.status="queued"
q._run_pipeline(job)

tree = job.features
f0 = tree.features[0]
sg = f0.sketch_geometry or {}
corners = sg.get("corners")
fillets = sg.get("fillets")
print("BASE feature:", f0.type, "n_corners", len(corners) if corners else 0)
if corners:
    for i,(c,r) in enumerate(zip(corners, fillets or [None]*len(corners))):
        mark = f"  R={r:.2f}" if (r and r>0.05) else "  (sharp)"
        print(f"  [{i:2d}] ({c[0]:8.2f},{c[1]:8.2f}){mark}")
print()
# Joins (tabs may be joins)
for jf in tree.features[1:]:
    jsg = jf.sketch_geometry or {}
    jc = jsg.get("corners"); jr = jsg.get("fillets")
    if jc:
        print(f"JOIN {jf.type} n_corners={len(jc)} rounded={sum(1 for r in (jr or []) if r and r>0.05)}")
