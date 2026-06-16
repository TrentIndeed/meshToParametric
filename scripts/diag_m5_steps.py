import sys, glob, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
import numpy as np
from mesh2param.viz_server.job_queue import JobQueue, Job
f = [c for c in glob.glob(str(ROOT/"models/Mesh5/*")) if c.lower().endswith((".3mf",".stl",".obj"))][0]
q = JobQueue(); job = Job(id="m5steps", filename=Path(f).name, file_path=f, step_through=False, pause_at_stage=None)
job.status="queued"; q._run_pipeline(job)
print("=== RECOGNIZED HOLE FEATURES ===")
if job.features:
    for ft in job.features.features:
        if "HOLE" in ft.type.name:
            d=ft.direction; ax=int(np.argmax(np.abs(d))) if d is not None else -1
            sg=ft.sketch_geometry or {}
            print(f"  {ft.type.name} axis={'XYZ'[ax]} r={sg.get('radius')} center={[round(float(x)) for x in (sg.get('center') or [0,0,0])]}")
print("=== ALL CONSTRUCTION STEPS ===")
for s in job.steps: print("  ", s.name)
