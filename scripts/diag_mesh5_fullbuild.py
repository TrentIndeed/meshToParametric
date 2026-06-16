import sys, glob, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
from mesh2param.viz_server.job_queue import JobQueue, Job
f = [c for c in glob.glob(str(ROOT/"models/Mesh5/*")) if c.lower().endswith((".3mf",".stl",".obj"))][0]
q = JobQueue()
job = Job(id="m5full", filename=Path(f).name, file_path=f, step_through=False, pause_at_stage=None)
job.status="queued"
try:
    q._run_pipeline(job)
except Exception as e:
    import traceback; traceback.print_exc()
print(f"\nSTATUS={job.status} steps={len(job.steps)} cad_vol={getattr(job,'cad_volume',0):.0f} mesh_vol={getattr(job,'mesh_volume',0):.0f}")
fil = [s.name for s in job.steps if s.name.startswith(("Fillet","Chamfer"))]
print(f"FILLET/CHAMFER STEPS ({len(fil)}):")
for n in fil: print("   ", n)
