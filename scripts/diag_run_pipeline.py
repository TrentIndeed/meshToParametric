import sys, time, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
from mesh2param.viz_server.job_queue import JobQueue, Job

q = JobQueue()
job = Job(id="diagtest", filename="MountingPlate.3mf",
          file_path="models/AI1/MountingPlate.3mf",
          step_through=False, pause_at_stage=None)
job.status = "queued"
t0 = time.time()
try:
    q._run_pipeline(job)
except Exception as e:
    import traceback; traceback.print_exc()
print(f"\n=== status={job.status} elapsed={time.time()-t0:.1f}s ===")
print(f"steps={len(job.steps)} cad_vol={round(job.cad_volume,1)} mesh_vol={round(job.mesh_volume,1)}")
print(f"building_live={job.building_live} steps_version={job.steps_version}")
print(f"early_launch_used={getattr(job,'_cad_launch','consumed')}")
for s in job.steps[:6]:
    print('  ', s.name)
