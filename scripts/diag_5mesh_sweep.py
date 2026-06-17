import sys, glob, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
from mesh2param.viz_server.job_queue import JobQueue, Job
for m in ["Mesh1","Mesh2","Mesh3","Mesh4","Mesh5"]:
    fs = [c for c in glob.glob(str(ROOT/f"models/{m}/*")) if c.lower().endswith((".3mf",".stl",".obj"))]
    if not fs: print(f"{m}: NO FILE"); continue
    q = JobQueue(); job = Job(id=f"sw{m}", filename=Path(fs[0]).name, file_path=fs[0], step_through=False, pause_at_stage=None)
    job.status="queued"
    try:
        q._run_pipeline(job)
        cv=getattr(job,'cad_volume',0) or 0; mv=getattr(job,'mesh_volume',0) or 0
        err = 100*(cv-mv)/mv if mv else 0
        nf = sum(1 for s in job.steps if s.name.startswith(("Fillet","Chamfer")))
        nh = sum(1 for s in job.steps if "Hole" in s.name and "Sketch" not in s.name)
        print(f"RESULT {m}: status={job.status} steps={len(job.steps)} volerr={err:+.1f}% fillets={nf} holecuts={nh}")
    except Exception as e:
        print(f"RESULT {m}: FAIL {type(e).__name__}: {str(e)[:70]}")
