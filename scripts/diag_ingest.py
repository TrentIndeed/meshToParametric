import sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mesh2param.viz_server.job_queue import _ingest_streamed_steps

class FakeJob:
    id = "test"
    steps = []
    steps_version = 0
    last_activity_at = 0.0

job = FakeJob()
sd = ROOT / "data" / "diag_stream_test"
n = _ingest_streamed_steps(job, sd, 0)
print(f"ingested={n} job.steps={len(job.steps)} version={job.steps_version}")
for s in job.steps:
    tv = s.tess_vertices
    print(f"  {s.name!r} verts={None if tv is None else len(tv)} vol={s.volume}")
# Idempotency: a second call from `n` should add nothing
n2 = _ingest_streamed_steps(job, sd, n)
print(f"second-call ingested={n2} (should equal {n}), steps={len(job.steps)}")
