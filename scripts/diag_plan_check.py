import sys, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import logging; logging.disable(logging.INFO)
from collections import Counter
from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.extrusion import plan_extrusions_components
for m in sys.argv[1:]:
    fs = [c for c in glob.glob(str(ROOT/f"models/{m}/*")) if c.lower().endswith((".3mf",".stl",".obj"))]
    if not fs: print(f"{m}: NO FILE"); continue
    plan = plan_extrusions_components(ingest_mesh(fs[0], IngestConfig(remesh=False)))
    print(f"RESULT {m}: {len(plan.segments)} segs axes={dict(Counter('XYZ'[s.axis_idx] for s in plan.segments))}")
