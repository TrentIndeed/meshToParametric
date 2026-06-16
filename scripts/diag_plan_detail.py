import sys, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import logging; logging.disable(logging.INFO)
from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.extrusion import plan_extrusions_components
m = sys.argv[1]
f = [c for c in glob.glob(str(ROOT/f"models/{m}/*")) if c.lower().endswith((".3mf",".stl",".obj"))][0]
plan = plan_extrusions_components(ingest_mesh(f, IngestConfig(remesh=False)))
print(f"{m}: {len(plan.segments)} segs")
for i,s in enumerate(plan.segments):
    print(f"  seg{i}: axis={'XYZ'[s.axis_idx]} span={s.start:.1f}..{s.end:.1f} area={s.area:.0f} npts={len(s.profile)}")
