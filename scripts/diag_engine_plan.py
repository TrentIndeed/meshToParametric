import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import logging; logging.disable(logging.INFO)
from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.extrusion import plan_extrusions_components
cleaned = ingest_mesh(str(ROOT/"models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False))
plan = plan_extrusions_components(cleaned)
print(f"base_axis should be Z(2). {len(plan.segments)} segments:")
for i,s in enumerate(plan.segments):
    print(f"  seg{i}: axis={s.axis_idx} ({'XYZ'[s.axis_idx]}) span={s.start:.1f}..{s.end:.1f} area={s.area:.0f}")
