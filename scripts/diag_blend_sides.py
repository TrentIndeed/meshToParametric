import sys, glob, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.disable(logging.INFO)
import numpy as np
from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.convert import recognize_body
f = [c for c in glob.glob(str(ROOT/"models/Mesh5/*")) if c.lower().endswith((".3mf",".stl",".obj"))][0]
cleaned = ingest_mesh(f, IngestConfig(remesh=False, scan_auto=True))
br = recognize_body(cleaned, scan_mode=True)
V = cleaned.vertices
print(f"blend records: {len(br.blends)}")
for i, rec in enumerate(br.blends):
    idx = rec.get("face_indices")
    kind = rec.get("kind")
    if idx is None or len(idx)==0:
        print(f"  rec{i} kind={kind} (no faces)"); continue
    fv = cleaned.faces[np.asarray(idx)]
    ctr = V[fv.reshape(-1)].mean(axis=0)
    bb_min = V[fv.reshape(-1)].min(axis=0); bb_max = V[fv.reshape(-1)].max(axis=0)
    side = "LEFT(x<0)" if ctr[0] < 0 else "RIGHT(x>0)"
    print(f"  rec{i} kind={kind} nfaces={len(idx)} ctr=({ctr[0]:.0f},{ctr[1]:.0f},{ctr[2]:.0f}) "
          f"xrange=({bb_min[0]:.0f}..{bb_max[0]:.0f}) {side}")
# Count blends on left vs right side plate (|x|>30)
left = sum(1 for r in br.blends if r.get("face_indices") is not None and len(r["face_indices"])>0
           and V[cleaned.faces[np.asarray(r["face_indices"])].reshape(-1)].mean(0)[0] < -30)
right = sum(1 for r in br.blends if r.get("face_indices") is not None and len(r["face_indices"])>0
            and V[cleaned.faces[np.asarray(r["face_indices"])].reshape(-1)].mean(0)[0] > 30)
print(f"\nSIDE-PLATE BLENDS: left(x<-30)={left}  right(x>30)={right}")
