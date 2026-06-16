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
print("FEATURES:")
for ft in br.tree.features:
    sg = ft.sketch_geometry or {}
    nm = ft.type.name
    if "HOLE" in nm:
        d = ft.direction; ax = int(np.argmax(np.abs(d))) if d is not None else -1
        c = sg.get("center"); pos = sg.get("positions")
        cc = [round(float(x),0) for x in c] if c is not None else None
        print(f"  {nm} axis={'XYZ'[ax]} r={sg.get('radius')} center={cc} npos={len(pos) if pos else 0}")
    else:
        print(f"  {nm}")
