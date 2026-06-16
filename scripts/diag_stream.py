import logging, sys, os, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
from mesh2param.config import IngestConfig
from mesh2param.pipeline.convert import convert_mesh
from mesh2param.pipeline.export_cadquery import export_cadquery

path = sys.argv[1] if len(sys.argv) > 1 else "models/Mesh1/Mesh1.stl"
cleaned, tree, bodies = convert_mesh(path)
sd = ROOT / "data" / "diag_stream_test"
import shutil; shutil.rmtree(sd, ignore_errors=True); sd.mkdir(parents=True)
res = export_cadquery(tree, cleaned, use_llm=False, num_variants=1, stream_dir=str(sd))
files = sorted(glob.glob(str(sd / "*.pkl")))
print(f"{Path(path).name}: result_steps={len(res.construction_steps)} streamed_files={len(files)}")
import pickle
for f in files:
    with open(f, "rb") as fh:
        p = pickle.load(fh)
    tv = p.get("tess_vertices")
    print(f"  {Path(f).name}: {p['name']!r} verts={None if tv is None else len(tv)}")
