import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import logging; logging.disable(logging.INFO)
from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh
import match_fillets_to_edges as mfe
cleaned = ingest_mesh(str(ROOT/"models/Mesh5/RackMount.3mf"), IngestConfig(remesh=False))
raw, body_results, _ = mfe.build_clusters(cleaned, single_body=True)
_,_,tree = body_results[0]
for f in tree.features:
    sg = f.sketch_geometry or {}
    has_c = bool(sg.get("corners")); has_f = bool(sg.get("fillets"))
    npts = len(sg.get("points") or [])
    print(f"{f.type.name:18s} pts={npts:3d} corners={has_c} fillets={has_f} "
          f"axis_idx={sg.get('axis_idx')} span={sg.get('span')}")
