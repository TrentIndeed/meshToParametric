"""Pinpoint WHY the 2 side-plate outer-edge wraps fail OCC apply.

Forces the export_cadquery logger to DEBUG (basicConfig is a no-op once an
imported lib has configured logging), so every per-bucket apply attempt —
radius steps, leave-one-out, greedy subset, shell-gate rejections — is shown.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
lg = logging.getLogger("mesh2param.pipeline.export_cadquery")
lg.setLevel(logging.DEBUG)
lg.addHandler(h)
lg.propagate = False

from mesh2param.config import IngestConfig
from mesh2param.pipeline.export_cadquery import export_cadquery
from mesh2param.pipeline.ingest import ingest_mesh
import match_fillets_to_edges as mfe

cleaned = ingest_mesh(str(ROOT / "models/Mesh5/RackMount.3mf"),
                      IngestConfig(remesh=False))
raw, body_results, _ = mfe.build_clusters(cleaned, single_body=True)
clusters = [{
    "group_id": cl["gid"], "vertices": cl["verts"],
    "radius": cl["radius"], "category": cl["category"],
    "kind": cl.get("kind", "fillet"), "n_records": 0,
} for cl in raw]
print(f"\n==== {len(clusters)} clusters in ====", flush=True)
_, _, tree = body_results[0]
res = export_cadquery(tree, cleaned, use_llm=False, stitched_clusters=clusters)
n = sum(1 for s in res.construction_steps
        if s.name.startswith(("Fillet", "Chamfer")))
print(f"\n==== fillet/chamfer steps applied: {n} ====", flush=True)
