"""Enable DEBUG logs and re-segment to see fragment-merger decisions."""
import logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")

from mesh2param.config import DecomposeConfig, IngestConfig, SegmentConfig
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh

cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
foots = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    size = sorted(sub.bbox[1] - sub.bbox[0])
    foots.append((size[-1] * size[-2], b, sub))
foots.sort(key=lambda x: -x[0])
# Body 0 only
_, body, sub = foots[1]
print(f"=== Segmenting body {body.body_id} ===")
regs = segment_mesh(sub, SegmentConfig())
print(f"Produced {len(regs)} regions")
