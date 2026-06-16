import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import logging; logging.basicConfig(level=logging.ERROR)
from mesh2param.pipeline.convert import convert_mesh
from mesh2param.pipeline.export_cadquery import _extract_profile_and_holes

cleaned, tree, bodies = convert_mesh('models/AI1/MountingPlate.3mf')
base = tree.features[0]
extrude_dir = np.array(base.direction)
profiles, start, holes, chamfers, fillets = _extract_profile_and_holes(tree, cleaned, extrude_dir)
print(f"axis_dir={extrude_dir.round(2)} extrude_start={start:.2f} n_profiles={len(profiles)}")
p0 = np.array(profiles[0])
print(f"outer profile: {len(p0)} pts")
print(f"  u range [{p0[:,0].min():.1f}, {p0[:,0].max():.1f}]  v range [{p0[:,1].min():.1f}, {p0[:,1].max():.1f}]")
closed = np.allclose(p0[0], p0[-1])
print(f"  first==last (explicitly closed)? {closed}")
# gap check: largest edge length vs typical
edges = np.linalg.norm(np.diff(np.vstack([p0, p0[0]]), axis=0), axis=1)
print(f"  edge lengths: min={edges.min():.2f} median={np.median(edges):.2f} max={edges.max():.2f}")
big = np.where(edges > 5*np.median(edges))[0]
print(f"  big-gap edges (>5x median): {big.tolist()}")
for i in big:
    a = p0[i]; b = p0[(i+1) % len(p0)]
    print(f"    edge {i}: {a.round(1)} -> {b.round(1)} len={edges[i]:.1f}")
