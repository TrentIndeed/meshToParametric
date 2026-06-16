import sys, numpy as np, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import logging; logging.basicConfig(level=logging.CRITICAL)
# capture the rect-symmetrize log
import mesh2param.pipeline.extrusion as ext
_fired = {'n': 0, 'r': []}
_orig = ext._symmetrize_rect_corners
def _wrap(coords, arc_samples=18):
    out = _orig(coords, arc_samples)
    if out is not coords and len(out) != len(coords):
        _fired['n'] += 1
    return out
ext._symmetrize_rect_corners = _wrap
from mesh2param.pipeline.convert import convert_mesh

import glob
meshes = sys.argv[1:]
for path in meshes:
    files = glob.glob(f"models/{path}/*.3mf") + glob.glob(f"models/{path}/*.stl") + glob.glob(f"models/{path}/*.obj")
    if not files: 
        print(f"{path}: NO FILE"); continue
    _fired['n']=0
    try:
        cleaned, tree, bodies = convert_mesh(files[0])
        base = tree.features[0]; bsg = base.sketch_geometry or {}
        from collections import Counter
        feats = dict(Counter(f.type.name for f in tree.features))
        ip = len(bsg.get('points') or [])
        holes = len(bsg.get('inner') or [])
        print(f"{path}: OK feats={feats} base_pts={ip} plate_holes={holes} rect_rebuilt={_fired['n']>0}")
    except Exception as e:
        print(f"{path}: FAIL {type(e).__name__}: {str(e)[:80]}")
