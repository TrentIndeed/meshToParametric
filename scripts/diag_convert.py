"""Dump the feature tree for a mesh using the SHARED pipeline core.

Calls mesh2param.pipeline.convert.convert_mesh — the exact same code the web
server runs — so this matches the app. Usage:
    python scripts/diag_convert.py models/AI1/MountingPlate.3mf
"""
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)

from mesh2param.pipeline.convert import convert_mesh  # noqa: E402


def main(path):
    cleaned, tree, bodies = convert_mesh(path)
    ext = (cleaned.vertices.max(0) - cleaned.vertices.min(0)).round(1)
    print(f"{Path(path).name}: bodies={len(bodies)} extents={ext} "
          f"scan_mode={cleaned.metadata.get('scan_mode')}")
    print(f"features: {dict(Counter(f.type.name for f in tree.features))}")
    for f in tree.features:
        if f.type.name == "BOSS":
            sg = f.sketch_geometry or {}
            print(f"  BOSS axis={sg.get('axis_idx')} "
                  f"span={[round(x, 1) for x in sg.get('span', [])]} "
                  f"loops={1 + len(sg.get('extra_loops', []))}")
        elif "HOLE" in f.type.name:
            sg = f.sketch_geometry or {}
            print(f"  {f.type.name} r={sg.get('radius')} center={sg.get('center')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/AI1/MountingPlate.3mf")
