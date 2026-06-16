import logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
from mesh2param.pipeline.convert import convert_mesh

def main(path):
    cleaned, tree, bodies = convert_mesh(path)
    body, sub, br = bodies[0]
    seg = br.seg_mesh
    nf = len(seg.faces) if seg is not None else len(cleaned.faces)
    print(f"{Path(path).name}: seg_faces={nf}")
    for f in tree.features:
        sg = f.sketch_geometry or {}
        print(f"  {f.type.name:14s} faces={len(f.face_indices):4d} "
              f"depth={f.depth:6.2f} r={sg.get('radius')}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/AI1/MountingPlate.3mf")
