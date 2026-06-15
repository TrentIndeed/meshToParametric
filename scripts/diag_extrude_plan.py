"""Print the extrusion-plan segments for one or more meshes.

Usage: python scripts/diag_extrude_plan.py models/AI1/MountingPlate.3mf [more...]
"""
import sys

from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.extrusion import plan_extrusions


def main(paths):
    for path in paths:
        print(f"\n=== {path} ===")
        try:
            cleaned = ingest_mesh(path, IngestConfig(remesh=False))
        except Exception as exc:  # noqa: BLE001
            print(f"  ingest failed: {exc}")
            continue
        bb = cleaned.vertices.max(0) - cleaned.vertices.min(0)
        print(f"  bbox = {bb.round(1)}  faces={len(cleaned.faces)}")
        plan = plan_extrusions(cleaned)
        print(f"  {len(plan.segments)} segment(s), coverage={plan.coverage_frac:.1%}")
        for k, s in enumerate(plan.segments):
            print(
                f"   [{k}] axis={s.axis_idx} span={s.start:7.2f}..{s.end:7.2f}"
                f" len={s.length:6.2f} area={s.area:9.1f}"
                f" gain={s.gain_frac:5.1%}"
            )


if __name__ == "__main__":
    args = sys.argv[1:] or ["models/AI1/MountingPlate.3mf"]
    main(args)
