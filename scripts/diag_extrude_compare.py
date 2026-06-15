"""Compare the volume-greedy planner vs the component-tracking planner.

Prints segment count + per-segment (axis, span, area, gain) for both, so we
can validate the new planner matches/beats the old on the reference meshes
before switching over. Target: AI1 -> 10 objects.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh2param.config import IngestConfig
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.extrusion import (
    plan_extrusions, plan_extrusions_components,
)


def _show(name, plan):
    print(f"  {name}: {len(plan.segments)} segs, coverage={plan.coverage_frac:.1%}")
    for k, s in enumerate(sorted(plan.segments, key=lambda s: (s.axis_idx, s.start))):
        print(f"    [{k}] axis={s.axis_idx} span={s.start:8.2f}..{s.end:8.2f}"
              f" area={s.area:9.1f} gain={s.gain_frac:5.1%}")


def main(paths):
    for path in paths:
        print(f"\n=== {Path(path).parent.name} ({Path(path).name}) ===")
        try:
            c = ingest_mesh(path, IngestConfig(remesh=False))
        except Exception as exc:  # noqa: BLE001
            print(f"  ingest failed: {exc}")
            continue
        _show("OLD volume-greedy ", plan_extrusions(c))
        _show("NEW component-track", plan_extrusions_components(c))


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "models/AI1/MountingPlate.3mf",
        "models/Mesh1/Body1Mesh.3mf",
        "models/Mesh2/BottomMount.3mf",
        "models/Mesh3/Front Plate.3mf",
        "models/Mesh4/tube.3mf",
        "models/Mesh5/RackMount.3mf",
    ]
    main(args)
