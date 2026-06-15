"""Validate the component-tracking planner across the broader AI/scan set.

Builds each part through the full pipeline twice -- default volume-greedy
planner vs the component planner -- and reports CAD volume error + segment count
for both, so we can see whether flipping use_component_planner on by default is
safe. Mirrors scripts/diag_ai_build's scan/AI front-end.

Run:  python -u scripts/diag_component_validate.py [AIn ... | all]
"""
from __future__ import annotations

import glob
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for n in ("trimesh", "matplotlib", "PIL", "urllib3", "pymeshlab", "cadquery"):
    logging.getLogger(n).setLevel(logging.ERROR)

from mesh2param.config import (  # noqa: E402
    FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
)
from mesh2param.pipeline.export_cadquery import export_cadquery  # noqa: E402
from mesh2param.pipeline.fit import fit_surfaces  # noqa: E402
from mesh2param.pipeline.ingest import ingest_mesh  # noqa: E402
from mesh2param.pipeline.recognize import recognize_features  # noqa: E402
from mesh2param.pipeline.regularize import regularize_mesh  # noqa: E402
from mesh2param.pipeline.segment import segment_mesh  # noqa: E402
from mesh2param.pipeline.topology import build_topology  # noqa: E402


def build(part_dir: Path) -> None:
    f = glob.glob(str(part_dir / "*.3mf")) + glob.glob(str(part_dir / "*.stl"))
    if not f:
        return
    nm = part_dir.name
    t = time.time()
    try:
        c = ingest_mesh(f[0], IngestConfig(remesh=False, scan_auto=True))
        dirty = bool(c.metadata.get("scan_mode"))
        seg = SegmentConfig(noise_robust=dirty)
        regions = segment_mesh(c, seg)
        fitted = fit_surfaces(regions, c, FitConfig())
        if dirty:
            regularize_mesh(c, regions, fitted, snap_angle_deg=12.0)
            regions = segment_mesh(c, seg)
            fitted = fit_surfaces(regions, c, FitConfig())
        graph, edges = build_topology(regions, fitted, c, TopologyConfig())
        out = []
        for flag in (False, True):
            tree = recognize_features(
                regions, fitted, graph, edges, c,
                RecognizeConfig(use_component_planner=flag),
            )
            res = export_cadquery(tree, c, use_llm=False, num_variants=1)
            out.append(f"{res.volume_error_pct:>6.1f}%/{res.status:<4}")
        print(f"{nm:<8} old={out[0]}  comp={out[1]}  ({time.time()-t:.0f}s)",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{nm:<8} ERR {str(e)[:70]}", flush=True)


def main() -> int:
    which = sys.argv[1:]
    if not which or which == ["all"]:
        which = ([f"AI{i}" for i in range(1, 16)]
                 + [f"Scan{i}" for i in range(1, 4)])
    print(f"{'part':<8}{'default vs component planner':>34}", flush=True)
    for w in which:
        d = ROOT / "models" / w
        if d.exists():
            build(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
