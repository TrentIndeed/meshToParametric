# TODO

## Test coverage
- [ ] **Add reference STEP files for Mesh2–Mesh5** (export from the original CAD
  into `models/MeshN/`). Only `models/Mesh1/Body1Step.step` exists today, so the
  per-feature accuracy gate (`tests/test_step_comparison.py::test_hole_accuracy`)
  covers one part. The tests auto-discover mesh+STEP pairs — dropping the files
  in extends hole-radius/axis accuracy checks to all models with zero code
  changes. (Owner: Trenton)

## Recognition
- [ ] Mesh1 hole recall is 2/6: the four Y-axis plate corner holes are not
  recognized as THROUGH_HOLE features (absorbed into other features). Measured
  by the hole-accuracy metric; needs a recognition fix.

## Pipeline (from the 2026-06-10 analysis)
- [ ] Scans end-to-end: decimation + segmentation are fast now, but stages 3–6
  on scan geometry are unvalidated (Scan3 → 228 regions).
- [ ] Feature-tree-driven CAD rebuild (replaces profile-extrusion + fillet
  matcher search; the structural fix for the blend apply ratio).
- [ ] Scale normalization at ingest so mm-absolute thresholds become ratios.
- [ ] Promote stitched-fillets (job_queue.py) and fillet apply
  (export_cadquery.py) into first-class pipeline stages with dataclass I/O.
