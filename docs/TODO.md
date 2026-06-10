# TODO

## Test coverage
- [x] Reference STEPs for Mesh3 (Front Plate) and Mesh5 (RackMount) — found in
  `Documents/CAD_3MF/march2026/stp/`, geometrically verified (0.4% / 0.0%
  volume diff, identical frames), copied into `models/`.
- [ ] **Reference STEPs still missing for Mesh2 (BottomMount) and Mesh4
  (Component3)**. Note: `Component3.step` in the stp folder is a DIFFERENT part
  (bbox matches the Mesh1 bracket family, 71% volume diff vs Component3.3mf);
  none of the bottom-mount candidates match BottomMount.3mf. (Owner: Trenton)
- [ ] **Multi-body test harness**: `test_step_comparison._run_pipeline` uses the
  single-body path, so RackMount (multi-body) gates are xfailed. Port the
  decompose + per-body-tree + stitched-clusters path into the harness so Mesh5
  compares against its reference fairly.
- [ ] **Mesh3 Front Plate rebuild is +36.7% volume, 72% bbox overlap vs its
  reference** (single-body part, so this is a real reconstruction failure —
  likely the same fragmented-section profile-extraction issue below).
  Gates xfailed in test_step_comparison.KNOWN_FAILURES; remove entries as fixed.

## Recognition
- [ ] Mesh1 hole recall is 2/6: the four Y-axis plate corner holes are not
  recognized as THROUGH_HOLE features (absorbed into other features). Measured
  by the hole-accuracy metric; needs a recognition fix.

## Profile extraction (root cause located 2026-06-10)
- [ ] RackMount side plates rebuild ~0.6mm short on every filleted side
  (profile u[51,195] vs true y[50,196]). Root cause: trimesh mid-slice
  sections of the plate come back FRAGMENTED (outer ring never closes as
  one polygon), so `_extract_profile_and_holes` reconstructs the boundary
  by unioning pieces + `buffer(0.5).buffer(-0.3)` — that reconstruction
  insets the outline. Slice ranking now prefers bbox extent over enclosed
  area (commit after 4f4042c), but the real fix is rebuilding the outer
  profile from the topology/peeled sharp outline instead of mesh slicing
  (part of the feature-tree-driven rebuild). Measure with
  `metrics.band_fidelity_report` — side-plate clusters read ~0.7mm vs
  ~0.08mm for base-plate clusters.

## Pipeline (from the 2026-06-10 analysis)
- [ ] Scans end-to-end: decimation + segmentation are fast now, but stages 3–6
  on scan geometry are unvalidated (Scan3 → 228 regions).
- [ ] Feature-tree-driven CAD rebuild (replaces profile-extrusion + fillet
  matcher search; the structural fix for the blend apply ratio).
- [ ] Scale normalization at ingest so mm-absolute thresholds become ratios.
- [ ] Promote stitched-fillets (job_queue.py) and fillet apply
  (export_cadquery.py) into first-class pipeline stages with dataclass I/O.
