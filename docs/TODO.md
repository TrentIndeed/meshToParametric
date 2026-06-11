# TODO

## Test coverage
- [x] Reference STEPs for Mesh3 (Front Plate) and Mesh5 (RackMount) — found in
  `Documents/CAD_3MF/march2026/stp/`, geometrically verified (0.4% / 0.0%
  volume diff, identical frames), copied into `models/`.
- [x] Reference STEPs for Mesh2 (BottomMount re-export) and Mesh4 (tube) —
  both verified 0.0% volume diff, identical frames. All 5 models now have
  ground truth.
- [x] Mesh2 BottomMount: base extrude axis was a 45-deg chamfer normal
  ([0.7,0,-0.72]) instead of the plate's thin Z axis, so it sketched the
  13mm edge and dropped all holes. Fixed in _find_base_extrusion (area-
  weighted dominant plane + principal-axis snap + thin-plate override).
  Volume -23.2% -> +9.8%; holes restored; volume/bbox gates now pass.
- [x] Mesh2 corner cuts via the trapezoid profile (user: lowest-step
  principle). The base profile now selects the slice whose outer area is
  closest to the part's AVERAGE cross-section (mesh volume / depth) — the
  single extrude that best reproduces the part — instead of the largest
  slice. For BottomMount that IS the trapezoid (cut corners baked in), so
  one extrude needs no separate corner cut. Also guarded the face-
  projection "clean exterior" replacement so it only de-slits the outline,
  never enlarges it to the full silhouette. Mesh2 9.8%->0.1%; the same
  principle fixed Mesh3 Front Plate 36.7%->0.8% and the tube 0%. Mesh3
  volume+bbox gates un-xfailed.
- [x] Tube annular profile (user request): inner cutouts that belong to
  the cross-section (the bore, windows) are now drawn as holes in the
  BASE sketch and extruded once (CadQuery even-odd), instead of
  extrude-then-cut. Tube: 4 steps (Sketch+Extrude+Sketch+Cut) -> 2
  (Sketch Profile with outer+inner circle, Extrude Base). 0% volume,
  no regression across meshes.
- [x] Single sketch for symmetric hole pattern (user request): the 2D
  profile hole detector in _extract_profile_and_holes replaced the
  recognition's grouped features with one feature per hole, so the build
  made N separate Sketch+Cut steps. It now re-groups the 2D-detected
  holes by radius+axis into multi-center features. RackMount bottom
  plate -> "Sketch Through Hole Pattern (4x r=4.0)" in one sketch; Mesh2
  -> two patterns (9x r=2.5, 4x r=4.0).
- [ ] **Mesh4 tube: new part class (annular/revolved)**. One seam region
  unfittable (7/8 fitted), the coaxial bore is not recognized as a hole, and
  the base profile should be an annulus (circle + concentric cutout).
  test_all_meshes fit/has_holes + hole-accuracy gates xfailed.
- [ ] **Tube CadQuery build takes ~73s** (800-face part; BottomMount takes
  ~6s) with no progress updates — the silent window reads as a hang in the
  UI. Profile the build (likely the ~100-segment circle polyline extrude +
  per-feature passes) and emit progress heartbeats during stage 6.
- [x] Multi-body test harness: tests/test_multibody.py (slow) runs RackMount
  through decompose -> per-body -> stitched clusters -> production
  export_cadquery, locking 3-body decomposition, 12/12 fillet apply
  (match_ratio 1.0), volume < 10%, the 4-hole bottom-plate pattern grouping,
  and whole-mm fillet radii. Guards the most-iterated subsystem (fillet/apply)
  against silent regression.
- [ ] test_step_comparison Mesh5 gates remain xfailed (single-body harness);
  could fold the multibody helper into that comparison to un-xfail them.
- [ ] **Mesh3 Front Plate rebuild is +36.7% volume, 72% bbox overlap vs its
  reference** (single-body part, so this is a real reconstruction failure —
  likely the same fragmented-section profile-extraction issue below).
  Gates xfailed in test_step_comparison.KNOWN_FAILURES; remove entries as fixed.

## Chamfers
- [x] Chamfers now ride the SAME cluster->match->robust-apply pipeline as
  fillets (face-wire edge selection, live edge re-resolution, leave-one-
  out), applying OCC .chamfer() instead of .fillet(). The stitched stage
  builds chamfer clusters (kind="chamfer") from topology chamfer records
  instead of skipping them. Validated by a synthetic box-edge chamfer
  test; fillets unchanged (RackMount 12/12).
- [ ] No current test MESH produces chamfer blend records (all 5 parts'
  bevels are rounded fillets or baked into profiles). Need a chamfered
  reference part to validate chamfer detection->apply end-to-end on real
  geometry; the chamfer "setback" size estimate (currently the record's
  band half-extent) should be checked against a known chamfer then.

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

## Scans
- [x] Decimation speedup: 2.5M-face scans spent ~60s in quadric edge-collapse.
  A cheap vertex-clustering pre-pass (when faces > 5x the cap) cuts the input
  ~10x in ~2s, so the quality quadric runs on a small mesh. Scan2 ingest
  82s -> 33s. CAD meshes (<cap) are unaffected.
- [x] Scan smoke tests (tests/test_scans.py, slow-marked): all 3 scans
  decimate to the cap and run stages 1-5 without crashing, ingest < 120s.
- [x] Scans made watertight via screened-Poisson reconstruction
  (pymeshlab, already a dependency; IngestConfig.poisson_reconstruct).
  Gated on non-watertight input ONLY — clean CAD meshes are untouched
  because Poisson smooths/rounds sharp edges. All 3 scans -> watertight
  in 1-4s; volume errors dropped Scan1 1002%->173%, Scan2 157%->134%,
  Scan3 112%->74% (now a real-volume comparison, not inflated convex
  hull). Smoke test asserts watertight after ingest.
- [ ] **Residual scan volume error is still 74-173%** — the organic-part
  ceiling: Poisson makes a spanner watertight, not prismatic. The genuinely
  prismatic scan (Scan2 Industrial Part) is the best candidate for further
  work; truly organic shapes (spanner) belong on /clean. No reference STEPs
  for scans, so accuracy is volume-only.

## Iterative extrusion engine (started 2026-06-11)
- [x] **Phase 1+4: engine + metric.** `pipeline/extrusion.py` decomposes a
  part into prismatic segments: slice every principal axis, track
  cross-section component runs by reference-IoU (taper breaks the run —
  no averaging through it), bridge hole-split stations, then greedily
  pick segments by residual voxel coverage (marginal-gain stop, not a
  fixed count; profile-area tie-break = "largest single extrude area").
  `metrics.cross_section_fidelity` scores CAD-vs-mesh by section IoU
  across all axes — the gate for modeling decisions (volume can be gamed
  by an averaged cross-section; sections cannot). Engine now drives the
  base-extrude axis in `_find_base_extrusion` (legacy logic = fallback
  for ragged scans). Results: Mesh2 1 segment Y 100% (trapezoid from
  first principles); Mesh3 is genuinely a STEPPED part (3 coaxial Y
  segments, 99.9% — its 0.686 worst-section score under single-extrude
  confirms the gap); Mesh5 base-Z + side joins, 90%.
- [x] **Phase 2a: JOIN extrudes in one feature tree (2026-06-11).**
  recognize emits the plan's non-primary segments as BOSS/JOIN features
  (profile polygon + axis span); the base carries its segment span so
  its extrude stops where joins take over. export builds Sketch Join +
  Extrude Join steps (own axis plane, sign by span overlap, union)
  BEFORE holes. RackMount single-body now builds Extrude Base (bottom
  plate) -> Join x2 (side plates) -> 4 hole patterns, vol err 10.1%,
  fidelity 0.79. All earlier engine misreads fixed the same day:
  per-axis station density (~3mm — 24 uniform stations can't see a 6.7mm
  side plate), step-merge area-ratio floor 0.3 (Front Plate's mid
  section is 49% of the plate = cuts; RackMount's side sections are 6%
  of the base = joins), near-complete-candidate rule (a ~100% run beats
  a bigger-profile 91% run: BottomMount trapezoid), windowed-area
  preference otherwise ("largest single extrude area": base plate over
  U-channel sections). Engine: 1/1/1/1/3 segments on Mesh1-5.
- [ ] **Phase 2b: single-body multi-extrude parity.** The new path lacks
  the stitched-cluster blend stages (job_queue) and pocket cuts: the
  side plates' corner chamfer cuts are missing (the +10% volume,
  fidelity 0.79 vs multibody's clusters 12/12). Production still routes
  RackMount through decompose, so the new path is dormant until parity:
  apply stitched fillets/chamfers + pockets on the joined solid, gate on
  cross_section_fidelity >= multibody baseline, then make multi-extrude
  the default and keep decompose ONLY for disconnected shells (true
  assemblies). Join profiles also need interior windows carried (a join
  with a large bore would re-fill it; none of the 5 parts hits this yet).

## Pipeline (from the 2026-06-10 analysis)
- [ ] Feature-tree-driven CAD rebuild (replaces profile-extrusion + fillet
  matcher search; the structural fix for the blend apply ratio).
- [ ] Scale normalization at ingest so mm-absolute thresholds become ratios.
- [ ] Promote stitched-fillets (job_queue.py) and fillet apply
  (export_cadquery.py) into first-class pipeline stages with dataclass I/O.
