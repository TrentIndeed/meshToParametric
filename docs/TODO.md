# TODO

## Product vision & release gates

**Target statement (the bar to hit before release):**
> Paramesh converts clean mechanical meshes into editable parametric CAD, shows
> you the recovered feature tree, measures the reconstruction error, and lets
> you correct the model.

Gates to clear before releasing the software (do NOT ship until these exist):

- [ ] **Demonstration videos (20–50, uncut).** Real screen captures, not
  cinematic animation: file selection → processing → generated feature tree →
  parameter editing → regeneration → mesh deviation overlay. Uncut = credible.
- [ ] **Public benchmark set** with easy / moderate / difficult / **failed**
  examples. Publishing the failures reads as more credible than competitors
  who only show ideal parts.
- [ ] **Numeric verification** reported per conversion: max deviation, mean
  surface deviation, volume difference, validity (watertight/manifold),
  processing time, feature-tree complexity. (We already compute volume error +
  bbox; ADD: max + mean surface deviation, validity flag, timing,
  feature-count — surface this in the UI and the report.)
- [ ] **Tightly defined support envelope**, stated up front: single solid,
  watertight mesh, primarily mechanical/prismatic geometry, no organic
  surfaces, limited/no assembly support, and an explicit list of supported
  feature types (extrude, hole, pocket, boss, slot, fillet, chamfer).
- [ ] **Immediate access** — even a paid alpha with usage limits beats another
  waitlist. (We have a waitlist + token tiers; flip on a gated alpha.)
- [ ] **Human correction loop** — let the user mark an axis, identify a hole,
  replace a feature, or pick among candidate strategies. Reducing a 2-hour
  rebuild to 5 minutes matters more than full autonomy. (Today's AI chat +
  dimension review is a start; ADD direct feature-tree edits + axis/hole
  picking + candidate-strategy selection.)
- [ ] **Failure-safe behavior** — "Unsupported geometry" is far better than a
  model that looks right but has dimensional/topological errors. Gate on the
  numeric verification: if max deviation or volume error exceeds a threshold,
  REFUSE with a reason instead of returning a wrong-but-plausible model.

Principle: **fully autonomous reconstruction is secondary.** The product wins
by reducing a 2-hour manual rebuild to ~5 minutes WITH a correction loop, and
by refusing rather than silently shipping wrong geometry.

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

## Onshape live export (started 2026-06-12)
- [x] **Live export works end-to-end.** `onshape/client.py` (API key
  pair via HTTP Basic — no HMAC needed; creds in .env, gitignored) +
  `onshape/feature_builder.py` (FeatureTree -> native BTM features) +
  `_live_export` in export_onshape.py. RackMount: 15/15 features OK in
  a real document — base + 2 joins (with baked windows/bores via
  filterInnerLoops) + 4 hole patterns; bbox matches the mesh to 0.1mm,
  volume -7.7% (fillets + relief counterbores unmodeled). CLI:
  `mesh2param convert x.3mf --onshape-doc "Name" --no-dry-run`-style
  flags already existed; document_name now plumbed through.
  Serialization details that cost probing time are documented in
  feature_builder.py's docstring and locked by tests/test_onshape_export.py
  (live smoke gated behind ONSHAPE_LIVE_TESTS=1). Probe scripts kept in
  scripts/onshape_*.py. NOTE: free-tier accounts can only create PUBLIC
  documents (private creation 409s; the client falls back automatically).
- [x] **AS-BUILT export (2026-06-12, round 2).** Exporting the raw
  recognition tree dropped 3 of 4 tab holes (hole recovery lives in
  export_cadquery) and had no fillets. CadQueryResult.as_built now
  records every hole group actually drilled and every blend actually
  applied (with pre-blend edge midpoints); the Onshape builder replays
  them — fillets/chamfers as native features with
  qUnion([qClosestTo(qEverything(EntityType.EDGE), vector)]) edge
  queries, which transfer because the two rebuilds are geometrically
  identical. Inner rings now use CadQuery's exact circle recognition
  (_clean_engine_polygon + algebraic LSQ + n>=12 + max-resid gate) so
  bores are true cylinders, not 60-gon prisms — that alone fixed all 4
  rim fillets. cPlanes get an explicit small "size" so they stop
  swamping the view. RackMount v4: 22/24 features OK — all 4 tab
  holes, 10/12 fillets, volume matches the CadQuery rebuild.
- [x] **Round 3 (2026-06-12): clean corners, full-ring fillets, build
  ON origin, hole sketches at the part.** All five user reports fixed;
  RackMount v7 = 23/23 features OK.
    - **Profile regularizer** (`_regularize_profile`): mesh sampling
      renders each corner as short 10-deg facets; the Onshape sketch
      drew them as a raw polyline ("slight angle ~1/10 from each
      corner"). Now: separate axis-aligned straight runs from corner
      arc-runs, SNAP straights to the sketch axes, intersect for exact
      90-deg sharp corners + circle-fit the rounding radius, then build
      EXACTLY tangent arcs. Verified offline: base 20 corners all 90
      deg, side plate 4 corners all 90 deg, areas within ~1%. Two
      fallbacks (fitted-arc walk, then raw polyline) for outlines it
      can't regularize.
    - **Full outer-edge fillet**: the outer edge is now a
      tangent-continuous loop (clean arcs), so a fillet seeded on EVERY
      recorded edge + tangentPropagation=true rolls the whole loop;
      single-seed stopped at the first sharp corner ("only the long
      edge filleted"). Sketch ARC serialization: BTCurveGeometryCircle
      with startParam/endParam CCW angles, clockwise=true REJECTED by
      the extrude (express CW by swapping the angles — probe6).
    - **Build ON origin** (not translated-near-it): anchor the BASE
      plate's own span-lo to 0 so its sketch sits on the Top default
      plane (deterministic JDC, no construction plane) — the side
      plates hang below into -Z. bbox-min anchoring had floated the
      base ~76mm up because the side plates are the bbox minimum.
    - **Through-hole sketches at the holes**: placed at
      min(center)-depth/2-2 on the hole axis instead of the global
      bbox extreme (which sat ~80mm 'in front' on the far side-plate
      edge).
- [ ] **Sketch visibility is not REST-settable.** Consumed sketches
  auto-hide; show/hide is client view-state, not in the feature model.
  No API path — UI only: select the sketches in the tree, right-click,
  Show. (Revisit if Onshape exposes a display-state endpoint.)
- [ ] **Relief counterbores under the tabs (~9% volume) unmodeled** in
  both exports — the residual that keeps the volume gate at FAIL.
  Recognize them as pocket/counterbore features.
- [ ] **Multi-user Onshape integration — see `docs/ONSHAPE_INTEGRATION_PLAN.md`.**
  Decisions locked 2026-06-13: per-user OAuth 2.0; full round-trip loop
  (export → re-sync → import from Onshape); design for the Onshape App
  Store (launch Paramesh from inside Onshape) from the start; Pro-tier,
  billing stays in Stripe. The live export engine is built and proven;
  what remains is per-user account plumbing (OAuth tokens on the User,
  job↔document link on MeshJob) and the round-trip UX. Phased: (1)
  connect + one-click export to the user's own account, (2) linked
  docs + re-sync + embed, (3) import + App Store extensions. Today the
  server .env key exports to the OWNER's account — fine only for
  single-user testing.
- [ ] Blind-hole opening derivation assumes the recognized center sits
  at the bore's axial midpoint — verify against a part with deep blind
  holes (Mesh1's tangent blind pair) before trusting depth direction.

## Scans + AI-generated ("dirty") meshes
> Market note: scans AND AI-generated meshes (Meshy/Tripo) are the SAME
> dirty-mesh problem and the same target market — see `docs/ROADMAP.md`.
> EYES-ON findings from real AI parts (regularize stops at interior flats,
> bumpy extruded profile, flat-fit-as-cylinder, AI13 non-watertight repair
> fails, Mesh5 cosmetic corner triangles) → `docs/AI_MESH_FINDINGS.md`.

**Front-end BUILT so far (2026-06-13/14) — `docs/SCAN_RECOVERY_PLAN.md`:**
- [x] Feature-preserving **denoise** — `ingest._bilateral_denoise` (Zheng
  bilateral normal filtering, default `denoise_method="bilateral"`):
  removes surface noise WITHOUT faceting curves (the pymeshlab two-step
  faceted: 0.6%→5.6% false edges; bilateral stays ~0.5%).
- [x] **Canonical-frame alignment** — `pipeline/align.py` (dominant normal
  triple + OBB fallback). Wrench 0%→92% engine coverage.
- [x] **Dirty-mesh detection by surface exactness** — `ingest._surface_wobble`
  (median dihedral within near-flat faces). Catches low-poly AI meshes the
  100k face-count threshold misses (clean CAD ~0°, dirty ~5°). Sets
  `cleanup_source="dirty_mesh"`.
- [x] **Regularize stage** — `pipeline/regularize.py`. Builds the part's own
  orthogonal frame, snaps planes to it (exact parallel/perp), and PROJECTS
  mesh vertices onto the consensus planes (flat faces, 90° corners). Noisy
  box → exactly axis-aligned. Wired into the live pipeline gated to
  scan_mode (12° conservative snap preserves real drafts/chamfers).
- [x] **Noise-robust segmentation** — `SegmentConfig.noise_robust` +
  `_merge_coplanar_fragments` (boundary dominance, cascade-safe). Absorbs
  noise slivers into the dominant face. NOTE: validated correct on
  embedded-sliver cases, but SIMULATED noise on coarse CAD shatters at
  region-GROWING (no big regions to merge into) — needs a REAL AI/scan mesh
  to tune (15 in `models/AI*/`).
- [x] **Cleanup viz stages** — Denoise / Align / Regularize each a separate
  pause-stage (gated to dirty inputs).

**Real AI-mesh validation (2026-06-14, `scripts/diag_ai_parts.py`, 15 parts
in `models/AI*/`):** Meshy meshes are DENSE (72k–1.14M faces), not low-poly —
14/15 are caught by the >100k face-count detector (NOT the wobble detector;
their wobble is low, 0.0–1.5). They DO fragment (seg 46→7341 regions), and
`noise_robust` cuts that 50–70% on real data (HexStandoff 46→21, PerforatedPlate
98→26, EnclosureLid 3110→1077) — it works here because real meshes keep big
regions + slivers (my earlier SIMULATED coarse-CAD shattered with no big
regions, which is why it didn't help there). So the merge is validated; the
remaining work is to push the counts further down on the dense/complex parts.

**Still TODO:**
- [ ] **Region-growing robustness for dense AI meshes** — the post-merge helps
  50–70% but complex parts (EnclosureLid 1077, TaperedHousing 2667) are still
  over-fragmented. Fix the GROWING/curvature-split that shatters them, not just
  the post-merge.
- [ ] **Detection gap: AI8 (72k faces, low wobble) slips through** — between
  the 30k decimation floor and the 100k scan threshold AND too smooth for the
  wobble detector. Consider lowering `scan_face_threshold` or detecting "dense
  relative to the decimation cap."
- [ ] **Stage E global constraint inference (beyond orientation):** equal-
  radius, symmetry (hole patterns / mirror planes), and the dimension
  network. Orientation (parallel/perp) is done in `regularize_planes`.
- [ ] **Stage F:** promote the Onshape `_regularize_profile` (axis-snapped
  straights + tangent arcs + circle recovery) into a shared sketch-
  regularization stage consumed by both exporters.
- [ ] **Prismatic/organic classifier (stage H)** — HARD: simple geometry
  metrics don't separate (the smooth spanner reads as MORE developable than
  the detailed industrial/gear scans). Needs noise-robust primitive-fitting
  coverage, not a one-shot threshold. Organic → /clean only.
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
- [~] **Scan intent recovery (user direction, 2026-06-12)** — LARGELY
  ADDRESSED by the front-end above. "Smoothing out surfaces, recovering
  intent such as circles, 90 degree angles" → denoise (smoothing) +
  regularize (90° / flat faces via consensus-frame snapping + vertex
  projection) are built and gated to dirty inputs, in the user-corrected
  order (clean/regularize FIRST, then recognition). REMAINING: near-circle
  → exact-circle recovery (part of stage F), and revisiting the extrusion
  engine's profile/span honesty on the now-cleaned wrench (the over-extrude
  is a symptom of an organic part — the spanner belongs on /clean per the
  prismatic/organic gate).
- [ ] Scan failures currently visible in tests: Scan3 fit drops 2/105
  regions, Scan1 recognizes only a BASE feature (no holes) — both
  pre-existing (confirmed on 4b51b30, 2026-06-12), xfail candidates
  until the intent-recovery work lands.

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
- [ ] **Phase 2b: single-body multi-extrude parity (diagnosed 2026-06-11).**
  Current state: RackMount single-body builds base + 2 joins + holes +
  stitched fillets (build_clusters(single_body=True)), join profiles
  carry interior windows, fidelity 0.855 mean / vol -12% (UNDER, not
  over). The root cause of the deficit is the SIDE READING: the engine
  picks the X-axis web faces (area 12366) for the side joins, but the
  real side structures are J-profiles with hooks, outer ear flanges and
  lower rails — constant along Y (see y=123 section) — i.e. the user's
  CAD extrudes the sides along Y with the full XZ profile. The X-web
  reading misses hooks/flanges/rails entirely. Fix chain:
    1. HONEST voxel claims: membership currently ORs +/-pitch (2.9mm)
       tolerance, which on a thin-walled part lets wrong candidates
       claim boundary bands (~87k mm^3 of slack) and hides real
       segments. Use strict interior membership + pitch sized to the
       thinnest feature (the 6mm webs need ~1.5mm pitch).
    2. With honest claims the Y side-runs out-gain the X webs and the
       0.95-residual rule (not the area window) decides — RackMount
       becomes base-Z + two Y-side extrudes matching the user's CAD.
    3. Tab undersides (sculpted 45deg reliefs under the 4 mounting tabs)
       remain as cut features after that.
  Then: gate on cross_section_fidelity >= multibody baseline, make
  multi-extrude the default, keep decompose ONLY for disconnected
  shells. Note: side-plate corner cuts are NOT chamfer blend records in
  either path (zero chamfer records on all 5 parts) — they ride the
  profiles.

- [ ] **Phase 2c: step-cut emission (the actual blocker, found 2026-06-11).**
  Branch wip/engine-honest-claims holds: honest voxel claims (half-cell
  symmetric tolerance, wall-thickness pitch, generous claim dilation),
  pure max-gain greedy (the area-window was a crutch for sloppy claims
  and mispicked once claims were honest), engine base profile + parity
  inner rings feeding the export, _clean_engine_polygon (self-
  intersecting mesh outlines extrude EMPTY in CadQuery, silently).
  With honest claims RackMount reads as U-channel-along-Y + base wings
  (2 extrudes, fewer steps than the user's 3) -- but a step-merged
  segment extrudes its LARGEST profile over the union span, and the
  mid-span differences (the merged runs' diffs vs that profile) must be
  emitted as CUT features. Nothing consumes the step_cuts flag yet, so
  the U build over-fills (19% vol) and scores worse end-to-end than the
  committed X-webs reading (fid 0.855). DO NOT merge the branch until
  cut emission exists: per merged run, diff each constituent run's
  profile against the segment profile and emit pocket/cut features over
  that run's span. Same machinery then handles Front Plate's window lips
  and the RackMount tab undersides.

- [ ] **Two side-plate rim creases still don't ROLL (matched 11/11, applied
  9/11).** Root cause chain fully diagnosed 2026-06-11: the join outline's
  corner arcs are threePointArc fits from mesh points, leaving micro-kinks
  at arc/straight junctions; OCC BRepFilletAPI refuses the 13-edge rim
  crease at any radius (leave-one-out exhausted). The proper fix — sharp
  polygon + OCC MakeFillet2d for EXACTLY tangent corner arcs — was
  prototyped and produces correct B-rep volumes, but the wire it returns
  corrupts the workplane tessellation chain (faces accumulate per step,
  final tess inverts to negative volume). Likely a pendingWires/object-
  stack pollution issue in _draw_corner_filleted_wire; the prototype is
  in the reflog (reverted working-tree change after a12837b). Resume in a
  worktree with time to study how polyline().close() threads the stack.

- [ ] **+X plate bore-rim bleed: per-face record separation for fused
  wrap bands (diagnosed 2026-06-12).** The single-body topology emits
  the +X plate's short-side/bottom outline-fillet bands as records that
  WRAP both plate faces with sparse-but-continuous bridge vertices (no
  void to split at). Midplane splitting fragments worse: at the plate
  corners the inner half of one band contacts the outer half of the
  neighboring band, and a global cross-side union guard reshuffles
  rather than fixes the chains (tried 2026-06-12, reverted). The robust
  approach is PLANE-ASSOCIATION: each blend record carries
  plane_region_ids — assign each face of a wrap record to the plate
  face whose plane it blends from, splitting by adjacency rather than
  geometry. The old body-decompose path got per-face records for free
  (per-piece topology); this reconstructs that property on the
  single-part path.

## Fillet UX/robustness round (user feedback 2026-06-12)
Done this round: apply order holes -> plate-edge wraps -> tab outlines
(bucket sort by category + extent); greedy maximal-subset fallback after
leave-one-out so a stitched group never fails outright (full broken-shell
gate per trial); topology viz repainted neutral gray (was stacked on the
fit-stage colors, hiding edge classification); >= 2.5s dwell on EVERY
stage visual (was only stitched/build/intent/apply/reconcile — others
flashed for a frame in non-step-through runs).

Deferred (need the de-stacked topology viz to diagnose):
- [ ] **Progressive build animation + feature tree.** Today both appear
  only at the very end. Wanted sequence: show the build animation +
  feature tree for the steps built so far, THEN the fillet intent stage,
  THEN extend the animation/tree as fillets get applied. Requires
  emitting timeline/tree snapshots incrementally from the export
  subprocess instead of one final payload.
- [x] **Intent overlay missing one side plate's edge group (root
  cause found 2026-06-12).** Not an overlay bug: the stitched stage's
  pass-2b axial-continuation union merged the +X plate's inner and
  outer outline rings into ONE group (11 instead of 12). Topology had
  fitted both faces' short-side bands to the same axis LINE between
  the plates, so the endpoint perpendicular-distance gate read ~0 mm
  while the actual vertex clouds sat a plate thickness apart. Fixed
  with a cloud-centroid perpendicular-offset gate (checked against
  both records' axes). The merged group had matched only one face's
  5-edge chain — the other face's ring was invisible to intent AND
  apply. After the fix + greedy subset: 12/12 matched, 12 applied.
  Found because the production grouping and the
  scripts/match_fillets_to_edges.py mirror DIVERGED (the mirror lacks
  pass 2b and was producing the correct 12) — see the extraction item
  under Pipeline.
- [ ] If OCC still refuses some chains on other parts, the next
  escalation is a full fillet-applicable-edge INVENTORY on the rebuilt
  solid (score every edge for blend feasibility up front) and match
  each stitched group to the most-similar feasible edge chain, so
  apply targets are valid-by-construction (user request).
- [ ] **Corner band enlargement.** The bled-fillet splitter works, but
  band length grows near corners (split halves keep corner-arc faces
  belonging to the adjacent band). Likely the same plane-association
  fix as the bore-rim item above.
- [ ] **Left and right edge fillets incorrectly grouped together** on
  the left side plate (two colors on the short sides). The user suspects
  the topology stage (now visually debuggable after de-stacking) — check
  whether one topology record genuinely spans both edges or the stitched
  union merges them.

## Pipeline (from the 2026-06-10 analysis)
- [ ] Feature-tree-driven CAD rebuild (replaces profile-extrusion + fillet
  matcher search; the structural fix for the blend apply ratio).
- [ ] Scale normalization at ingest so mm-absolute thresholds become ratios.
- [ ] Promote stitched-fillets (job_queue.py) and fillet apply
  (export_cadquery.py) into first-class pipeline stages with dataclass I/O.
  Urgency raised 2026-06-12: the grouping logic now exists in THREE
  copies (job_queue stitched block, scripts/match_fillets_to_edges.py,
  scripts/diag_plate_edge_groups.py) and they drifted — production had
  pass-2b axial unions the mirror lacked, which hid the ring-merge bug
  from every test while the user saw it on every run.
