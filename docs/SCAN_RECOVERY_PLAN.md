# Scan Cleaning & Design-Intent Recovery Plan

How to turn a noisy 3D scan of a real mechanical part into the clean
parametric model it was manufactured from — "reverse engineering" in
the CAD sense. The clean-CAD path (Mesh1–5) is solid; this is the
scan-specific front-end + intent recovery that feeds into it.

Scope: **prismatic / mechanical parts** (brackets, housings, plates,
the Scan2 industrial part). Genuinely organic shapes (the Scan1
spanner) get CLEANING only and belong on `/clean`, not parametric
recovery — a parametric feature tree is the wrong model for them.

---

## 0. The principle

A scan violates every assumption the clean pipeline makes: surfaces are
bumpy not planar, holes are lumpy not circular, edges are rounded/jagged
not sharp, density is uneven, the part is arbitrarily oriented, there's
measurement noise and occlusion gaps, and the scale may be off. The job
is to **recover the design intent** — the small set of clean primitives
and the constraints between them (parallel, perpendicular, coaxial,
equal, symmetric, round dimensions) that the part was designed with —
and rebuild from THAT, not from the triangle soup.

Guiding rule (user, confirmed): **clean + align + regularize FIRST,
then run recognition.** Trying to read features off raw scan noise is
working on garbage (the wrench over-extrude is a symptom).

---

## 1. What already exists vs. what's missing

Already in the pipeline (the bones of intent recovery):
- `ingest.py`: merge/decimate/fix-normals/fill-holes, optional Poisson
  watertighting, remesh, `_normalize_scale` (snaps to a canonical size).
- `fit.py`: RANSAC + least-squares plane/cyl/cone/sphere fitting;
  `_snap_params` snaps scalars to round values (int/½/¼) and directions
  to PRINCIPAL AXES; `_merge_coincident_cylinders` (coaxiality);
  `snapped_params` on each FittedSurface.
- `onshape/feature_builder._regularize_profile`: axis-snapped straight
  runs + exact tangent corner arcs (2D intent recovery for outlines).
- `metrics.cross_section_fidelity`: the scan-vs-model score to gate on.

The three gaps that make all of the above fail on scans:
1. **No feature-preserving denoise.** Poisson rounds sharp edges; naive
   smoothing rounds everything. Scans need denoising that KEEPS edges.
2. **No canonical-frame alignment.** `_snap_params` snaps to the WORLD
   axes — meaningless unless the part is already world-aligned. Scans
   are arbitrarily oriented, so axis snapping, the extrusion engine's
   axis pick, and "prismatic" assumptions all misfire (this is the
   wrench: a non-aligned part whose silhouette is read on the wrong
   axis and extruded past its true width).
3. **No GLOBAL constraint inference.** Snapping is per-primitive. Real
   intent is relational: faces parallel/perpendicular to EACH OTHER,
   holes coaxial / equal-radius / in a symmetric pattern, dimensions
   that repeat. Recovering those relations is what makes the result
   look designed instead of approximated.

---

## 2. The recovery pipeline (new/extended stages)

Ordered; each imposes more structure than the last.

### A. Scan-aware cleaning (mesh level)
- **Feature-preserving denoise — IMPLEMENTED 2026-06-14.**
  `ingest._bilateral_denoise` (Zheng et al. bilateral normal filtering,
  vectorized numpy): soft space+normal weighting removes surface noise
  while preserving true sharp edges and, crucially, WITHOUT faceting
  smooth curves. Default `denoise_method="bilateral"` (knobs:
  `denoise_sigma_s`=0.3, `denoise_normal_iter`, `denoise_vertex_iter`);
  the pymeshlab two-step is kept as an opt-in fallback. Validated on a
  synthetic noised box (ground truth): bilateral keeps edge/corner error
  ~25x below Laplacian; on the spanner it smooths surface noise (median
  adj-normal angle 4.9->4.0 deg) while the two-step facets it (false >30
  deg edges 0.6%->5.6%, bilateral stays 0.5%). Tests in
  `tests/test_scan_denoise.py`; dev harness `scripts/diag_scan_denoise.py`.
- Remaining stage-A gaps: explicit sharp-feature tagging to protect edges
  through remesh/Poisson; uniform feature-preserving remesh.
- **Sharp-feature detection**: tag edges where the dihedral angle is
  consistently high → protect them through denoise/remesh/Poisson.
- **Repair**: islands/spikes/non-manifold removal, occlusion-gap hole
  fill (existing), make watertight. Keep Poisson as a LAST resort,
  feature-aware (don't round tagged edges).
- **Uniform remesh** to a consistent density (existing `_remesh`),
  feature-preserving.
- **Scale/units**: keep `_normalize_scale`; add unit inference where
  possible.
- Output: a clean, watertight, uniform, edge-preserved manifold.

### B. Canonical-frame alignment (THE missing front-end) — IMPLEMENTED 2026-06-13
`pipeline/align.py::align_to_canonical_frame` (dominant orthogonal
normal triple, OBB fallback, axis-reorder so length→X/thickness→Z) +
feature-preserving denoise (`ingest._denoise_scan`, two-step
smoothing), both gated by `IngestConfig.align_canonical` /
`denoise_scan`. Validated on Scan1: the wrench went from engine
coverage **0% → 92%**, dominant face normal **19.6° → 0.6°** off-axis,
bbox resolved to a flat **165×27×9.3 mm** part. The engine still
over-segments the thickness (stage 5 / the engine span-honesty work) —
but the part is now in a readable, axis-aligned pose. Original design
below:
- Find the part's natural orthogonal frame and rotate it onto the world
  axes, so every downstream "snap to axis" / "prismatic" assumption
  holds and the extrusion engine reads the right directions.
- Method: collect large planar regions (or face-normal clusters), find
  the dominant mutually-orthogonal normal triple (RANSAC over normal
  histogram / Gaussian-sphere clustering), build a rotation that maps
  it to X/Y/Z. Fall back to oriented-bbox / PCA for parts without three
  clean planes. Record the transform so the model can be returned in
  the original pose if needed.
- This alone should fix the wrench class of failure.

### C. Robust segmentation (on the cleaned + aligned mesh)
- Existing region-growing, but with noise-tolerant normal thresholds
  and curvature-aware splitting tuned for residual scan scatter.
- The cleaning + alignment upstream is what makes this tractable.

### D. Primitive fitting + per-primitive snapping (extend existing)
- Keep RANSAC plane/cyl/cone/sphere + `_snap_params`, now meaningful
  because the part is aligned. Robust thresholds scaled to scan noise.
- Snap each cylinder to a clean radius + axis = a canonical direction;
  each plane normal to an axis.

### E. Global constraint inference (the heart of intent recovery)
A pass over the fitted primitives that infers and ENFORCES relations:
- **Parallel / perpendicular**: near-parallel plane normals → exactly
  parallel; near-perpendicular → exactly perpendicular (snap as a set,
  not individually).
- **Coaxial** (extend `_merge_coincident_cylinders`): cylinders with
  near-coincident axes → one shared axis.
- **Equal radius / equal dimension**: holes/bosses within tolerance →
  the same radius; repeated lengths → equal (build a dimension graph
  and solve for the consensus values).
- **Symmetry**: detect mirror planes and rotational symmetry (hole
  patterns, symmetric outlines) → enforce so a 4-bolt pattern is
  exactly symmetric.
- **Angle snapping**: dihedral / outline angles near 90°/45° → exact.
- **Dimension rationalization**: snap consensus dimensions to round
  numbers / common fractions, optionally to a detected grid.
- Implementation: a constraint graph + a small iterative solver
  (snap → propagate → re-snap) rather than per-primitive one-shots.
  Start greedy/pragmatic; a real least-squares constraint solver is the
  full-ambition version.

### F. Sketch / profile regularization (first-class stage)
- Promote `onshape/feature_builder._regularize_profile` (axis-snapped
  straights + tangent corner arcs + circle recovery) out of the Onshape
  exporter into a shared pipeline stage, and run it on the recovered
  cross-section profiles. This is "recover circles and 90° corners" in
  2D. Both CadQuery and Onshape export consume the regularized sketch.

### G. Reconstruction
- Feed the cleaned + aligned + constraint-snapped primitives and
  regularized sketches into the EXISTING feature-tree builder + export
  (CadQuery / Onshape). No new export machinery — the win is that the
  inputs are now clean and intent-bearing.

### H. Confidence scoring + validation + iterate
- Score each recovered feature by fit residual + how much snapping
  moved it; surface low-confidence features for review instead of
  silently fabricating.
- Gate the whole result on `cross_section_fidelity` + volume vs. the
  scan; loop (re-segment / re-fit / relax constraints) where it's poor.
- Honest reporting: scans won't hit clean-CAD accuracy; show the gap.

---

## 3. How this fixes the known failures
- **Wrench extrudes a jagged side profile past the true width** →
  canonical-frame alignment (B) + profile regularization (F) + the
  honest extrusion-span work already noted; the engine reads the right
  axis on an aligned, regularized outline.
- **Residual volume error 74–173%** → the part is reconstructed as the
  right prismatic shape once primitives are snapped to clean canonical
  forms and constrained (E), instead of approximated from noise.
- **Poisson rounds sharp edges** → feature-preserving denoise +
  sharp-edge protection (A) replaces blind watertighting.
- **Scan1 only recognizes a BASE feature; Scan3 drops fit regions** →
  robust segmentation/fitting on the cleaned mesh (C/D).

---

## 3b. "Needs cleaning" detection — IMPLEMENTED 2026-06-14
A scan isn't the only dirty input: **AI-generated meshes** (Meshy, Tripo)
are LOW-poly but have wobbly near-flat faces and approximate angles, so the
`scan_face_threshold` (raw faces > 100k) misses them. `ingest._surface_wobble`
adds a face-count-INDEPENDENT trigger: median dihedral among near-flat
adjacent faces — clean CAD ~0 deg, AI/scan ~5 deg. With `scan_auto`, a mesh
above `dirty_wobble_deg` (1.5) gets the cleanup front-end too
(`cleanup_source="dirty_mesh"` in metadata). Validated: clean Mesh1-5 stay
at 0.0-0.4 deg (untouched); a 0.3%-noised low-poly mesh reads 4.9 deg and is
cleaned. NOTE: the front-end currently does denoise + align; actually
SNAPPING near-flat -> flat and near-90 -> 90 (what makes AI meshes crisp) is
stage E/F regularization, still to build.

## 4. Prismatic vs organic
A classifier up front (planarity ratio, how much of the surface is
explained by a few primitives, symmetry): prismatic → full recovery;
organic → cleaning only, routed to `/clean` with an honest "this part
isn't a good fit for parametric recovery." The spanner is the canonical
organic case; the Scan2 industrial part is the canonical prismatic one.

---

## 5. Tooling
- pymeshlab (already a dep) — denoise, feature-preserving smoothing,
  remesh, Poisson. trimesh — repair/sections. scipy/sklearn — normal
  clustering, RANSAC, constraint solving. numpy throughout. Possibly
  open3d for robust normals/denoise (evaluate; avoid a heavy new dep if
  pymeshlab covers it).

---

## 6. Where each stage lands
- `pipeline/ingest.py` — A (feature-preserving denoise, sharp-edge
  protection); keep Poisson as feature-aware last resort.
- `pipeline/align.py` (new) — B canonical-frame alignment.
- `pipeline/segment.py` / `fit.py` — C/D noise-robust tuning; D already
  has `_snap_params` to extend.
- `pipeline/intent.py` (new) — E global constraint inference.
- `pipeline/regularize.py` (new) or shared module — F, generalizing
  `_regularize_profile` (move it out of `onshape/feature_builder.py`).
- `pipeline/recognize.py` + exports — G (unchanged consumers).
- `metrics.py` + recognize — H confidence + gating.
- `config.py` — a `ScanConfig` / extend IngestConfig with denoise +
  alignment + constraint tolerances; a prismatic/organic gate.
- `tests/` — per-stage on Scan2 first; volume-only gates (no reference
  STEP for scans) + the fidelity metric.

---

## 7. Suggested sequencing (pragmatic first, ship per-scan wins)
1. **Canonical-frame alignment (B)** + feature-preserving denoise (A).
   Biggest single lever; directly attacks the wrench failure. Validate
   on Scan2 (prismatic) — expect a large volume-error drop.
2. **Promote the sketch regularizer (F)** into a shared stage; run it
   on recovered profiles.
3. **Global constraint inference (E)** — start with parallel/perp +
   coaxial + equal-radius (greedy), then symmetry, then the dimension
   network.
4. **Confidence scoring + gating (H)**; prismatic/organic classifier (4).
5. Revisit the extrusion-engine axis/span honesty on the now-cleaned,
   aligned input (the wrench's second half).

The full-ambition version (a least-squares constraint solver, learned
symmetry, dimension grids) layers on top of this pragmatic spine — same
stages, deeper solvers.

---

## 7b. Locked decisions (2026-06-13)
- **First target: Scan1 (wrench/spanner)** — the part with the visible
  failure; alignment + denoise attack it directly. (Semi-organic, so
  it's also a stress test of the prismatic/organic gate.)
- **Ambition: full constraint solver** — build toward the least-squares
  constraint graph (E), not just greedy per-primitive snaps.
- **Start with: canonical-frame alignment + feature-preserving denoise**
  (stages B + A). Biggest lever, the wrench fix, and the foundation
  everything else stands on.

## 8. Open decisions
- First target part + ambition for v1 (pragmatic snapping spine vs.
  full constraint solver) — drives where to start.
- Denoise library: pymeshlab-only vs. add open3d.
- How aggressive to snap dimensions (round-number bias can fight a part
  designed in another unit system) — make it tolerance-driven and
  reportable, never silent.
