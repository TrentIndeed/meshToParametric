# Eyes-on findings from real parts (2026-06-14)

Visual review of the pipeline on real meshes (Mesh5 RackMount + AI/Meshy parts
in `models/AI*/`). These are observed, concrete issues to fix — recorded so
they aren't lost. Tagged **[cosmetic]** (output is right, visual is messy — low
priority) vs **[structural]** (affects the output — real blocker), per the
quality framework in `ROADMAP.md`.

---

## Mesh5 / RackMount (segmentation visuals)

1. **[cosmetic] Sharp triangles at fillet-fillet corner junctions.** On the
   side-tab fillet transitions, flat triangles get pulled into the curved
   fillet region and render as wedges poking into the plate. Already documented
   in `SEGMENTATION_HANDOFF.md` (marked "stuck"). Fillet recognition still
   succeeds and Mesh5 builds correctly (235k vol, 1.3%, all 12 fillets OK) — so
   this is a classification/visual artifact, NOT a structural blocker. Do not
   over-invest; let general segmentation robustness subsume it.

2. **[tech-debt] Brittle corner-split fix.** `segment._split_long_fillet_into_
   corners` separates the left side plate's converging corner fillets using
   HARDCODED magic numbers (CORNER_RADIUS=2.5mm, long ≥ 4× mid, short ≥ 5mm,
   fillet r 2-3mm). Works for RackMount, will NOT generalize to other fillet
   radii / plate thicknesses / AI parts. Acceptable now (gated to a specific
   long-thin-curved shape, low risk elsewhere), but should be replaced by a
   geometry/topology-driven corner separation, not per-part thresholds.

---

## AI15 / RibbedBlock (full pipeline, ends "pretty good")

The overall result is decent, but the intermediate stages show where the
front-end stops short:

3. **[structural] Regularize only crisps INTERIOR flat faces.** The big inner
   flats become very flat (good), but the OUTER areas and fillets stay rough/
   bumpy. Root cause: `regularize_mesh` only projects a vertex when ALL its
   incident regions are planar+snapped (conservative, to protect curves) — so
   the outer boundary, fillets, and any rough/curved region are left untouched.
   → Extend regularization to the outer boundary (and feed a regularized
   silhouette to the base profile), not just confident interior planes.

4. **[structural] Fillets between surfaces stay rough / inconsistent.**
   Regularize only snaps PLANES; it does nothing to fillets, so they remain
   bumpy and varying instead of consistent constant-radius rolls. → A fillet/
   radius regularization sub-stage (snap a rolled-edge region to a single
   clean radius arc) is missing.

5. **[structural] A flat side got fit as a CYLINDER.** On a rough segment, the
   primitive fit picked a large-radius cylinder instead of a plane. → Fit
   robustness on noisy regions (bias toward plane when the cylinder radius is
   huge / residual is marginal), or run fit AFTER better regularization.

6. **[structural] Extruded base SKETCH PROFILE is bumpy; corners not 90°.** The
   outer profile that gets extruded is wavy in spots and the lines into
   filleted corners don't meet at clean right angles. This is EXACTLY what
   Stage F (promote `onshape/feature_builder._regularize_profile` — axis-snapped
   straight runs + tangent corner arcs + 90° snapping — into a shared sketch-
   regularization stage) is for. The outer-boundary regularization (#3) feeds
   it. → Highest-leverage visible-quality fix for the "looks bumpy" complaint.

---

## AI13 / TaperedHousing (upload)

7. **[UX + repair] Non-watertight mesh rejected on upload; the mesh is badly
   broken (not just open).** DIAGNOSED 2026-06-14: AI13 has **0 open boundary
   edges but 23,217 NON-MANIFOLD edges and ~18,000 components** — it's
   degenerate "soup" (overlapping/self-intersecting surfaces, T-junctions), not
   a mesh with holes. So hole-closing is irrelevant, and Poisson does NOT close
   it either (poisson-only result still non-watertight; quadric decimation
   after it also breaks watertightness). trimesh voxel-remesh needs `skimage`
   (not installed). **Robustly repairing arbitrary broken soup needs a manifold
   library** (`manifold3d`, or pymeshlab alpha-wrap whose plugin DLL is missing
   here) — a dependency decision. TWO problems: (a) repair this class of mesh
   (needs the lib); (b) the preview HARD-rejects (`can_process=False`,
   app.py:436-441) and bounces the user to /clean — should instead either auto-
   repair or let the build attempt it with a warning. (b) is the quick UX win;
   (a) is the real repair work and needs a dep.

---

## CONCLUSION (2026-06-14): the bottleneck is UPSTREAM (segmentation + fit)
Built three downstream cleanup passes — mesh-plane regularization, 2D sketch-
profile regularization (Stage F), and fillet-cylinder regularization — and ALL
THREE no-op on the real AI parts for the same reason: segmentation/fit don't
hand them clean regions to work on. The profile was already cleaned by the mesh
pass (or not cleanly detectable); the fillets are never isolated as cylinder
regions (so the fillet pass finds nothing); flats get mis-fit as cylinders
(#5). The downstream regularizers are correct and SAFE (self-gated, guarded,
no-op when input is bad), but they're starved. So the #1 lever is unambiguously
**segmentation + primitive-fit robustness on rough/dense AI meshes** — once that
produces clean plane + fillet regions, the (already-built) regularizers fire.
Stop adding downstream passes; fix the input.

## How these map to the roadmap
- #3, #4 → extend Stage E (regularization) to outer boundary + fillets.
- #5 → Stage D fit robustness on noisy regions.
- #6 → **Stage F sketch-profile regularizer** (the biggest win for the
  "bumpy / not-90°" visual complaint).
- #7 → upload/preview UX (auto-repair, don't reject).
- #1, #2 → subsumed by general segmentation robustness (the AI-data lever);
  do not hand-tune Mesh5 further.
