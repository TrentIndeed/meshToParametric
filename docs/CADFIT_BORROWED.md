# Techniques borrowed from CADFit

Three capabilities ported from CADFit (github.com/ghadinehme/CADFit, an
IoU-driven CAD-program optimizer) after a head-to-head algorithm comparison.
This repo is a deterministic feed-forward *recognition* pipeline; CADFit is
*synthesis-by-search* scored on volumetric IoU. These modules import the parts
of CADFit's approach that plug cleanly into our recognition flow.

**Status: standalone + unit-tested, NOT yet wired into the default pipeline.**
They are pure, dependency-light functions with their own tests; each has a
documented integration point but is intentionally opt-in until validated
end-to-end the way the align-before-decimate change was (build quality on the
AI/scan set, not just unit tests). Wiring any of them into `convert.py` /
`recognize.py` / `export_cadquery.py` should be gated behind the same
`cad_vol_err` validation (`scripts/diag_ai_build.py`).

## 1. `pipeline/iou_refine.py` — residual-refinement post-pass (highest value)
CADFit's core loop: build the result, diff it against the target, patch the
difference (`over -> cut`, `under -> union`) until the residual is small.
Closes the gap this repo has — no global feedback, so a mis-recognized feature
silently propagates. `refine_against_target(target_mesh, recon_mesh)` finds the
largest under/over residual via **voxel IoU** (manifold3d isn't installed; voxel
IoU matches our existing `extrusion.py` occupancy machinery), proposes an
axis-aligned add/cut box, and accepts it only if IoU measurably improves.
- **Wire-in:** after `export_cadquery` builds the solid, run this against the
  cleaned mesh; replay accepted `ops` as CadQuery box add/cut, or (better)
  extrude the residual silhouette instead of a bbox. Promote the existing
  `volume_error_pct` *flag* into this *driver*.
- **Tests:** `tests/test_iou_refine.py` (missing-boss → add, extra-block → cut,
  perfect recon → no-op, IoU never decreases).

## 2. `pipeline/contour_fit.py` — greedy 2D primitive fitter
CADFit's `fit_primitives.py`: greedy segmentation of a 2D loop into lines + arcs
(PCA line fit, Kasa algebraic circle fit, spline fallback). Hardens base-profile
extraction on noisy AI/scan silhouettes (jittery slice outlines → a few clean
lines and arcs).
- **Wire-in:** post-process the base-profile loop in the extrusion/sketch path
  before it becomes a CAD sketch; emit arcs as real arc segments instead of
  polylines.
- **Tests:** `tests/test_contour_fit.py` (square→4 lines, circle→arcs,
  rounded-rect→lines+arcs, noise tolerance).

## 3. `pipeline/revolve_detect.py` — revolve-axis detection
CADFit's `analyze_revolve_axis.py`: a solid of revolution cut through its axis
yields a mirror-symmetric profile, so we find the profile's reflective-symmetry
axis (4 fixed + 2 PCA candidate axes, refined). Fills this repo's biggest
operation gap — it is otherwise extrusion-only, so turned/round parts can't be
modelled.
- **Wire-in:** in `recognize.py`, when a body/region isn't well explained by
  extrusion, test its profile with `detect_revolve_axis`; add a REVOLVE feature
  type (`models/feature.py`) + a CadQuery `revolve` branch in
  `export_cadquery.py` (and FeatureScript in `export_onshape.py`).
- **Tests:** `tests/test_revolve_detect.py` (rectangle/circle → candidate,
  L-shape → none, tilted axis recovered, 3D lift).

See the full comparison in the conversation that produced these; the one-line
summary: *CADFit measures fit-to-target by IoU and searches; this repo
recognizes named features and trusts its priors.* These three are the cleanest
borrowings in the search direction.
