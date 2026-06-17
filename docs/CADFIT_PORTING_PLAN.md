# Plan: port CADFit's robust algorithms; decide on segmentation

Sequel to `docs/CADFIT_BORROWED.md`. That doc landed three standalone modules
(`iou_refine`, `contour_fit`, `revolve_detect`) + wired the residual-refinement
post-pass into `export_cadquery`. This plan takes the next steps: make the
residual loop actually *reconstruct* (not just patch volume), wire in the other
borrowed algorithms, and run the experiment that decides whether to keep, demote,
or replace this repo's segmentation stage.

## Guiding principle
This repo's bet was "get every stage right, no global correction." CADFit's was
"forgive weak stages with a global IoU loop." We now have the loop
(`iou_refine`). The strategy is therefore: **lean on the loop so the fragile
stages (segmentation especially) stop being load-bearing**, and port CADFit's
pieces that make the loop strong. Every phase is gated behind the two metrics we
already trust:
- `scripts/diag_ai_build.py` → `cad_vol_err` + PASS/FAIL on the AI set (and add
  the Scan set) — the end-to-end accuracy oracle.
- `tests/test_step_comparison.py` must stay **18 passed / 7 xfailed** — the
  clean-CAD non-regression guardrail.

Region count is NOT a target (it misled us once; see
`memory/align-before-decimate-outcome`). `cad_vol_err` is the score.

---

## Phase 1 — DONE
Residual-refinement post-pass with axis-aligned box patches, gated on volume-
error improvement, skipped when the build is already good. Foundation + safety
net. Validated: AI6 20.3%→13.2%, no regressions, guardrail intact.

## Phase 2 — IN PROGRESS (2026-06-17)
Landed so far:
- **Greedy box-cover** (`iou_refine._box_cover`) replaced the single overshoot
  bbox — tracks L/stepped/distributed residuals. End-to-end: **AI6 20.3% → 9.6%
  (now PASS)**. AI1/AI14 correctly untouched.
- **Fillet-by-IoU sweep** (`export_cadquery._fillet_refine_solid`, OPT-IN, off
  by default): the volume/IoU-based alternative to blend-graph + stitching —
  sweep a global fillet radius, keep what improves voxel IoU. Validated (sharp
  target stays sharp; filleted target pulls in a fillet) but SLOW (~rebuild +
  voxelize per radius), hence off by default and v1 = one global radius.

Still open in Phase 2 (below): silhouette/revolve residual candidates (AI11 49.7%
and AI13 22.3% don't move — their residual is a shape/topology mismatch, not a
chunk a box-cover can patch), and PER-EDGE, locally-scored fillets.

### Original Phase 2 spec — Upgrade residual candidates: box → real geometry
The v1 box patches only help box-shaped residuals (why AI11/AI13 didn't move).
Replace/augment the box candidate in `iou_refine` / `_residual_refine_solid`:
1. **Extrude the residual silhouette.** For an under/over residual component,
   take its dominant axis (it's already aligned), slice it, extract the
   cross-section outline, run it through `contour_fit` (Phase 3), and
   extrude/cut that profile instead of a bbox. This is CADFit's per-residual
   re-analysis. Reuse `pipeline/extrusion.py` machinery on the residual mesh.
2. **Revolve candidate.** If the residual silhouette is mirror-symmetric
   (`revolve_detect`), propose a revolve instead of an extrude.
3. **Fillet/chamfer by IoU** (CADFit `refine_edges`): after the solid matches in
   volume, sweep fillet/chamfer sizes on candidate edges and keep those that
   improve IoU. NB volume is a weak signal for small fillets — score with a
   surface-distance / shape IoU on the blended region, not bulk volume.
4. **Iterate** like CADFit (`run_pipeline` loop): re-tessellate after each
   accepted op, recompute residual, repeat until residual < tol or no op helps.
**Exit metric:** AI11/AI13 `cad_vol_err` drops materially; AI6 improves further;
no clean-CAD regression.

## Phase 3 — Wire `contour_fit` into base-profile extraction
Post-process slice/base-profile loops through `segment_greedy_fit` before they
become CadQuery sketches, emitting real lines + arcs instead of dense polylines.
Hardens noisy AI/scan silhouettes and feeds Phase 2's residual silhouettes.
- Integration point: the profile path in `export_cadquery._extract_profile_and_holes`
  / `_draw_profile_with_arcs`, and the extrusion engine's outline extraction.
- Risk: clean CAD already has crisp profiles — gate so contour_fit only re-fits
  when the loop is "noisy" (many near-collinear points), or run it only on the
  `scan_mode` path. **Must hold 18/7.**

## Phase 4 — Add REVOLVE as a first-class feature
Close the operation gap. `revolve_detect` exists; wire it through:
1. `models/feature.py`: add `FeatureType.REVOLVE` (+ axis/profile in
   `sketch_geometry`).
2. `recognize.py`: when a body/region is poorly explained by extrusion (low
   `coverage_frac`), test its profile with `detect_revolve_axis`; emit a REVOLVE.
3. `export_cadquery.py`: a `revolve` build branch (CadQuery `.revolve(angle,
   axisStart, axisEnd)`); `export_onshape.py` / `feature_builder.py`:
   FeatureScript `opRevolve`.
- Validate on a turned part (add one to `models/`); ensure prismatic parts are
  unaffected (revolve only fires on the low-coverage branch).

---

## Phase 5 — RESULT (2026-06-17): KEEP segmentation; add a contour fallback

A/B ran (`scripts/diag_seg_ab.py`, `pipeline/contour_recon.py`) on clean CAD
(Mesh1–5) + AI (1,6,11,13,14). Path A = segment→fit→topology→recognize→export;
Path B = plan_extrusions→extrude→residual, NO segmentation. Both share the
residual loop, so the delta is segmentation's contribution. cad_vol_err %:

| part | A (recog) | B (no seg) | | part | A | B |
|---|---|---|---|---|---|---|
| Mesh1 | 9.5 | **2.9** | | AI1 | 0.9 | 1.1 |
| Mesh2 | **0.5** | 1.6 | | AI6 | **9.6** | 22.0 |
| Mesh3 | **3.0** | 23.1 | | AI11 | 49.7 | **21.0** |
| Mesh4 | **0.0** | 0.8 | | AI13 | **22.3** | 132.8 |
| Mesh5 | **2.2** | 5.9 | | AI14 | **1.3** | 18.8 |

**Verdict (pre-committed gate → "A wins broadly → KEEP"):** recognition wins
8/10, incl. 4/5 AI. Segmentation earns its keep. Do NOT replace it.

**Two findings that refine the plan:**
1. **Segmentation's value is concentrated in HOLE/POCKET identification, not the
   base shape.** Path B's failures (AI13 132%, AI14 18.8%, Mesh3 23%) are
   OVER-builds: the extrusion engine reproduces the base prism fine, but B
   can't cut holes precisely (the residual box-CUT is too crude for round
   holes), so it fills them. Recognition finds holes precisely — that's the
   win. So the base-shape half is segmentation-free-capable TODAY; the
   feature-subtraction half is where segmentation is load-bearing.
2. **There is a real subset where recognition mis-fires and B is better**
   (AI11: A 49.7% → B 21%). So the contour path is a valuable FALLBACK, not a
   replacement.

**Actionable outcome → "best-of-both" router (recommended next):** run
recognition; when its cad_vol_err is poor (e.g. > 15%), also run
`reconstruct_segmentation_free` and keep whichever matches the target better.
Cheap (only failing parts pay for B), self-correcting (the residual loop's
IoU-pick-the-better philosophy at the path level), and it rescues recognition's
worst cases (AI11) with zero risk to the wins. Wiring lives in `convert.py` /
`job_queue`; gate behind cad_vol_err + 18/7 like everything else.

**Caveat:** n=10, and Path B is handicapped (extrude-only, no hole cutting), so
the experiment slightly favors A. A fairer B (silhouette/cylinder residual
candidates that can cut round holes — Phase 2 remainder) might shift the AI
numbers; revisit after that lands. But the headline ("keep segmentation, add a
contour fallback") is robust to that caveat.

### Original Phase 5 design

**Question:** is this repo's segmentation stage (region-growing + ~10 peeling
passes + noise-robust merge) *better in the long run* than CADFit's
"don't-segment" approach (slice → planar-contour extraction → IoU loop)? It is
this repo's most fragile, highest-maintenance stage (confetti on AI meshes,
fillet-junction bugs — see `docs/SEGMENTATION_HANDOFF.md`). CADFit sidesteps it
entirely and lets the IoU loop tolerate the errors.

**Build the alternative path (behind a flag), don't rip anything out yet:**
A `contour_recon` path that mirrors CADFit:
1. Extract planar contours directly from the cleaned mesh (axis-aligned slices +
   DBSCAN-on-normals planes — port `extract_planar_contours` / `extract_sketches`),
   `contour_fit` for 2D primitives.
2. `analyze_loop_translation`-style depth (or reuse this repo's superior
   cross-section depth engine — it's already better, see the comparison).
3. Greedy IoU assembly + the Phase 2 residual loop. NO segment/fit/topology.

**A/B experiment** (`scripts/diag_ai_build.py` extended to run both paths):
run the recognition path and the contour_recon path on the **full set** — clean
CAD (Mesh1–5), AI1–15, Scan1–3 — and tabulate `cad_vol_err`, PASS rate, build
time, and feature-tree editability (named features vs boolean ops).

**Decision gate (pre-committed so it's honest):**
| Outcome on the data | Action |
|---|---|
| contour_recon ≥ recognition on AI/Scan `cad_vol_err` AND recognition still wins on clean-CAD editability | **Route by input:** clean CAD → recognition (named features); dirty/AI/scan → contour_recon. Demote segmentation to the clean path only. |
| contour_recon ≥ recognition everywhere (incl. clean) | **Replace** segmentation as primary; keep recognition only if its named-feature tree is worth maintaining for Onshape export. |
| recognition still wins broadly | **Keep** segmentation; invest the saved effort in the region-growing robustness TODO instead. |

**Why this is the right shape:** it's reversible (flag-gated), data-driven (the
same oracle we trust), and it refuses to delete a working stage on a hunch — but
it commits, in advance, to replacing it if the contour+IoU path actually wins
where segmentation hurts most (the AI/scan parts). The strong suspicion from the
code comparison is the "route by input" outcome: segmentation earns its keep for
clean prismatic parts with rich named features, while the contour+IoU loop is
more robust for the dirty inputs segmentation shatters on.

---

## Sequencing & risk
1. **Phase 2** first — biggest accuracy win, builds on shipped code, no new
   feature types. 2. **Phase 3** (contour_fit) — feeds Phase 2, low risk if
   scan-gated. 3. **Phase 4** (revolve) — additive, isolated branch. 4. **Phase
   5** — the big one; only worth it after the loop (Phase 2) is strong, because
   the whole premise is "the loop makes segmentation non-load-bearing."

Every phase: `cad_vol_err` must improve-or-hold on AI/Scan, and
`test_step_comparison` must stay 18/7. Validate end-to-end before flipping any
default, exactly as the align-before-decimate change was.
