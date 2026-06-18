# Plan: make this pipeline better & more robust than CADFit

Strategy doc from the head-to-head study (2026-06-18) vs CADFit
(github.com/ghadinehme/CADFit). Companion to `docs/CADFIT_BORROWED.md` (what was
ported) and `docs/CADFIT_PORTING_PLAN.md` (residual loop / segmentation A/B).

## TL;DR
CADFit's one structural edge is a **global IoU loop** that forgives weak
per-stage estimates. This repo's edges are **strong specialized stages + dirty
(scan/AI) input cleanup + named-feature editability + determinism + a deployed
product**. The residual loop is already ported. The single highest-leverage move
to pull ahead is **IoU-driven BASE selection** (Pillar 2) — bring CADFit's
"try many candidates, keep the best by IoU" to the base shape, not just the
residual.

## Both are sketch + operation systems
Neither models freeform/organic geometry (lofts, sweeps, NURBS, scanned organic
shapes). That is the shared ceiling. Everything below is within "designed
mechanical parts."

## Model-input limitations

| Dimension | This repo | CADFit |
|---|---|---|
| Domain | Prismatic parts + bodies of revolution | General mechanical CAD (extrude+revolve+boolean) |
| Operations | Extrude, holes, pockets, bosses, slots, fillets, chamfers, revolve | Extrude, revolve, fillet, chamfer, union, cut |
| Input prep | Heavy: denoise, Poisson/voxel repair, canonical alignment, deterministic decimation — built for dirty scans/AI meshes | Alpha-wrap manifoldify; assumes cleaner input; image->3D front-end (Hunyuan) |
| Watertight | Needs it (repairs if not) | Needs it (alpha-wraps) |
| Orientation | World-axis-relative; needs alignment; arbitrary per-feature tilt breaks it | Pose-tolerant (DBSCAN on normals) |
| Multi-body | Splits connected components; tangent/touching bodies NOT separated | Boolean composition |
| Complexity | Many-feature parts -> wrong base recognition; UI caps >10K faces | Scales via greedy search but cost balloons |
| Determinism | Deterministic (fast-simplification) | Greedy + optional DINOv2 (stochastic-ish) |
| Platform | Cross-platform, productized (web UI, Onshape) | Linux-oriented (SIGALRM build-timeouts, Xvfb); does NOT port cleanly to Windows |
| Needs target mesh | No (recognition) | Yes — IoU-to-target is the optimizer oracle |

## Head-to-head finding (2026-06-18)
`scripts/diag_cadfit_compare.py`, bbox-normalized voxel IoU, 6 meshes:
- **CADFit completed 0/6 within 180s on Windows** (4 timeouts, 2 no-output) — its
  per-build timeouts use Unix SIGALRM, so slow CadQuery builds hang. NOT a
  verdict on its algorithm — a verdict on its portability.
- This repo built all 6 deterministically (IoU 0.75-1.0 except AI13 0.15).
- Only clean head-to-head (AI3, untimed earlier run): CADFit 0.973 / this 0.953
  — comparable, CADFit marginally ahead.
- => A fair ALGORITHMIC benchmark requires CADFit on **Linux (WSL/Docker)**.

## The plan — five pillars

### Pillar 1 — Finish closing the IoU loop (match CADFit's self-correction)
Residual loop is shipped (box/prism/cylinder, dual real-volume op-set, contour +
revolve routers). Remaining:
- **Per-edge fillet-by-IoU** to replace the fragile blend-graph/stitching path
  (`_fillet_refine_solid` exists but is global-radius + opt-in; make it
  per-edge, locally scored, default-able).
- **Extrude the residual SILHOUETTE** (not just box/cylinder) for arbitrary
  prismatic residuals.

### Pillar 2 — IoU-driven BASE selection (highest leverage)
The 5 hard AI fails (AI2/4/6/11/12) fail because recognition commits ONE base
shape early and it's wrong (AI11's base had ~0 IoU). CADFit avoids this by
building many candidate ops and keeping the best by IoU. Bring that to the base:
generate several base candidates (largest-plane extrude, alt-axis extrude,
revolve, contour), run each through the residual loop, keep the best final IoU.
Converts "recognize and hope" into "fit until it matches" where it matters.

### Pillar 3 — Exact-boolean IoU (manifold3d now installed)
Swap the residual loop's VOXEL IoU for exact-boolean IoU (manifold3d) — sharper
decisions; fixes artifacts like AI13's "right volume, wrong shape" (0.15 voxel
IoU at 8.6% vol-err). Matches CADFit's metric precision.

### Pillar 4 — Push the robustness lead CADFit can't match
- **Tangent multi-body split** (ShaftAdapter: two stacked tangent revolves in one
  connected component) — split by geometry/feature, not just components.
- **Sub-feature revolves** (small screws revolved at small scale) — per-feature
  revolve detection, not just whole-body.
- **Scan/AI cleanup front-end** — keep investing (CADFit lacks it).
- **Determinism** — keep it (CADFit is stochastic-ish). Chase the residual
  non-determinism downstream of decimation (AI3 batch anomaly: 1.2% standalone
  vs 23% when built 6th in a batch).

### Pillar 5 — Keep the structural advantages
Named feature tree + live Onshape (CADFit emits less-editable boolean programs);
the deployed product. Don't trade these for raw IoU.

## Sequencing
Pillar 3 (exact-boolean IoU) first — cheap, sharpens every loop decision and the
benchmark metric. Then Pillar 2 (base selection) — the real lever for the hard
parts. Then Pillar 1 remainder + Pillar 4 coverage. Every change gated on the
clean-CAD guardrail (test_step_comparison 18/7) + the AI cad_vol_err set.

## Open: fair benchmark
Run CADFit in a Linux container (its deps have Linux wheels; SIGALRM works there;
skip the optional DINOv2 `--filter-sketches`). Reuse `diag_cadfit_compare.py`'s
bbox-normalized voxel-IoU metric on both outputs for an apples-to-apples result.
