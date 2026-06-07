# Fillet Detection & Hole Recognition Handoff

> ## ⚠️ SUPERSEDED — read this banner before the body
>
> The pipeline below this banner describes the **collinear-merge / reconciliation** flow that was used before the **stitched-fillets stage** replaced it. The notes are kept for historical context but the code they reference has been removed or rewritten. Current state, as of the rewrite:
>
> - `reconcile_fillets.py`, `merge_blend_records_for_export`, `_detect_mesh_fillets`, `_detect_mesh_fillet_clusters`, `_detect_fillet_strips_from_mesh`, `_filter_edges_by_mesh_fillets`, `_estimate_fillet_radius_from_mesh`, `_apply_top_fillet_to_solid`, `_find_body_top_face`, the swept-tube and edgewalk debug stages, `swept_tube.py`, `fit_fillets_along_edges.py`, `extract_cad_edges.py`, `_diag_fillet_match.py`, `tests/test_export_fillet_consistency.py`, and `tests/test_fillet_detection.py` are all **deleted**.
> - The stitched-fillets stage in [`mesh2param/viz_server/job_queue.py`](../mesh2param/viz_server/job_queue.py) groups every per-body fillet record into cross-body clusters via vertex KDTree proximity + axis-aligned continuation, then packages each cluster as `{group_id, vertices: (N,3) ndarray, n_records}` onto `job._stitched_clusters`.
> - The CAD subprocess pickle now carries `stitched_clusters` instead of `peeled_fillets`. `export_cadquery(..., stitched_clusters=...)` calls a rewritten `_apply_recognized_fillets_to_solid` that matches sharp solid edges to clusters by vertex proximity and **measures the radius per edge** from the cluster's vertices (90th percentile perpendicular distance from edge line). No upstream record radius is ever read.
> - **Known broken (next session):** the apply pass is still placing fillets at short corner-meeting edges instead of along the long edges that should actually be filleted. Suspected root cause is the matcher's "best mean proximity" metric favouring short edges that sit fully inside a cluster's vertex bbox over long edges that only have their middle inside it. The intent overlay also doesn't render any surface patches because no edges are matching the way we'd expect. Investigate the edge-selection metric (mean → median or coverage-fraction?) and the edge enumeration on the rebuilt LLM-built solid before anything else.

**Original status (collinear-merge era):** in-progress. The blend-graph (Stage 4.5) and feature tree (Stage 5) now show the fillets the user expects, but the CAD build (Stage 6) applies wrong fillet count/sizes and the through-hole detection produces inconsistent radii because Stage 2 segmentation is over-merging bore + adjacent surfaces. This doc gives a new Claude session the context to attack it without re-running dead ends.

**Scope:** Stages 3 (`fit.py`), 4.5 (`peel_via_topology.py`), 5 (`recognize.py`), and 6 (`export_cadquery.py`) of the meshToParametric pipeline, focused on the rackmount mesh (`models/Mesh5/RackMount.3mf`).

A separate handoff exists for the upstream segmentation/corner-triangle issue: see `docs/SEGMENTATION_HANDOFF.md`.

---

## The user's expectation for the rackmount mesh

The reference part is `models/Mesh5/RackMount.3mf` — a U-bracket with:
- 2 vertical side plates (Body 0 and Body 1)
- 1 horizontal base plate (Body 2)
- 4 mounting tabs on the perimeter of the base plate, each with a through-hole + countersink
- 2 main through-holes through the side plates
- Rolled-edge fillets on all plate perimeters and 3D corner blends where they meet

**What the user wants in the final CAD build:**
- Roughly **12 fillets total** — not the 26 that come out of Stage 4.5. The 26 are fragments of the same physical fillet, split across bodies and within-body by segmentation.
- The 4 base-plate through-holes recognized as **one symmetric pattern**, not 4 separate features. They should all have the *same* radius (~6mm, the user asserts; the cylinder fitter currently gives 4mm-vs-6.37mm).
- Through-holes on the side plates should pass through unchanged.
- Each side plate has 2 perimeter rolled-edge fillets that should each combine from ~5–7 segments.

---

## Current end-to-end state on the rackmount

### Stage 4.5 (Blend graph) — `mesh2param/pipeline/peel_via_topology.py`

Working. Produces 26 cyan blend records covering:
- 8 small corner bevels (4 per side plate)
- 4 perimeter rolled edges of the base plate
- 3 base-plate corner curls (the small 2-3mm curls the user originally requested)
- Several large 3D-corner sphere fits and perimeter wrap regions
- 2 sub-mm slivers (mesh tessellation noise)

The visualization in the web UI is the source of truth for "what the user saw cyan".

### Stage 5 (Feature recognition) — `mesh2param/pipeline/recognize.py`

Now wires Stage 4.5 records through `_find_fillets_and_chamfers(..., topo_blends=...)` so any blend not absorbed by Strategy 3's stricter strip-cylinder check ends up as a `FILLET` feature rather than silently dumped into the base extrusion.

### Stage 6 (CAD build) — `mesh2param/pipeline/export_cadquery.py`

Two issues here that we've made progress on but haven't fully solved:

1. **Most recognized fillets fail to apply** because the matcher in `_apply_recognized_fillets_to_solid` looks for an existing CAD edge that's parallel to the record's `axis_dir` and at perpendicular distance ≈ `radius`. The base-extrusion CAD solid has only 12-20 sharp edges per body; many recorded fillets have no corresponding edge at the right distance, so they get rejected.
2. **The radii are sometimes wildly wrong** because cylinder fits on small partial-arc patches inflate (12–12000mm radii on regions that actually have ~1.4mm curl thickness).

---

## What we've changed in this round (what to NOT redo)

These are all in place and the fillet-consistency test (`tests/test_export_fillet_consistency.py`) verifies the export records still trace back to Stage 4.5 records.

### `mesh2param/pipeline/peel_via_topology.py`

- **Bbox-min radius cap:** when the cylinder/cone fitter gives a radius >3× the region's smallest bbox dimension, fall back to `bbox_min` as the radius. This catches the "12000mm radius from a 1.4mm curl" case. Lines ~498-509.
- **Records carry `source_fit_type`:** so downstream merge can drop SPHERE-fit records (their `axis_dir` is a plane neighbor's normal, not a real edge tangent — useless to the CAD matcher). Line ~526.

### `mesh2param/pipeline/reconcile_fillets.py`

Added `merge_blend_records_for_export(records)`:
- Drops SPHERE-fit records.
- Drops records with `radius < 1.0mm` (mesh tessellation noise — sub-mm "fillets" are not real design intent).
- Union-finds collinear records (same axis line, comparable radii) and emits one merged record per cluster with arc-length-weighted radius.
- Does NOT do same-body-same-radius grouping. We tried that and it broke the matcher: pre-merging 4 corner records into 1 dropped 3 edges from getting filleted (the matcher finds one best edge per record, then buckets).

### `mesh2param/pipeline/export_cadquery.py`

- **LLM call disabled** at lines ~196-217. Heuristic-only construction until the heuristic baseline is solid enough that an LLM reorder is a measurable improvement.
- **Matcher fan-out** rewritten to find ALL parallel edges per record within ±15% of best perpendicular distance (was: single best edge). Lines ~3509-3565.
- **Per-record bucketing** instead of cross-record bucketing. Each record's matched edges form their own OCC fillet call. Different records that happen to share a radius stay separate operations. Lines ~3517-3565.

### `mesh2param/pipeline/recognize.py`

- **`_group_symmetric_holes(hole_features)`** added at line ~294. Pairwise union-find: same hole type, same principal axis index, radii within 5% (or 60% for THROUGH_HOLE — the cylinder fitter is unstable for short stubby cylinders). Anti-parallel axes (+X / −X) treated as the same axis for through-holes since both faces are valid entry points. Multiple matched holes get emitted as ONE feature with `sketch_geometry["centers"] = [...]`.
- **Thin-plate axis override** at line ~487: when the body is plate-shaped (smallest dim ≪ middle dim by 3×), through-hole axes are forced to the body's thin direction. Caught the case where the cylinder fitter returned axis Y for one hole and Z for its mirror. **NOTE: this might over-correct on parts where through-holes legitimately go through tab side faces, not through the plate face. Reconsider if other meshes regress.**
- **Group radius taken as MAX** of fitted radii. **The user has questioned this** — for the rackmount, MAX gives 6.37mm but that value comes from over-merged regions, not the true 4mm bore. Switching to MIN, MEDIAN, or "lowest-RMS-fit wins" would all be defensible improvements.

### `mesh2param/pipeline/segment.py`

- **`_classify_surface_type` SVD check** added at line ~1995: a region with `mean_curv < 0.1` and `normal_spread < 0.1` is now classified as PLANE only if its normals' second-largest singular value is < 0.5× the largest. Otherwise it falls through to CYLINDER/SPHERE classification. This caught small 3D corner fillets whose curvature was too gentle for the old PLANE threshold.

### `mesh2param/viz_server/job_queue.py`

`peeled_fillets` (the input to the CAD subprocess) is now built by:
1. Filtering raw blend records by absolute radius bounds `[0.2, bbox_diag * 0.25]`mm.
2. Tagging each with `body_index` for downstream lookup.
3. Calling `merge_blend_records_for_export` to drop spheres/sub-mm and merge collinear records.

### Test added

`tests/test_export_fillet_consistency.py` — 5 tests that verify:
- Every export record traces back to a Stage 4.5 blend record (no synthetic records).
- Export radius matches the length-weighted source radius within 5%.
- Export length covers the source span.
- Export `axis_point` lies within 2.5mm of every source's axis line.
- Sphere fits and sub-mm records get dropped.

These pass; they don't catch the actual user-facing problems (wrong radii from upstream, wrong matcher behavior in the CAD build).

---

## What's still wrong, and what the user has said

### Symptom 1: 4 base-plate through-holes detected at 2 different radii

The user says all 4 mounting holes are physically identical (~6mm radius, ~45° countersinks at the top). Stage 3 fits give:

| rid | r (fit) | bbox | faces | vertex distances to axis |
|---|---|---|---|---|
| 1 (left bore) | **4.10** | 8 × 8 × 6.1 | 48 | bimodal: 3.06–3.30 cluster + 4.78–5.03 cluster |
| 4 (right bore) | **4.00** | 8 × 8 × 6.1 | 48 | all exactly 4.00mm — clean cylinder |
| 24 (back bore) | **6.37** | 25 × 19.8 × 8.5 | 116 | strongly bimodal: ~4mm peak + ~8mm peak |
| 25 (front bore) | **6.37** | 25 × 19.8 × 8.5 | 116 | same as rid 24 |

**Diagnosis (the user agreed):** the cylindrical bore region in each hole only spans 6.1mm of the 10mm plate thickness. The remaining ~4mm is the **countersink cone** at the top, which Stage 3 fits as a separate region. For rid 24/25, the segmenter has additionally **merged the bore wall with the tab's outer cylindrical surface** (the bimodal distance distribution at ~4mm and ~8mm is two concentric cylinders glued together), and the fitter averages across both → median = 5.99 → fit reports 6.37.

**rid 4's fit is geometrically correct — the bore really IS 4mm.** rid 1 is mostly correct with some contamination. rid 24/25 are over-merged.

So the user's "all should be 6mm" is the *outer countersink opening* size, not the bore. The bore is ~4mm.

### Symptom 2: CAD build shows only 2-3 fillets, often wrong

The matcher's per-record fan-out + per-record bucketing now passes more candidate edges to OCC, but most of the ~17 export records still don't find any matching edge. The fundamental constraint:

- The CAD solid is built as `base_extrusion + holes` — typically 12-20 sharp edges per body.
- Each fillet record can only attach to a sharp edge that exists in the CAD solid.
- Many recorded fillets have no corresponding sharp edge because the base extrusion doesn't model the cantilever tabs, the corner cuts, etc.

This is structural, not a tuning issue.

### Symptom 3: User questioned the "MAX radius" rule

After the radius unification picked 6.37mm as the canonical value for the 4 grouped through-holes, the user pointed out that 6.37 comes from over-merged regions and the clean fit (rid 4) gives 4mm. The MAX rule rewards the *worst* fit. **Switch to "lowest-RMS fit wins"** — rid 4 has near-zero RMS (perfect cylinder), rid 24/25 will have very high RMS due to bimodal vertex distribution.

---

## What to attack next, in order

### 1. Switch `_group_symmetric_holes` radius unification from MAX to lowest-RMS

In `mesh2param/pipeline/recognize.py` around line 393. The merged group's `sketch_geometry["radius"]` should come from the cluster member whose Stage 3 fit had the smallest `rms_error`. The fit's RMS isn't currently passed into the hole feature; you'd either:
- Plumb `f.rms_error` into the hole feature's `sketch_geometry["fit_rms"]` in `_find_holes`, then pick min there in `_group_symmetric_holes`, OR
- Recompute vertex-to-axis std from the region's faces in the merge step.

The first option is simpler.

### 2. Detect countersunk through-holes as compound features

In `_find_holes` (`recognize.py` line ~293):
- After the cylinder hole-detection loop, look for unassigned CONE regions whose axis aligns with an already-detected THROUGH_HOLE's axis AND whose center is within ~radius of the hole's center.
- When found, attach the cone as a countersink: store cone's `half_angle`, top radius, and depth in the hole feature's `sketch_geometry["countersink"]`.
- The cone's wide half-angle (currently rejected at line ~448 with `half_angle > 45° → skip`) is fine for a countersink — just don't reject it when it's adjacent to a through-hole bore.

Then in `export_cadquery.py`'s hole loop, after cutting the bore cylinder, also cut a cone for the countersink.

### 3. Investigate why rid 24/25 over-merge bore + tab outer cylinder

In Stage 2 (`segment.py`'s `_classify_surface_type` and the region-grow code), the bore wall and the tab's outer cylindrical surface are getting flood-filled into one region. The flood-fill probably walks across because:
- The dihedral angle between bore wall and tab outer surface is gentle at the rim where they meet.
- Both surfaces are CYLINDER-classified by `_classify_surface_type`.

Investigate what region IDs are adjacent to rid 24/25 that should be the tab outer surface, and look at the dihedral-angle sharp-edge detection (`_detect_sharp_edges` in segment.py). Likely fix: tighten the curved-surface region-grow threshold OR add a "different cylinder axis" boundary check during region growing.

This is the **proper** fix for the radius-inflation issue and would also fix Symptom 1 cleanly without needing the lowest-RMS workaround.

### 4. Consider whether the matcher's "find an existing edge" approach is the right architecture

Symptom 2's structural limit (only edges in the simple base extrusion get filleted) suggests a different approach: instead of matching mesh-detected fillets to CAD edges, **build the CAD solid in a way that has all the right edges**. For each blend record, identify which two adjacent flat regions it sits between, and ensure both flat regions exist as faces in the CAD solid before applying the fillet.

This is a much larger rework (effectively a new export strategy that's feature-tree-driven rather than profile-extrusion-driven), so consider it only after #1–3 are tried.

---

## Useful diagnostic scripts kept in the repo root

These got created during this session for inspecting the rackmount; safe to delete once the issues are fixed:

- `_diag_all_regions.py` — dumps all regions per body with bbox/normal stats
- `_diag_blend_graph.py` — runs through Stage 4.5 and Stage 5 for one mesh
- `_diag_corner_bridge.py` — shows which corner regions pass the bridge test
- `_diag_full_corner.py` — diagnoses the small 4-corner bevels at the side plate corners
- `_diag_pull_check.py` — shows edges around bevel regions
- `_diag_rid12.py` — checks why a specific region's neighbor count is too high
- `_diag_side_tab.py` — lists all blends with positions for each body
- `_diag_stage_trace.py` — traces a region through all pipeline stages
- `_diag_tab_top.py` — dumps all blend records sent to CAD export
- `_diag_unify.py` — attempts to find user-specified corner positions
- `_check_body2.py` — groups records by axis line to see what would merge
- `_diag_bottom_tab.py` — diagnoses the small base-plate corner curls

Run any with `PYTHONIOENCODING=utf-8 python _diag_NAME.py` from the repo root.

---

## Running and testing

```bash
# Web UI (the user's primary interface)
PYTHONIOENCODING=utf-8 python -m mesh2param.viz_server.app
# Open http://127.0.0.1:8080, upload models/Mesh5/RackMount.3mf

# Pipeline tests
PYTHONIOENCODING=utf-8 python -m pytest tests/test_export_fillet_consistency.py -v
PYTHONIOENCODING=utf-8 python -m pytest tests/test_segment.py -q

# All-mesh integration (slower)
PYTHONIOENCODING=utf-8 python -m pytest tests/test_all_meshes.py
```

The user has been driving everything via the web UI's "Build" timeline + blend-graph stage view, then telling Claude what looks wrong. Re-running the rackmount and observing the construction sequence is the fastest feedback loop.

---

## What NOT to do

- **Don't re-enable the LLM in `export_cadquery.py`** until heuristic CAD construction is reliable. We disabled it intentionally so debugging the matcher isn't confounded by AI variability.
- **Don't pre-merge same-body-same-radius records before the matcher** — the matcher finds one best edge per record, so collapsing 4 corner records into 1 record drops 3 edges from getting filleted.
- **Don't take MAX radius** when unifying a hole group. The user explicitly pushed back on this. The largest fitted radius typically comes from the most over-merged region.
- **Don't paper over the radius issue with a wider radius tolerance** in the grouping. The 60% tolerance currently in `through_hole_radius_tol_pct` is already on the edge of being permissive enough to merge truly different holes; the right fix is upstream (clean segmentation), not downstream (loose grouping).
