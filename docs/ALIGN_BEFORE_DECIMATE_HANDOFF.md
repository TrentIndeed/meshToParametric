# Handoff: Align on the high-res mesh, decimate only for segmentation

**For:** a fresh Claude instance picking up this fix.
**Status:** diagnosed + planned, NOT implemented. Listed in `docs/TODO.md` under
"Scans + AI-generated … Still TODO".
**Scope:** `mesh2param/pipeline/ingest.py` (`ingest_mesh`) ordering only. No new
algorithms — a reorder + one cost guard.

---

## 1. The problem (one sentence)

For dirty/AI/scan meshes the pipeline **decimates to `dirty_max_faces` (12k)
FIRST and computes the canonical-frame ALIGNMENT on that ≤12k mesh** — but
alignment derives its rotation from area-weighted face normals, so coarse
triangles give a worse frame, and every downstream world-axis assumption (axis
snapping, extrusion-engine silhouettes, prismatic reasoning, the whole re-axis
work) starts from a slightly-wrong rotation.

**Alignment should run on the HIGHER-triangle mesh; the aggressive 12k
decimation should happen ONLY to feed segmentation** (which is the only reason
the cap exists — flat faces over-fragment in region-growing at high density).

---

## 2. Current state (exact, verified 2026-06-16)

`mesh2param/pipeline/ingest.py`, function `ingest_mesh` (~line 23 onward). The
order inside the cleanup block + after it:

```
line  97  _remove_degenerate_faces
line  98  _merge_vertices
line 102  _cap = dirty_max_faces (12k) if needs_frontend else max_faces (30k)
line 104  _decimate(mesh, _cap)          <-- AGGRESSIVE DECIMATION (too early)
line 105  _fix_normals
line 109  _denoise_scan        (if do_denoise)        # on the ≤12k mesh
line 117  _fill_holes
line 121  _poisson_reconstruct (if not watertight)
line 124  _voxel_remesh_repair (if not watertight)
line 133  _remesh              (if config.remesh — FALSE on the dirty path)
line 146  align_to_canonical_frame(mesh)  <-- ALIGNMENT (on the ≤12k mesh)  ← BUG
line 154  _normalize_scale
line 156  _build_cleaned_mesh
```

Key facts:
- `do_align = config.align_canonical or needs_frontend`. `needs_frontend =
  is_scan or is_dirty`. So **alignment ONLY runs for dirty/scan inputs** — clean
  CAD (Mesh1–5) never aligns and is < `max_faces`, so the reorder CANNOT affect
  the reference meshes. (Confirm: `IngestConfig(remesh=False, scan_auto=True)`
  is what the server + `convert_mesh` use.)
- `align_to_canonical_frame` (`mesh2param/pipeline/align.py`) returns
  `(rotated_mesh, transform, method)`. **`transform` is a pure rotation** (3×3,
  from `_frame_from_normals` dominant-normal triple, else OBB fallback). It uses
  `mesh.face_normals` area-weighted — strictly better with more triangles.
- Because the transform is a pure rotation, it is **resolution-independent once
  computed**: you may compute it on the high-res mesh and apply the same
  rotation to the decimated mesh.
- `_decimate` (line ~319) already has a cheap **vertex-clustering pre-pass** when
  `faces > 5 * max_faces` (knocks a 2.5M scan to ~250k in ~2s), then a quality
  quadric collapse. Use this for cost control.
- `dirty_max_faces = 12_000`, `max_faces = 30_000` (in `config.py`,
  `IngestConfig`).

Why 12k exists: segmentation (`segment.py` region-growing) was tuned on the
clean-CAD triangle regime (1.6k–9.5k faces). At 30k+ a flat AI/scan face has far
more triangles for noise to shatter into fragments. 12k lands dirty inputs near
that regime. **This rationale is about SEGMENTATION, not alignment** — that's the
crux of the fix.

---

## 3. The fix (recommended plan)

**Reorder so alignment sees a high-res mesh, and apply the 12k cap only just
before segmentation. Add a moderate cost-guard cap so denoise/repair don't run
on a multi-million-face scan.**

Target order inside `ingest_mesh`:

```
_remove_degenerate_faces
_merge_vertices
# COST GUARD ONLY (new): bound denoise/repair cost on huge scans WITHOUT
# crushing the normals alignment needs. Most AI meshes (72k–1.14M) and all
# clean CAD are under this; only big scans hit it.
if needs_frontend: _decimate(mesh, config.align_max_faces)   # e.g. 120k
_fix_normals
_denoise_scan                                   # now on ~100k, not 12k
_fill_holes
_poisson_reconstruct
_voxel_remesh_repair
_remesh                                          # (no-op on dirty path; remesh=False)
align_to_canonical_frame(mesh)                   # GOOD normals  ✓
# SEGMENTATION CAP (moved): only now drop to 12k, for segment cost/quality
if needs_frontend: _decimate(mesh, config.dirty_max_faces)   # 12k
_fix_normals
_normalize_scale
_build_cleaned_mesh
```

### Config change (`config.py`, `IngestConfig`)
Add:
```python
# Cost-guard cap applied BEFORE denoise/repair/align on dirty/scan inputs, so
# those steps + the canonical-frame alignment run on a HIGH-resolution mesh
# (good normals) rather than the aggressive segmentation cap. Only huge scans
# (>this) get pre-decimated here; the aggressive dirty_max_faces cap is applied
# AFTER align, to feed segmentation. 0 = no cost guard (align on full mesh).
align_max_faces: int = 120_000
```
Tune `align_max_faces`: high enough that alignment normals are accurate
(≥ ~80k is plenty for a prismatic part's dominant normals), low enough that
bilateral denoise + Poisson/voxel-repair stay reasonable (~seconds, not
minutes). 120k is a safe start; measure on the densest scan (Scan2/2.5M) and the
densest AI part (~1.14M).

### Implementation notes / gotchas
- **`_decimate` is idempotent below the cap** (`len(faces) <= max_faces` returns
  as-is), so calling it twice with two caps is clean: the first call only fires
  for meshes above `align_max_faces`, the second only for meshes above 12k.
- **The clustering pre-pass keys off `5 * max_faces`.** With the cost-guard cap
  at 120k it pre-clusters meshes > 600k — good (only the truly huge scans). The
  second `_decimate(…, 12k)` call after align operates on a ≤120k mesh, so its
  pre-pass (> 60k) fires and is cheap.
- **`on_stage("denoise", …)` / `on_stage("align", …)`** viz callbacks: keep them
  firing where the corresponding step runs. The "align" pause-stage now shows
  the high-res aligned mesh; fine.
- **`align_transform` metadata** (`result.metadata["align_transform"]`) is still
  the rotation; the STEP export un-rotates with it. Unchanged — still correct
  because the rotation is identical whatever resolution computed it.
- **Order vs. repair:** alignment MUST stay AFTER denoise + Poisson/voxel-repair
  (those change geometry/normals). The plan keeps that — only the 12k decimation
  moves.
- **Don't double-apply the rotation.** `align_to_canonical_frame` already returns
  the rotated mesh; the post-align `_decimate` just decimates the
  already-rotated mesh. No transform bookkeeping needed.

### Alternative (smaller change, if `align_max_faces` cost is acceptable on the
full mesh): skip the cost-guard cap entirely and just MOVE line-104's
`_decimate(…, dirty cap)` to after `align_to_canonical_frame`. Simpler, but
denoise + Poisson + voxel-repair then run on the full 1M–2.5M-face mesh — profile
`_denoise_scan` and `_voxel_remesh_repair` first; if either is > ~20s on the
densest scan, use the cost-guard cap.

---

## 4. How to verify

1. **Alignment quality (the point of the fix).** Before/after the reorder, for a
   few AI parts in `models/AI*/`, compare the canonical frame. The aligned part's
   dominant faces should sit closer to the world axes. Quick check: after ingest,
   measure how axis-aligned the big planar faces are — e.g. the max over faces of
   `|normal · nearest_world_axis|` area-weighted, or simpler, the fraction of
   face area whose normal is within N° of a world axis. Higher (and ≥ the old
   value) = better. `scripts/diag_ai_parts.py` already exercises the AI parts and
   is the place to add this metric.
2. **Segmentation still gets 12k.** Assert the mesh handed to segmentation is
   ≤ 12k for dirty inputs (log `len(faces)` right before `_build_cleaned_mesh`,
   or in the segment stage). Region counts should be unchanged vs. today (the
   segmentation input is the same 12k mesh, just aligned from better normals).
3. **No clean-CAD regression.** `pytest tests/test_step_comparison.py -q` must
   stay **18 passed / 7 xfailed**. Clean CAD never enters `needs_frontend`, so it
   should be untouched — this test is the guardrail that the reorder didn't leak
   into the clean path.
4. **Cost.** Time ingest on the densest scan (Scan2, ~2.5M) and densest AI part.
   Should not be meaningfully slower than today; if it is, lower `align_max_faces`
   or profile the repair steps.
5. **End-to-end on a real AI part** that today aligns poorly: build it via the
   server (`job_queue._run_pipeline`) and confirm the extrusion engine picks
   sensible axes (the re-axis / base-axis-preference work depends on a correct
   frame).

Useful existing diag scripts: `scripts/diag_ai_parts.py` (AI validation),
`scripts/diag_scan_align.py` (alignment), `scripts/diag_robust.py` (per-mesh
feature sweep), `scripts/diag_ingest.py`.

---

## 5. Risks / things that will bite

- **Bilateral denoise / voxel-remesh cost on high-res.** The whole reason the
  decimation was early. The cost-guard cap (`align_max_faces`) is the mitigation;
  do NOT remove it without profiling the densest inputs.
- **Watertightness gating.** `_poisson_reconstruct` and `_voxel_remesh_repair`
  only fire `if not mesh.is_watertight`. Decimating differently can flip
  watertightness at the margins; sanity-check that dirty meshes still reach a
  watertight state before align (align doesn't require watertight, but the
  component planner downstream does).
- **`align_max_faces` too low** re-introduces the bug (coarse normals). Too high
  blows up repair cost. Tune empirically.
- **Two `_decimate` calls** — make sure the FIRST is gated on `needs_frontend`
  (so clean CAD, which shouldn't be decimated below `max_faces`, is untouched)
  and the SECOND uses `dirty_max_faces` and is also `needs_frontend`-gated.
- **Mirror the change nowhere else** — ingest order lives only in `ingest_mesh`.
  (No 3-copy drift problem here, unlike the fillet grouping.)

---

## 6. Definition of done

- Alignment runs on a ≥ ~80k-face mesh for dirty/scan inputs (or the full mesh
  if smaller); segmentation still receives ≤ 12k.
- A measurable alignment-quality metric improves (or holds) on AI parts.
- `test_step_comparison` unchanged (18/7).
- Ingest time on the densest scan not materially worse.
- `dirty_max_faces` semantics narrowed to "segmentation cap" in code comments;
  new `align_max_faces` documented in `config.py`.
