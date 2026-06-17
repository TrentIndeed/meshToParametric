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

### ALSO do this while you're in here: GATE the alignment (skip when already
aligned). Many AI meshes (Meshy/Tripo) arrive roughly axis-aligned, so the
canonical rotation is unnecessary and only introduces a small pose error — this
is most of why "align makes AI models look worse." In `align_to_canonical_frame`,
after computing the candidate frame, measure how axis-aligned the dominant face
normals ALREADY are (area-weighted fraction of faces whose normal is within ~3–5°
of a world axis, or `max |R - I|`). If already aligned, return the IDENTITY
transform and don't rotate. Combined with the high-res fix above, this keeps
alignment for the genuinely tilted inputs that need it and leaves clean AI meshes
untouched. See the Design Q&A (§7, Q2) for the rationale.

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

---

## 7. Design Q&A (conceptual context for whoever picks this up)

These are the design tensions behind this fix and the AI-mesh path generally.
Captured here so a fresh instance has the *why*, not just the *what*. Answers are
grounded in the code as of 2026-06-16.

### Q1 — A curved edge in a mesh could be a FILLET (blend between two faces) OR the side-profile of a real designed RADIUS. How does the blend stage tell them apart, and how does the program decide which way to model it?

It's genuinely ambiguous in a raw mesh. `peel_via_topology.identify_blends_from_topology`
resolves it with a **geometric "fillet signature"**, NOT by trusting edge tags
(the topology fillet-tag is unreliable — fit.py often classifies a partial arc as
a sphere/cone, or merges a through-hole cylinder to 360°). A curved region is
called a **fillet** only if ALL of:

- **Small, mm-scale radius** — measured downstream from median perpendicular
  distance to the edge line, not the raw cylinder-fit radius (which is garbage
  for partial arcs).
- **Bridges two flat faces tangentially** — the "bridge test": some region face
  normals align with an adjacent plane's normal (dot > 0.866) AND the region's
  normals sweep a **bounded arc ≤ ~90°** (a real fillet rolls at most a
  quarter-circle).
- **Axis doesn't pierce an adjacent plane** — bores/boss-walls/countersinks
  pierce; a fillet runs tangent.
- **Long-thin-strip shape** — runs *along* an edge (length ≫ radius), not a ring
  or a blocky 3D corner.

What this REJECTS (→ treated as real geometry, not a fillet):
- **Bore/hole walls** — radial normals never align → recognized as a hole.
- **Boss / cylindrical side walls** — normals sweep 180°+ (fail ≤90° spread) →
  real cylinder surface, becomes an extrude/boss.
- **Countersink cones** — uniform half-angle normals fail tangency.
- **A designed curved radius that's part of the silhouette** — too big / sweeps
  too far to be a fillet, so it stays in the **cross-section profile** and gets
  extruded as the outline (the way base-plate corner radii become sketch arcs).

**Decision rule in one line:** *small tangent arc bridging two faces with bounded
sweep → fillet (peeled off, re-applied as a blend feature after the build);
everything else curved → part of the shape itself* (profile arc, cylinder, cone,
revolve). HONEST CAVEAT: this is a heuristic threshold cascade, and the
borderline cases (a generous rolled edge vs. a small designed radius) are exactly
where it's fragile — it's the recurring source of the side-plate fillet bugs and
the left/right asymmetry (see `docs/TODO.md` "Asymmetric corner-thickness blend
detection" and `docs/SEGMENTATION_HANDOFF.md`).

### Q2 — Do AI meshes even need the align stage? It's the main stage making the model look worse.

**Conceptually yes, but right now it does more harm than good — fix it, don't
remove it.** Two separate reasons it hurts:

1. **It computes the frame on the over-decimated mesh** — the bug this whole doc
   is about. Coarse normals → slightly-wrong rotation → every world-axis
   assumption inherits a tilt/skew → "looks worse."
2. **Many AI meshes arrive roughly axis-aligned**, so the canonical rotation is
   unnecessary and only adds a small pose error.

**Why it exists:** the extrusion engine, axis snapping, and prismatic reasoning
are all *world-axis-relative* (slice along X/Y/Z, assume dominant faces are
axis-parallel). A genuinely tilted scan/AI mesh MUST be rotated onto its own
frame or all of that breaks.

**Recommendation:** (a) run it on high-res (this doc's main fix), and (b) **gate
it** — make it a no-op when the part is already aligned (return identity if the
dominant normals are already within a few degrees of world axes; see §3 "ALSO do
this"). That keeps alignment for the tilted inputs that need it and stops it from
nudging already-clean AI meshes off-true.

### Q3 — How clean do segmentation + topology need to look on AI meshes for the program to recreate the part?

**Not pixel-perfect — but they must recover the part's dominant PRISMATIC
STRUCTURE.** Concretely the program needs:

- **The major planar faces as single, large regions** (base plate, each wall,
  the big flats). This is the hard requirement and the main AI-mesh failure: a
  flat AI face shatters into hundreds/thousands of slivers (seg 46→7341 on a real
  part), so the extrusion engine's cross-section tracking + "biggest face"
  selection have nothing to lock onto. `noise_robust` merge recovers 50–70%; the
  dense/complex parts are still over-fragmented (the open "region-growing
  robustness" TODO).
- **Curved strips identifiable per Q1** — over-fragmentation breaks the fillet
  bridge test (the arc gets split, normals don't form a clean bounded sweep),
  which is part of why corner-fillet detection goes asymmetric.
- **Topology edges good enough to find holes + the sharp outline** — tolerant of
  noise on smooth/fillet edges, but needs the sharp silhouette and hole rims.

**Does NOT need:** every triangle classified right, perfectly closed regions, or
exact fillet strips. The build is driven by (extrusion profile + holes + blend
records); if the big faces, outline, holes, and dominant blends survive, recovery
works (the detail finders are deliberately conservative on dirty parts — drop
weak evidence rather than hallucinate).

**Practical bar:** if a human glancing at the segmentation viz can pick out "base
plate / two walls / holes / rolled edges" as distinct regions, the program can
rebuild it. When the flats look like confetti, it can't. So **getting
segmentation into the clean-CAD triangle regime is the single highest-leverage
AI-mesh lever** (the `dirty_max_faces` cap + the region-growing robustness work)
— more than any individual feature fix. This is also exactly why getting the
alignment+decimation ordering right (this doc) matters: a good frame makes the
segmentation cleaner, which makes everything downstream tractable.
