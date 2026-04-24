# Segmentation Handoff: Corner Triangle Problem

**Status:** stuck. Per-body segmentation is stable for the main structure, but one class of edge case keeps regressing when tuned against. This doc gives a new Claude session the context to attack it fresh without repeating dead ends.

**Scope:** this document concerns `mesh2param/pipeline/segment.py`'s `_region_growing_by_axis_class` function and its smoothing passes. The broader pipeline restructure (per-body segment → reconcile fillets → export) is already landed and working.

---

## The problem in one sentence

On the RackMount test mesh, flat triangles that sit at **fillet-fillet corner junctions** (where two rolled-edge fillet strips meet at a plate corner) get classified as "curved" and show up with the fillet's color, creating visible wedges poking into the plate face.

## The mesh that triggers it

`models/Mesh5/RackMount.3mf` — a U-bracket decomposed into three bodies:
- Body 0, Body 1: the two vertical side plates (mirror symmetric)
- Body 2: the bottom plate (connecting bar)

The side plates have rolled-edge fillets on all four edges of the front/back faces. At the plate corners, two fillets meet. The tessellation at that junction is a mix of:
- Genuinely flat plate triangles (their own normal ≈ plate normal)
- Fillet strip triangles from each of the two adjacent rolled edges (normals rotate around the edge axis)
- Transition triangles connecting the two fillet strips

The user wants the flat triangles classified as plate, and the fillet triangles classified as the fillet regions. What's happening is that flat triangles get pulled into the fillet region via mesh adjacency.

## The user also cares about through-hole bore detection

Body 0/1 each have a central through-hole (~2mm radius). The hole's **cylindrical bore wall** must be detected as a single `CYLINDER` region with ~2mm radius so downstream stages (feature recognition, export) can treat it as a through-hole feature. Axis-class segmentation naturally splits a 360° cylinder into 4 parts along ±Y ±Z (the cardinal normal directions). The `_merge_adjacent_cylinder_fragments` post-pass stitches them back into one region.

**The hard constraint:** any fix for the corner-triangle issue must not destroy bore detection. Every attempt to aggressively reclassify curved faces based on their own normal ends up eating into the bore's cardinal patches and either:
- Fragmenting the bore into 4+ regions again
- Over-merging the bore with the plate, producing a giant misfit region

## Current state (as of handoff)

`_region_growing_by_axis_class` does:

1. **Classify each face** by its raw normal into one of 7 classes (±X, ±Y, ±Z, curved) with a 5° axis-threshold.
2. **Option 1 (per-region coherence):** for each class-6 connected component, if its mean-normal magnitude > 0.9 and near an axis, reclassify the whole region to that axis class. Catches flat patches that landed in class 6 due to 5-8° noise.
3. **Option 2 (per-face SVD test):** for each class-6 face, gather 3-hop neighborhood normals, run SVD. If `s_norm[1] < 0.12` (rank-1 = locally flat neighborhood), flip the face to its neighborhood's dominant axis class. Rank-2 (cylinder) and rank-3 (sphere) are left alone.
4. Rebuild connectivity from the updated classes.

After this:
- `_merge_small_regions` with two-pass fallback (similar normal → longest-shared-boundary)
- `_merge_adjacent_cylinder_fragments` stitches cylinder quadrants back together

Tests in [tests/test_fillet_detection.py](../tests/test_fillet_detection.py):
- **3 of 4 pass.**
- `test_regions_are_not_spiky` — passes at compactness ≥ 0.08. Acts as a regression guard.
- `test_bottom_plate_has_tab_fillets` — passes (≥1 tab fillet detected).
- `test_fillets_constant_width` — passes (trivially on this mesh).
- `test_side_plates_have_expected_fillets` — **fails**. Expects 6 fillets per side plate (4 long edges + 2 hole rim); currently 0-2 plausible fillets detected. This has been failing the whole session; fixing it is a separate problem, not the one this doc is about.

## What's been tried (don't repeat these)

1. **Raise axis threshold from 5° to 10° / 12° / 15°.** Breaks cylinder detection — bore faces 5-15° off axis all collapse into one aligned class, merging with plate.
2. **2-hop smoothed normals used for classification.** Flat triangles near a fillet get averaged-down into curved direction, pulling them into fillet region (opposite of what we want).
3. **Mesh subdivision** (`_subdivide_long_edges`, trimesh `subdivide_to_size`). Fragments segmentation further because new sub-triangles preserve parent normals, revealing per-triangle variation that was hidden by coarse tessellation. Face count exploded 4-14× without solving the problem.
4. **Isotropic remesh** (pymeshlab `meshing_isotropic_explicit_remeshing`). Even with `smoothflag=False, reprojectflag=True`, this destroyed mesh geometry — volume dropped 11% on RackMount. Smoothed over small fillets and made the mesh look "already peeled".
5. **Curvature-drift guard** in region growing. Fragile on noisy per-face curvature estimates — fragmented cylinders unpredictably.
6. **Reverse smoothing: flip curved → aligned if neighbors agree.** Eroded cylinder walls at cardinal positions, broke bore detection.
7. **Per-face "flip if own normal ≤ X° off axis".** With X=7°, fixes corners but breaks bore. With X=5°, misses corners. With X=10°, corners fixed, bore completely fragmented.
8. **Region-level cylinder protection** (mean-normal magnitude of whole region). Too coarse — rolled fillet strip's mean magnitude is also low, gets protected too, blocking the fix.
9. **Per-face 1-hop neighborhood magnitude gate.** Too small a window — bore cardinal face's 3 direct neighbors can be coherent enough to pass the gate.
10. **SVD rank test at various thresholds** (0.05 / 0.07 / 0.08 / 0.10 / 0.12 / 0.15).
    - Tight (0.05): corners miss.
    - Loose (0.12+): fillet-strip faces get flipped, back-face region becomes spiky (c < 0.08).
    - 0.08-0.10 is the "best tradeoff" but some corner triangles still stay in curved.

**The core tension:** any rule aggressive enough to reclassify a corner-fan triangle (flat own-normal, curved neighborhood) also affects cylinder-wall triangles at cardinal positions (flat own-normal, curved neighborhood). The two cases look similar at the local level.

## Why the SVD test is the "closest" attempt but still fails

SVD on a 3-hop neighborhood distinguishes:
- Rank 1 (all normals parallel): flat surface
- Rank 2 (normals in a plane): cylinder
- Rank 3 (normals span 3D): sphere / noisy / mixed

For a **bore cardinal face**: neighborhood normals rotate in a plane → rank 2 → correctly protected.

For a **plate face in the middle**: normals mostly parallel → rank 1 → flipped correctly.

For a **corner-fan flat triangle where two fillets meet**: 3-hop neighborhood includes triangles from TWO different fillet strips, each rotating in its own plane → mixture of rank-2 signals → combined signal looks rank-2 or rank-3. The SVD test says "curved" and doesn't flip, so the user sees it wedged into the fillet region.

## What to try next

### Idea 1: Plane-fit residual test (most principled)

For each class-6 face, fit a plane through the face's vertices + 2-hop neighborhood vertices (least-squares). Compute per-vertex residual.
- Small residual: the neighborhood genuinely lies on a plane → flat → flip to axis class that matches plane normal.
- Large residual: there's a real geometric curve → leave as curved.

This is direction-invariant (doesn't care about which way the plane points) and captures actual local planarity rather than normal-direction agreement. Would need to be careful about the fit's robustness on 9-15 vertices.

### Idea 2: Two-pass with different neighborhood radii

Use a SMALL neighborhood (1-hop) to see if the face's immediate surroundings are flat. Use a LARGER neighborhood (3-hop) to confirm it's not on a curve that happens to be flat locally.

- Small neighborhood rank-1 AND large neighborhood rank-1 → truly flat → flip.
- Small rank-1 but large rank-2+ → on a fillet strip, looks flat locally → don't flip.

### Idea 3: Distinguish corner-junction from fillet-strip

A **fillet strip**'s neighborhood normals rotate around ONE axis (the edge direction). SVD rank-2 with one specific principal direction.

A **corner junction** where two fillets meet has normals rotating around TWO different axes. SVD may still be rank-2 or rank-3 but the principal directions are less "pure".

Check `s_norm[2] / s_norm[1]` ratio:
- Clean cylinder: s_norm[2] ≈ 0, ratio ≈ 0
- Spheroid corner junction: s_norm[2] comparable to s_norm[1], ratio > 0.5

If the ratio is close to zero (clean cylinder), protect. If ratio is higher (corner), allow flipping based on OWN normal.

### Idea 4: Graph-based region growing that respects the flat-plate topology

Instead of classifying each face independently, identify the flat-plate region first (largest connected component of axis-aligned faces) and then greedily absorb any face whose own normal is within say 8° of the plate's normal AND whose mesh-distance to the plate region is small. This is a morphological grow of the plate region into ambiguous boundary faces.

### Idea 5: Decouple classification from region assignment

Currently, class determines which faces can be connected in a region. Separate these: first classify (curved vs flat), then do region growing where crossing a class boundary is allowed if the next face has similar normal to the running region. That way a flat-class corner triangle connected only to curved-class neighbors via mesh adjacency can still be in its own region (class 0 connected component) rather than absorbed into the curved region.

---

## Architecture reference

Key files:
- [mesh2param/pipeline/segment.py](../mesh2param/pipeline/segment.py) — all the axis-class logic
  - `_region_growing_by_axis_class` — line ~245, the main classification function
  - `_merge_small_regions` — line ~410, two-pass fallback merger
  - `_merge_adjacent_cylinder_fragments` — line ~740, cylinder quadrant stitcher
  - `_smooth_region_boundaries` — line ~155, post-pass label smoothing
- [tests/test_fillet_detection.py](../tests/test_fillet_detection.py) — the regression suite

Running the diag loop this session used:

```python
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.config import IngestConfig, SegmentConfig, FitConfig, DecomposeConfig
cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
for bi, body in enumerate(decomp.bodies):
    sub = extract_sub_mesh(cleaned, body.face_indices, body=body)
    regs = segment_mesh(sub, SegmentConfig())
    fits = fit_surfaces(regs, sub, FitConfig())
    # render with pyvista + clipped bboxes for zoom-ins
```

Render corner zoom-ins with `mesh.clip_box([xmin, xmax, ymin, ymax, zmin, zmax], invert=False)` then a PyVista plotter — that was the only reliable way to inspect the problem.

## Process guidance

**Verify both cases on every change.** The session consistently regressed one case by fixing the other. Minimum checks per iteration:

1. Body 0 top-left corner — no color wedges in the plate face
2. Body 0 through-hole — one continuous ring, no 90° breaks
3. `pytest tests/test_fillet_detection.py` — at least 3/4 still pass (the spiky guard is the key regression signal)
4. `fit_surfaces` returns a cylinder at radius ≈ 2mm on Body 0 (the bore)

If any of those four regresses, revert the change before continuing.

**Don't iterate thresholds.** Every setting on the current architecture has been tried 0.05 → 0.20 in multiple combinations. The threshold surface is well explored; new value won't help. A new architectural approach (ideas above) is needed.

**Target the actual mesh that shows the bug.** Mesh5/RackMount. Don't debug on other meshes that don't exhibit the problem — they'll pass trivially and mislead you.
