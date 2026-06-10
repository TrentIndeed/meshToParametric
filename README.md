# Paramesh AI

Reverse-engineering pipeline that converts triangulated mesh files (STL/OBJ/PLY/3MF) of prismatic mechanical parts into fully parametric CAD models. AI-powered feature recognition, construction timeline, and interactive web viewer.

**Live at [parameshai.com](https://parameshai.com)**

**Target parts:** brackets, mounts, plates — prismatic components with planar faces, cylindrical holes, pockets, bosses, fillets, and chamfers.

## Pipeline

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `ingest.py` | Load mesh, repair topology, fix normals, remove degenerate faces |
| 2 | `segment.py` | Region-growing segmentation with normal clustering, curvature splitting |
| 3 | `fit.py` | RANSAC + least-squares fitting for plane/cylinder/cone/sphere |
| 4 | `topology.py` | Surface adjacency graph, dihedral angles, edge classification |
| 5 | `recognize.py` | Feature recognition: extrusions, holes, pockets, chamfers, fillets |
| 6a | `export_onshape.py` | FeatureScript generation + Onshape REST API (dry-run or live) |
| 6b | `export_cadquery.py` | CadQuery solid: sketch → extrude → cut holes → STEP export |

## Web App

Three-panel CAD viewer with AI assistant at [parameshai.com/app](https://parameshai.com/app).

- **Left**: Feature tree with typed icons (sketch, extrude, hole, chamfer)
- **Center**: Interactive Three.js 3D viewer with orbit controls, XYZ grid
- **Right**: AI chatbot — describe parts, suggest edits, analyze dimensions
- **Bottom**: Timeline with 3D-rendered thumbnails per construction step
- **Upload flow**: preview mesh → complexity meter → AI thinking process → CAD model

### Free Tools

- **Mesh Cleaner** ([parameshai.com/clean](https://parameshai.com/clean)) — fix normals, remove degenerate faces, merge vertices, fill holes, resize. No signup required.
- **Learn** ([parameshai.com/learn](https://parameshai.com/learn)) — 20 SEO guides on mesh-to-CAD conversion

### Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with video hero, pricing, FAQ, waitlist |
| `/app` | CAD viewer tool (requires login) |
| `/clean` | Free mesh cleaner (requires login) |
| `/home` | Signed-in home — tool picker with usage bar |
| `/learn` | 20 SEO guide pages |
| `/changelog` | What's New + Roadmap + Git History |
| `/contact` | Support form (saves to DB + emails) |
| `/login` | Sign in / Sign up |
| `/privacy` | Privacy policy |
| `/terms` | Terms of service |

## Setup

```bash
pip install -e ".[dev]"
pip install cadquery shapely
```

For AI features:
```bash
npm install -g @anthropic-ai/claude-code
claude "hello"  # one-time auth
```

## Running

```bash
# Web app
python -m mesh2param.viz_server.app
# Open http://127.0.0.1:8080

# CLI
mesh2param convert input.stl --dry-run --output feature_tree.json
```

## Testing

```bash
pytest                                    # all tests
python run_visuals.py                     # visual reports
python run_visuals.py --stage 4 5 6       # specific stages
pytest tests/test_step_comparison.py -v   # STEP comparison
pytest tests/test_all_meshes.py -v        # all 5 meshes
```

### Model Pairs

Drop a folder into `models/` with a mesh + reference STEP to auto-test:

```
models/Mesh1/Body1Mesh.3mf + Body1Step.step
models/Mesh2/BottomMount.3mf
models/Mesh3/Front Plate.3mf
models/Mesh4/Bottom.3mf
models/Mesh5/Component3.3mf
```

## Architecture

```
mesh2param/
  pipeline/      — 6 stage modules (ingest → segment → fit → topology → recognize → export)
  models/        — dataclasses (CleanedMesh, FaceRegion, FittedSurface, Feature, FeatureTree)
  viz/           — PyVista 3D + matplotlib 2D visualizations per stage
  viz_server/    — FastAPI web app + Three.js viewer + job queue
  db/            — SQLite (User, MeshJob, JobStep, WaitlistEntry, SupportTicket)
  api/           — REST endpoints (auth, chat, settings)
  ai.py          — Claude CLI wrapper with injection protection + budget caps
  config.py      — Pydantic settings for all stages
tests/           — pytest suite (visual, correctness, STEP comparison, all-mesh)
models/          — mesh + reference STEP pairs
```

## Pricing (Token-Based)

| Plan | Price | Tokens/mo | Upload Limit | Features |
|------|-------|-----------|-------------|----------|
| Free | $0 | 100 | 10 MB | Cleaner unlimited, 3-5 CAD conversions |
| Starter | $9/mo | 1,000 | 50 MB | STEP export, AI dimension planner |
| Pro | $29/mo | 5,000 | 200 MB | Onshape + AI chat + priority support |
| Unlimited | $79/mo | Unlimited | 500 MB | Everything |

Token cost: ~20 for a simple bracket, ~55 for a complex mount with chamfers.

## Security

- Per-IP token tracking prevents free trial abuse (100 tokens + 10 jobs per IP)
- Complexity meter rejects meshes >10,000 faces or non-watertight >5,000 faces
- AI calls: system prefix, prompt sanitization, 8K cap, 20 calls/session, 5s rate limit
- Auth: PBKDF2-SHA256 salted password hashes, signed expiring tokens, per-IP login rate limiting
- Database: SQLite on Docker volume, survives rebuilds

## Deployment

### Local Development Server

```bash
python -m mesh2param.viz_server.app
# Running at http://127.0.0.1:8080/app
```

### Deploy to VPS

Deploys via SSH to a Hetzner VPS running Docker + Caddy (port 8081), alongside operatorDashboard.

```bash
# Deploy (VPS IP: 5.161.100.230)
python deploy.py --ip 5.161.100.230 --domain parameshai.com
```

The deploy script will:
1. SSH into the VPS
2. Clone or pull the `main` branch to `/opt/meshToParametric`
3. Write `.env` with `AUTH_SECRET` and `DATABASE_URL`
4. Run `docker compose up -d --build` (container port 8080 → host port 8081)
5. Append Caddy reverse proxy config for the domain if not already present

**Note:** Container restart clears in-memory job data. User accounts persist in SQLite (`./data/mesh2param.db` volume).

### Both at Once

```bash
# Start local server and deploy to VPS in parallel
python -m mesh2param.viz_server.app &
python deploy.py --ip 5.161.100.230 --domain parameshai.com
```

## Key Libraries

trimesh, cadquery, pyvista, matplotlib, numpy, scipy, scikit-learn, networkx, fastapi, sqlalchemy, three.js

## Progress & Engineering Notes

This section tracks major changes to the core pipeline, problems hit, and how they were addressed. The pipeline works end-to-end on simple prismatic parts; the main ongoing work is **reliable segmentation on coarse, radial-fan triangulations** (common in 3MF exports from slicers).

### Pipeline restructure — per-body segmentation + stitched fillets

The original flow segmented the whole mesh, peeled its fillets, then decomposed into bodies. That made the per-body segmenter see an already-peeled mesh with no fillets to find, so fillet records for CAD export came only from the whole-mesh peel and had no symmetry between mirror features.

Restructured to: **ingest → decompose → per-body segment/fit/topology/identify_blends → stitched-fillets stage (groups per-body records into cross-body clusters via vertex KDTree proximity + axis-aligned continuation) → CAD apply (vertex-proximity edge matching with per-edge measured radius)**. The stitched stage replaced the earlier collinear-merge reconciliation step. The apply pass no longer trusts upstream record radii — it measures each CAD edge's radius from the same mesh vertex set the stitched group carries.

### Segmentation: axis-class + smoothing

The original flood-fill segmenter crept across rolled edges (plate → fillet → next plate all ended up in one chaotic region). Replaced `_region_growing` with **axis-class segmentation**: every face classifies into one of 7 classes (±X, ±Y, ±Z, curved) by principal-axis alignment within 5°. Mesh connectivity within each class gives regions. Prismatic parts have principal-axis-aligned faces as a hard constraint, so this cuts cleanly where the flood-fill failed.

Two smoothing passes run on the class assignment:

1. **Aligned → curved** — isolated axis-aligned faces (no same-class neighbors) get reassigned to curved. Catches pie-slice fan triangulation on cones where each slice lands in a different cardinal class.
2. **Curved → aligned** — curved faces whose majority of neighbors share one aligned class get pulled back to it. Catches plate triangles 5-10° off axis (tessellation noise) that initially landed in class 6 but belong to the plate.

Post-passes kept:
- `_merge_small_regions` with two-pass fallback: similar-normal neighbor first, then **longest-shared-boundary neighbor** if nothing matches within 75°. Absorbs long-thin sliver triangles that have no neighbor with matching normal.
- `_merge_adjacent_cylinder_fragments` stitches class-split cylinder quadrants back together, using the cumulative normal spread across the group to verify cylinder signature.
- `_smooth_region_boundaries` flips faces whose strict majority of neighbors belong to a different region.

### Blend detection filters (`peel_via_topology.py`)

The fillet-identification filter evolved through several generations. Original version used edge-tag transitions from topology — unreliable because the fitter sometimes classifies fillet arcs as SPHERE/CONE or merges through-holes into 360° cylinders, so topology tags adjacent edges CHAMFER or SHARP instead of FILLET.

Current filter walks the region adjacency graph and accepts a region as a fillet when:
- Non-plane (CYLINDER/SPHERE/CONE), small radius (≤ 8% bbox diag), reasonable face count (≤ 300).
- **For CYLINDER/CONE, at least one plane neighbor is perpendicular to the surface axis** — positive geometric signature of an edge blend. Hole bores, boss walls, and countersinks (axis parallel to plate normal) fail this and are correctly rejected.
- Falls back to using the cylinder axis as the edge line when adjacency gives only one plane neighbor or two parallel planes (rolled-over thin plates, rim fillets).

Chamfer detection requires a narrow plane region whose normal roughly bisects its two plane neighbors — rejects thin flat strips alongside rounded edges that have plate-matching normals.

### Issues hit and how they were resolved

| Issue | Root cause | Fix |
|-------|-----------|-----|
| Symmetric features produced asymmetric regions | Source 3MF triangulation is asymmetric; flood-fill amplified that | Axis-class segmentation deterministic per face — symmetric geometry produces symmetric regions when the triangulation is |
| Remesh step silently failed | `pymeshfix.repair(verbose=False)` — unsupported kwarg in installed version | Dropped the kwarg; later confirmed remesh destroyed mesh geometry (-11% volume), so disabled in viz-server path |
| Isotropic remesh (pymeshlab) corrupted small features | Vertex smoothing + reprojection tolerance too wide | Kept the code gated behind `IngestConfig(remesh=True)` for future use; viz-server path uses `remesh=False` |
| Through-hole wall split into 4 cardinal blotches | Axis-class segmentation naturally splits a 360° cylinder along ±Y ±Z | `_merge_adjacent_cylinder_fragments` stitches fragments whose combined normal spread exceeds 0.2 |
| Pink triangular wedges poking into flat plate | Smoothing flipped flat triangles to curved whenever >50% of neighbors were curved | Tightened: only flip aligned→curved on **zero** same-class neighbors; added reverse smoothing for slightly-tilted plate triangles |
| Tab cylinders pie-sliced into 8 wedges | Radial-fan triangulation: each slice's normal in a different cardinal class | Iterative aligned→curved smoothing dissolves isolated slices; fragment-merger re-unifies |
| 139 segfaults after pipeline runs | Native-code crash (likely OCP/CadQuery on some mesh); no Python trace | Added `faulthandler.enable()` to server startup — next segfault will print a C-level stack trace |
| Fillet radii mismatched across body seams | Per-body RANSAC fits each edge independently | Apply stage measures radius per CAD edge from the stitched group's mesh vertex set; the upstream fit radius is no longer read |

### Side-plate through-hole rim fillets — six-pass segmentation refinement

The original segmenter produced through-hole regions where the bore wall, both rim fillets, and the surrounding tangent strips on the plate face all collapsed into one or two giant labels. Visually that meant a single color blob around each hole instead of distinct highlights for each feature, with one plate looking different from the other depending on tessellation luck. Working through the failure modes one at a time produced this stack of passes in `mesh2param/pipeline/segment.py`:

1. **`_split_disconnected_components`** — `_split_flat_from_curved` peels flat faces out of a mixed region but leaves the remaining curved faces under one label even when they form mesh-disconnected components. Splits each label into one label per mesh-connected component so the bore wall and the rim fillet (separated by the plate in the mesh) don't share a label.
2. **`_unify_cylinder_features`** — finds cylinder-face candidates via 2-hop SVD rank-2 neighborhoods, DBSCAN-clusters them spatially, and unifies each cluster into one label. Uses a world-axis (X/Y/Z) bore-axis pick and filters to `|n·axis| < 0.25` so the bore wall is unified without absorbing the rim fillet's tilted-normal faces.
3. **`_split_disconnected_components` (second pass, `min_component_faces=50`)** — `_unify_cylinder_features`'s DBSCAN can bridge mesh-disconnected components if their centroids fall within `eps` of each other through the bore (front rim fillet ↔ back rim fillet on a thin plate). The post-unify split ensures front rim and back rim end up as distinct labels even when DBSCAN merged them.
4. **`_peel_axial_tangents_from_curves`** — for every region with mean-normal magnitude in `[0.3, 0.9]` (a tilted curved region, i.e. fillet arc), finds mesh-connected components of nearly-axial faces (`|n·axis| ≥ 0.985`) and peels them into their own label. Catches the very-first-row tangent strip of a fillet that otherwise visually merges with the adjacent plate. Tight 10° threshold so it doesn't eat mid-arc faces and split rolled-edge fillets along their length.
5. **`_peel_hole_rim_fillets`** — detects every through-hole by clustering radial-normal faces via DBSCAN per world axis, gated by `mean_normal_magnitude < 0.35` to reject partial arcs (rounded plate corners that are not actual through-holes). For each genuine hole, pulls hole-local rim faces (`|n·axis|` in `[0.3, 0.95]`) split by front-vs-back face side, but only when the dominant source label is plate-spanning (its bbox > 2× hole diameter). Eliminates the case where small through-holes' rim fillets sat inside the rolled-edge perimeter region.
6. **Cylinder-parent guard in `_split_flat_from_curved`** — checks the parent region's normals for rank-2 coplanar structure before peeling axial faces, so a 360° bore's cardinal-direction strips (where the arc crosses ±Y, ±Z) don't get yanked out as "flat" sub-regions and split the bore into 1 curved + 4 cardinal pieces.

End result for `models/Mesh5/RackMount.3mf`:

- Side plates: 17 / 15 regions each (was 1-2 chaotic regions for everything curved). Main through-hole has bore wall + front-rim + back-rim as three distinct labels.
- Bottom plate: 41 regions (was 52 before the round/bounded/mean-mag gates ruled out plate-corner false positives). Tabs each have 1 bore wall, 2 rims, and per-tab fillets.
- Rolled-edge fillets along plate perimeters stay unbroken along their length (the 10° tangent-peel threshold leaves mid-fillet arcs intact).

### Three other infrastructure fixes that came out of this work

- **`scipy.optimize.least_squares` method `lm` → `trf`** in all three call sites (`_fit_sphere`, `_fit_cylinder`, `_fit_cone`). The C-level MINPACK backend behind `lm` segfaulted with a Windows access violation on degenerate inputs (narrow tangent-strip regions fitted as spheres during `_disambiguate_sphere_cylinder`) and was also not thread-safe. `trf` is pure-Python trust-region reflective and handles singular Jacobians gracefully. Added `np.isfinite` input guards as defense in depth.
- **`viz_server/job_queue.py`: parallel body processing → sequential.** The `ThreadPoolExecutor` running one worker per body let segmentation on body B race against fitting on body A through MKL/BLAS shared state, producing intermittent crashes even after the fit-stage lock. Sequential processing costs 1-2s on a 3-body part and eliminates the crash class.
- **`_color_regions_no_adjacent_match` greedy graph-coloring** for the segmentation viz. Replaced `palette[ri % len(palette)]` round-robin with an algorithm that visits regions largest-first, builds region-region adjacency from edge-shared faces, and assigns each region the first palette index not used by any already-colored mesh-adjacent neighbor. Guarantees no two adjacent regions render in the same color (non-adjacent regions can still share a color with a 12-color palette).

### Test suite

Fillet-specific regression tests (`tests/test_fillet_detection.py`, `tests/test_export_fillet_consistency.py`) were removed when the upstream collinear-merge reconciliation pass was replaced by the stitched-fillets stage; the invariants they encoded (radius agreement across body seams, sphere-fit filtering at export time) no longer apply because the apply pass measures radii per edge from raw mesh vertices rather than reading any upstream fit. Future regression coverage for the new vertex-proximity matcher is open work.

Per-feature invariants in `tests/test_rackmount_segmentation.py` (20 tests, all passing) covering each body of `models/Mesh5/RackMount.3mf`:

- Side plates: two large flat regions, main through-hole bore wall in ONE region, rim fillets exist as labels distinct from the bore, no tiny axis-aligned fragments, rim fillets are hole-local (bbox ≲ 3× diameter), exactly one genuine 360° through-hole per plate.
- Bottom plate: total region count 10–60, large flat base face exists, all 4 tab through-hole bore walls each in ONE region, no excessive tiny fragments, tab flat-face fragmentation bounded.
- Cross-body: 3-body decomposition; side plates have matching region counts and matching curved-region counts (mirror-symmetry check).

### Shortcuts / open issues

- **Side-plate long-edge fillets** on the RackMount test mesh aren't captured as regions — they're absorbed into the plate or the "curved" class by the segmenter. No good fix without either a cleaner input mesh or per-edge fillet fitting.
- **Tab cylinders** on bottom plate (Body 2 of RackMount) still produce ~30 regions due to radial-fan triangulation; the fragment merger only catches the obviously-cylinder groupings.
- **Curvature estimate** is noisy on coarse meshes; the curvature-drift guard I tried in region growing fragmented cylinders, so it's disabled. Re-enable with smoothed curvature if we want to handle rolled-edge detection more aggressively.
- **Triangle subdivision during ingest** was tried and reverted — even with `process=False` preserving coplanarity, subdivision exposed per-triangle normal noise in the source mesh that fragmented segmentation further.
- **Session stability:** occasional segfaults in the FastAPI worker after long sessions. Faulthandler armed for next crash; no stack trace yet.
