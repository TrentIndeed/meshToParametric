# Paramesh AI — Market & Roadmap

Strategic direction: who the app is for, and the difficulty ladder we climb
as the mesh→CAD engine matures. (Product strategy, not a task list — see
`TODO.md` for the engineering backlog.)

---

## 1. The core question
Not "which industries use CAD," but: **who has a MESH and needs it as editable,
manufacturable parametric CAD?** Three inflows of meshes-needing-CAD:
- **Physical-part scans** (phone / structured-light / laser) — reverse engineering.
- **Downloaded / sculpted STLs** that need to become editable.
- **AI-generated meshes** (Meshy, Tripo, …) — a brand-new, fast-growing inflow.

## 2. Target market
**Mechanical is the heartland.** Mechanical parts ARE prismatic — brackets,
plates, mounts, housings, fixtures — which is exactly the feature set
(base extrude, holes, pockets, bosses, fillets, chamfers) the pipeline models.

Two wedges that share the same engine:
- **Prosumer / maker / 3D-printing** — *largest user base by count.* They
  download STLs, generate parts with AI, or phone-scan objects, then want to
  EDIT them (resize, change a hole, make it parametric). Underserved because
  pro mesh→solid tools (Fusion, etc.) are clunky. Low ACV, high volume, viral,
  AI-chat-friendly → the **growth engine**.
- **Reverse engineering / repair / aftermarket / machine shops** — scan a
  broken or legacy part, remanufacture it. Higher value per job, real
  willingness to pay, historically painful (Geomagic-class tools) → the
  **monetization engine**.

**Electronics** contributes only via its *mechanical* side — enclosures,
brackets, connector housings, mounting plates. The PCB itself is born in EDA
tools (Altium/KiCad) and is never a mesh→CAD job. So electronics ≈ a subset of
the mechanical/maker crowd, not a separate market.

## 3. Architecture — real industry, wrong product
Reverse-engineering meshes in architecture is a big, real business, but **not
ours**:
- New buildings are designed in CAD/BIM (Revit, ArchiCAD) from scratch — no
  mesh→CAD step.
- The mesh→CAD that exists is **scan-to-BIM** (laser-scan an existing building
  → as-built BIM for renovation/heritage). Different geometry problem entirely
  — walls/floors/slabs/MEP at building scale, cm tolerances, BIM semantics —
  and a different competitive field (Matterport, Leica). Our prismatic feature
  tree doesn't map onto it; chasing it = a rebuild, not a feature.
- The ONLY architecture overlap that fits: **mechanical components used in
  buildings** (a bracket, fixture, façade clip, hardware) — but that's just
  "mechanical parts," already covered.

## 4. The AI-generation → CAD tailwind (the big future bet)
As text/image→3D generation explodes, a flood of meshes will exist that someone
wants to actually BUILD — and the bridge from "blob I generated" to "parametric
CAD I can fabricate" is largely unbuilt. Owning that bridge is the biggest
*future* growth vector, and it's exactly where this pipeline sits. Legacy tools
ignore it; some competitors generate CAD directly from text and skip the mesh —
our niche is strongest where a **mesh already exists** (scan / download / prior
AI gen) and the part is **prismatic**.

Caveat that is the whole game: **conversion QUALITY is the moat.** "Convert it
quickly" is easy to say and hard to do — for complex/curved/assembly geometry
it is precisely the unsolved part. We win by being *correct*, per part, first.

## 5. The difficulty ladder (how we climb)
Aim the AI-gen→CAD tailwind at the rung that's ready, climb as the engine
hardens:

1. **NOW — single prismatic parts.** Brackets, plates, flanges, standoffs,
   mounts. The market that's ready today; the 15 AI test parts live here.
2. **NEXT — multi-feature single parts + simple weldments.** Hole patterns,
   pockets, bosses, mixed fillets/chamfers, multi-body unions (the Mesh5 side-
   plate work). Mostly done.
3. **LATER — small assemblies / architectural-metal components.** Railings,
   stairs *decomposed into their fabricated parts* (a tread is a plate, a
   stringer is a cut plate, a bracket is a bracket). Architectural METAL
   FABRICATION (stairs, railings, gates, façade panels, structural brackets) is
   closer to mechanical fabrication than to building BIM — a real adjacent
   market, reached per-component.
4. **SOMEDAY — full curved assemblies.** Whole staircases/railings as single
   curved structures. Genuinely hard: assembly decomposition + freeform +
   structural. This is where the "AI-gen a whole staircase → CAD" idea lives —
   a long-game extension once the per-part engine is rock-solid, NOT a near-term
   target. Whole-assembly + freeform is the problem reverse-engineering tools
   have always struggled with.

**On "AI-generate a staircase → CAD":** great instinct on the tailwind, but a
staircase is rung 4 (assembly + often curved + structural), and stairs are
usually *configured* parametrically (rise/run codes matter more than matching a
blob) rather than reverse-engineered. The realistic near-term version is rung 3:
AI-gen or scan the **individual fabricated parts** of stairs/railings/brackets →
clean CAD → laser/CNC/welder.

## 6. Scope guardrails
- **In:** prismatic mechanical parts and fabricated components (incl.
  architectural metal-fab parts), scans + downloads + AI-generated meshes.
- **Out (for now):** building-scale scan-to-BIM, freeform/organic shapes, whole
  curved assemblies. Saying "for mechanical parts" sharpens the product; trying
  to also do buildings or sculptures dilutes everything.

## 7. Why the current engineering work is dead-center
Both target wedges — reverse-engineering scans AND AI-generated meshes — produce
the same **dirty meshes** (wobbly faces, approximate angles, noise). So the
scan/AI front-end we're building (denoise → align → regularize → noise-robust
segmentation, plus the dirty-mesh detector) is not a side quest: it's the moat
("conversion quality") for the exact inflows that define the market. See
`SCAN_RECOVERY_PLAN.md` for the technical plan and current status.
