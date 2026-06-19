# Claude Usage Plan — how to use Claude on this repo

The biggest debugging bottleneck is that reconstruction failures are **geometric**
(mispositioned extrusion, uncut cavity, fillet on the wrong edge) and a single
IoU/volume number can't localize them. Claude is **multimodal** — it can read
rendered images — so the fix is to render the mesh-vs-model and let Claude *see*.
This doc is the reference for the protocol summarized in `CLAUDE.md`.

Target: the **$200/mo Max** plan. The budget is a usage allowance over rolling
windows, not metered dollars — so optimize for **repo improvement per unit of
allowance, with zero blind rework**.

## Three principles
1. **See before you fix.** Never diagnose a geometry problem from numbers alone.
   Blind fixes that need rework are the biggest allowance waste.
2. **Verify before you claim.** No "it works" without a before/after render + a
   metric delta (guardrail and/or container exact IoU).
3. **Cheap by default, escalate on a trigger.** Solo visual loop is the norm;
   multi-agent fan-out and full AI-set runs are earned, not reflexive.

## The mandatory loop (every session)
```
0. ORIENT    (free)  CLAUDE.md + the stage's handoff doc + recalled memory.
                     Classify: GEOMETRY/recon  vs  WEB/infra/docs.
1. SEE       (cheap) GEOMETRY only: scripts/diag_render_sheet.py <part>; READ png.
2. DIAGNOSE  (free)  Localize the failing STAGE from heatmap+overlay, not numbers.
3. FIX       (targeted) one right stage module.
4. VERIFY    (cheap) re-render before/after; clean-CAD → test_step_comparison
                     (18/7); AI → container diag_ai_build on affected parts.
5. RECORD    (free)  memory + handoff: only what was non-obvious.
```

### The keystone tool — `scripts/diag_render_sheet.py`
Headless (matplotlib Agg, no VTK/display) so it runs in the lean `mesh2param`
container. Emits `tests/visual_reports/diagnosis/<part>.png`: 2 camera angles ×
[cleaned mesh | reconstruction | overlay | error heatmap]. Heatmap = recon
coloured by signed distance to the mesh — **RED = extra material outside the true
surface, BLUE = inside** (blue voids where the mesh has material = recon is
missing it). Title carries IoU / vol_err / feature list / dims.

Run (Git Bash on Windows; mounts source + models + output):
```bash
W="$(pwd -W)"; MSYS_NO_PATHCONV=1 docker run --rm \
  -v "${W}/models":/app/models -v "${W}/mesh2param":/app/mesh2param \
  -v "${W}/scripts":/app/scripts -v "${W}/tests":/app/tests \
  mesh2param python scripts/diag_render_sheet.py AI13 AI9
```
Then `Read` the PNG. Worked example: **AI13** shows a thin-walled hollow
enclosure rebuilt as a near-solid block with 8 pockets — vol_err 1.0% (looked
fine) but IoU 0.55, because the single interior cavity was fragmented. The number
hid it; the render made it obvious.

## Escalate only on a trigger
| Trigger | Action |
|---|---|
| "audit / all parts / comprehensive" | subagent per AI part → synthesis |
| change to a load-bearing stage (ingest/segment/fit/recognize) | adversarial-verify subagent: try to *refute* the fix before commit |
| any pipeline behavior change | exactly ONE full AI-set container run (not per tweak) |
| pure WEB/infra/docs | standard loop, no renders, no fan-out |

## Budget rules ($200 Max)
- One composited sheet per part (4 panels, one ≤1568px image ≈ ~1.5k tokens).
- One AI-set container run per substantive change; tweaks verify on the 2–3
  affected parts only.
- Memory is the anti-waste tool: recall before re-investigating, record after.
- Batch, don't drip: web/MCP/multi-part diagnosis in one call.
- Fan-out is rationed — only on the "audit" trigger.

## Keep product AI off the dev budget
The app's user-facing Claude (chat, dimension planner, a future "AI Review of the
render") should run on the **metered API** (Haiku default + Batch −50% + prompt
caching), NOT the OAuth subscription, so end-user traffic never competes with the
dev allowance. Rule: **dev work = subscription; product features = API.** `ai.py`
already enforces per-session caps.

## Modern Claude features → this repo
- **Vision (multi-image, ≤1568px, up to 100/req)** — the diagnosis sheet; pack
  many parts in one request for cross-image grounding.
- **Structured outputs / tool use** — force JSON (`failure_mode`, `suggested_edits`)
  so diagnosis is actionable into the `/api/job/{id}/edit` rebuild API.
- **Subagents (Agent SDK)** — one diagnosis agent per part for "audit all 15".
- **MCP** — expose the pipeline (run part / get IoU / apply fillet) as tools so an
  agent can operate it, not just read it.
- **Computer use** — skip for now: a canvas screenshot + vision gets ~90% of the
  value at ~5% of the cost.

## Cost facts (for any move to the metered API)
Batch API −50%, prompt caching −90% input (but TTL dropped to ~5 min in 2026, so
batch tightly). Per-M tokens: Haiku $1/$5, Sonnet $3/$15, Opus $5/$25. Render
locally (free); call Claude only on the assembled sheet, gated by a numeric
trigger (e.g. IoU < 0.7 or vol_err > 10%).
