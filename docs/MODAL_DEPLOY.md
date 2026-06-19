# Modal deploy — parallel pipeline for dev + public launch

One Modal function, `build_mesh` (in `modal_app.py`), is the unit of parallelism
for everything: each call runs the full mesh→CAD pipeline in its **own isolated
container**, so the pymeshlab-can't-run-concurrently wall is gone, and Modal
autoscales 0→N and bills per second.

## Setup (one time)
```bash
pip install modal
modal setup          # authenticate in the browser
```

## Dev — the 15 AI meshes in parallel (~minutes, not 15-20)
```bash
modal run modal_app.py                        # all AI1..AI15
modal run modal_app.py --parts AI9,AI12,AI13  # a subset
```
PROVEN: full 15-part run = **mean IoU 0.780, wall ~195s** (~3.3 min) in parallel
isolated containers — vs 15-20 min sequential, and no pymeshlab stalls.

### Windows gotchas (one time)
- Set UTF-8 or the Modal CLI crashes printing its `✓`:
  `setx PYTHONUTF8 1`  (new shells), or per-shell `$env:PYTHONUTF8=1`.
- `modal` isn't on PATH after `pip install` — use `python -m modal ...`.
- A burst of `Task was destroyed ... / aclose(): asynchronous generator is
  already running` AFTER `✓ App completed` is HARMLESS Modal-on-Windows-3.13
  shutdown noise — the run already succeeded. (Filter it: `... | Select-String IoU`.)
The local entrypoint reads `models/<part>/*.3mf` on your machine, ships the bytes
to Modal, and `build_mesh.starmap` fans them out across containers in parallel.
Prints each part's IoU + a mean. First run builds the image (~few min, cached
after). Cost: roughly **$0.10–0.30** for a full 15-part sweep, scale-to-zero.

## Production architecture (the public launch)
```
 [FastAPI web server]      [Modal: build_mesh]
   upload  ───────────►  .spawn(bytes)  ─► isolated container (autoscaled 0→N)
   poll    ◄───────────  call.get()     ◄─ {tess_*, step, iou, vol_err, ...}
   serve UI + results
```
The web server stays as the front-end (uploads, auth, UI, serving results); the
**heavy pipeline moves to Modal**, one container per customer job — concurrency-
safe and parallel by construction. No in-process worker queue to overload.

### Wiring the server to Modal (replaces the local job worker)
1. `modal deploy modal_app.py` — deploys `build_mesh` as a callable cloud function.
2. In the server, look it up once and dispatch per job:
   ```python
   import modal
   _build = modal.Function.from_name("mesh2param-pipeline", "build_mesh")

   # in the run-pipeline endpoint, instead of enqueuing to the local worker:
   call = _build.spawn(mesh_bytes, filename, want_geometry=True)  # async, isolated
   job.modal_call_id = call.object_id                              # store on the job

   # status/poll endpoint:
   call = modal.FunctionCall.from_id(job.modal_call_id)
   try:
       result = call.get(timeout=0)        # ready? -> {tess_vertices, tess_faces, step, iou}
       # build ConstructionSteps / final mesh from result, mark job done
   except modal.exceptions.OutputExpiredError:
       ...                                  # not ready yet -> keep polling
   ```
3. Concurrency is automatic: N customers → N Modal containers. Set
   `@app.function(..., max_containers=...)` to cap spend; scale-to-zero when idle.
4. Auth/billing/token-metering stay in the web server (unchanged). Only the
   compute moves out.

### Live build streaming (the dual-pane "watch it build" view)
Two options, in order of effort:
- **Simple:** return the final result from `build_mesh` and skip per-step streaming
  for cloud jobs (show a progress spinner). Fastest to ship.
- **Streaming:** have `build_mesh` write each `ConstructionStep` to a `modal.Queue`
  or `modal.Dict` keyed by call id; the server's status endpoint drains it. This
  preserves the existing live-build UX. (Do this after the simple path works.)

## Cost (per the 2026 research)
- Pay-per-second CPU, scale-to-zero — no idle charge.
- A customer job (~2 min, 4 vCPU) ≈ a couple of cents; **1,000 jobs/mo ≈ $20–50**.
- A full 15-part dev sweep ≈ **$0.10–0.30**.
- Cap spend with `max_containers` and a per-job `timeout`.

## Notes
- The Modal image mirrors `_docker/Dockerfile` (the validated lean pipeline stack
  with the alpha-wrap system libs). Keep them in sync if deps change.
- `add_local_python_source("mesh2param")` ships the live package, so code edits
  don't require an image rebuild — only dependency changes do.
- The product's user-facing Claude features still use the metered API, not this.
