"""Fast dev render against a DEPLOYED warm app (see the warm-pool toggle in
modal_app.py). Reuses hot containers, so each sweep is ~3 min instead of the cold
6-15 min ramp — worth it during an active testing session.

Workflow (one-time deploy, then repeat the render as often as you like, then stop):
    # 1. deploy a SEPARATE dev app with a warm pool (does NOT touch prod):
    WARM_POOL=12 MODAL_APP_NAME=mesh2param-dev python -m modal deploy modal_app.py
    # 2. fast, repeatable sweeps (full 15 AI parts, or pass a subset):
    MODAL_APP_NAME=mesh2param-dev python scripts/warm_render.py
    MODAL_APP_NAME=mesh2param-dev python scripts/warm_render.py AI6,AI4
    # 3. when done — scale to zero so billing STOPS (~$4/hr while warm):
    python -m modal app stop mesh2param-dev

Cost: ~$0.25-0.30/hr per warm container (4 core + 8 GB). WARM_POOL=12 ≈ $3-4/hr
while deployed; the per-sweep compute is only ~$0.15. Keep it warm only during a
session. Leave WARM_POOL unset (or just use `modal run modal_app.py::render`) for
the free cold path.
"""
import glob
import os
import sys
import time

import modal

APP = os.environ.get("MODAL_APP_NAME", "mesh2param-dev")
arg = sys.argv[1] if len(sys.argv) > 1 else ""
names = [p.strip() for p in arg.split(",") if p.strip()] or \
    [f"AI{i}" for i in range(1, 16)]

try:
    render_all = modal.Function.from_name(APP, "render_all")
except Exception as exc:
    sys.exit(f"could not find deployed app '{APP}' / render_all — deploy it first:\n"
             f"  WARM_POOL=12 MODAL_APP_NAME={APP} python -m modal deploy modal_app.py\n"
             f"({type(exc).__name__}: {exc})")

print(f"warm render of {len(names)} parts against deployed app '{APP}'...")
t0 = time.time()
call = render_all.spawn(names)
res = None
while time.time() - t0 < 1500:
    try:
        res = call.get(timeout=20)
        break
    except Exception as exc:
        print(f"  ...waiting ({time.time() - t0:.0f}s) [{type(exc).__name__}]")
        time.sleep(2)
if not res:
    sys.exit("render did not complete in time")

out_dir = "tests/visual_reports/diagnosis/_parts"
os.makedirs(out_dir, exist_ok=True)
for p, b in res.get("parts", {}).items():
    with open(f"{out_dir}/{p}.png", "wb") as fh:
        fh.write(b)
with open("tests/visual_reports/diagnosis/ALL_parts.png", "wb") as fh:
    fh.write(res["montage"])
for p, v in sorted(res.get("vol_errs", {}).items()):
    print(f"{p}: vol_err={v:.1f}%")
cc = res.get("concurrency", {})
if cc:
    print(f"CONCURRENCY: {cc['peak_containers']}/{cc['n_parts']} containers at once "
          f"(effective_parallelism={cc['effective_parallelism']})")
print(f"wrote tests/visual_reports/diagnosis/ALL_parts.png | wall {time.time() - t0:.0f}s")
