"""Modal app — run the mesh->CAD pipeline in ISOLATED, PARALLEL cloud containers.

Why Modal: pymeshlab (alpha-wrap, the slow ingest stage) can't run concurrently
in one machine's processes, so neither local threads nor a local process pool
can parallelize the AI parts. On Modal each call runs in its OWN container —
isolated, so the conflict vanishes — and Modal autoscales 0->N and bills per
second. The SAME `build_mesh` function serves BOTH:
  * dev      — build_mesh.starmap(...) over the 15 AI meshes, in parallel
  * prod     — the web server calls build_mesh.spawn(bytes) per customer upload

── Setup (one time) ─────────────────────────────────────────────────────────
  pip install modal
  modal setup                      # opens a browser to authenticate

── Dev: build the AI meshes in parallel + print IoUs ────────────────────────
  modal run modal_app.py                          # all AI1..AI15
  modal run modal_app.py --parts AI9,AI12,AI13    # a subset
  (runs from the repo root so it can read models/<part>/*.3mf locally and ship
   the bytes to the cloud — models/ is NOT baked into the image.)

── Prod: deploy so the web server can dispatch jobs to it ───────────────────
  modal deploy modal_app.py
  # then in the server:
  #   fn = modal.Function.from_name("mesh2param-pipeline", "build_mesh")
  #   call = fn.spawn(mesh_bytes, filename, want_geometry=True)   # async, isolated
  #   result = call.get()                                          # {tess_*, step, iou, ...}
  # See docs/MODAL_DEPLOY.md for the full production wiring.
"""
import os

import modal

# --- Warm-pool toggle (dev only) -------------------------------------------
# WARM_POOL=N keeps N build_mesh containers hot so a DEPLOYED dev app serves the
# sweep in ~3 min instead of the 6-15 min cold-start ramp. It only takes effect on
# `modal deploy` (an ephemeral `modal run` cold-starts regardless). Default 0 =
# normal autoscale, so prod is unaffected. To avoid warming the PROD app by
# accident, deploy the warm pool under a SEPARATE name via MODAL_APP_NAME, e.g.:
#   WARM_POOL=12 MODAL_APP_NAME=mesh2param-dev modal deploy modal_app.py
#   python scripts/warm_render.py            # fast sweeps against the warm app
#   modal app stop mesh2param-dev            # scale to zero, billing stops
# scaledown_window keeps a container warm this many seconds after its last call
# (so back-to-back sweeps stay hot, then it scales down on its own).
_WARM = int(os.environ.get("WARM_POOL", "0") or "0")
_SCALEDOWN = int(os.environ.get("WARM_SCALEDOWN", "600") or "600")
_APP_NAME = os.environ.get("MODAL_APP_NAME", "mesh2param-pipeline")

# Mirror the validated lean pipeline stack from _docker/Dockerfile: the apt libs
# pymeshlab's alpha-wrap plugin needs + the pip pins proven to work in the
# container. (No server/pyvista — this is just the pipeline.)
_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1", "libglib2.0-0", "libxrender1", "libxext6", "libsm6",
        "libgomp1", "libegl1", "libcom-err2", "libkrb5-3", "libgssapi-krb5-2",
        "libk5crypto3", "libxcb1", "libxkbcommon0", "libdbus-1-3",
        "libfontconfig1", "libfreetype6", "libopengl0", "libglu1-mesa",
    )
    .pip_install(
        "numpy>=2,<3", "scipy", "scikit-learn", "scikit-image", "networkx",
        "shapely", "pydantic", "click", "pymeshfix", "matplotlib",
        "trimesh==4.11.5", "fast-simplification", "cadquery==2.7.0",
        "manifold3d", "pymeshlab", "lxml", "distinctipy",
    )
)
# Lean image for build_mesh (server + prod jobs): just the live pipeline source.
image = _base.add_local_python_source("mesh2param")
# Render image for the dev sweep: ALSO bakes the model files in, so the cloud
# orchestrator (render_all) reads meshes from the image instead of the client
# uploading them — the client never ships mesh bytes over its flaky link.
# EXCLUDE the scans (Scan1/2/3 = 220 MB of the 282 MB models dir, e.g. the 104 MB
# Cam-gear): they balloon the image + are the slowest builds, and the standard
# dev sweep is the 15 AI parts (per CLAUDE.md). Scans render via --parts on the
# slower per-mesh path, not this fast image.
# (copy=True layer goes BEFORE the python-source mount, which must stay last.)
render_image = (
    _base.add_local_dir("models", remote_path="/root/models", copy=True,
                        ignore=["Scan*", "Scan*/**"])
    .add_local_python_source("mesh2param")
)

app = modal.App(_APP_NAME, image=image)


STEP_QUEUE_NAME = "mesh2param-steps"
_STEP_MAX_BYTES = 950_000  # modal.Queue items cap at 1 MiB; stay under it


def _make_step_sink(stream_key: str):
    """Build a `sink(payload, n)` that forwards each construction step to the
    shared step Queue, partitioned by `stream_key`. The web server's poller
    drains that partition live so the cloud build streams step-by-step, exactly
    like a local build. Best-effort: a dropped step just reappears in the final
    result. Returns (sink, queue) or (None, None) if the Queue is unavailable."""
    import numpy as np
    try:
        q = modal.Queue.from_name(STEP_QUEUE_NAME, create_if_missing=True)
    except Exception:
        return None, None

    def _sink(payload: dict, n: int) -> None:
        v, f = payload.get("tess_vertices"), payload.get("tess_faces")
        if v is None or f is None:
            return
        va = np.asarray(v, np.float32).reshape(-1, 3)
        fa = np.asarray(f, np.int32).reshape(-1, 3)
        if va.nbytes + fa.nbytes > _STEP_MAX_BYTES:
            return  # too big for a Queue item; final result will carry it
        msg = {
            "i": n,
            "name": payload.get("name", f"Step {n + 1}"),
            "tess_vertices": va,
            "tess_faces": fa,
            "volume": float(payload.get("volume", 0.0) or 0.0),
        }
        try:
            q.put(msg, partition=stream_key, partition_ttl=3600)
        except Exception:
            pass

    return _sink, q


@app.function(cpu=4.0, memory=8192, timeout=900,
              min_containers=_WARM, scaledown_window=_SCALEDOWN)
def build_mesh(mesh_bytes: bytes, filename: str, want_geometry: bool = False,
               stream_key: str | None = None, editor: bool = False,
               want_iou: bool = True) -> dict:
    """Full mesh->CAD pipeline on ONE mesh, in an isolated container. The unit of
    parallelism for both dev validation and production jobs. Returns metrics; with
    want_geometry=True also returns the tessellated recon solid (light). editor=True
    ADDITIONALLY ships the heavy rebuild bundle + STEP bytes for the viewer/editor —
    server jobs set it; the dev render path leaves it off so the result stays small
    (the big bundle transfer over a flaky link is what blew past 5 min).
    With stream_key set, each construction step is also pushed live to the shared
    step Queue (partition=stream_key) so the server can stream the build."""
    import os
    import tempfile
    import time

    import numpy as np

    t0 = time.time()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "stl"
    with tempfile.NamedTemporaryFile(suffix="." + ext, delete=False) as fh:
        fh.write(mesh_bytes)
        path = fh.name

    step_sink, step_queue = (None, None)
    if stream_key:
        step_sink, step_queue = _make_step_sink(stream_key)
    step_out = None

    try:
        from mesh2param.config import IngestConfig
        from mesh2param.pipeline.convert import analyze_part
        from mesh2param.pipeline.export_cadquery import export_cadquery

        # SAME pipeline the server runs: ingest -> decompose -> per-body
        # recognize_body -> cross-body stitched fillets. analyze_part is the
        # single source of truth (mesh2param/pipeline/convert.py), so a cloud
        # build produces the same geometry a local job does — multi-body parts
        # and stitched fillets included.
        analysis = analyze_part(path, IngestConfig(remesh=False, scan_auto=True))
        c = analysis.cleaned
        tree = analysis.tree
        if tree is None:
            return {"filename": filename, "ok": False,
                    "error": "no body recognized", "elapsed": round(time.time() - t0, 1)}
        # Only the editor needs a STEP (edge-picking + download); skip writing it
        # on the dev render path to keep the call light + fast.
        if editor:
            fd, step_out = tempfile.mkstemp(suffix=".step")
            os.close(fd)
        # EXACT same export call as the local CAD subprocess (job_queue line ~727):
        # body_results only when multi-body, plus the stitched-fillet clusters.
        res = export_cadquery(
            tree, c, use_llm=False, num_variants=1,
            body_results=analysis.body_results if analysis.multi_body else None,
            stitched_clusters=analysis.stitched_clusters,
            output_path=step_out, stream_sink=step_sink)

        iou = None
        # The exact-IoU arbiter is a manifold boolean over both solids — useful for
        # validation (main/server) but NOT shown in the visual render montage (which
        # uses vol_err). want_iou=False on the render path skips it to cut per-part
        # cost while keeping the full mesh-vs-recon comparison.
        try:
            import trimesh
            from mesh2param.pipeline.iou_refine import exact_iou_available, mesh_iou_exact
            if want_iou and exact_iou_available() and res.tess_vertices is not None:
                tgt = trimesh.Trimesh(vertices=c.vertices, faces=c.faces, process=False)
                rec = trimesh.Trimesh(
                    vertices=np.asarray(res.tess_vertices, float).reshape(-1, 3),
                    faces=np.asarray(res.tess_faces, np.int64).reshape(-1, 3),
                    process=False)
                iou = float(mesh_iou_exact(tgt, rec))
        except Exception:
            pass

        out = {
            "filename": filename, "ok": True, "iou": iou,
            "vol_err": float(res.volume_error_pct),
            "n_regions": analysis.n_regions, "n_features": tree.num_features,
            "elapsed": round(time.time() - t0, 1),
            # absolute epochs so the orchestrator can compute real container
            # concurrency (peak interval overlap) vs cold-start/queue waves.
            "t_start": round(t0, 3), "t_end": round(time.time(), 3),
        }
        if want_geometry:
            # Light geometry: just the recon tessellation + volumes. Enough for the
            # dev render montage (which loads the MESH panel from the local file).
            out["tess_vertices"] = np.asarray(res.tess_vertices, np.float32).reshape(-1).tolist()
            out["tess_faces"] = np.asarray(res.tess_faces, np.int32).reshape(-1).tolist()
            out["cad_volume"] = float(res.cad_volume)
            try:
                import trimesh
                out["mesh_volume"] = abs(float(trimesh.Trimesh(
                    vertices=c.vertices, faces=c.faces, process=False).volume))
            except Exception:
                out["mesh_volume"] = 0.0
        if editor:
            # Editor parity (server jobs only): the durable rebuild bundle — the
            # SAME 4-tuple the local worker pickles, (tree, cleaned, body_results,
            # decomposition) — so /api/job/{id}/edit rebuilds multi-body parts too,
            # plus the STEP bytes for edge-picking + download. Heavy (~MBs/part),
            # so it is OFF on the dev render sweep.
            import pickle
            try:
                out["bundle_bytes"] = pickle.dumps(
                    (tree, c, analysis.body_results, analysis.decomposition),
                    protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                out["bundle_bytes"] = None
            if step_out and os.path.exists(step_out):
                try:
                    with open(step_out, "rb") as sfh:
                        out["step_bytes"] = sfh.read()
                except Exception:
                    pass
        return out
    except Exception as exc:
        import traceback
        return {"filename": filename, "ok": False, "error": str(exc),
                "trace": traceback.format_exc()[-1500:],
                "elapsed": round(time.time() - t0, 1)}
    finally:
        # Signal end-of-stream so the poller stops draining this partition
        # promptly (it also stops when the call result arrives).
        if step_queue is not None and stream_key:
            try:
                step_queue.put({"__done__": True}, partition=stream_key,
                               partition_ttl=3600)
            except Exception:
                pass
        for _p in (path, step_out):
            if _p:
                try:
                    os.unlink(_p)
                except Exception:
                    pass


@app.function(image=render_image, cpu=2.0, memory=8192, timeout=600)
def diagnose(names: list) -> dict:
    """Lean Linux-side diagnostic (no full recon, so fast): for each baked AI part,
    report whether the two new cut-extrude detectors fire and whether the MESH
    actually has interior voids at mid-thickness. Pins down whether a suspect recon
    feature (e.g. a recess on a flat plate) is a real mesh feature or a detector
    mis-fire — runs the SAME Linux pipeline as the render, so it matches."""
    import glob

    import numpy as np
    from shapely.geometry import MultiPolygon
    import trimesh

    from mesh2param.config import IngestConfig
    from mesh2param.pipeline.ingest import ingest_mesh
    from mesh2param.pipeline.recognize import _section_material
    from mesh2param.pipeline.export_cadquery import (
        _detect_face_pockets, _detect_tangent_holes)

    out = {}
    for p in names:
        hits = sorted(
            glob.glob(f"/root/models/{p}/*.3mf") + glob.glob(f"/root/models/{p}/*.stl")
            + glob.glob(f"/root/models/{p}/*.obj") + glob.glob(f"/root/models/{p}/*.ply"))
        if not hits:
            out[p] = {"error": "no mesh in image"}; continue
        try:
            c = ingest_mesh(hits[0], IngestConfig(remesh=False, scan_auto=True))
            m = trimesh.Trimesh(vertices=c.vertices, faces=c.faces, process=False)
            lo, hi = np.asarray(c.bbox[0]), np.asarray(c.bbox[1])
            dims = hi - lo
            axis = int(np.argmin(dims))
            mmid = _section_material(m, axis, 0.5 * (lo[axis] + hi[axis]))
            nvoid = 0
            if mmid is not None:
                polys = list(mmid.geoms) if isinstance(mmid, MultiPolygon) else [mmid]
                nvoid = sum(len(pp.interiors) for pp in polys)
            ext = np.zeros(3); ext[axis] = 1.0
            fp = _detect_face_pockets(c, ext)
            th = _detect_tangent_holes(c, ext)
            out[p] = {
                "dims": [round(float(x), 1) for x in dims],
                "thin_axis": "XYZ"[axis],
                "mesh_mid_interior_voids": int(nvoid),
                "face_pockets_detected": len(fp),
                "tangent_holes_detected": len(th),
                "pocket_depths": [round(float(d.get("depth", 0)), 2) for d in fp][:6],
            }
        except Exception as exc:
            import traceback
            out[p] = {"error": str(exc), "trace": traceback.format_exc()[-400:]}
    return out


@app.local_entrypoint()
def diag(parts: str = "AI1,AI10,AI15"):
    """Run the lean detector diagnostic on the named parts (cloud, resilient
    spawn+poll). Tells us if suspect recon features are real or detector mis-fires."""
    import json
    import time

    names = [p.strip() for p in parts.split(",") if p.strip()]
    print(f"diagnosing {names}...")
    t0 = time.time()
    call = diagnose.spawn(names)
    res = None
    while time.time() - t0 < 900:
        try:
            res = call.get(timeout=20); break
        except Exception as exc:
            print(f"  ...waiting ({time.time() - t0:.0f}s) [{type(exc).__name__}]")
            time.sleep(2)
    if not res:
        print("diagnose did not complete"); return
    print(json.dumps(res, indent=2))
    print(f"wall {time.time() - t0:.0f}s")


@app.function(image=render_image, cpu=4.0, memory=8192, timeout=900)
def split_ab(part: str, lean: bool = False) -> dict:
    """Build ONE baked AI part twice — with the segmentation split pass ON and OFF
    (PARAMESH_NO_SPLIT) — and return both recons' stats (+ geometry unless lean).
    Pins down whether a suspect base-recognition feature (e.g. AI15's top grooves)
    is caused by the split pass added this session or is pre-existing."""
    import glob
    import os

    import numpy as np

    from mesh2param.config import IngestConfig
    from mesh2param.pipeline.convert import analyze_part
    from mesh2param.pipeline.export_cadquery import export_cadquery

    hits = sorted(
        glob.glob(f"/root/models/{part}/*.3mf") + glob.glob(f"/root/models/{part}/*.stl")
        + glob.glob(f"/root/models/{part}/*.obj") + glob.glob(f"/root/models/{part}/*.ply"))
    if not hits:
        return {"part": part, "error": "no mesh"}
    path = hits[0]

    from mesh2param.pipeline import segment as _seg

    out = {"part": part}
    for tag, no_split in (("split_on", "0"), ("split_off", "1")):
        os.environ["PARAMESH_NO_SPLIT"] = no_split
        _seg._SPLIT_EVENTS.clear()
        try:
            a = analyze_part(path, IngestConfig(remesh=False, scan_auto=True))
            res = export_cadquery(
                a.tree, a.cleaned, use_llm=False, num_variants=1,
                body_results=a.body_results if a.multi_body else None,
                stitched_clusters=a.stitched_clusters)
            rec = {
                "vol_err": float(res.volume_error_pct),
                "n_regions": a.n_regions, "n_features": a.tree.num_features if a.tree else 0,
                "split_events": list(_seg._SPLIT_EVENTS),
            }
            if not lean:
                rec["tess_vertices"] = np.asarray(res.tess_vertices, np.float32).reshape(-1).tolist()
                rec["tess_faces"] = np.asarray(res.tess_faces, np.int32).reshape(-1).tolist()
            out[tag] = rec
        except Exception as exc:
            import traceback
            out[tag] = {"error": str(exc), "trace": traceback.format_exc()[-400:]}
    return out


@app.function(image=render_image, cpu=2.0, memory=8192, timeout=1800)
def split_sweep(names: list) -> list:
    """Net effect of the segmentation split pass across the AI set: build each part
    split-ON vs split-OFF (lean) in parallel sibling containers, return the table."""
    return list(split_ab.starmap([(n, True) for n in names]))


@app.local_entrypoint()
def splitsweep(parts: str = ""):
    """Quantify the split pass across all AI parts: per-part vol_err on vs off, so I
    can decide remove-vs-guard and catch regressions beyond AI15."""
    import time

    names = [p.strip() for p in parts.split(",") if p.strip()] or \
        [f"AI{i}" for i in range(1, 16)]
    print(f"split-sweep over {len(names)} parts...")
    t0 = time.time()
    call = split_sweep.spawn(names)
    res = None
    while time.time() - t0 < 1500:
        try:
            res = call.get(timeout=20); break
        except Exception as exc:
            print(f"  ...waiting ({time.time() - t0:.0f}s) [{type(exc).__name__}]")
            time.sleep(2)
    if not res:
        print("split_sweep did not complete"); return
    print(f"\n{'part':<7} {'ON%':>7} {'OFF%':>7} {'delta':>7}  verdict")
    helped = hurt = 0
    for r in sorted(res, key=lambda x: x.get("part", "")):
        on = r.get("split_on", {}).get("vol_err")
        off = r.get("split_off", {}).get("vol_err")
        if on is None or off is None:
            print(f"{r.get('part',''):<7}  err {r.get('split_on',{}).get('error','') or r.get('split_off',{}).get('error','')}"); continue
        delta = on - off  # positive => split made it WORSE
        v = "split HURTS" if delta > 1.0 else ("split helps" if delta < -1.0 else "~same")
        if delta > 1.0: hurt += 1
        elif delta < -1.0: helped += 1
        print(f"{r['part']:<7} {on:>7.1f} {off:>7.1f} {delta:>+7.1f}  {v}")
    print(f"\nsplit HELPS {helped} part(s), HURTS {hurt} part(s) | wall {time.time()-t0:.0f}s")


@app.local_entrypoint()
def splitdiff(part: str = "AI15"):
    """Render mesh | split-ON recon | split-OFF recon for one part, to SEE whether
    the segmentation split pass is responsible for a suspect feature."""
    import glob
    import io
    import os
    import time

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    print(f"A/B split diff for {part}...")
    t0 = time.time()
    call = split_ab.spawn(part)
    res = None
    while time.time() - t0 < 900:
        try:
            res = call.get(timeout=20); break
        except Exception as exc:
            print(f"  ...waiting ({time.time() - t0:.0f}s) [{type(exc).__name__}]")
            time.sleep(2)
    if not res or "error" in res:
        print(f"failed: {res}"); return

    hits = sorted(glob.glob(f"models/{part}/*.3mf") + glob.glob(f"models/{part}/*.stl")
                  + glob.glob(f"models/{part}/*.obj") + glob.glob(f"models/{part}/*.ply"))
    Vm = Fm = None
    if hits:
        import trimesh
        ext = hits[0].rsplit(".", 1)[-1].lower()
        try:
            obj = trimesh.load(hits[0], file_type=ext, process=False)
            if hasattr(obj, "geometry"):
                obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
            Vm, Fm = np.asarray(obj.vertices, float), np.asarray(obj.faces, np.int64)
        except Exception:
            pass

    def _panel(ax, V, F, col, title):
        if V is None or F is None or len(F) == 0:
            ax.set_title(title + " (none)", fontsize=8); ax.set_axis_off(); return
        if len(F) > 4000:
            F = F[::int(np.ceil(len(F) / 4000))]
        ax.add_collection3d(Poly3DCollection(V[F], alpha=1, facecolor=col,
                                             edgecolor="#33414f", linewidths=0.05))
        mn, mx = V.min(0), V.max(0)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
        try:
            ax.set_box_aspect(mx - mn)
        except Exception:
            pass
        ax.view_init(28, -52); ax.set_axis_off(); ax.set_title(title, fontsize=9)

    fig = plt.figure(figsize=(13, 4))
    _panel(fig.add_subplot(1, 3, 1, projection="3d"), Vm, Fm, "#9fb4c7", f"{part} MESH")
    for i, tag in enumerate(("split_on", "split_off")):
        d = res.get(tag, {})
        V = F = None
        if "tess_vertices" in d:
            V = np.asarray(d["tess_vertices"], float).reshape(-1, 3)
            F = np.asarray(d["tess_faces"], np.int64).reshape(-1, 3)
        _panel(fig.add_subplot(1, 3, i + 2, projection="3d"), V, F, "#7fa8d8",
               f"{tag} {d.get('vol_err', 0):.0f}% f={d.get('n_features', 0)} "
               f"reg={d.get('n_regions', 0)}")
    os.makedirs("tests/visual_reports/diagnosis", exist_ok=True)
    op = f"tests/visual_reports/diagnosis/{part}_splitdiff.png"
    fig.tight_layout(); fig.savefig(op, dpi=120); plt.close(fig)
    for tag in ("split_on", "split_off"):
        d = res.get(tag, {})
        print(f"{tag}: vol_err={d.get('vol_err')}% regions={d.get('n_regions')} "
              f"features={d.get('n_features')}")
        for ev in d.get("split_events", []):
            print(f"    split-candidate {ev}")
    print(f"wrote {op} | wall {time.time() - t0:.0f}s")


@app.local_entrypoint()
def main(parts: str = ""):
    """Dev: ship each AI mesh to its own Modal container, in parallel, print IoUs."""
    import glob
    import os
    import time

    names = [p.strip() for p in parts.split(",") if p.strip()] or \
        [f"AI{i}" for i in range(1, 16)]
    jobs, labels = [], []
    for p in names:
        hits = (glob.glob(f"models/{p}/*.3mf") + glob.glob(f"models/{p}/*.stl")
                + glob.glob(f"models/{p}/*.obj") + glob.glob(f"models/{p}/*.ply"))
        if not hits:
            print(f"{p}: no mesh under models/{p}/")
            continue
        with open(hits[0], "rb") as fh:
            jobs.append((fh.read(), os.path.basename(hits[0]), False))
            labels.append(p)

    print(f"dispatching {len(jobs)} parts to Modal (parallel, isolated)...")
    t0 = time.time()
    ious = []
    for label, r in zip(labels, build_mesh.starmap(jobs)):
        if r.get("ok"):
            iou = r["iou"]
            ious.append(iou if iou is not None else float("nan"))
            iou_s = "n/a" if iou is None else f"{iou:.3f}"
            print(f"{label:<8} IoU={iou_s:<6} vol_err={r['vol_err']:>6.1f}% "
                  f"regions={r['n_regions']:>3} feats={r['n_features']:>3} "
                  f"({r['elapsed']:.0f}s)")
        else:
            print(f"{label:<8} ERR {r.get('error', '')[:80]}")
    good = [x for x in ious if x == x]
    if good:
        print(f"\nmean IoU = {sum(good) / len(good):.3f} over {len(good)} parts "
              f"| wall {time.time() - t0:.0f}s")


@app.function(image=render_image, cpu=2.0, memory=8192, timeout=1800,
              min_containers=min(_WARM, 1), scaledown_window=_SCALEDOWN)
def render_all(names: list) -> dict:
    """Cloud-orchestrated dev render. Runs ENTIRELY in the cloud: fans build_mesh
    out over all parts (sibling containers, fast INTERNAL Modal network — no held
    client stream, no heartbeat stalls), renders each mesh-vs-recon panel here, and
    returns one small bundle of PNG bytes. The client makes a single resilient
    spawn+get round trip — so a degraded client<->Modal link can't stall the sweep
    the way starmap did (442s..1186s with 67s heartbeat timeouts). Meshes are read
    from the image (baked in), so the client uploads nothing."""
    import glob
    import io
    import math
    import os

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    jobs, labels = [], []
    for p in names:
        hits = sorted(
            glob.glob(f"/root/models/{p}/*.3mf") + glob.glob(f"/root/models/{p}/*.stl")
            + glob.glob(f"/root/models/{p}/*.obj") + glob.glob(f"/root/models/{p}/*.ply"))
        if not hits:
            continue
        with open(hits[0], "rb") as fh:
            jobs.append((fh.read(), os.path.basename(hits[0]), True))
            labels.append(p)

    # Fan out CLOUD-SIDE: this starmap's stream is container<->Modal (internal,
    # reliable), not over the user's link. All parts build in parallel containers.
    # want_iou=False: skip the exact-IoU boolean (not shown in the montage) to cut
    # per-part time while keeping the mesh-vs-recon comparison.
    results = list(build_mesh.starmap(jobs, kwargs={"want_iou": False}))

    # Concurrency telemetry: peak overlap of the build intervals = how many
    # containers actually ran at once. If << len(jobs), we're queue/cap-bound, not
    # network-bound — the lever is the workspace container limit, not the client.
    ivs = sorted((r["t_start"], r["t_end"]) for r in results
                 if r and r.get("ok") and "t_start" in r)
    peak = 0
    ev = []
    for s, e in ivs:
        ev.append((s, 1)); ev.append((e, -1))
    cur = 0
    for _, d in sorted(ev):
        cur += d; peak = max(peak, cur)
    sum_build = sum(r["elapsed"] for r in results if r and r.get("ok"))
    span = (max(e for _, e in ivs) - min(s for s, _ in ivs)) if ivs else 0.0
    concurrency = {"peak_containers": peak, "n_parts": len(ivs),
                   "sum_build_s": round(sum_build, 1), "build_span_s": round(span, 1),
                   "effective_parallelism": round(sum_build / span, 1) if span else 0.0}
    print(f"[render_all] peak_containers={peak}/{len(ivs)} "
          f"sum_build={sum_build:.0f}s span={span:.0f}s "
          f"effective_parallelism={concurrency['effective_parallelism']}")

    def _deci(V, F, n=4000):
        if len(F) <= n:
            return V, F
        return V, F[::int(np.ceil(len(F) / n))]

    def _panel(ax, V, F, col, title):
        if V is None or F is None or len(F) == 0:
            ax.set_title(title + " (none)", fontsize=8); ax.set_axis_off(); return
        V, F = _deci(V, F)
        ax.add_collection3d(Poly3DCollection(V[F], alpha=1, facecolor=col,
                                             edgecolor="#33414f", linewidths=0.04))
        mn, mx = V.min(0), V.max(0)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
        try:
            ax.set_box_aspect(mx - mn)
        except Exception:
            pass
        ax.view_init(24, -58); ax.set_axis_off(); ax.set_title(title, fontsize=9)

    def _load_mesh(data, name):
        import trimesh
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else "stl"
        try:
            obj = trimesh.load(io.BytesIO(data), file_type=ext, process=False)
            if hasattr(obj, "geometry"):  # Scene (multi-body / sub-.model 3MF)
                obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
            return np.asarray(obj.vertices, float), np.asarray(obj.faces, np.int64)
        except Exception:
            return None, None

    parts_png, vol_errs = {}, {}
    for (data, name, _), p, r in zip(jobs, labels, results):
        fig = plt.figure(figsize=(7, 3.4))
        try:
            if not r or not r.get("ok"):
                fig.text(0.5, 0.5, f"{p} FAILED\n{(r or {}).get('error', '')[:60]}",
                         ha="center", color="red")
            else:
                Vr = np.asarray(r["tess_vertices"], float).reshape(-1, 3)
                Fr = np.asarray(r["tess_faces"], np.int64).reshape(-1, 3)
                Vm, Fm = _load_mesh(data, name)
                _panel(fig.add_subplot(1, 2, 1, projection="3d"), Vm, Fm,
                       "#9fb4c7", f"{p} MESH")
                _panel(fig.add_subplot(1, 2, 2, projection="3d"), Vr, Fr, "#7fa8d8",
                       f"RECON {r.get('vol_err', 0):.0f}% f={r.get('n_features', 0)}")
                vol_errs[p] = float(r.get("vol_err", 0))
        except Exception as exc:
            fig.text(0.5, 0.5, f"{p} render err\n{str(exc)[:50]}", ha="center", color="red")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=85); plt.close(fig)
        parts_png[p] = buf.getvalue()

    valid = [p for p in labels if p in parts_png]
    nc = 3
    nr = max(1, math.ceil(len(valid) / nc))
    fig = plt.figure(figsize=(nc * 7, nr * 3.6))
    for i, p in enumerate(valid):
        ax = fig.add_subplot(nr, nc, i + 1)
        ax.imshow(plt.imread(io.BytesIO(parts_png[p]))); ax.set_axis_off()
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    montage = buf.getvalue()

    return {"montage": montage, "parts": parts_png, "vol_errs": vol_errs,
            "concurrency": concurrency}


@app.local_entrypoint()
def render(parts: str = ""):
    """Dev: render the mesh-vs-recon montage for all parts. The build + render run
    cloud-side (render_all); this client just spawns it and polls for the single
    result bundle with retries — resilient to a flaky client<->Modal link. Writes
    tests/visual_reports/diagnosis/ALL_parts.png + per-part PNGs."""
    import os
    import time

    # Default = the 15 AI parts (the proven ~3-min sweep). Scans are excluded from
    # the fast baked image (220 MB, slowest builds) — request them explicitly with
    # --parts Scan1,Scan2,Scan3 only if needed (slower, not in the image).
    names = [p.strip() for p in parts.split(",") if p.strip()] or \
        [f"AI{i}" for i in range(1, 16)]
    out_dir = "tests/visual_reports/diagnosis/_parts"
    os.makedirs(out_dir, exist_ok=True)

    print(f"spawning cloud render of {len(names)} parts...")
    t0 = time.time()
    call = render_all.spawn(names)
    res = None
    while time.time() - t0 < 1500:
        try:
            res = call.get(timeout=20)
            break
        except Exception as exc:
            # TimeoutError (not done yet) OR a transient link hiccup — the work is
            # detached cloud-side, so just keep polling. One good 20s window wins.
            print(f"  ...waiting ({time.time() - t0:.0f}s) [{type(exc).__name__}]")
            time.sleep(2)
    if not res:
        print("render_all did not complete within 25 min"); return

    for p, b in res.get("parts", {}).items():
        with open(f"{out_dir}/{p}.png", "wb") as fh:
            fh.write(b)
    with open("tests/visual_reports/diagnosis/ALL_parts.png", "wb") as fh:
        fh.write(res["montage"])
    for p, v in sorted(res.get("vol_errs", {}).items()):
        print(f"{p}: vol_err={v:.1f}%")
    cc = res.get("concurrency", {})
    if cc:
        print(f"CONCURRENCY: {cc['peak_containers']}/{cc['n_parts']} containers ran "
              f"at once | sum_build={cc['sum_build_s']}s over span={cc['build_span_s']}s "
              f"(effective_parallelism={cc['effective_parallelism']})")
    print(f"wrote tests/visual_reports/diagnosis/ALL_parts.png | wall {time.time() - t0:.0f}s")
