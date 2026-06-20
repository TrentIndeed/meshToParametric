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
import modal

# Mirror the validated lean pipeline stack from _docker/Dockerfile: the apt libs
# pymeshlab's alpha-wrap plugin needs + the pip pins proven to work in the
# container. (No server/pyvista — this is just the pipeline.)
image = (
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
    .add_local_python_source("mesh2param")   # the pipeline package (live source)
)

app = modal.App("mesh2param-pipeline", image=image)


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


@app.function(cpu=4.0, memory=8192, timeout=900)
def build_mesh(mesh_bytes: bytes, filename: str, want_geometry: bool = False,
               stream_key: str | None = None) -> dict:
    """Full mesh->CAD pipeline on ONE mesh, in an isolated container. The unit of
    parallelism for both dev validation and production jobs. Returns metrics; with
    want_geometry=True also returns the tessellated solid + STEP for the viewer.
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
        from mesh2param.config import (
            FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig)
        from mesh2param.pipeline.ingest import ingest_mesh
        from mesh2param.pipeline.segment import segment_mesh
        from mesh2param.pipeline.fit import fit_surfaces
        from mesh2param.pipeline.topology import build_topology
        from mesh2param.pipeline.recognize import recognize_features
        from mesh2param.pipeline.export_cadquery import export_cadquery

        c = ingest_mesh(path, IngestConfig(remesh=False, scan_auto=True))
        dirty = bool((c.metadata or {}).get("scan_mode"))
        seg = SegmentConfig(noise_robust=dirty)
        regions = segment_mesh(c, seg)
        fitted = fit_surfaces(regions, c, FitConfig())
        if dirty:
            from mesh2param.pipeline.regularize import regularize_mesh
            regularize_mesh(c, regions, fitted, snap_angle_deg=12.0)
            regions = segment_mesh(c, seg)
            fitted = fit_surfaces(regions, c, FitConfig())
        graph, edges = build_topology(regions, fitted, c, TopologyConfig())
        tree = recognize_features(regions, fitted, graph, edges, c, RecognizeConfig())
        # When the server wants the full job (want_geometry), also write a STEP so
        # we can ship its bytes back for edge-picking + download (editor parity).
        if want_geometry:
            fd, step_out = tempfile.mkstemp(suffix=".step")
            os.close(fd)
        res = export_cadquery(tree, c, output_path=step_out, use_llm=False,
                              num_variants=1, stream_sink=step_sink)

        iou = None
        try:
            import trimesh
            from mesh2param.pipeline.iou_refine import exact_iou_available, mesh_iou_exact
            if exact_iou_available() and res.tess_vertices is not None:
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
            "n_regions": len(regions), "n_features": tree.num_features,
            "elapsed": round(time.time() - t0, 1),
        }
        if want_geometry:
            import pickle
            out["tess_vertices"] = np.asarray(res.tess_vertices, np.float32).reshape(-1).tolist()
            out["tess_faces"] = np.asarray(res.tess_faces, np.int32).reshape(-1).tolist()
            out["cad_volume"] = float(res.cad_volume)
            # Editor parity: the durable rebuild bundle (same tuple the local
            # worker pickles) so /api/job/{id}/edit can re-run the build, plus the
            # STEP bytes for edge-picking + download, plus the source mesh volume.
            # The cloud flat pipeline has no decomposition, so body_results /
            # decomposition are None — rebuild_from_bundle tolerates that.
            try:
                out["bundle_bytes"] = pickle.dumps((tree, c, None, None),
                                                   protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                out["bundle_bytes"] = None
            if step_out and os.path.exists(step_out):
                try:
                    with open(step_out, "rb") as sfh:
                        out["step_bytes"] = sfh.read()
                except Exception:
                    pass
            try:
                import trimesh
                out["mesh_volume"] = abs(float(trimesh.Trimesh(
                    vertices=c.vertices, faces=c.faces, process=False).volume))
            except Exception:
                out["mesh_volume"] = 0.0
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
