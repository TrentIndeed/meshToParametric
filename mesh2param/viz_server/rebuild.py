"""Interactive feature-tree rebuild + edge extraction (StudioCAD-style editing).

After a build the web UI lets the user edit the recognized feature tree — delete
features and add finishing features (fillet/chamfer), either globally or on a
SPECIFIC edge picked in the 3D viewer — then rebuild. CadQuery must run in a
SUBPROCESS (OCC/VTK isolation), so:

* `rebuild_from_bundle(bundle, edits, out_step)` loads the durable build bundle
  (`(tree, cleaned, body_results, decomposition)`), drops deleted features,
  re-runs export_cadquery, applies the finishing ops (in order; each may target
  one edge by its midpoint), writes the result STEP, and returns tessellation.
* `extract_edges(step_path)` loads a STEP and returns its edges as polylines +
  midpoints so the frontend can hover/highlight/pick them.

Both are standalone (no Job/server state) so they can be tested in the container.
"""
from __future__ import annotations

import logging
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


# ── child: rebuild from an edited tree ───────────────────────────────────────
_CHILD_REBUILD = r"""
import pickle, sys, traceback, os
import numpy as np
from collections import defaultdict
IN, OUT = sys.argv[1], sys.argv[2]
try:
    with open(IN, 'rb') as f:
        bundle_path, edits, out_step, base_step, base_valid = pickle.load(f)
    import cadquery as cq
    from mesh2param.pipeline.export_cadquery import _tessellate_solid, _solid_volume

    # FAST PATH: a finish-only edit (deletes unchanged) reuses the cached base
    # solid STEP — skips the ~20-40s export_cadquery re-run, just re-imports and
    # re-applies the finishing ops. The slow export only happens when the set of
    # deleted features changes (then the base is rebuilt + re-cached).
    n_features = None
    if base_valid and base_step and os.path.exists(base_step):
        wp = cq.importers.importStep(base_step)
    else:
        with open(bundle_path, 'rb') as f:
            tree, cleaned, body_results, decomposition = pickle.load(f)
        drop = set(edits.get('deletes', [])) | set(edits.get('suppress', []))
        if drop:
            tree.features = [ft for i, ft in enumerate(tree.features) if i not in drop]
        n_features = len(tree.features)
        from mesh2param.pipeline.export_cadquery import export_cadquery
        multi = getattr(decomposition, 'is_multi_body', False)
        result = export_cadquery(tree, cleaned, use_llm=False, num_variants=1,
                                 body_results=body_results if multi else None)
        if result.solid is None:
            raise RuntimeError('export produced no solid')
        wp = result.solid if hasattr(result.solid, 'edges') else cq.Workplane(obj=result.solid)
        if base_step:
            try:
                cq.exporters.export(wp, base_step)   # cache base for fast finishes
            except Exception:
                traceback.print_exc(file=sys.stdout)

    def _diag(shape):
        try:
            bb = shape.BoundingBox()
            return ((bb.xlen)**2 + (bb.ylen)**2 + (bb.zlen)**2) ** 0.5
        except Exception:
            return 1.0

    def _do(w, kind, rr):
        return w.fillet(rr) if kind == 'fillet' else w.chamfer(rr)

    def _vkey(p):
        return (round(p.x, 3), round(p.y, 3), round(p.z, 3))

    def _utan(e, t):
        try:
            v = e.tangentAt(t)
            L = (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5 or 1.0
            return (v.x/L, v.y/L, v.z/L)
        except Exception:
            return None

    def _tangent_chain(shape, mid, dot_tol=0.985):
        # Grow a chain of TANGENT-CONTINUOUS edges out from the edge nearest
        # `mid`. Filleting a smooth chain together is what CAD does (and what OCC
        # accepts) — it rounds a whole rounded-slot loop or a run of collinear
        # segments at once, where a lone edge fails. Sharp 90° corners are NOT
        # tangent, so a plate edge stays a singleton (handled by backoff).
        edges = list(shape.Edges())
        best, bd = None, 1e18
        for e in edges:
            try:
                c = e.positionAt(0.5)
                d = (c.x-mid[0])**2 + (c.y-mid[1])**2 + (c.z-mid[2])**2
            except Exception:
                continue
            if d < bd:
                bd, best = d, e
        if best is None:
            return [], 1e18
        vmap = defaultdict(list)
        for e in edges:
            try:
                for t in (0.0, 1.0):
                    vmap[_vkey(e.positionAt(t))].append(e)
            except Exception:
                pass
        chain, seen, stack = [best], {id(best)}, [best]
        while stack:
            e = stack.pop()
            for t in (0.0, 1.0):
                try:
                    vp = e.positionAt(t)
                except Exception:
                    continue
                te = _utan(e, t)
                if te is None:
                    continue
                for nb in vmap.get(_vkey(vp), ()):
                    if id(nb) in seen:
                        continue
                    for s in (0.0, 1.0):
                        try:
                            if _vkey(nb.positionAt(s)) != _vkey(vp):
                                continue
                        except Exception:
                            continue
                        tn = _utan(nb, s)
                        if tn is None:
                            break
                        if abs(te[0]*tn[0] + te[1]*tn[1] + te[2]*tn[2]) > dot_tol:
                            seen.add(id(nb)); chain.append(nb); stack.append(nb)
                        break
        return chain, bd ** 0.5

    def _finish_one_edge(wp, kind, r, mid):
        # Apply to the picked edge's TANGENT CHAIN (falls back to the single edge
        # if the chain won't take it). Radius backs off on OCC StdFail at corners.
        shape = wp.val()
        tol = max(0.5, 0.05 * _diag(shape))
        chain, dist = _tangent_chain(shape, mid)
        if not chain or dist > tol:
            return wp, 'edge not found (model changed)', 0
        for sel, label in ((chain, len(chain)), ([chain[0]], 1)):
            for factor in (1.0, 0.6, 0.35):
                rr = r * factor
                try:
                    w2 = _do(cq.Workplane(obj=shape).newObject(sel), kind, rr)
                    notes = []
                    if factor != 1.0:
                        notes.append('radius reduced to %.3f' % rr)
                    if label > 1:
                        notes.append('tangent chain of %d edges' % label)
                    return w2, ('; '.join(notes) or None), label
                except Exception:
                    continue
        return wp, 'this edge cannot take a %s here' % kind, 0

    def _finish_global(wp, kind, r):
        # No edge picked: round every edge long enough for the radius. OCC is
        # all-or-nothing on a multi-edge fillet, so fall back to per-edge.
        shape = wp.val()
        cand = [e for e in shape.Edges() if e.Length() > 2.2 * r]
        if not cand:
            return wp, 'no edges long enough for %.3f' % r, 0
        try:
            return _do(cq.Workplane(obj=shape).newObject(cand), kind, r), None, len(cand)
        except Exception:
            pass
        cur = shape
        targets = [(e, e.positionAt(0.5)) for e in cand]
        n = 0
        for _e, p0 in targets:
            tgt, td = None, 1e18
            for ec in cur.Edges():
                try:
                    c = ec.positionAt(0.5)
                    d = ((c.x-p0.x)**2 + (c.y-p0.y)**2 + (c.z-p0.z)**2) ** 0.5
                except Exception:
                    continue
                if d < td:
                    td, tgt = d, ec
            if tgt is None or td > 1e-4:
                continue
            try:
                cur = _do(cq.Workplane(obj=cur).newObject([tgt]), kind, r).val()
                n += 1
            except Exception:
                continue
        if n == 0:
            return wp, 'could not apply (radius too large)', 0
        note = None if n == len(targets) else ('applied to %d of %d edges' % (n, len(targets)))
        return cq.Workplane(obj=cur), note, n

    applied, warnings = [], []
    for fin in edits.get('finishes', []):
        kind = fin.get('type')
        if kind not in ('fillet', 'chamfer'):
            continue
        r = float(fin.get('value', 0))
        if r <= 0:
            continue
        mid = fin.get('edge_mid')
        if mid:
            wp, note, n = _finish_one_edge(wp, kind, r, mid)
        else:
            wp, note, n = _finish_global(wp, kind, r)
        if n > 0:
            applied.append(fin)
        if note:
            warnings.append('%s r=%.3f: %s' % (kind, r, note))

    if out_step:
        try:
            cq.exporters.export(wp, out_step)
        except Exception:
            traceback.print_exc(file=sys.stdout)

    v, fc = _tessellate_solid(wp)
    if v is None or fc is None:
        raise RuntimeError('tessellation failed')
    with open(OUT, 'wb') as f:
        pickle.dump({'ok': True,
                     'verts': np.asarray(v, dtype=np.float64),
                     'faces': np.asarray(fc, dtype=np.int64),
                     'cad_volume': float(_solid_volume(wp)),
                     'n_features': n_features,
                     'applied': applied, 'warnings': warnings}, f)
except BaseException as exc:
    with open(OUT, 'wb') as f:
        pickle.dump({'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc),
                     'trace': traceback.format_exc()}, f)
    sys.exit(1)
"""


# ── child: extract pickable edges from a STEP ────────────────────────────────
_CHILD_EDGES = r"""
import pickle, sys, traceback
IN, OUT = sys.argv[1], sys.argv[2]
try:
    with open(IN, 'rb') as f:
        step_path, = pickle.load(f)
    import cadquery as cq
    shape = cq.importers.importStep(step_path)
    val = shape.val()
    bb = val.BoundingBox()
    diag = ((bb.xlen)**2 + (bb.ylen)**2 + (bb.zlen)**2) ** 0.5 or 1.0
    edges = []
    for i, e in enumerate(val.Edges()):
        try:
            L = float(e.Length())
            if L < 1e-6:
                continue
            n = max(2, min(28, int(L / (0.015 * diag)) + 2))
            pts = []
            for k in range(n):
                p = e.positionAt(k / (n - 1))
                pts.append([round(p.x, 4), round(p.y, 4), round(p.z, 4)])
            m = e.positionAt(0.5)
            edges.append({'id': i, 'mid': [round(m.x, 4), round(m.y, 4), round(m.z, 4)],
                          'pts': pts, 'length': round(L, 3),
                          'kind': e.geomType()})
        except Exception:
            continue
    with open(OUT, 'wb') as f:
        pickle.dump({'ok': True, 'edges': edges,
                     'bbox_min': [bb.xmin, bb.ymin, bb.zmin],
                     'bbox_max': [bb.xmax, bb.ymax, bb.zmax]}, f)
except BaseException as exc:
    with open(OUT, 'wb') as f:
        pickle.dump({'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc),
                     'trace': traceback.format_exc()}, f)
    sys.exit(1)
"""


def _run_child(script: str, payload, timeout: float):
    in_f = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    out_f = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    in_f.close(); out_f.close()
    try:
        with open(in_f.name, 'wb') as f:
            pickle.dump(payload, f)
        proc = subprocess.run(
            [sys.executable, '-c', script, in_f.name, out_f.name],
            capture_output=True, text=True, timeout=timeout, cwd='.')
        try:
            with open(out_f.name, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {'ok': False, 'error': 'subprocess failed',
                    'detail': (proc.stderr or proc.stdout or '')[-800:]}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'timed out'}
    finally:
        for p in (in_f.name, out_f.name):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def rebuild_from_bundle(bundle_path: str, edits: dict, out_step: str | None = None,
                        base_step: str | None = None, base_valid: bool = False,
                        timeout: float = 240.0) -> dict:
    """Re-run the CAD build with an edited feature tree + finishing ops.

    edits = {
        'deletes':  [feature_index, ...],
        'suppress': [feature_index, ...],
        'finishes': [{'type': 'fillet'|'chamfer', 'value': mm,
                      'edge_mid': [x,y,z] | None}, ...],  # edge_mid -> one edge
    }
    base_step + base_valid enable the FAST PATH: when base_valid (the delete set
    is unchanged) and base_step exists, the cached base solid is re-imported
    instead of re-running export_cadquery — finishing edits then take ~seconds
    instead of ~minutes. When base_valid is False the base is rebuilt and
    written to base_step for next time.

    Returns {'ok', 'verts', 'faces', 'cad_volume', 'applied', 'warnings'} or
    {'ok': False, 'error'}. Writes `out_step` if given.
    """
    if not bundle_path or not Path(bundle_path).exists():
        return {'ok': False, 'error': 'no saved build to edit (rebuild the model first)'}
    return _run_child(_CHILD_REBUILD,
                      (bundle_path, edits, out_step, base_step, base_valid), timeout)


def extract_edges(step_path: str, timeout: float = 60.0) -> dict:
    """Load a STEP and return its edges as polylines + midpoints for picking."""
    if not step_path or not Path(step_path).exists():
        return {'ok': False, 'error': 'no model STEP available yet'}
    return _run_child(_CHILD_EDGES, (step_path,), timeout)
