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
import pickle, sys, traceback
import numpy as np
IN, OUT = sys.argv[1], sys.argv[2]
try:
    with open(IN, 'rb') as f:
        bundle_path, edits, out_step = pickle.load(f)
    with open(bundle_path, 'rb') as f:
        tree, cleaned, body_results, decomposition = pickle.load(f)

    drop = set(edits.get('deletes', [])) | set(edits.get('suppress', []))
    if drop:
        tree.features = [ft for i, ft in enumerate(tree.features) if i not in drop]

    from mesh2param.pipeline.export_cadquery import (
        export_cadquery, _tessellate_solid, _solid_volume)
    import cadquery as cq
    multi = getattr(decomposition, 'is_multi_body', False)
    result = export_cadquery(tree, cleaned, use_llm=False, num_variants=1,
                             body_results=body_results if multi else None)
    solid = result.solid
    if solid is None:
        raise RuntimeError('export produced no solid')
    wp = solid if hasattr(solid, 'edges') else cq.Workplane(obj=solid)

    def _diag(shape):
        try:
            bb = shape.BoundingBox()
            return ((bb.xlen)**2 + (bb.ylen)**2 + (bb.zlen)**2) ** 0.5
        except Exception:
            return 1.0

    def _do(w, kind, rr):
        return w.fillet(rr) if kind == 'fillet' else w.chamfer(rr)

    def _finish_one_edge(wp, kind, r, mid):
        # Fillet/chamfer the single edge nearest `mid` (the edge the user picked
        # in the viewer). Matched on the CURRENT solid so it composes with
        # earlier ops. Tolerance scales with the part so picks always land.
        shape = wp.val()
        tol = max(0.5, 0.03 * _diag(shape))
        best, bestd = None, 1e18
        for e in shape.Edges():
            try:
                c = e.positionAt(0.5)
                d = ((c.x - mid[0])**2 + (c.y - mid[1])**2 + (c.z - mid[2])**2) ** 0.5
            except Exception:
                continue
            if d < bestd:
                bestd, best = d, e
        if best is None or bestd > tol:
            return wp, 'edge not found (model changed)', 0
        # A single-edge fillet that runs into a sharp corner fails (OCC
        # StdFail_NotDone); back the radius off before giving up.
        for factor in (1.0, 0.6, 0.35):
            rr = r * factor
            try:
                w2 = _do(cq.Workplane(obj=shape).newObject([best]), kind, rr)
                note = None if factor == 1.0 else ('radius reduced to %.3f' % rr)
                return w2, note, 1
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
                     'n_features': len(tree.features),
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
                        timeout: float = 240.0) -> dict:
    """Re-run the CAD build with an edited feature tree + finishing ops.

    edits = {
        'deletes':  [feature_index, ...],
        'suppress': [feature_index, ...],
        'finishes': [{'type': 'fillet'|'chamfer', 'value': mm,
                      'edge_mid': [x,y,z] | None}, ...],  # edge_mid -> one edge
    }
    Returns {'ok', 'verts', 'faces', 'cad_volume', 'applied', 'warnings'} or
    {'ok': False, 'error'}. Writes `out_step` if given.
    """
    if not bundle_path or not Path(bundle_path).exists():
        return {'ok': False, 'error': 'no saved build to edit (rebuild the model first)'}
    return _run_child(_CHILD_REBUILD, (bundle_path, edits, out_step), timeout)


def extract_edges(step_path: str, timeout: float = 60.0) -> dict:
    """Load a STEP and return its edges as polylines + midpoints for picking."""
    if not step_path or not Path(step_path).exists():
        return {'ok': False, 'error': 'no model STEP available yet'}
    return _run_child(_CHILD_EDGES, (step_path,), timeout)
