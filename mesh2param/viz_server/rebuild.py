"""Interactive feature-tree rebuild (StudioCAD-style editing).

The web UI lets the user edit the recognized feature tree after a build —
delete/suppress features and add finishing features (fillet/chamfer) from a top
bar — then rebuild. CadQuery must run in a SUBPROCESS (OCC/VTK isolation, same
as the pipeline build), so this loads the durable bundle the build persisted
(`data/cad_bundle_{id}.pkl` = (tree, cleaned, body_results, decomposition)),
mutates the tree in the child, re-runs export_cadquery, applies the finishing
ops to the resulting solid, tessellates, and hands the new geometry back.

`rebuild_from_bundle(bundle_path, edits)` is deliberately standalone (no Job /
server state) so it can be unit-tested directly in the container.
"""
from __future__ import annotations

import logging
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Child script: load bundle, drop deleted/suppressed features, export, apply the
# finishing ops to result.solid, tessellate. Runs in its own interpreter so the
# heavy OCC import never touches the server process.
_CHILD = r"""
import pickle, sys, traceback
import numpy as np
IN, OUT = sys.argv[1], sys.argv[2]
try:
    with open(IN, 'rb') as f:
        bundle_path, edits = pickle.load(f)
    with open(bundle_path, 'rb') as f:
        tree, cleaned, body_results, decomposition = pickle.load(f)

    drop = set(edits.get('deletes', [])) | set(edits.get('suppress', []))
    if drop:
        tree.features = [ft for i, ft in enumerate(tree.features) if i not in drop]

    from mesh2param.pipeline.export_cadquery import (
        export_cadquery, _tessellate_solid, _solid_volume)
    multi = getattr(decomposition, 'is_multi_body', False)
    result = export_cadquery(tree, cleaned, use_llm=False, num_variants=1,
                             body_results=body_results if multi else None)
    solid = result.solid
    if solid is None:
        raise RuntimeError('export produced no solid')

    import cadquery as cq
    wp = solid if hasattr(solid, 'edges') else cq.Workplane(obj=solid)

    def _do(w, kind, rr):
        return w.fillet(rr) if kind == 'fillet' else w.chamfer(rr)

    def _apply_finish(wp, kind, r):
        # Filleting/chamfering EVERY edge at once is all-or-nothing in OCC and
        # usually fails on real parts (a too-short edge, or several filleted
        # edges meeting at one vertex -> StdFail_NotDone). Strategy: only touch
        # edges long enough to take the radius; try them all at once (fast path
        # for simple parts), and if the kernel refuses, fall back to per-edge,
        # skipping the ones it can't do. Returns (wp, note, n_applied).
        shape = wp.val()
        cand = [e for e in shape.Edges() if e.Length() > 2.2 * r]
        if not cand:
            return wp, 'no edges long enough for radius %.3f' % r, 0
        try:                                    # fast path: all at once
            w2 = _do(cq.Workplane(obj=shape).newObject(cand), kind, r)
            return w2, None, len(cand)
        except Exception:
            pass
        # Robust path: fillet edges one at a time on the accumulating shape,
        # re-locating each original edge by its midpoint (topology shifts as we
        # go; edges adjacent to an applied one vanish and are simply skipped).
        cur = shape
        targets = [(e, e.Center()) for e in cand]
        n = 0
        for _e, p0 in targets:
            tgt = None
            for ec in cur.Edges():
                try:
                    if ec.Center().sub(p0).Length < 1e-5:
                        tgt = ec
                        break
                except Exception:
                    pass
            if tgt is None:
                continue
            try:
                cur = _do(cq.Workplane(obj=cur).newObject([tgt]), kind, r).val()
                n += 1
            except Exception:
                continue
        if n == 0:
            return wp, 'could not apply (radius too large for the edges)', 0
        note = None if n == len(targets) else ('applied to %d of %d edges' % (n, len(targets)))
        return cq.Workplane(obj=cur), note, n

    applied, warnings = [], []
    for fin in edits.get('finishes', []):
        r = float(fin['value'])
        kind = fin['type']
        if kind not in ('fillet', 'chamfer'):
            continue
        wp, note, n = _apply_finish(wp, kind, r)
        if n > 0:
            applied.append(fin)
            if note:
                warnings.append('%s r=%.3f: %s' % (kind, r, note))
        else:
            warnings.append('%s r=%.3f: %s' % (kind, r, note))

    v, fc = _tessellate_solid(wp)
    if v is None or fc is None:
        raise RuntimeError('tessellation failed')
    out = {'ok': True,
           'verts': np.asarray(v, dtype=np.float64),
           'faces': np.asarray(fc, dtype=np.int64),
           'cad_volume': float(_solid_volume(wp)),
           'n_features': len(tree.features),
           'applied': applied, 'warnings': warnings}
    with open(OUT, 'wb') as f:
        pickle.dump(out, f)
except BaseException as exc:
    with open(OUT, 'wb') as f:
        pickle.dump({'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc),
                     'trace': traceback.format_exc()}, f)
    sys.exit(1)
"""


def rebuild_from_bundle(bundle_path: str, edits: dict, timeout: float = 240.0) -> dict:
    """Re-run the CAD build with an edited feature tree + finishing ops.

    edits = {
        'deletes':  [feature_index, ...],   # remove these recognized features
        'suppress': [feature_index, ...],   # same effect, kept distinct for UI
        'finishes': [{'type': 'fillet'|'chamfer', 'value': radius_mm}, ...],
    }
    Returns {'ok': True, 'verts': (V,3), 'faces': (F,3), 'cad_volume', ...} or
    {'ok': False, 'error': str}.
    """
    if not bundle_path or not Path(bundle_path).exists():
        return {'ok': False, 'error': 'no saved build to edit (rebuild the model first)'}

    in_f = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    out_f = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    in_f.close(); out_f.close()
    try:
        with open(in_f.name, 'wb') as f:
            pickle.dump((bundle_path, edits), f)
        proc = subprocess.run(
            [sys.executable, '-c', _CHILD, in_f.name, out_f.name],
            capture_output=True, text=True, timeout=timeout, cwd='.')
        try:
            with open(out_f.name, 'rb') as f:
                res = pickle.load(f)
        except Exception:
            return {'ok': False,
                    'error': 'rebuild subprocess failed',
                    'detail': (proc.stderr or proc.stdout or '')[-800:]}
        if not res.get('ok'):
            logger.warning("rebuild failed: %s", res.get('error'))
        return res
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'rebuild timed out'}
    finally:
        for p in (in_f.name, out_f.name):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
