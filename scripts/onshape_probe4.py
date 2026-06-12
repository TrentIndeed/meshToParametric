"""Probe round 4: sign machinery.
a. cPlane with NEGATIVE offset expression -> allowed?
b. extrude "oppositeDirection" boolean -> flips direction?
c. Front-plane (Y axis) full case: plane at world y=+12, extrude 8mm
   toward +Y -> bbox y 12..20?
d. THROUGH_ALL cut from a cPlane above a plate, oppositeDirection=True
   -> cuts downward through the part?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onshape_iterate import (  # noqa: E402
    BASE, DID, EID, WID, add, delete_feature,
    plane_q_deterministic, sketch,
)
from onshape_probe2 import bbox, circle, cplane, rect  # noqa: E402
from onshape_probe3 import (  # noqa: E402
    extrude2, sketch_on_query_string, volume,
)
from onshape_scratch import _session  # noqa: E402


def extrude3(name, sketch_fid, op="NEW", end_bound="BLIND",
             depth_mm=None, filter_inner=False, opposite=False):
    f = extrude2(name, sketch_fid, op, end_bound, depth_mm, filter_inner)
    f["parameters"].append({
        "btType": "BTMParameterBoolean-144",
        "parameterId": "oppositeDirection",
        "value": opposite,
    })
    return f


def main() -> int:
    s = _session()

    # --- a+b+c: Y-axis profile at +12, extruded +Y by 8 ---
    # Front normal is -Y, so world y=+12 means offset 12 with
    # oppositeDirection=True (negative expressions violate the
    # feature spec), and +Y extrusion means oppositeDirection=True.
    cp, st = add(s, cplane("y plane", "JCC", 12.0, opposite=True))
    sk, st2 = add(s, sketch_on_query_string("y sketch", cp,
                                            rect(0.040, 0.030)))
    ex, st3 = add(s, extrude3("y ex", sk, depth_mm=8.0, opposite=True))
    print(f"neg offset cPlane {st}, sketch {st2}, opposite extrude {st3}, "
          f"bbox={bbox(s)} (expect y 12..20)")
    for fid in (ex, sk, cp):
        if fid:
            delete_feature(s, fid)

    # --- d: plate on Top, hole sketch on cPlane at z=+20, cut DOWN ---
    sk1, _ = add(s, sketch("plate", plane_q_deterministic("JDC"),
                           rect(0.040, 0.030)))
    ex1, _ = add(s, extrude3("plate ex", sk1, depth_mm=5.0))
    cp2, _ = add(s, cplane("above", "JDC", 20.0))
    sk2, st = add(s, sketch_on_query_string(
        "hole above", cp2, [circle("h1", 0.020, 0.015, 0.005)],
    ))
    ex2, st2 = add(s, extrude3("cut down", sk2, op="REMOVE",
                               end_bound="THROUGH_ALL", opposite=True))
    print(f"down-cut: sketch {st}, remove {st2}, volume={volume(s)} "
          f"(expect ~5607)")
    for fid in (ex2, sk2, cp2, ex1, sk1):
        if fid:
            delete_feature(s, fid)
    print("cleaned up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
