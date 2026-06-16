import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
txt = (ROOT/"models"/"Mesh5"/"RackMountStep.step").read_text(errors="ignore")
# STEP CIRCLE: #n=CIRCLE('',#axis,radius);
rads = [float(m) for m in re.findall(r"CIRCLE\('[^']*',#\d+,([0-9.]+)\)", txt)]
import collections
c = collections.Counter(round(r,2) for r in rads)
print("CIRCLE radii (rounded 2dp) -> count, sorted by radius:")
for r in sorted(c):
    print(f"  R={r:7.2f}  x{c[r]}")
print("total circles:", len(rads))
