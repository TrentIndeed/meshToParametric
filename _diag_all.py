"""Full diagnostic: per-body feature inventory for RackMount."""
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter

from mesh2param.config import DecomposeConfig, IngestConfig, SegmentConfig
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh


def analyze_body(sub, body):
    print(f"\n{'='*60}")
    print(f"Body {body.body_id}: {len(sub.faces)} faces, bbox={(sub.bbox[1]-sub.bbox[0]).round(1)}")
    print(f"{'='*60}")

    regs = segment_mesh(sub, SegmentConfig())
    verts = np.asarray(sub.vertices)
    normals = np.asarray(sub.normals)

    print(f"\n{len(regs)} regions:")
    for r in sorted(regs, key=lambda r: -r.num_faces):
        mean_n = normals[r.face_indices].mean(axis=0)
        mag = float(np.linalg.norm(mean_n))
        rv = verts[np.unique(sub.faces[r.face_indices].ravel())]
        bb = rv.max(axis=0) - rv.min(axis=0)
        kind = "PLATE" if mag > 0.95 else ("CYL" if mag < 0.3 else "curve")
        print(f"  rid {r.region_id:3d}  faces={r.num_faces:5d}  mag={mag:.2f}  {kind}  bbox={bb.round(1)}")

    # Find cylindrical holes by DBSCAN on radial face centroids per major axis
    centroids = np.array([verts[sub.faces[fi]].mean(axis=0) for fi in range(len(sub.faces))])
    for axis_idx, axis_name in enumerate(["X", "Y", "Z"]):
        radial = np.where(np.abs(normals[:, axis_idx]) < 0.3)[0]
        if len(radial) < 10:
            continue
        # Use perpendicular plane centroids for clustering
        perp_idx = [i for i in range(3) if i != axis_idx]
        db = DBSCAN(eps=3.0, min_samples=10).fit(centroids[radial][:, perp_idx])
        clusters = [cl for cl in set(db.labels_) if cl != -1]
        if not clusters:
            continue
        print(f"\n  Cylindrical features with axis~{axis_name} (perpendicular normals):")
        lbl_of = np.full(len(sub.faces), -1, dtype=int)
        for r in regs:
            lbl_of[r.face_indices] = r.region_id
        for cl in clusters:
            mask = db.labels_ == cl
            faces = radial[mask]
            cc = centroids[faces]
            bb = cc.max(axis=0) - cc.min(axis=0)
            # Diameter check: cylindrical features have bbox extent
            # similar in 2 axes (diameter) with 3rd being thickness
            if bb[perp_idx[0]] < 1.0 or bb[perp_idx[1]] < 1.0:
                continue
            region_dist = Counter(lbl_of[faces].tolist())
            dist_str = ", ".join(f"rid{lbl}:{n}" for lbl, n in region_dist.most_common(5))
            print(f"    cluster: {len(faces)} faces, bbox={bb.round(1)} yz-center=[{cc[:, perp_idx[0]].mean():.1f},{cc[:, perp_idx[1]].mean():.1f}] -> {dist_str}")


cleaned = ingest_mesh("models/Mesh5/RackMount.3mf", IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
for body in decomp.bodies:
    sub = extract_sub_mesh(cleaned, body.face_indices, body=body)
    analyze_body(sub, body)
