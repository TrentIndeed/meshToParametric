"""Trace why Stage 5 (recognize) doesn't catch the corner bevels."""
from collections import defaultdict
from mesh2param.config import (
    DecomposeConfig, IngestConfig, SegmentConfig, FitConfig, TopologyConfig, RecognizeConfig,
)
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh
from mesh2param.pipeline.fit import fit_surfaces
from mesh2param.pipeline.topology import build_topology
from mesh2param.pipeline.recognize import recognize_features
from mesh2param.models.region import SurfaceType
from mesh2param.models.feature import FeatureType
import numpy as np

cleaned = ingest_mesh('models/Mesh5/RackMount.3mf', IngestConfig(remesh=False))
decomp = decompose_bodies(cleaned, DecomposeConfig())
entries = []
for b in decomp.bodies:
    sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
    sz = sub.bbox[1] - sub.bbox[0]
    entries.append((sz, b, sub))
entries.sort(key=lambda e: -sorted(e[0])[-1] * sorted(e[0])[-2])
for sz, body, sub in entries:
    if sub.bbox[0][0] < -40 and sub.bbox[1][0] < 0:
        regs = segment_mesh(sub, SegmentConfig())
        fits = fit_surfaces(regs, sub, FitConfig())
        graph, edges = build_topology(regs, fits, sub, TopologyConfig())
        from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
        topo_blends_pre = identify_blends_from_topology(regs, fits, edges, sub)
        tree = recognize_features(regs, fits, graph, edges, sub, RecognizeConfig(),
                                  topo_blends=topo_blends_pre)
        print(f'Body 0: {len(tree.features)} features total')
        n_fillet = sum(1 for f in tree.features if f.type == FeatureType.FILLET)
        n_other = sum(1 for f in tree.features if f.type != FeatureType.FILLET)
        print(f'  {n_fillet} fillets, {n_other} other')

        # For each blend rid in Stage 4.5, check if it landed in Stage 5
        from mesh2param.pipeline.peel_via_topology import identify_blends_from_topology
        blends = identify_blends_from_topology(regs, fits, edges, sub)
        blend_rids = sorted([r.get('region_id') for r in blends])
        print(f'\nStage 4.5 blends: {blend_rids}')
        for brid in blend_rids:
            r = next((rr for rr in regs if rr.region_id==brid), None)
            if not r: continue
            target_faces = set(int(fi) for fi in r.face_indices)
            features_with = []
            for f in tree.features:
                feat_faces = set(int(fi) for fi in f.face_indices)
                overlap = target_faces & feat_faces
                if overlap:
                    features_with.append(f"{f.type.name}({len(overlap)}/{len(target_faces)}f)")
            in_feature = ",".join(features_with) if features_with else None
            verts = np.asarray(sub.vertices)
            rverts = verts[np.unique(sub.faces[r.face_indices].ravel())]
            bb = rverts.max(axis=0) - rverts.min(axis=0)
            fit = next((f for f in fits if f.region_id == brid), None)
            ftype = fit.surface_type.value if fit else "?"
            mark = "OK " if in_feature and "FILLET" in in_feature else "MISS"
            print(f'  {mark} rid {brid:3d} ({r.num_faces:4d}f, {ftype:8s}) bbox={bb.round(1)} -> {in_feature}')

        for target in (12, 13, 15, 16):
            r = next((rr for rr in regs if rr.region_id==target), None)
            if not r: continue
            target_faces = set(int(fi) for fi in r.face_indices)
            print(f'\nrid {target} ({r.num_faces}f, {r.surface_type.value}):')
            in_feature = False
            for f in tree.features:
                feat_faces = set(int(fi) for fi in f.face_indices)
                if target_faces & feat_faces:
                    print(f'  -> in {f.type.name} feature ({len(f.face_indices)} faces)')
                    in_feature = True
            if not in_feature:
                print(f'  -> NOT in any feature')

            tangent_n = 0
            if graph.has_node(target):
                for nbr in graph.neighbors(target):
                    edata = graph.get_edge_data(target, nbr) or {}
                    tt = str(edata.get('transition_type', '')).upper()
                    if tt in ('SMOOTH', 'FILLET'):
                        tangent_n += 1
            print(f'  tangent neighbors: {tangent_n}')

            face_norms = np.asarray(sub.normals)[r.face_indices]
            n_unit = face_norms / np.linalg.norm(face_norms, axis=1)[:, None]
            M = n_unit.T @ n_unit
            ev, _ = np.linalg.eigh(M)
            lam = np.sort(np.abs(ev))
            print(f'  eigenvalues sorted: {lam.round(3)}')
            print(f'  lam[0]/lam[2]={lam[0]/max(lam[2],1e-12):.3f} (need <=0.2)')
            print(f'  lam[1]/lam[2]={lam[1]/max(lam[2],1e-12):.3f} (need <=0.85)')
        break
