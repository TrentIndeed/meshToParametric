"""Per-feature segmentation invariants for Mesh5/RackMount.3mf.

A regression harness codifying what each body should look like after
segmentation, so a fix to one feature can be verified against every
other feature at the same time. Each test targets ONE specific
feature area and fails independently — the output tells you exactly
what regressed.

RackMount structure:
  Body 0 & Body 1: two side plates (mirror symmetric)
    - 2 large flat faces (front ±X, back ∓X), each ~600-700 faces
    - 4 rolled-edge perimeter fillets (top/bottom/left/right)
    - 1 main through-hole (~29mm diameter, axis ±X)
    - 2 smaller through-holes (~9mm diameter, axis ±X)
    - Rim fillets on every through-hole (one per face side)
  Body 2: bottom plate
    - 1 large flat base face
    - 4 tabs (corner extensions), each with:
      - Flat faces on top/bottom
      - 1 tab through-hole (~8mm diameter, axis ±Z)
      - Fillets where tab meets base
"""
from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from mesh2param.config import DecomposeConfig, IngestConfig, SegmentConfig
from mesh2param.pipeline.decompose import decompose_bodies, extract_sub_mesh
from mesh2param.pipeline.ingest import ingest_mesh
from mesh2param.pipeline.segment import segment_mesh


RACKMOUNT = "models/Mesh5/RackMount.3mf"


class BodyData(NamedTuple):
    body: object
    sub: object
    regions: list
    face_label: np.ndarray  # face index -> region_id


@pytest.fixture(scope="module")
def bodies() -> dict[str, BodyData]:
    """Load RackMount, decompose, segment — classify bodies by role."""
    cleaned = ingest_mesh(RACKMOUNT, IngestConfig(remesh=False))
    decomp = decompose_bodies(cleaned, DecomposeConfig())
    assert decomp.is_multi_body, "RackMount must decompose into multiple bodies"

    # Classify bodies by footprint: 2 side plates (long+narrow) + 1
    # bottom plate (wide+shallow).
    entries = []
    for b in decomp.bodies:
        sub = extract_sub_mesh(cleaned, b.face_indices, body=b)
        size = sub.bbox[1] - sub.bbox[0]
        entries.append((size, b, sub))
    # Side plates have similar shape and thickness along shortest axis;
    # bottom plate has largest 2D footprint along its shortest axis.
    entries.sort(key=lambda e: -sorted(e[0])[-1] * sorted(e[0])[-2])
    bottom = entries[0]  # largest 2D footprint
    side1 = entries[1]
    side2 = entries[2]

    out: dict[str, BodyData] = {}
    for label, (_, body, sub) in [("bottom", bottom), ("side1", side1), ("side2", side2)]:
        regs = segment_mesh(sub, SegmentConfig())
        face_label = np.full(len(sub.faces), -1, dtype=int)
        for r in regs:
            face_label[r.face_indices] = r.region_id
        out[label] = BodyData(body=body, sub=sub, regions=regs, face_label=face_label)
    return out


def _find_bore_features(sub, axis_idx: int, eps: float = 3.0,
                       min_samples: int = 10) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Find actual through-hole cylindrical features with given
    axis by clustering radial-normal face centroids in the
    perpendicular plane.

    A real through-hole's bore-wall faces cover the full 360°
    around the axis, so their mean normal has magnitude ~0. A
    rounded perimeter corner or partial arc of the plate's
    rolled edge gives a cluster that passes the shape tests
    (round, thin) but has mean-normal magnitude ≥ 0.5 because
    the arc covers only 90-180°. That mean_n check is what
    distinguishes real through-holes from plate corners.

    Returns list of (face_indices, centroid, bbox) per cylinder.
    """
    verts = np.asarray(sub.vertices)
    normals = np.asarray(sub.normals)
    centroids = np.array([
        verts[sub.faces[fi]].mean(axis=0) for fi in range(len(sub.faces))
    ])
    # Radial: normal perpendicular to axis
    radial = np.where(np.abs(normals[:, axis_idx]) < 0.25)[0]
    if len(radial) < min_samples:
        return []
    perp = [i for i in range(3) if i != axis_idx]
    proj = centroids[radial][:, perp]
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(proj)

    features = []
    for cl in set(db.labels_):
        if cl == -1:
            continue
        mask = db.labels_ == cl
        faces = radial[mask]
        cc = centroids[faces]
        bb = cc.max(axis=0) - cc.min(axis=0)
        # Real cylinder: perpendicular dimensions are similar (diameter)
        perp_bb = [bb[p] for p in perp]
        if min(perp_bb) < 1.0:
            continue  # too thin, not a cylinder
        # Reject partial arcs — a real 360° bore wall's mean
        # normal cancels to ~0 magnitude. A 90° or 180° rounded
        # corner gives ~0.6+. Threshold 0.35 cleanly separates
        # the two on this model.
        mean_mag = float(np.linalg.norm(normals[faces].mean(axis=0)))
        if mean_mag > 0.35:
            continue
        features.append((faces, cc.mean(axis=0), bb))
    return features


# --------------------------------------------------------------------
# Side plate tests
# --------------------------------------------------------------------

class TestSidePlateStructure:
    """Invariants for the two side plates (mirror symmetric)."""

    @pytest.mark.parametrize("which", ["side1", "side2"])
    def test_has_two_large_plate_regions(self, bodies, which):
        """Each side plate has 2 big planar regions (front + back face),
        each >400 faces with tight axis-aligned normal."""
        bd = bodies[which]
        normals = np.asarray(bd.sub.normals)
        large_plates = []
        for r in bd.regions:
            if r.num_faces < 400:
                continue
            mag = float(np.linalg.norm(normals[r.face_indices].mean(axis=0)))
            if mag >= 0.95:
                large_plates.append(r)
        assert len(large_plates) == 2, (
            f"{which}: expected 2 large axis-aligned plate regions "
            f"(front + back face), got {len(large_plates)}: "
            f"{[(r.region_id, r.num_faces) for r in large_plates]}"
        )

    @pytest.mark.parametrize("which", ["side1", "side2"])
    def test_main_through_hole_bore_is_one_region(self, bodies, which):
        """The main through-hole's cylinder wall should be in ONE
        region. At least 90% of its radial faces share the same
        label."""
        bd = bodies[which]
        # Main hole: biggest radial cluster along X axis
        features = _find_bore_features(bd.sub, axis_idx=0)
        assert features, f"{which}: no X-axis bore detected"
        features.sort(key=lambda f: -len(f[0]))
        main_faces, _center, bb = features[0]
        # Must be the big one: bbox > 20mm diameter
        assert max(bb) > 20, (
            f"{which}: biggest X-axis bore is only {bb} — no main hole?"
        )
        cnt = Counter(bd.face_label[main_faces].tolist())
        top_lbl, top_n = cnt.most_common(1)[0]
        frac = top_n / len(main_faces)
        assert frac >= 0.90, (
            f"{which}: main bore wall split across regions — "
            f"{len(cnt)} regions, top holds {frac:.0%} of faces. "
            f"Distribution: {dict(cnt.most_common(5))}"
        )

    @pytest.mark.parametrize("which", ["side1", "side2"])
    def test_one_main_through_hole_per_side_plate(self, bodies, which):
        """Each side plate has exactly ONE genuine through-hole
        (the ~29mm central hole). The 9mm features at the plate
        corners are ROUNDED PERIMETER CORNERS, not holes — their
        radial-normal clusters have partial-arc normals
        (mean_mag ≥ 0.5), which `_find_bore_features` filters out.
        """
        bd = bodies[which]
        features = _find_bore_features(bd.sub, axis_idx=0)
        assert len(features) == 1, (
            f"{which}: expected 1 genuine X-axis through-hole, "
            f"got {len(features)}: bboxes={[f[2].tolist() for f in features]}"
        )
        _, _, bb = features[0]
        assert max(bb) > 20, (
            f"{which}: the one detected through-hole is small "
            f"(bbox={bb}) — expected ~29mm main hole"
        )

    @pytest.mark.parametrize("which", ["side1", "side2"])
    def test_rim_fillets_exist_on_through_holes(self, bodies, which):
        """Each through-hole has rim fillets visible as separate
        regions (not swallowed into the bore wall region)."""
        bd = bodies[which]
        normals = np.asarray(bd.sub.normals)

        # For each X-axis through-hole, find faces near it with
        # normals that are TILTED toward ±X (not purely radial) —
        # those are rim-fillet faces.
        features = _find_bore_features(bd.sub, axis_idx=0)
        assert features, f"{which}: no through-holes detected"

        main_feature = max(features, key=lambda f: len(f[0]))
        main_faces_set = set(main_feature[0].tolist())
        _, center, bb = main_feature

        # Near-hole region in YZ, slightly wider than the bore
        verts = np.asarray(bd.sub.vertices)
        centroids = np.array([
            verts[bd.sub.faces[fi]].mean(axis=0)
            for fi in range(len(bd.sub.faces))
        ])
        dist_yz = np.linalg.norm(centroids[:, 1:] - center[1:], axis=1)
        near_mask = dist_yz < (max(bb[1], bb[2]) * 0.7)
        # Rim fillets: near hole AND normal tilted from radial
        # (|Nx| > 0.3 but not pure axial)
        rim_mask = near_mask & (np.abs(normals[:, 0]) > 0.3) & \
            (np.abs(normals[:, 0]) < 0.95)
        rim_faces = np.where(rim_mask)[0]
        # Exclude bore wall faces from rim
        rim_faces = [int(f) for f in rim_faces if f not in main_faces_set]
        rim_faces = np.array(rim_faces, dtype=int)
        assert len(rim_faces) >= 10, (
            f"{which}: main hole has no tilted-normal rim faces"
        )
        # Rim faces should belong to region(s) separate from the bore wall
        bore_label = Counter(
            bd.face_label[main_feature[0]].tolist()
        ).most_common(1)[0][0]
        rim_labels = set(bd.face_label[rim_faces].tolist())
        rim_distinct = rim_labels - {bore_label}
        assert len(rim_distinct) >= 1, (
            f"{which}: main hole's rim fillet is absorbed into the "
            f"bore wall region (label {bore_label}) — expected "
            f"at least one distinct rim region"
        )

    @pytest.mark.parametrize("which", ["side1", "side2"])
    def test_through_hole_rim_fillets_are_local(self, bodies, which):
        """Every through-hole's rim fillet should be in HOLE-LOCAL
        regions (bbox ≲ 3× hole diameter), NOT absorbed into a
        plate-spanning frame region. A rim fillet that ends up in
        a 145×75mm "rolled-edge perimeter" label paints with the
        same color as the entire plate perimeter — the user sees
        the rim as "connected to the outside flat plane" instead
        of as its own distinct highlight.
        """
        bd = bodies[which]
        normals = np.asarray(bd.sub.normals)
        verts = np.asarray(bd.sub.vertices)
        centroids = np.array([
            verts[bd.sub.faces[fi]].mean(axis=0)
            for fi in range(len(bd.sub.faces))
        ])

        features = _find_bore_features(bd.sub, axis_idx=0)
        assert features, f"{which}: no X-axis through-holes detected"

        offenders = []
        for i, (hole_faces, hole_center, hole_bb) in enumerate(features):
            hole_diam = max(hole_bb[1], hole_bb[2])
            max_local_dim = hole_diam * 3.0  # rim ~2-3× hole radius

            # Find the rim fillet faces for this hole (near the
            # hole, normals tilted ±30-80° from the bore axis).
            dyz = np.linalg.norm(
                centroids[:, 1:] - hole_center[1:], axis=1
            )
            near = dyz < hole_diam * 1.2
            rim_mask = near & \
                (np.abs(normals[:, 0]) > 0.3) & \
                (np.abs(normals[:, 0]) < 0.95)
            rim_faces = np.where(rim_mask)[0]
            if len(rim_faces) < 10:
                continue  # no rim fillet on this hole

            # Group rim faces by their assigned label. For each
            # label, check if it's plate-local (bbox ≲ 3× diameter).
            from collections import Counter as C
            rim_label_counts = C(bd.face_label[rim_faces].tolist())
            for lbl, n in rim_label_counts.items():
                if n < 20:
                    continue  # tiny leak, ignore
                lbl_faces = np.where(bd.face_label == lbl)[0]
                lbl_centroids = centroids[lbl_faces]
                lbl_bb = lbl_centroids.max(axis=0) - lbl_centroids.min(axis=0)
                lbl_max_dim = float(max(lbl_bb[1], lbl_bb[2]))
                if lbl_max_dim > max_local_dim:
                    offenders.append({
                        "hole_idx": i,
                        "hole_yz": hole_center[1:].round(1).tolist(),
                        "hole_diam": round(hole_diam, 1),
                        "label": int(lbl),
                        "rim_faces_in_label": int(n),
                        "label_bbox": lbl_bb.round(1).tolist(),
                    })
        assert not offenders, (
            f"{which}: {len(offenders)} rim fillets absorbed into "
            f"plate-spanning regions. Each rim fillet should be in "
            f"a hole-local region (bbox ≲ 3× hole diameter). "
            f"Offenders: {offenders}"
        )

    @pytest.mark.parametrize("which", ["side1", "side2"])
    def test_no_tiny_plate_fragments(self, bodies, which):
        """Large flat surfaces shouldn't fragment into many tiny
        axis-aligned regions. Count regions that are axis-aligned
        (mag≥0.95) AND small (<10 faces) — should be few."""
        bd = bodies[which]
        normals = np.asarray(bd.sub.normals)
        tiny_axis_regions = []
        for r in bd.regions:
            mag = float(np.linalg.norm(normals[r.face_indices].mean(axis=0)))
            if mag >= 0.95 and r.num_faces < 10:
                tiny_axis_regions.append((r.region_id, r.num_faces))
        assert len(tiny_axis_regions) <= 3, (
            f"{which}: {len(tiny_axis_regions)} tiny axis-aligned "
            f"fragments — plates are over-segmented. {tiny_axis_regions}"
        )


# --------------------------------------------------------------------
# Bottom plate tests
# --------------------------------------------------------------------

class TestBottomPlateStructure:
    """Invariants for the bottom plate with 4 tabs."""

    def test_total_region_count_reasonable(self, bodies):
        """Bottom plate has expected features:
          1 main base (top face) + 1 base bottom face
          4 tabs × 2 faces (top + underside) = 8 tab plate faces
          4 tabs × 1 through-hole bore wall = 4 bore walls
          4 tabs × 2 rim fillets per hole = 8 rims
          Some perimeter fillets + tab-to-body fillets
        Estimated sensible range: 15-35 regions. More than ~40
        suggests over-segmentation.
        """
        bd = bodies["bottom"]
        n_regions = len(bd.regions)
        assert 10 <= n_regions <= 60, (
            f"bottom plate has {n_regions} regions — outside sensible "
            f"range 10-60. Likely over- or under-segmented."
        )

    def test_has_large_base_face(self, bodies):
        """The bottom plate's base has a large flat top face
        (one big region, hundreds of faces)."""
        bd = bodies["bottom"]
        normals = np.asarray(bd.sub.normals)
        large_flats = []
        for r in bd.regions:
            mag = float(np.linalg.norm(normals[r.face_indices].mean(axis=0)))
            if mag >= 0.95 and r.num_faces >= 50:
                large_flats.append((r.region_id, r.num_faces))
        assert len(large_flats) >= 1, (
            f"bottom plate has no large flat region (>=50 faces, "
            f"mag>=0.95). All big regions fragmented? "
            f"Region sizes: {[r.num_faces for r in bd.regions[:10]]}"
        )

    def test_tab_bore_walls_each_one_region(self, bodies):
        """Each of the 4 tab through-holes (axis Z) has its
        cylinder wall in ONE region (>= 80% of its radial faces
        share a label).

        Tab bores are round (X≈Y bbox dim) and have a small Z
        extent (only the tab thickness, ~2-3mm). Other
        cylindrical features (curved tab edges, fillets) have
        non-round bboxes or extend further along Z, which
        excludes them from this test.
        """
        bd = bodies["bottom"]
        features = _find_bore_features(bd.sub, axis_idx=2)
        # Filter to round bores: bbox X≈Y (within 20%) AND
        # round-ness >5mm AND Z extent <4mm (just tab thickness)
        tab_bores = []
        for f in features:
            faces, center, bb = f
            x, y, z = bb
            if z >= 4.0:
                continue
            if not (5 <= x <= 12 and 5 <= y <= 12):
                continue
            if abs(x - y) / max(x, y) > 0.2:
                continue
            tab_bores.append(f)
        assert len(tab_bores) == 4, (
            f"expected 4 tab through-holes (Z axis, round, ~8mm), "
            f"got {len(tab_bores)} bbox sizes: "
            f"{[f[2].tolist() for f in tab_bores]}"
        )
        for i, (faces, _center, bb) in enumerate(tab_bores):
            cnt = Counter(bd.face_label[faces].tolist())
            top_n = cnt.most_common(1)[0][1]
            frac = top_n / len(faces)
            assert frac >= 0.80, (
                f"tab {i} (bbox={bb}) bore wall split across "
                f"{len(cnt)} regions. Top holds {frac:.0%}"
            )

    def test_no_excessive_tiny_fragments(self, bodies):
        """Bottom plate shouldn't have dozens of single-digit-face
        fragments. Some small fragments are OK but not the majority."""
        bd = bodies["bottom"]
        tiny = [r for r in bd.regions if r.num_faces < 5]
        total = len(bd.regions)
        frac_tiny = len(tiny) / max(total, 1)
        assert frac_tiny < 0.33, (
            f"bottom plate has {len(tiny)}/{total} ({frac_tiny:.0%}) "
            f"regions with <5 faces — too much fragmentation"
        )

    def test_tab_flat_faces_not_fragmented(self, bodies):
        """Each tab has flat top + bottom + maybe sides. Count the
        "tab-local" PLATE regions per tab (within 15mm of the bore
        center). Should be at most 3-4, NOT dozens of tiny flats.
        """
        bd = bodies["bottom"]
        normals = np.asarray(bd.sub.normals)
        verts = np.asarray(bd.sub.vertices)
        centroids = np.array([
            verts[bd.sub.faces[fi]].mean(axis=0)
            for fi in range(len(bd.sub.faces))
        ])

        features = _find_bore_features(bd.sub, axis_idx=2)
        tab_bores = [
            f for f in features
            if (f[2][2] < 4.0
                and 5 <= f[2][0] <= 12 and 5 <= f[2][1] <= 12
                and abs(f[2][0] - f[2][1]) / max(f[2][0], f[2][1]) <= 0.2)
        ]
        assert len(tab_bores) == 4, "need 4 tab bores to locate tabs"

        offenders = []
        for i, (_bore_faces, tab_center, _bb) in enumerate(tab_bores):
            dxy = np.linalg.norm(centroids[:, :2] - tab_center[:2], axis=1)
            near = dxy < 12.0
            near_labels = set(bd.face_label[near].tolist())
            plate_near_labels = []
            for lbl in near_labels:
                lbl_faces = np.where(bd.face_label == lbl)[0]
                lbl_mean_n = normals[lbl_faces].mean(axis=0)
                if float(np.linalg.norm(lbl_mean_n)) < 0.95:
                    continue  # not a plate region
                # Faces of this label that are near the tab
                n_near = int(np.isin(lbl_faces, np.where(near)[0]).sum())
                if n_near < 1:
                    continue
                plate_near_labels.append((lbl, n_near, len(lbl_faces)))
            if len(plate_near_labels) > 5:
                offenders.append((
                    i, tab_center[:2].round(1).tolist(),
                    len(plate_near_labels),
                    sorted(plate_near_labels, key=lambda x: -x[1])[:8],
                ))
        assert not offenders, (
            f"{len(offenders)} tabs have >5 plate-like regions "
            f"within 12mm of the bore center — flat faces "
            f"over-segmented. Offenders: {offenders}"
        )


# --------------------------------------------------------------------
# Cross-body sanity
# --------------------------------------------------------------------

class TestCrossBody:
    def test_decomposition(self, bodies):
        """Body count is 3 (2 side plates + 1 bottom plate)."""
        assert set(bodies.keys()) == {"side1", "side2", "bottom"}

    def test_side_plates_are_symmetric(self, bodies):
        """Side plates should have similar region counts (mirror
        symmetric bodies)."""
        n1 = len(bodies["side1"].regions)
        n2 = len(bodies["side2"].regions)
        assert abs(n1 - n2) <= 3, (
            f"side plates have very different region counts: "
            f"side1={n1}, side2={n2} — segmentation not symmetric"
        )

    def test_side_plates_have_matching_curved_region_count(self, bodies):
        """Both side plates should have the same number of CURVED
        regions (mean-normal magnitude < 0.95) within ±1. A
        structural asymmetry in curved features (e.g. one plate
        has its rim fillet as a separate region, the other has
        it absorbed into the bore) would show up here.
        """
        def curved_count(bd):
            normals = np.asarray(bd.sub.normals)
            n = 0
            for r in bd.regions:
                mag = float(np.linalg.norm(
                    normals[r.face_indices].mean(axis=0)
                ))
                if mag < 0.95:
                    n += 1
            return n
        c1 = curved_count(bodies["side1"])
        c2 = curved_count(bodies["side2"])
        assert abs(c1 - c2) <= 1, (
            f"side plates have different curved-region counts: "
            f"side1={c1}, side2={c2}. A fillet on one side is "
            f"separated differently than on the other — "
            f"segmentation is not symmetric."
        )
