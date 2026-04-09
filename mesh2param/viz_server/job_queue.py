"""Job queue for concurrent mesh-to-CAD pipeline runs.

Each upload creates a Job with a UUID. A background worker thread
processes jobs one at a time. Frontend polls by job ID.

Completed jobs are persisted to SQLite so they survive server restarts.
On startup, completed jobs are loaded from the DB back into memory.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue

import numpy as np

logger = logging.getLogger(__name__)

MAX_JOBS = 50  # max completed jobs to keep in memory


@dataclass
class Job:
    id: str
    filename: str
    file_path: str  # temp file path
    client_ip: str = ""  # for per-IP billing
    status: str = "queued"  # queued | processing | done | error
    progress: int = 0
    progress_stage: str = "Queued"
    error: str | None = None
    # Pipeline results
    features: object = None
    cleaned: object = None
    steps: list = field(default_factory=list)
    variants: list = field(default_factory=list)  # list of {name, strategy, steps}
    active_variant: int = 0  # index of currently selected variant
    mesh_volume: float = 0.0
    cad_volume: float = 0.0
    # Progress visualization — updated per stage for live rendering
    progress_vertices: list | None = None  # flat float list
    progress_faces: list | None = None     # flat int list
    progress_colors: list | None = None    # flat float list (per-face RGB)
    progress_colors_per_face: bool = False
    progress_edges: list | None = None     # flat float list (line segment pairs)
    progress_edge_colors: list | None = None  # flat float list (per-edge-vertex RGB)
    progress_version: int = 0  # incremented on each viz update
    bbox_min: list = field(default_factory=lambda: [0, 0, 0])
    bbox_max: list = field(default_factory=lambda: [0, 0, 0])
    token_cost: int = 0
    created_at: float = field(default_factory=time.time)


class JobQueue:
    """Thread-safe job queue with background worker."""

    def __init__(self):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: Queue[str] = Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._tokens_spent = 0
        self._on_complete = None  # callback: fn(job) called when job finishes
        self._db_loaded = False  # True after initial DB load

    def set_on_complete(self, fn):
        """Set a callback for when a job completes. fn(job) is called."""
        self._on_complete = fn

    @property
    def tokens_spent(self) -> int:
        return self._tokens_spent

    def submit(self, filename: str, file_path: str, client_ip: str = "", num_variants: int = 1) -> str:
        """Submit a new job. Returns job ID."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, filename=filename, file_path=file_path, client_ip=client_ip)
        job._num_variants = max(1, min(4, num_variants))
        logger.info("Job submitted: id=%s filename=%r ip=%s", job_id, filename, client_ip)

        with self._lock:
            self._jobs[job_id] = job
            # Evict oldest completed jobs if over limit
            if len(self._jobs) > MAX_JOBS:
                to_remove = []
                for jid, j in self._jobs.items():
                    if j.status in ("done", "error") and len(self._jobs) - len(to_remove) > MAX_JOBS:
                        to_remove.append(jid)
                for jid in to_remove:
                    del self._jobs[jid]
                if to_remove:
                    logger.debug("Evicted %d old jobs from memory", len(to_remove))

        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        # Try loading from DB if not in memory
        return self._load_single_from_db(job_id)

    def _worker_loop(self):
        """Process jobs one at a time."""
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._run_pipeline(job)
            except Exception as e:
                job.status = "error"
                job.error = str(e)
                logger.exception("Job %s failed", job_id)
            finally:
                # Clean up temp file
                Path(job.file_path).unlink(missing_ok=True)

    def _run_pipeline(self, job: Job):
        """Run the full mesh-to-CAD pipeline for a job."""
        import pickle

        from mesh2param.config import (
            FitConfig, IngestConfig, RecognizeConfig, SegmentConfig, TopologyConfig,
        )
        from mesh2param.pipeline.ingest import ingest_mesh
        from mesh2param.pipeline.segment import segment_mesh
        from mesh2param.pipeline.fit import fit_surfaces
        from mesh2param.pipeline.topology import build_topology
        from mesh2param.pipeline.recognize import recognize_features

        job.status = "processing"
        logger.info("Pipeline started for job %s (file=%r)", job.id, job.filename)

        # Helper: set mesh visualization on the job for live progress
        def _set_mesh_viz(cleaned_mesh, face_colors=None, edge_pts=None, edge_colors=None):
            """Update the job's progress geometry for the frontend to render.
            Pauses briefly so the frontend poll can pick up each stage."""
            job.progress_vertices = cleaned_mesh.vertices.flatten().tolist()
            job.progress_faces = cleaned_mesh.faces.flatten().tolist()
            if face_colors is not None:
                # Send per-FACE colors (num_faces, 3) — frontend builds
                # non-indexed geometry and assigns each face's 3 vertices
                # the same color for sharp per-face coloring.
                job.progress_colors = face_colors.flatten().tolist()
                job.progress_colors_per_face = True
            else:
                job.progress_colors = None
                job.progress_colors_per_face = False
            job.progress_edges = edge_pts.flatten().tolist() if edge_pts is not None else None
            job.progress_edge_colors = edge_colors.flatten().tolist() if edge_colors is not None else None
            job.progress_version += 1
            logger.info("Progress viz v%d: %d verts, %d faces, colors=%s, edges=%s", job.progress_version,
                        len(job.progress_vertices), len(job.progress_faces),
                        'yes' if job.progress_colors else 'no',
                        'yes' if job.progress_edges else 'no')
            time.sleep(0.8)  # pause so frontend poll catches this stage

        # Stage 1: Ingest
        job.progress = 5
        job.progress_stage = "Ingesting mesh..."
        logger.debug("Job %s: stage=ingest file=%s", job.id, job.file_path)
        cleaned = ingest_mesh(job.file_path, IngestConfig(remesh=False))
        _set_mesh_viz(cleaned)

        # Stage 2: Segment — color each region distinctly
        job.progress = 8
        job.progress_stage = "Segmenting surfaces..."
        logger.debug("Job %s: stage=segment faces=%d", job.id, cleaned.num_faces)
        regions = segment_mesh(cleaned, SegmentConfig())

        # Generate per-face colors from region IDs
        region_colors_map = {}
        palette = [
            [0.36, 0.67, 0.93], [0.93, 0.46, 0.36], [0.40, 0.85, 0.55],
            [0.90, 0.75, 0.30], [0.70, 0.45, 0.85], [0.95, 0.55, 0.65],
            [0.30, 0.80, 0.80], [0.85, 0.60, 0.40], [0.55, 0.70, 0.35],
            [0.65, 0.55, 0.90], [0.45, 0.65, 0.75], [0.80, 0.40, 0.50],
        ]
        face_colors_seg = np.full((len(cleaned.faces), 3), 0.6)
        for ri, region in enumerate(regions):
            color = palette[ri % len(palette)]
            region_colors_map[region.region_id] = color
            for fi in region.face_indices:
                if fi < len(face_colors_seg):
                    face_colors_seg[fi] = color
        _set_mesh_viz(cleaned, face_colors=face_colors_seg)

        # Stage 3: Fit — color by surface type
        job.progress = 12
        job.progress_stage = "Fitting primitives..."
        logger.debug("Job %s: stage=fit regions=%d", job.id, len(regions))
        fitted = fit_surfaces(regions, cleaned, FitConfig())

        type_colors = {
            'PLANE': [0.36, 0.56, 0.93],     # blue
            'CYLINDER': [0.30, 0.78, 0.40],   # green
            'CONE': [0.93, 0.65, 0.30],       # orange
            'SPHERE': [0.93, 0.35, 0.35],     # red
        }
        face_colors_fit = np.full((len(cleaned.faces), 3), 0.6)
        fitted_lookup = {f.region_id: f for f in fitted}
        for region in regions:
            f = fitted_lookup.get(region.region_id)
            if f:
                color = type_colors.get(f.surface_type.name, [0.6, 0.6, 0.6])
                for fi in region.face_indices:
                    if fi < len(face_colors_fit):
                        face_colors_fit[fi] = color
        _set_mesh_viz(cleaned, face_colors=face_colors_fit)

        # Stage 4: Topology — show edges colored by classification
        job.progress = 18
        job.progress_stage = "Building topology..."
        logger.debug("Job %s: stage=topology fitted=%d", job.id, len(fitted))
        graph, edges_list = build_topology(regions, fitted, cleaned, TopologyConfig())

        edge_type_colors = {
            'SHARP': [1.0, 1.0, 1.0],      # white
            'FILLET': [0.35, 0.65, 1.0],    # blue
            'CHAMFER': [1.0, 0.85, 0.25],   # yellow
            'SMOOTH': [0.5, 0.5, 0.5],      # gray
        }
        edge_pts_list = []
        edge_col_list = []
        region_lookup = {r.region_id: r for r in regions}
        for edge in edges_list:
            ra = region_lookup.get(edge.surface_a_id)
            rb = region_lookup.get(edge.surface_b_id)
            if ra is None or rb is None:
                continue
            # Find shared boundary vertices between the two regions
            set_a = set(cleaned.faces[ra.face_indices].flatten()) if len(ra.face_indices) > 0 else set()
            set_b = set(cleaned.faces[rb.face_indices].flatten()) if len(rb.face_indices) > 0 else set()
            shared = set_a & set_b
            if len(shared) < 2:
                continue
            shared_verts = cleaned.vertices[list(shared)]
            # Draw a line from min to max along the longest axis
            extent = shared_verts.max(axis=0) - shared_verts.min(axis=0)
            ax = int(np.argmax(extent))
            sorted_idx = np.argsort(shared_verts[:, ax])
            p0 = shared_verts[sorted_idx[0]]
            p1 = shared_verts[sorted_idx[-1]]
            color = edge_type_colors.get(edge.transition_type.name, [0.5, 0.5, 0.5])
            edge_pts_list.extend([p0, p1])
            edge_col_list.extend([color, color])

        edge_pts = np.array(edge_pts_list) if edge_pts_list else None
        edge_cols = np.array(edge_col_list) if edge_col_list else None
        _set_mesh_viz(cleaned, face_colors=face_colors_fit, edge_pts=edge_pts, edge_colors=edge_cols)

        # Stage 5: Recognize features
        job.progress = 22
        job.progress_stage = "Recognizing features..."
        logger.debug("Job %s: stage=recognize edges=%d", job.id, len(edges_list))
        tree = recognize_features(regions, fitted, graph, edges_list, cleaned, RecognizeConfig())

        job.features = tree
        job.cleaned = cleaned
        job.mesh_volume = self._estimate_volume(cleaned)
        job.bbox_min = cleaned.bbox[0].tolist()
        job.bbox_max = cleaned.bbox[1].tolist()

        # Stage 5 visualization: color mesh by recognized feature type
        from mesh2param.models.feature import FeatureType
        feature_type_colors = {
            FeatureType.BASE_EXTRUSION: [0.35, 0.55, 0.90],  # blue
            FeatureType.THROUGH_HOLE:   [0.30, 0.80, 0.40],  # green
            FeatureType.BLIND_HOLE:     [0.25, 0.70, 0.50],  # teal
            FeatureType.CHAMFER:        [1.00, 0.85, 0.25],  # yellow
            FeatureType.FILLET:         [0.95, 0.55, 0.25],  # orange
            FeatureType.POCKET:         [0.70, 0.35, 0.80],  # purple
            FeatureType.BOSS:           [0.85, 0.45, 0.45],  # red
        }
        face_colors_feat = np.full((len(cleaned.faces), 3), 0.55)  # gray default
        for feat in tree.features:
            color = feature_type_colors.get(feat.type, [0.55, 0.55, 0.55])
            try:
                fi_arr = feat.face_indices
                if fi_arr is not None and hasattr(fi_arr, '__len__') and len(fi_arr) > 0:
                    for fi in fi_arr:
                        if int(fi) < len(face_colors_feat):
                            face_colors_feat[int(fi)] = color
            except Exception:
                pass
        _set_mesh_viz(cleaned, face_colors=face_colors_feat, edge_pts=edge_pts, edge_colors=edge_cols)

        # Keep feature colors but remove edge lines for the build stage.
        # The colored mesh shows what CadQuery is constructing.
        job.progress_edges = None
        job.progress_edge_colors = None
        job.progress = 30
        job.progress_stage = "Building CAD model..."
        time.sleep(0.8)
        logger.debug("Job %s: stage=cadquery features=%d", job.id, len(tree.features))

        # CadQuery in subprocess
        try:
            tmp_in = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            pickle.dump((tree, cleaned), tmp_in)
            tmp_in.close()
        except Exception:
            logger.exception("Job %s: failed to serialize pipeline data for CadQuery subprocess", job.id)
            raise

        tmp_out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp_out.close()

        num_variants = getattr(job, '_num_variants', 1)
        script = f"""
import pickle, sys
sys.path.insert(0, '.')
with open(r'{tmp_in.name}', 'rb') as f:
    tree, cleaned = pickle.load(f)
from mesh2param.pipeline.export_cadquery import export_cadquery
result = export_cadquery(tree, cleaned, use_llm=True, num_variants={num_variants})
result.solid = None
with open(r'{tmp_out.name}', 'wb') as f:
    pickle.dump(result, f)
"""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=600,  # 10 min for 4 variants
            )
        except subprocess.TimeoutExpired:
            logger.error("Job %s: CadQuery subprocess timed out after 180s", job.id)
            Path(tmp_in.name).unlink(missing_ok=True)
            Path(tmp_out.name).unlink(missing_ok=True)
            raise RuntimeError("CadQuery subprocess timed out after 180s")
        except Exception:
            logger.exception("Job %s: CadQuery subprocess failed to launch", job.id)
            Path(tmp_in.name).unlink(missing_ok=True)
            Path(tmp_out.name).unlink(missing_ok=True)
            raise
        Path(tmp_in.name).unlink(missing_ok=True)

        if proc.returncode != 0:
            logger.error("Job %s: CadQuery subprocess failed (exit %d): %s",
                         job.id, proc.returncode, proc.stderr[-300:])
            Path(tmp_out.name).unlink(missing_ok=True)
            raise RuntimeError(f"CadQuery failed: {proc.stderr[-300:]}")

        try:
            with open(tmp_out.name, "rb") as f:
                result = pickle.load(f)
        except Exception:
            logger.exception("Job %s: failed to deserialize CadQuery output", job.id)
            raise
        finally:
            Path(tmp_out.name).unlink(missing_ok=True)

        job.steps = result.construction_steps
        job.cad_volume = result.cad_volume
        # Store variants
        job.variants = []
        for v in getattr(result, 'variants', []):
            job.variants.append({
                'name': v.name,
                'strategy': v.strategy,
                'steps': v.construction_steps,
                'cad_volume': v.cad_volume,
            })
        if job.variants:
            job.steps = job.variants[0]['steps']  # default to first variant
        job.progress = 100
        job.progress_stage = "Done"

        # Compute token cost — tuned so 2×Mesh1 + 1×Mesh2 ≈ 100 free tokens
        from mesh2param.models.feature import FeatureType
        cost = max(1, cleaned.num_faces // 150)  # base: face complexity
        for feat in tree.features:
            if feat.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
                cost += 2
            elif feat.type in (FeatureType.CHAMFER, FeatureType.FILLET):
                cost += 2
            elif feat.type in (FeatureType.POCKET, FeatureType.BOSS, FeatureType.SLOT):
                cost += 3
        # Round to nearest 5
        cost = max(5, round(cost / 5) * 5)
        job.token_cost = cost
        self._tokens_spent += cost

        job.status = "done"
        logger.info("Pipeline completed for job %s: %d steps, token_cost=%d",
                     job.id, len(job.steps), job.token_cost)

        # Persist to SQLite
        self._save_to_db(job)

        # Notify callback (for per-IP billing)
        if self._on_complete:
            try:
                self._on_complete(job)
            except Exception:
                logger.exception("Job %s: on_complete callback failed", job.id)

    # ── Database persistence ──

    def load_completed_from_db(self):
        """Load completed jobs from SQLite into memory on startup."""
        if self._db_loaded:
            return
        self._db_loaded = True
        try:
            from mesh2param.db.database import SessionLocal, MeshJob as DBMeshJob, JobStep as DBJobStep
            from mesh2param.pipeline.export_cadquery import ConstructionStep
            db = SessionLocal()
            try:
                db_jobs = (
                    db.query(DBMeshJob)
                    .filter(DBMeshJob.status == "done", DBMeshJob.job_uuid.isnot(None))
                    .order_by(DBMeshJob.created_at.desc())
                    .limit(MAX_JOBS)
                    .all()
                )
                loaded = 0
                for db_job in reversed(db_jobs):  # oldest first so OrderedDict order is correct
                    job_id = db_job.job_uuid
                    with self._lock:
                        if job_id in self._jobs:
                            continue

                    job = Job(
                        id=job_id,
                        filename=db_job.filename or "",
                        file_path="",
                        status="done",
                        progress=100,
                        progress_stage="Done",
                        mesh_volume=db_job.mesh_volume or 0.0,
                        cad_volume=db_job.cad_volume or 0.0,
                        token_cost=db_job.token_cost or 0,
                        created_at=db_job.created_at.timestamp() if db_job.created_at else time.time(),
                    )

                    # Restore bbox
                    try:
                        job.bbox_min = json.loads(db_job.bbox_min_json) if db_job.bbox_min_json else [0, 0, 0]
                        job.bbox_max = json.loads(db_job.bbox_max_json) if db_job.bbox_max_json else [0, 0, 0]
                    except Exception:
                        logger.debug("Failed to parse bbox JSON for job %s, using defaults", job_id)

                    # Restore construction steps
                    db_steps = (
                        db.query(DBJobStep)
                        .filter(DBJobStep.job_id == db_job.id)
                        .order_by(DBJobStep.step_index)
                        .all()
                    )
                    steps = []
                    for db_step in db_steps:
                        tess_verts = None
                        tess_faces = None
                        if db_step.vertices_blob and db_step.vertices_shape:
                            try:
                                shape = tuple(int(x) for x in db_step.vertices_shape.split(","))
                                tess_verts = np.frombuffer(db_step.vertices_blob, dtype=np.float64).reshape(shape)
                            except Exception:
                                logger.debug("Failed to deserialize vertices for job %s step %s",
                                             job_id, db_step.step_index)
                        if db_step.faces_blob and db_step.faces_shape:
                            try:
                                shape = tuple(int(x) for x in db_step.faces_shape.split(","))
                                tess_faces = np.frombuffer(db_step.faces_blob, dtype=np.int64).reshape(shape)
                            except Exception:
                                logger.debug("Failed to deserialize faces for job %s step %s",
                                             job_id, db_step.step_index)
                        overlay = None
                        if getattr(db_step, "sketch_overlay_blob", None) and getattr(db_step, "sketch_overlay_shape", None):
                            try:
                                shape = tuple(int(x) for x in db_step.sketch_overlay_shape.split(","))
                                overlay = np.frombuffer(db_step.sketch_overlay_blob, dtype=np.float64).reshape(shape)
                            except Exception:
                                pass
                        steps.append(ConstructionStep(
                            name=db_step.name,
                            tess_vertices=tess_verts if tess_verts is not None else np.array([]),
                            tess_faces=tess_faces if tess_faces is not None else np.array([]),
                            volume=db_step.volume or 0.0,
                            sketch_overlay=overlay,
                        ))
                    job.steps = steps

                    # Restore variants from pickled blob
                    if getattr(db_job, 'variants_blob', None):
                        try:
                            import pickle as _pkl
                            job.variants = _pkl.loads(db_job.variants_blob)
                            if job.variants:
                                job.steps = job.variants[0]['steps']
                        except Exception:
                            logger.debug("Failed to restore variants for job %s", job_id)

                    with self._lock:
                        self._jobs[job_id] = job
                    self._tokens_spent += job.token_cost
                    loaded += 1

                if loaded:
                    logger.info("Loaded %d completed jobs from database", loaded)
            finally:
                db.close()
        except Exception:
            logger.exception("Failed to load jobs from database")

    def _save_to_db(self, job: Job):
        """Persist a completed job to SQLite."""
        try:
            from mesh2param.db.database import SessionLocal, MeshJob as DBMeshJob, JobStep as DBJobStep
            db = SessionLocal()
            try:
                # Check if already saved
                existing = db.query(DBMeshJob).filter(DBMeshJob.job_uuid == job.id).first()
                if existing:
                    return

                num_features = 0
                features_summary = None
                if job.features and hasattr(job.features, "features"):
                    num_features = len(job.features.features)
                    try:
                        features_summary = json.dumps([
                            {"type": f.type.value if hasattr(f.type, "value") else str(f.type),
                             "name": getattr(f, "name", ""),
                             "params": {k: v for k, v in (getattr(f, "params", {}) or {}).items()
                                        if isinstance(v, (int, float, str, bool))}}
                            for f in job.features.features
                        ])
                    except Exception:
                        pass

                # Pickle variants for DB persistence
                variants_blob = None
                if job.variants:
                    try:
                        import pickle as _pkl
                        variants_blob = _pkl.dumps(job.variants)
                    except Exception:
                        logger.warning("Failed to pickle variants for job %s", job.id)

                db_job = DBMeshJob(
                    user_id=1,  # default user; TODO: pass actual user from request
                    job_uuid=job.id,
                    filename=job.filename,
                    status=job.status,
                    error=job.error,
                    mesh_volume=job.mesh_volume,
                    cad_volume=job.cad_volume,
                    num_features=num_features,
                    num_steps=len(job.steps),
                    token_cost=job.token_cost,
                    ai_tokens_used=job.token_cost,
                    bbox_min_json=json.dumps(job.bbox_min),
                    bbox_max_json=json.dumps(job.bbox_max),
                    features_json=features_summary,
                    variants_blob=variants_blob,
                    completed_at=datetime.utcnow(),
                )
                db.add(db_job)
                db.flush()  # get db_job.id

                for i, step in enumerate(job.steps):
                    verts_blob = None
                    verts_shape = None
                    faces_blob = None
                    faces_shape = None
                    if step.tess_vertices is not None and len(step.tess_vertices) > 0:
                        arr = np.asarray(step.tess_vertices, dtype=np.float64)
                        verts_blob = arr.tobytes()
                        verts_shape = ",".join(str(s) for s in arr.shape)
                    if step.tess_faces is not None and len(step.tess_faces) > 0:
                        arr = np.asarray(step.tess_faces, dtype=np.int64)
                        faces_blob = arr.tobytes()
                        faces_shape = ",".join(str(s) for s in arr.shape)

                    overlay_blob = None
                    overlay_shape = None
                    if getattr(step, "sketch_overlay", None) is not None and len(step.sketch_overlay) > 0:
                        arr = np.asarray(step.sketch_overlay, dtype=np.float64)
                        overlay_blob = arr.tobytes()
                        overlay_shape = ",".join(str(s) for s in arr.shape)

                    db_step = DBJobStep(
                        job_id=db_job.id,
                        step_index=i,
                        name=step.name,
                        volume=step.volume,
                        is_sketch="sketch" in step.name.lower(),
                        vertices_blob=verts_blob,
                        vertices_shape=verts_shape,
                        faces_blob=faces_blob,
                        faces_shape=faces_shape,
                        sketch_overlay_blob=overlay_blob,
                        sketch_overlay_shape=overlay_shape,
                    )
                    db.add(db_step)

                db.commit()
                logger.info("Saved job %s to database (%d steps)", job.id, len(job.steps))
            finally:
                db.close()
        except Exception:
            logger.exception("Failed to save job %s to database", job.id)

    def _load_single_from_db(self, job_id: str) -> Job | None:
        """Try to load a single job from the DB (cache miss)."""
        try:
            from mesh2param.db.database import SessionLocal, MeshJob as DBMeshJob, JobStep as DBJobStep
            from mesh2param.pipeline.export_cadquery import ConstructionStep
            db = SessionLocal()
            try:
                db_job = db.query(DBMeshJob).filter(
                    DBMeshJob.job_uuid == job_id, DBMeshJob.status == "done"
                ).first()
                if db_job is None:
                    return None

                job = Job(
                    id=job_id,
                    filename=db_job.filename or "",
                    file_path="",
                    status="done",
                    progress=100,
                    progress_stage="Done",
                    mesh_volume=db_job.mesh_volume or 0.0,
                    cad_volume=db_job.cad_volume or 0.0,
                    token_cost=db_job.token_cost or 0,
                    created_at=db_job.created_at.timestamp() if db_job.created_at else time.time(),
                )
                try:
                    job.bbox_min = json.loads(db_job.bbox_min_json) if db_job.bbox_min_json else [0, 0, 0]
                    job.bbox_max = json.loads(db_job.bbox_max_json) if db_job.bbox_max_json else [0, 0, 0]
                except Exception:
                    logger.debug("Failed to parse bbox JSON for single-load job %s", job_id)

                db_steps = (
                    db.query(DBJobStep)
                    .filter(DBJobStep.job_id == db_job.id)
                    .order_by(DBJobStep.step_index)
                    .all()
                )
                steps = []
                for db_step in db_steps:
                    tess_verts = None
                    tess_faces = None
                    if db_step.vertices_blob and db_step.vertices_shape:
                        try:
                            shape = tuple(int(x) for x in db_step.vertices_shape.split(","))
                            tess_verts = np.frombuffer(db_step.vertices_blob, dtype=np.float64).reshape(shape)
                        except Exception:
                            logger.debug("Failed to deserialize vertices for job %s step %s (single-load)",
                                         job_id, db_step.step_index)
                    if db_step.faces_blob and db_step.faces_shape:
                        try:
                            shape = tuple(int(x) for x in db_step.faces_shape.split(","))
                            tess_faces = np.frombuffer(db_step.faces_blob, dtype=np.int64).reshape(shape)
                        except Exception:
                            logger.debug("Failed to deserialize faces for job %s step %s (single-load)",
                                         job_id, db_step.step_index)
                    steps.append(ConstructionStep(
                        name=db_step.name,
                        tess_vertices=tess_verts if tess_verts is not None else np.array([]),
                        tess_faces=tess_faces if tess_faces is not None else np.array([]),
                        volume=db_step.volume or 0.0,
                    ))
                job.steps = steps

                # Cache in memory
                with self._lock:
                    self._jobs[job_id] = job
                return job
            finally:
                db.close()
        except Exception:
            logger.exception("Failed to load job %s from database", job_id)
            return None

    @staticmethod
    def _estimate_volume(cleaned) -> float:
        import trimesh
        try:
            mesh = trimesh.Trimesh(
                vertices=cleaned.vertices, faces=cleaned.faces, process=False,
            )
            if mesh.is_watertight:
                return abs(float(mesh.volume))
            return abs(float(mesh.convex_hull.volume))
        except Exception:
            logger.exception("Failed to estimate mesh volume")
            return 0.0


# Singleton
queue = JobQueue()
