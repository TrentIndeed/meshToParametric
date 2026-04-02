"""Job queue for concurrent mesh-to-CAD pipeline runs.

Each upload creates a Job with a UUID. A background worker thread
processes jobs one at a time. Frontend polls by job ID.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

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
    mesh_volume: float = 0.0
    cad_volume: float = 0.0
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

    def set_on_complete(self, fn):
        """Set a callback for when a job completes. fn(job) is called."""
        self._on_complete = fn

    @property
    def tokens_spent(self) -> int:
        return self._tokens_spent

    def submit(self, filename: str, file_path: str, client_ip: str = "") -> str:
        """Submit a new job. Returns job ID."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, filename=filename, file_path=file_path, client_ip=client_ip)

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

        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

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

        job.progress = 10
        job.progress_stage = "Ingesting mesh..."
        cleaned = ingest_mesh(job.file_path, IngestConfig(remesh=False))

        job.progress = 25
        job.progress_stage = "Segmenting surfaces..."
        regions = segment_mesh(cleaned, SegmentConfig())

        job.progress = 40
        job.progress_stage = "Fitting primitives..."
        fitted = fit_surfaces(regions, cleaned, FitConfig())

        job.progress = 55
        job.progress_stage = "Building topology..."
        graph, edges = build_topology(regions, fitted, cleaned, TopologyConfig())

        job.progress = 65
        job.progress_stage = "Recognizing features..."
        tree = recognize_features(regions, fitted, graph, edges, cleaned, RecognizeConfig())

        job.features = tree
        job.cleaned = cleaned
        job.mesh_volume = self._estimate_volume(cleaned)
        job.bbox_min = cleaned.bbox[0].tolist()
        job.bbox_max = cleaned.bbox[1].tolist()

        job.progress = 75
        job.progress_stage = "Building CAD model..."

        # CadQuery in subprocess
        tmp_in = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        pickle.dump((tree, cleaned), tmp_in)
        tmp_in.close()

        tmp_out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp_out.close()

        script = f"""
import pickle, sys
sys.path.insert(0, '.')
with open(r'{tmp_in.name}', 'rb') as f:
    tree, cleaned = pickle.load(f)
from mesh2param.pipeline.export_cadquery import export_cadquery
result = export_cadquery(tree, cleaned, use_llm=False)
result.solid = None
with open(r'{tmp_out.name}', 'wb') as f:
    pickle.dump(result, f)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=180,
        )
        Path(tmp_in.name).unlink(missing_ok=True)

        if proc.returncode != 0:
            raise RuntimeError(f"CadQuery failed: {proc.stderr[-300:]}")

        with open(tmp_out.name, "rb") as f:
            result = pickle.load(f)
        Path(tmp_out.name).unlink(missing_ok=True)

        job.steps = result.construction_steps
        job.cad_volume = result.cad_volume
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

        # Notify callback (for per-IP billing)
        if self._on_complete:
            try:
                self._on_complete(job)
            except Exception:
                pass

    @staticmethod
    def _estimate_volume(cleaned) -> float:
        import trimesh
        mesh = trimesh.Trimesh(
            vertices=cleaned.vertices, faces=cleaned.faces, process=False,
        )
        if mesh.is_watertight:
            return abs(float(mesh.volume))
        return abs(float(mesh.convex_hull.volume))


# Singleton
queue = JobQueue()
