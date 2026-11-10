"""
Job management.

A job is one upload. Files live in ``<storage>/jobs/<job_id>/``:
    input/      uploads exactly as received (never modified)
    stages/     stage images            layers/   RGBA map layers
    gaps/G01/   per-gap analysis and reconstruction images + analysis.json
    masks/      original_road_mask, current_road_mask, reconstructed_only
    reports/    CSV / JSON / log, and reports/pdf/ for generated PDF reports

The analysis (stages 1-9) runs in a background thread with live stage status.
Each user decision is a separate request that reconstructs at most one gap.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import AppSettings, PipelineConfig
from core.errors import ProcessingError, RoadHealingError
from core.export import FILE_PATTERN, graph_payload, results_payload, review_files, stage_files, write_reports
from core.pipeline import STAGE_LABELS, STAGES, ReconstructionPipeline
from core.preprocessing import LoadedImage, load_image, load_mask

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
PDF_NAME = re.compile(r"^report_\d{8}-\d{6}\.pdf$")


class JobNotFound(RoadHealingError):
    pass


class JobBusy(RoadHealingError):
    pass


@dataclass
class Job:
    id: str
    name: str
    created: float
    folder: Path
    image: LoadedImage | None
    mask: np.ndarray | None
    reference: np.ndarray | None
    source: str
    state: str = "uploaded"            # uploaded | running | review | done | error
    error: str = ""
    stage_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    file_versions: dict[str, int] = field(default_factory=dict)
    pipeline: ReconstructionPipeline | None = None
    reports: list[str] = field(default_factory=list)
    runs: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_status(self) -> None:
        self.stage_status = {s: {"status": "pending", "seconds": None} for s in STAGES}
        self.stage_status["upload"] = {"status": "done", "seconds": None}

    def file_urls(self) -> dict[str, str]:
        return {k: f"/api/jobs/{self.id}/files/{k}?v={v}" for k, v in self.file_versions.items()}

    def status(self) -> dict[str, Any]:
        return {"job_id": self.id, "name": self.name, "state": self.state, "error": self.error, "runs": self.runs,
                "stages": [{"key": s, "label": STAGE_LABELS[s], **self.stage_status.get(s, {"status": "pending"})}
                           for s in STAGES],
                "files": self.file_urls(), "reports": list(self.reports)}


class JobStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.root = Path(settings.storage_dir)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.history_path = self.root / "history.json"

    # ------------------------------------------------------------------ #
    def get(self, job_id: str) -> Job:
        if not JOB_ID.match(job_id or ""):
            raise JobNotFound("Unknown job id.")
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise JobNotFound("This job is not in memory (the server may have restarted). Open it from the "
                              "processing history to view its saved results.")
        return job

    def job_folder(self, job_id: str) -> Path:
        if not JOB_ID.match(job_id or ""):
            raise JobNotFound("Unknown job id.")
        folder = self.root / "jobs" / job_id
        if not folder.is_dir():
            raise JobNotFound("No saved results for this job.")
        return folder

    def create(self, image_bytes, image_name, mask_bytes, mask_name, ref_bytes, ref_name, source: str) -> Job:
        if not image_bytes and not mask_bytes:
            raise ProcessingError("Choose a satellite image to upload.")
        cfg = PipelineConfig().preprocess
        image = load_image(image_bytes, image_name or "image.png", cfg) if image_bytes else None
        shape = (image.height, image.width) if image is not None else None
        mask = load_mask(mask_bytes, mask_name or "mask.png", shape, "road mask") if mask_bytes else None
        ref_shape = shape or (mask.shape if mask is not None else None)
        reference = load_mask(ref_bytes, ref_name or "reference.png", ref_shape, "reference mask") if ref_bytes else None
        job_id = uuid.uuid4().hex
        folder = self.root / "jobs" / job_id
        (folder / "input").mkdir(parents=True)
        if image_bytes:
            (folder / "input" / f"image{Path(image_name or 'image.png').suffix.lower()}").write_bytes(image_bytes)
        if mask is not None:
            (folder / "input" / "road_mask.png").write_bytes(cv2.imencode(".png", mask)[1].tobytes())
        if reference is not None:
            (folder / "input" / "reference_mask.png").write_bytes(cv2.imencode(".png", reference)[1].tobytes())
        job = Job(job_id, image.name if image is not None else (mask_name or "road mask"), time.time(), folder,
                  image, mask, reference, source)
        job.reset_status()
        if image is not None:
            self._save(job, {"stages/original.png": image.image})
        with self.lock:
            self.jobs[job_id] = job
        return job

    # ------------------------------------------------------------------ #
    def _save(self, job: Job, files: dict[str, np.ndarray]) -> None:
        for rel, img in files.items():
            if img is None or not FILE_PATTERN.match(rel):
                continue
            ok, buf = cv2.imencode(".png", img)
            if ok:
                path = job.folder / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(buf.tobytes())
                job.file_versions[rel] = job.file_versions.get(rel, 0) + 1

    def _progress(self, job: Job, started: dict[str, float]):
        def callback(stage: str, status: str) -> None:
            if status == "running":
                started[stage] = time.perf_counter()
            elapsed = round(time.perf_counter() - started.get(stage, time.perf_counter()), 3)
            job.stage_status[stage] = {"status": status, "seconds": None if status == "running" else elapsed}
            if status == "done" and job.pipeline is not None:
                self._save(job, stage_files(job.pipeline, stage))
        return callback

    def _review_status(self, job: Job) -> None:
        s = job.pipeline.summary()
        job.stage_status["review"] = {"status": "done" if s["complete"] else "waiting", "seconds": None}
        job.stage_status["final"] = {"status": "done" if s["complete"] else "pending", "seconds": None}
        job.state = "done" if s["complete"] else "review"

    def start_run(self, job: Job, cfg: PipelineConfig) -> None:
        if not job.lock.acquire(blocking=False):
            raise JobBusy("This job is already running.")
        try:
            job.state, job.error = "running", ""
            job.reset_status()
            job.file_versions = {k: v for k, v in job.file_versions.items() if k == "stages/original.png"}
            for sub in ("stages", "layers", "gaps", "masks"):
                if sub != "stages":
                    shutil.rmtree(job.folder / sub, ignore_errors=True)
            pipe = ReconstructionPipeline(cfg, on_progress=self._progress(job, {}))
            job.pipeline = pipe
            pipe.load(job.image, job.mask, job.reference)
        except Exception:
            job.lock.release()
            raise
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def _run(self, job: Job) -> None:
        try:
            job.pipeline.run_all()
            write_reports(job.pipeline, job.folder / "reports")
            self._review_status(job)
            self._record_history(job)
        except RoadHealingError as exc:
            job.state, job.error = "error", str(exc)
        except Exception as exc:  # keep the server alive and report the problem
            job.state, job.error = "error", f"Unexpected error: {exc}"
            if job.pipeline is not None:
                job.pipeline.logger.exception("Unexpected error")
        finally:
            job.runs += 1
            job.lock.release()

    def decide(self, job: Job, gap_id: str, action: str) -> dict[str, Any]:
        if not job.lock.acquire(blocking=False):
            raise JobBusy("This job is busy; wait for the current step to finish.")
        try:
            if job.pipeline is None or job.state not in ("review", "done"):
                raise ProcessingError("Run the analysis before reviewing gaps.")
            outcome = job.pipeline.decide_gap(gap_id, action)
            self._save(job, review_files(job.pipeline, gap_id))
            write_reports(job.pipeline, job.folder / "reports")
            self._review_status(job)
            self._record_history(job)
            return outcome
        finally:
            job.lock.release()

    def make_pdf(self, job: Job) -> str:
        from core.pdf_report import build_pdf
        if not job.lock.acquire(blocking=False):
            raise JobBusy("This job is busy; try again in a moment.")
        try:
            if job.pipeline is None or "recommend" not in job.pipeline.state.completed:
                raise ProcessingError("Run the analysis before generating a PDF report.")
            name = f"report_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
            try:
                build_pdf(job.pipeline, job.folder / "reports" / "pdf" / name)
            except Exception as exc:
                raise ProcessingError(f"The PDF report could not be generated: {exc}") from exc
            job.reports.append(name)
            job.pipeline.logger.info(f"PDF report generated: reports/pdf/{name}")
            return name
        finally:
            job.lock.release()

    def payload(self, job: Job) -> dict[str, Any]:
        return {"status": job.status(), "results": results_payload(job.pipeline) if job.pipeline else {}}

    def graph(self, job: Job) -> dict[str, Any]:
        if job.pipeline is None:
            raise ProcessingError("Run the analysis first.")
        return graph_payload(job.pipeline)

    # ------------------------------------------------------------------ #
    def _load_history(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8")) if self.history_path.exists() else []
        except (OSError, json.JSONDecodeError):
            return []

    def _record_history(self, job: Job) -> None:
        s = job.pipeline.summary() if job.pipeline and job.pipeline.state.recon else {}
        entry = {"job_id": job.id, "name": job.name, "source": job.source,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.created)),
                 "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "gaps": s.get("total"), "approved": s.get("approved"),
                 "rejected": s.get("rejected"), "pending": s.get("pending"), "reconstructed": s.get("reconstructed"),
                 "components_before": s.get("components_before"), "components_after": s.get("components_after"),
                 "complete": s.get("complete")}
        with self.lock:
            history = [h for h in self._load_history() if h.get("job_id") != job.id]
            history.insert(0, entry)
            dropped, history = history[self.settings.keep_jobs:], history[:self.settings.keep_jobs]
            self.history_path.write_text(json.dumps(history, indent=1), encoding="utf-8")
            for old in dropped:
                if JOB_ID.match(old.get("job_id", "")):
                    self.jobs.pop(old["job_id"], None)
                    shutil.rmtree(self.root / "jobs" / old["job_id"], ignore_errors=True)

    def history(self) -> list[dict[str, Any]]:
        with self.lock:
            return self._load_history()

    def saved_results(self, job_id: str) -> dict[str, Any]:
        folder = self.job_folder(job_id)
        path = folder / "reports" / "results.json"
        if not path.exists():
            raise JobNotFound("This job has no saved results.")
        data = json.loads(path.read_text(encoding="utf-8"))
        files = {p.relative_to(folder).as_posix(): f"/api/jobs/{job_id}/files/{p.relative_to(folder).as_posix()}"
                 f"?v={int(p.stat().st_mtime)}" for sub in ("stages", "layers", "gaps", "masks")
                 for p in (folder / sub).rglob("*.png") if FILE_PATTERN.match(p.relative_to(folder).as_posix())}
        reports = sorted(p.name for p in (folder / "reports" / "pdf").glob("report_*.pdf")) \
            if (folder / "reports" / "pdf").is_dir() else []
        status = {"job_id": job_id, "name": (data.get("input") or {}).get("name", job_id), "state": "saved", "error": "",
                  "runs": 1, "stages": [{"key": s["key"], "label": s["label"], "status": "done" if s["done"] else "pending",
                                         "seconds": None} for s in data.get("stages", [])],
                  "files": files, "reports": reports}
        return {"status": status, "results": data}
