"""
REST API. Errors are JSON {"error": "..."} with 400 / 404 / 409 / 413 / 500.

    GET  /api/config | /api/samples | /api/samples/<name>/thumbnail.png | /api/history
    POST /api/jobs                       multipart: image, optional mask, optional reference
    POST /api/jobs/sample                {"name", "use_mask", "use_reference"}
    POST /api/jobs/<id>/run              {"params": {"section.field": value}}  -> analysis (stages 1-9)
    GET  /api/jobs/<id>/status           live stage status and file URLs
    GET  /api/jobs/<id>/results          everything the interface shows
    POST /api/jobs/<id>/gaps/<gap>/decision   {"action": "approve" | "reject" | "skip"}
    GET  /api/jobs/<id>/graph            the current road graph (nodes, edges, components, statistics)
    POST /api/jobs/<id>/report           generate a PDF report of the current session
    GET  /api/jobs/<id>/files/<path>     stage, layer, mask and gap images (whitelisted paths)
    GET  /api/jobs/<id>/download/<file>  CSV / JSON / log / PDF / all.zip
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import config_from_flat, schema_with_defaults
from core.errors import ConfigError, ImageLoadError, ProcessingError, RoadHealingError, StageOrderError
from core.export import FILE_PATTERN, zip_folder
from core.pipeline import DecisionRefused
from web.jobs import PDF_NAME, JobBusy, JobNotFound, JobStore

api = Blueprint("api", __name__, url_prefix="/api")
REPORT_FILES = {
    "gap_analysis.csv": "text/csv", "user_decisions.csv": "text/csv", "endpoints.csv": "text/csv",
    "fragments.csv": "text/csv", "excluded_pairs.csv": "text/csv", "evaluation_metrics.csv": "text/csv",
    "evaluation_metrics.json": "application/json", "results.json": "application/json",
    "road_graph.json": "application/json", "configuration.json": "application/json",
    "processing_log.txt": "text/plain",
}


def store() -> JobStore:
    return current_app.extensions["jobs"]


def _error(message: str, status: int):
    return jsonify({"error": message}), status


@api.errorhandler(JobNotFound)
def _not_found(exc):
    return _error(str(exc), 404)


@api.errorhandler(JobBusy)
def _busy(exc):
    return _error(str(exc), 409)


@api.errorhandler(StageOrderError)
def _order(exc):
    return _error(str(exc), 409)


@api.errorhandler(DecisionRefused)
def _refused(exc):
    return _error(str(exc), 409)


@api.errorhandler(RoadHealingError)
def _invalid(exc):
    return _error(str(exc), 400)


@api.errorhandler(RequestEntityTooLarge)
def _too_large(exc):
    limit = current_app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
    return _error(f"The upload is too large. The limit is {limit} MB for all files together.", 413)


@api.errorhandler(HTTPException)
def _http(exc):
    return _error(exc.description or exc.name, exc.code or 500)


@api.errorhandler(Exception)
def _unexpected(exc):
    current_app.logger.exception("Unexpected API error")
    return _error("Something went wrong on the server. The processing log has details.", 500)


def _file(field: str):
    f = request.files.get(field)
    if f is None or not f.filename:
        return None, None
    name = secure_filename(f.filename) or f"{field}.png"
    data = f.read()
    if not data:
        raise ImageLoadError(f"The {field} file '{name}' is empty.")
    return data, name


def _json_body() -> dict:
    data = request.get_json(silent=True)
    if data is None:
        if request.data:
            raise ProcessingError("The request body must be valid JSON.")
        return {}
    if not isinstance(data, dict):
        raise ProcessingError("The request body must be a JSON object.")
    return data


def _sample_folder(name: str) -> Path:
    root: Path = current_app.config["SAMPLES_DIR"]
    safe = secure_filename(name or "")
    folder = root / safe
    if not safe or safe != name or not folder.is_dir():
        raise JobNotFound(f"Unknown sample '{name}'.")
    return folder


# --------------------------------------------------------------------------- #
@api.get("/config")
def get_config():
    return jsonify({"groups": schema_with_defaults()})


@api.get("/samples")
def list_samples():
    root: Path = current_app.config["SAMPLES_DIR"]
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        loadable = (folder / "image.png").exists() and "load_error" not in meta.get("expected", {})
        out.append({"name": folder.name, "description": meta.get("description", ""), "loadable": loadable,
                    "thumbnail": f"/api/samples/{folder.name}/thumbnail.png" if loadable else None})
    return jsonify({"samples": out, "note": "Synthetic test scenes, not real satellite imagery."})


@api.get("/samples/<name>/thumbnail.png")
def sample_thumbnail(name: str):
    path = _sample_folder(name) / "image.png"
    if not path.exists():
        raise JobNotFound("This sample has no preview image.")
    return send_file(path, mimetype="image/png", max_age=3600)


@api.post("/jobs")
def upload():
    image, image_name = _file("image")
    mask, mask_name = _file("mask")
    ref, ref_name = _file("reference")
    job = store().create(image, image_name, mask, mask_name, ref, ref_name, source="upload")
    return jsonify(store().payload(job)), 201


@api.post("/jobs/sample")
def from_sample():
    body = _json_body()
    folder = _sample_folder(str(body.get("name", "")))
    read = lambda n: (folder / n).read_bytes() if (folder / n).exists() else None  # noqa: E731
    image = read("image.png")
    if image is None:
        raise ImageLoadError("This sample has no image (it may be the intentionally broken test file).")
    job = store().create(image, f"{folder.name}.png",
                         read("observed_mask.png") if body.get("use_mask", False) else None, "observed_mask.png",
                         read("reference_mask.png") if body.get("use_reference", True) else None, "reference_mask.png",
                         source=f"sample:{folder.name}")
    if job.image is not None:
        job.image.notes.append("Synthetic test scene (not real satellite imagery).")
    return jsonify(store().payload(job)), 201


@api.post("/jobs/<job_id>/run")
def run(job_id: str):
    job = store().get(job_id)
    params = _json_body().get("params", {})
    if not isinstance(params, dict):
        raise ConfigError("'params' must be an object of parameter values.")
    cfg = config_from_flat(params)
    notes = cfg.validate()
    store().start_run(job, cfg)
    return jsonify({"state": "running", "notes": notes, "status": job.status()}), 202


@api.get("/jobs/<job_id>/status")
def status(job_id: str):
    return jsonify(store().get(job_id).status())


@api.get("/jobs/<job_id>/results")
def results(job_id: str):
    try:
        job = store().get(job_id)
    except JobNotFound:
        return jsonify(store().saved_results(job_id))
    if job.state == "running":
        raise JobBusy("The analysis is still running.")
    return jsonify(store().payload(job))


@api.post("/jobs/<job_id>/gaps/<gap_id>/decision")
def gap_decision(job_id: str, gap_id: str):
    job = store().get(job_id)
    action = str(_json_body().get("action", ""))
    outcome = store().decide(job, gap_id, action)
    return jsonify({"outcome": outcome, **store().payload(job)})


@api.get("/jobs/<job_id>/graph")
def graph(job_id: str):
    return jsonify(store().graph(store().get(job_id)))


@api.post("/jobs/<job_id>/report")
def report(job_id: str):
    job = store().get(job_id)
    name = store().make_pdf(job)
    return jsonify({"name": name, "url": f"/api/jobs/{job_id}/download/{name}"}), 201


@api.get("/jobs/<job_id>/files/<path:relpath>")
def files(job_id: str, relpath: str):
    if not FILE_PATTERN.match(relpath):
        raise JobNotFound("Unknown file.")
    path = store().job_folder(job_id) / relpath
    if not path.exists():
        raise JobNotFound("This image has not been produced yet.")
    return send_file(path, mimetype="image/png", max_age=0)


@api.get("/jobs/<job_id>/download/<filename>")
def download(job_id: str, filename: str):
    folder = store().job_folder(job_id)
    try:
        job = store().get(job_id)
        if job.state == "running":
            raise JobBusy("The analysis is still running; download when it has finished.")
        stem = secure_filename(Path(job.name).stem) or "result"
    except JobNotFound:
        stem = job_id[:8]
    if filename == "all.zip":
        return Response(zip_folder(folder, stem), mimetype="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_results.zip"'})
    if PDF_NAME.match(filename):
        path, mime = folder / "reports" / "pdf" / filename, "application/pdf"
    elif filename in REPORT_FILES:
        path, mime = folder / "reports" / filename, REPORT_FILES[filename]
    else:
        raise JobNotFound(f"Unknown file '{filename}'.")
    if not path.exists():
        raise JobNotFound(f"'{filename}' is not available for this job yet.")
    return send_file(path, mimetype=mime, as_attachment=True, download_name=f"{stem}_{filename}")


@api.get("/history")
def history():
    return jsonify({"history": store().history()})
