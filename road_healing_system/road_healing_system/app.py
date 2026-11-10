"""
Context-Aware Shortest-Path Healing System for Occlusion-Robust Road Network Reconstruction.

    python app.py            (or double-click run.bat on Windows)

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from flask import Flask, render_template  # noqa: E402

from config import AppSettings  # noqa: E402
from web.jobs import JobStore  # noqa: E402
from web.routes import api  # noqa: E402


def ensure_samples(folder: Path) -> None:
    """Write the synthetic test scenes on first start if they are missing."""
    if folder.is_dir() and any(folder.glob("*/meta.json")):
        return
    from scripts.generate_samples import generate
    generate(folder)


def create_app(settings: AppSettings | None = None) -> Flask:
    settings = settings or AppSettings.from_env()
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    app.config["SAMPLES_DIR"] = Path(settings.samples_dir)
    app.json.sort_keys = False                  # keep stage order in JSON responses
    app.extensions["jobs"] = JobStore(settings)
    app.register_blueprint(api)

    @app.get("/")
    def index():
        return render_template("index.html", max_upload_mb=settings.max_upload_mb)

    return app


if __name__ == "__main__":
    s = AppSettings.from_env()
    ensure_samples(s.samples_dir)
    print(f"Road network reconstruction running at http://{s.host}:{s.port}  (Ctrl+C to stop)")
    create_app(s).run(host=s.host, port=s.port, debug=s.debug, threaded=True, use_reloader=False)
