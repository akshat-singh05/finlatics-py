"""Web UI routes: the single-page app, its assets and the written reports."""

from pathlib import Path

from flask import Blueprint, abort, jsonify, send_from_directory

from api.services.data_service import REPORTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = PROJECT_ROOT / "web"
REPORT_FILES = {"eda": "eda_findings.md", "model": "model_report.md"}

ui_bp = Blueprint("ui", __name__)


def _plotly_dir() -> Path | None:
    """Folder holding the plotly.js bundle shipped with the plotly package (offline use)."""
    try:
        import plotly
    except ImportError:
        return None
    folder = Path(plotly.__file__).resolve().parent / "package_data"
    return folder if (folder / "plotly.min.js").exists() else None


@ui_bp.route("/", methods=["GET"])
def index():
    """The web app."""
    return send_from_directory(WEB_DIR, "index.html")


@ui_bp.route("/ui/<path:filename>", methods=["GET"])
def asset(filename):
    """Static assets (CSS, JS) for the web app."""
    return send_from_directory(WEB_DIR / "static", filename)


@ui_bp.route("/ui/vendor/plotly.min.js", methods=["GET"])
def plotly_js():
    """plotly.js from the installed plotly package, so charts work without internet."""
    folder = _plotly_dir()
    if folder is None:
        abort(404)
    return send_from_directory(folder, "plotly.min.js", mimetype="text/javascript")


@ui_bp.route("/report/<name>", methods=["GET"])
def report(name):
    """A written report as Markdown: eda (the 20 questions) or model."""
    filename = REPORT_FILES.get(name)
    if filename is None:
        return jsonify({"error": f"Unknown report '{name}'", "valid": sorted(REPORT_FILES)}), 404
    if not (REPORTS_DIR / filename).exists():
        return jsonify({"error": f"{filename} not found. Run `python main.py` first."}), 404
    return send_from_directory(REPORTS_DIR, filename, mimetype="text/markdown")


@ui_bp.route("/report/figures/<path:filename>", methods=["GET"])
def report_figure(filename):
    """Charts referenced by the reports (reports/figures/...)."""
    return send_from_directory(REPORTS_DIR / "figures", filename)
