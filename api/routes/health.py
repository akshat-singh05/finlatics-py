"""Health, API info, and report routes."""

from flask import Blueprint, jsonify

from analysis.data_loader import DEFAULT_RAW_PATH
from api.services.data_service import (
    GROUP_DIMENSIONS, filter_options, get_dataset, load_report, to_jsonable,
)
from models import predictor

health_bp = Blueprint("health", __name__)

API_VERSION = "3.0.0"


@health_bp.route("/health", methods=["GET"])
def health():
    """Liveness check plus availability of data and model."""
    return jsonify({
        "status": "healthy",
        "data_available": DEFAULT_RAW_PATH.exists(),
        "model_available": predictor.model_available(),
        "api_version": API_VERSION,
    })


@health_bp.route("/api/info", methods=["GET"])
def api_info():
    """Dataset summary, valid filter values, model metadata and endpoints."""
    df = get_dataset()
    meta = predictor.get_metadata()
    return jsonify({
        "api_version": API_VERSION,
        "dataset": {
            "rows": len(df),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
        "filters": filter_options(),
        "dimensions": GROUP_DIMENSIONS,
        "model": {
            "available": predictor.model_available(),
            "best_model": meta.get("best_model"),
            "target": meta.get("target_type"),
            "features": meta.get("feature_names"),
            "train_samples": meta.get("train_samples"),
            "test_samples": meta.get("test_samples"),
            "trained_at": meta.get("timestamp"),
            "request_fields": predictor.REQUEST_FIELDS,
        },
        "endpoints": [
            "GET  /health",
            "GET  /api/info",
            "GET  /metrics",
            "GET  /metrics/<dimension>",
            "GET  /pivot?rows=<dimension>&cols=<dimension>",
            "GET  /data",
            "GET  /data.csv",
            "GET  /insights",
            "GET  /model",
            "GET  /report/<eda|model>",
            "POST /predict",
        ],
    })


@health_bp.route("/insights", methods=["GET"])
def insights():
    """EDA findings computed by the pipeline (reports/eda_findings.json)."""
    try:
        return jsonify(load_report("eda_findings"))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@health_bp.route("/model", methods=["GET"])
def model_report():
    """Held-out evaluation of the model vs baselines (reports/model_evaluation.json)."""
    try:
        report = load_report("model_evaluation")
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    report["metadata"] = to_jsonable(predictor.get_metadata())
    return jsonify(report)
