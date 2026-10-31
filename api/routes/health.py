"""Health check route."""

from pathlib import Path
from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "synthetic_advertising_data.csv"


@health_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model_available = any(MODEL_DIR.glob("*.joblib")) if MODEL_DIR.exists() else False
    data_available = DATA_PATH.exists()
    return jsonify({
        "status": "healthy",
        "data_available": data_available,
        "model_available": model_available,
        "api_version": "2.0.0",
    })


@health_bp.route("/api/info", methods=["GET"])
def api_info():
    """Return model metadata, feature names, and valid categories."""
    from api.utils.validation import VALID_CAMPAIGNS, VALID_CHANNELS, VALID_DEVICES
    from api.services.prediction_service import get_model_metadata

    metadata = get_model_metadata("high_roi")

    return jsonify({
        "api_version": "2.0.0",
        "valid_campaigns": VALID_CAMPAIGNS,
        "valid_channels": VALID_CHANNELS,
        "valid_devices": VALID_DEVICES,
        "model_metadata": {
            "best_model": metadata.get("best_model"),
            "task_type": metadata.get("task_type"),
            "target_type": metadata.get("target_type"),
            "feature_count": metadata.get("feature_count"),
            "train_samples": metadata.get("train_samples"),
            "test_samples": metadata.get("test_samples"),
            "trained_at": metadata.get("timestamp"),
        } if metadata else None,
    })
