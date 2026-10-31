"""Predict route: ML inference endpoint."""

from flask import Blueprint, request, jsonify

from api.services.prediction_service import predict
from api.utils.validation import validate_predict_input
from utils.logger import get_logger

logger = get_logger("routes.predict")

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Predict campaign profitability.

    JSON body:
        {
            "campaign": "Brand Awareness",
            "channel": "Google Ads",
            "device": "Mobile",
            "impressions": 5000,
            "clicks": 80,
            "spend": 150.0,
            "conversions": 5,
            "bounce_rate": 0.5,
            "avg_session_duration": 120.0
        }

    Optional:
        "target": "high_roi" (default) | "revenue" | "high_conversion"

    Returns:
        {
            "prediction": 1,
            "label": "Profitable",
            "confidence": 0.87,
            "target_type": "high_roi",
            "interpretation": { ... }
        }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        target = data.pop("target", "high_roi")

        validation_errors = validate_predict_input(data)
        if validation_errors:
            logger.warning("Prediction validation failed: %s", validation_errors)
            return jsonify({
                "error": "Validation failed",
                "messages": validation_errors,
            }), 400

        result = predict(data, target)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": "Validation error", "message": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500
