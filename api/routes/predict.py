"""Predict route: revenue / margin for planned inventory buys."""

from flask import Blueprint, request, jsonify

from models.predictor import (
    MAX_BATCH, ModelNotAvailable, predict, valid_categories, validate_request,
)
from utils.logger import get_logger

logger = get_logger("routes.predict")

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Predict revenue, margin and profitability for one or more planned buys.

    JSON body: one object or a list of up to MAX_BATCH objects, e.g.
        {
            "date": "2020-06-15",
            "campaign_number": "camp 1",
            "user_engagement": "High",
            "banner": "240 x 400",
            "placement": "ghi",
            "displays": 20000,
            "cost": 25.0
        }

    Returns a result object (or list) with predicted_revenue, cost,
    predicted_margin, predicted_roi, profitable and warnings.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    single = isinstance(data, dict)
    items = [data] if single else data
    if not isinstance(items, list) or not items:
        return jsonify({"error": "Body must be an object or a non-empty list"}), 400
    if len(items) > MAX_BATCH:
        return jsonify({"error": f"At most {MAX_BATCH} items per request"}), 400

    try:
        categories = valid_categories()
    except ModelNotAvailable as e:
        return jsonify({"error": str(e)}), 503

    errors = {i: errs for i, item in enumerate(items)
              if (errs := validate_request(item, categories))}
    if errors:
        logger.warning("Prediction validation failed: %s", errors)
        messages = errors[0] if single else {str(k): v for k, v in errors.items()}
        return jsonify({"error": "Validation failed", "messages": messages}), 400

    results = predict(items)
    return jsonify(results[0] if single else results)
