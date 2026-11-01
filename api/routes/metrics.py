"""Metrics routes: overall KPIs and per-dimension breakdowns."""

from flask import Blueprint, request, jsonify

from api.services.data_service import (
    GROUP_DIMENSIONS, apply_filters, overall_metrics, resolve_dimension, summary_records,
)
from utils.logger import get_logger

logger = get_logger("routes.metrics")

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/metrics", methods=["GET"])
def metrics():
    """
    Overall totals and KPIs.

    Optional query params: campaign_number, user_engagement, banner,
    placement, start, end (YYYY-MM-DD, inclusive).
    """
    df, errors = apply_filters(request.args)
    if errors:
        return jsonify({"error": "Invalid filters", "messages": errors}), 400
    if df.empty:
        return jsonify({"error": "No data matches filters"}), 404

    result = overall_metrics(df)
    result["filters"] = dict(request.args)
    return jsonify(result)


@metrics_bp.route("/metrics/<dimension>", methods=["GET"])
def metrics_by(dimension):
    """
    Totals and KPIs grouped by a dimension.

    dimension: campaign_number (alias campaign), user_engagement (alias
    engagement), banner, placement, month, weekday, date. Accepts the
    same filters as /metrics.
    """
    column = resolve_dimension(dimension)
    if column is None:
        return jsonify({"error": f"Unknown dimension '{dimension}'",
                        "valid": GROUP_DIMENSIONS}), 404

    df, errors = apply_filters(request.args)
    if errors:
        return jsonify({"error": "Invalid filters", "messages": errors}), 400
    if df.empty:
        return jsonify({"error": "No data matches filters"}), 404

    return jsonify({"dimension": column, "filters": dict(request.args),
                    "groups": summary_records(df, column)})
