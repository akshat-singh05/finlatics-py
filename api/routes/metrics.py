"""Metrics routes: overall, by campaign, by channel, by device."""

import json
from flask import Blueprint, request, jsonify

from api.services.data_service import get_filtered_dataset, get_overall_metrics, get_summary_by
from utils.logger import get_logger

logger = get_logger("routes.metrics")

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/metrics", methods=["GET"])
def metrics():
    """Return overall KPI metrics. Optional query params: campaign, channel, device."""
    try:
        campaign = request.args.get("campaign")
        channel = request.args.get("channel")
        device = request.args.get("device")

        df = get_filtered_dataset(campaign, channel, device)

        if len(df) == 0:
            return jsonify({"error": "No data matches filters"}), 404

        result = get_overall_metrics(df)
        result["filters"] = {"campaign": campaign, "channel": channel, "device": device}
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Error in /metrics endpoint")
        return jsonify({"error": str(e)}), 500


@metrics_bp.route("/metrics/campaign", methods=["GET"])
def metrics_by_campaign():
    """Return metrics grouped by campaign."""
    try:
        summary = get_summary_by("campaign")
        return jsonify(json.loads(summary.reset_index().to_json(orient="records")))
    except Exception as e:
        logger.exception("Error in /metrics/campaign endpoint")
        return jsonify({"error": str(e)}), 500


@metrics_bp.route("/metrics/channel", methods=["GET"])
def metrics_by_channel():
    """Return metrics grouped by channel."""
    try:
        summary = get_summary_by("channel")
        return jsonify(json.loads(summary.reset_index().to_json(orient="records")))
    except Exception as e:
        logger.exception("Error in /metrics/channel endpoint")
        return jsonify({"error": str(e)}), 500


@metrics_bp.route("/metrics/device", methods=["GET"])
def metrics_by_device():
    """Return metrics grouped by device."""
    try:
        summary = get_summary_by("device")
        return jsonify(json.loads(summary.reset_index().to_json(orient="records")))
    except Exception as e:
        logger.exception("Error in /metrics/device endpoint")
        return jsonify({"error": str(e)}), 500
