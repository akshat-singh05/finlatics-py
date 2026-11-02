"""Metrics routes: overall KPIs, per-dimension breakdowns, pivots and row data."""

from flask import Blueprint, Response, request, jsonify

from api.services.data_service import (
    DATA_COLUMNS, GROUP_DIMENSIONS, PIVOT_DIMENSIONS, apply_filters, data_csv, data_page,
    overall_metrics, pivot_records, resolve_dimension, summary_records,
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


@metrics_bp.route("/pivot", methods=["GET"])
def pivot():
    """
    Totals and KPIs for every pair of two dimensions (e.g. banner x placement).

    Query params: rows, cols (campaign_number, user_engagement, banner,
    placement, month, weekday; aliases allowed) plus the /metrics filters.
    """
    rows = resolve_dimension(request.args.get("rows", ""))
    cols = resolve_dimension(request.args.get("cols", ""))
    if rows not in PIVOT_DIMENSIONS or cols not in PIVOT_DIMENSIONS or rows == cols:
        return jsonify({"error": "rows and cols must be two different dimensions",
                        "valid": PIVOT_DIMENSIONS}), 400

    df, errors = apply_filters(request.args)
    if errors:
        return jsonify({"error": "Invalid filters", "messages": errors}), 400
    if df.empty:
        return jsonify({"error": "No data matches filters"}), 404

    return jsonify({"rows": rows, "cols": cols, "filters": dict(request.args),
                    "cells": pivot_records(df, rows, cols)})


@metrics_bp.route("/data", methods=["GET"])
def data():
    """
    Row-level data, paginated.

    Query params: page (1-based), page_size (max 500), sort (a column name),
    order (asc | desc) plus the /metrics filters.
    """
    df, errors = apply_filters(request.args)
    if errors:
        return jsonify({"error": "Invalid filters", "messages": errors}), 400
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 50))
    except ValueError:
        return jsonify({"error": "page and page_size must be integers"}), 400
    sort = request.args.get("sort", "date")
    if sort not in DATA_COLUMNS:
        return jsonify({"error": f"Unknown sort column '{sort}'", "valid": DATA_COLUMNS}), 400

    return jsonify(data_page(df, page, page_size, sort,
                             descending=request.args.get("order") == "desc"))


@metrics_bp.route("/data.csv", methods=["GET"])
def data_download():
    """Filtered rows as a CSV download (same filters as /metrics)."""
    df, errors = apply_filters(request.args)
    if errors:
        return jsonify({"error": "Invalid filters", "messages": errors}), 400
    return Response(data_csv(df), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=advertising_data_filtered.csv"})
