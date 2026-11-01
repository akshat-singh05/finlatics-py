"""
Data service: dataset loading, caching, filtering and aggregation.

Runs the same load -> clean -> KPI pipeline as main.py so API numbers
match the reports exactly.
"""

import io
import json
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_loader import DEFAULT_RAW_PATH, load_data
from analysis.data_cleaning import clean_data
from analysis.metrics import compute_kpis, summarize_by, get_overall_summary
from utils.logger import get_logger

logger = get_logger("data_service")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

FILTER_COLUMNS = ["campaign_number", "user_engagement", "banner", "placement"]
GROUP_DIMENSIONS = FILTER_COLUMNS + ["month", "weekday", "date"]
DIMENSION_ALIASES = {"campaign": "campaign_number", "engagement": "user_engagement"}

_dataset_cache = None


def get_dataset() -> pd.DataFrame:
    """Load and cache the cleaned, KPI-enriched dataset."""
    global _dataset_cache
    if _dataset_cache is None:
        if not DEFAULT_RAW_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DEFAULT_RAW_PATH}")
        with contextlib.redirect_stdout(io.StringIO()):  # pipeline prints reports
            _dataset_cache = compute_kpis(clean_data(load_data(DEFAULT_RAW_PATH)))
        logger.info("Dataset loaded and cached: %d rows", len(_dataset_cache))
    return _dataset_cache


def filter_options() -> dict:
    """Valid values for every filter column."""
    df = get_dataset()
    return {col: sorted(df[col].astype(str).unique()) for col in FILTER_COLUMNS}


def apply_filters(args) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter the dataset by query parameters.

    Supports equality filters on FILTER_COLUMNS and ``start`` / ``end``
    dates (inclusive).

    Returns
    -------
    (pd.DataFrame, list[str])
        Filtered frame and validation errors (frame is empty on error).
    """
    df = get_dataset()
    options = filter_options()
    errors = []

    for col in FILTER_COLUMNS:
        value = args.get(col)
        if value is None:
            continue
        if value not in options[col]:
            errors.append(f"Invalid {col} '{value}'. Valid: {options[col]}")
        else:
            df = df[df[col].astype(str) == value]

    for key, op in (("start", "ge"), ("end", "le")):
        value = args.get(key)
        if value is None:
            continue
        try:
            df = df[getattr(df["date"], op)(pd.Timestamp(value))]
        except (ValueError, TypeError):
            errors.append(f"{key} must be in YYYY-MM-DD format")

    return (df.iloc[0:0] if errors else df), errors


def resolve_dimension(name: str) -> str | None:
    """Map a URL dimension name (or alias) to a column, or None if unknown."""
    name = DIMENSION_ALIASES.get(name, name)
    return name if name in GROUP_DIMENSIONS else None


def overall_metrics(df: pd.DataFrame) -> dict:
    """Totals and ratio-of-totals KPIs for a (filtered) frame."""
    return to_jsonable(get_overall_summary(df, verbose=False))


def summary_records(df: pd.DataFrame, dimension: str) -> list[dict]:
    """Per-group totals and KPIs as JSON-ready records."""
    summary = summarize_by(df, dimension).reset_index()
    if dimension == "date":
        summary = summary.sort_values("date")
        summary["date"] = summary["date"].dt.strftime("%Y-%m-%d")
    return json.loads(summary.to_json(orient="records"))  # NaN -> null


def load_report(name: str) -> dict:
    """Read a JSON report written by main.py (reports/<name>.json)."""
    path = REPORTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python main.py` first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(obj):
    """Convert numpy scalars and NaN so the result is valid JSON."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    return obj
