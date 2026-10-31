"""
Data service: dataset loading, caching, and filtering.

Extracts data access logic from the monolithic app.py into a
reusable service consumed by both the metrics and predict routes.
"""

from pathlib import Path
import pandas as pd

from analysis.data_loader import load_data
from analysis.data_cleaning import clean_data
from analysis.metrics import compute_kpis, summarize_by
from utils.logger import get_logger

logger = get_logger("data_service")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "synthetic_advertising_data.csv"

_dataset_cache = None


def get_dataset() -> pd.DataFrame:
    """Load and cache the processed dataset."""
    global _dataset_cache
    if _dataset_cache is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        df = load_data(DATA_PATH)
        df = clean_data(df)
        df = compute_kpis(df)
        _dataset_cache = df
        logger.info("Dataset loaded and cached: %d rows", len(df))
    return _dataset_cache.copy()


def get_filtered_dataset(campaign=None, channel=None, device=None) -> pd.DataFrame:
    """Load dataset and apply optional filters."""
    df = get_dataset()
    if campaign:
        df = df[df["campaign"] == campaign]
    if channel:
        df = df[df["channel"] == channel]
    if device:
        df = df[df["device"] == device]
    return df


def get_summary_by(group_col: str) -> pd.DataFrame:
    """Return summarized metrics grouped by a column."""
    df = get_dataset()
    return summarize_by(df, group_col)


def get_overall_metrics(df: pd.DataFrame) -> dict:
    """Compute overall KPI metrics from a (possibly filtered) DataFrame."""
    total_spend = float(df["spend"].sum())
    total_revenue = float(df["revenue"].sum())
    total_clicks = int(df["clicks"].sum())
    total_impressions = int(df["impressions"].sum())
    total_conversions = int(df["conversions"].sum())

    return {
        "records": len(df),
        "total_spend": round(total_spend, 2),
        "total_revenue": round(total_revenue, 2),
        "net_profit": round(total_revenue - total_spend, 2),
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "total_conversions": total_conversions,
        "ctr": round(total_clicks / total_impressions, 6) if total_impressions else 0,
        "cpc": round(total_spend / total_clicks, 4) if total_clicks else 0,
        "roas": round(total_revenue / total_spend, 4) if total_spend else 0,
        "roi": round((total_revenue - total_spend) / total_spend, 4) if total_spend else 0,
    }
