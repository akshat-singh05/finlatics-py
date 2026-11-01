"""
Prediction service shared by the REST API and the dashboard.

Loads the trained revenue pipeline, validates planned-buy requests and
returns predicted revenue, margin and profitability. Features are built
with ``models.feature_engineering.build_features`` — the same function
used in training — so serving cannot drift from training.

Request fields (one planned buy = one row):
    date             "YYYY-MM-DD"
    campaign_number  e.g. "camp 1"
    user_engagement  "Low" | "Medium" | "High"
    banner           e.g. "240 x 400"
    placement        e.g. "ghi"
    displays         ads to be shown (>= 0)
    cost             placement cost in $ (>= 0)
"""

import json
import numbers
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.feature_engineering import (
    CATEGORICAL_FEATURES, PERIOD_START, build_features, target_to_revenue,
)
from utils.logger import get_logger

logger = get_logger("predictor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "revenue_model.joblib"
META_PATH = MODEL_PATH.with_suffix(".meta.json")

# Last day in the dataset; later dates are extrapolation for tree models
PERIOD_END = pd.Timestamp("2020-06-30")

REQUEST_CATEGORICAL = CATEGORICAL_FEATURES[:4]   # weekday is derived from date
REQUEST_FIELDS = ["date", *REQUEST_CATEGORICAL, "displays", "cost"]
MAX_BATCH = 1000


class ModelNotAvailable(FileNotFoundError):
    """Raised when no trained model exists (run `python main.py`)."""


@lru_cache(maxsize=1)
def load_model():
    """Load the trained pipeline once per process."""
    if not MODEL_PATH.exists():
        raise ModelNotAvailable(
            f"No trained model at {MODEL_PATH}. Run `python main.py` to train it."
        )
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Model loaded: %s", MODEL_PATH.name)
    return pipeline


def model_available() -> bool:
    return MODEL_PATH.exists()


def get_metadata() -> dict:
    """Training metadata written next to the model (empty if missing)."""
    if not META_PATH.exists():
        return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def valid_categories() -> dict:
    """Category levels the model was trained on, read from its encoder."""
    preprocessor = load_model().named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]
    levels = dict(zip(CATEGORICAL_FEATURES, (list(c) for c in encoder.categories_)))
    return {k: levels[k] for k in REQUEST_CATEGORICAL}


def validate_request(item, categories: dict | None = None) -> list[str]:
    """
    Validate one planned-buy request.

    Parameters
    ----------
    item : dict
        Request body for one row.
    categories : dict, optional
        Valid levels per categorical field (defaults to the model's).

    Returns
    -------
    list[str]
        Error messages; empty if valid.
    """
    if not isinstance(item, dict):
        return ["Each request must be a JSON object"]

    missing = [f for f in REQUEST_FIELDS if f not in item]
    if missing:
        return [f"Missing required fields: {missing}"]

    errors = []
    categories = categories if categories is not None else valid_categories()
    for field in REQUEST_CATEGORICAL:
        if str(item[field]) not in categories[field]:
            errors.append(f"Invalid {field} '{item[field]}'. Valid: {categories[field]}")

    for field in ["displays", "cost"]:
        val = item[field]
        # numbers.Real also covers numpy scalars (e.g. values read from a CSV)
        if isinstance(val, bool) or not isinstance(val, numbers.Real) or not np.isfinite(val) or val < 0:
            errors.append(f"{field} must be a non-negative number")

    try:
        date = pd.to_datetime(item["date"], format="%Y-%m-%d")
        if date < PERIOD_START:
            errors.append(f"date must be on or after {PERIOD_START.date()}")
    except (ValueError, TypeError):
        errors.append("date must be in YYYY-MM-DD format")

    return errors


def predict(items: list[dict]) -> list[dict]:
    """
    Predict revenue, margin and profitability for validated requests.

    Parameters
    ----------
    items : list[dict]
        Requests that passed ``validate_request``.

    Returns
    -------
    list[dict]
        One result per request.
    """
    df = pd.DataFrame(items)
    df["date"] = pd.to_datetime(df["date"])
    revenue = target_to_revenue(load_model().predict(build_features(df)))

    results = []
    for (_, row), rev in zip(df.iterrows(), revenue):
        cost = float(row["cost"])
        margin = float(rev) - cost
        warnings = []
        if row["date"] > PERIOD_END:
            warnings.append(
                f"date is after the training data ({PERIOD_END.date()}); the model "
                "assumes conditions stay as at the end of the data period"
            )
        if cost < 1:
            warnings.append("very small buy (cost < $1): per-click billing makes it noisy")
        results.append({
            "predicted_revenue": round(float(rev), 4),
            "cost": round(cost, 4),
            "predicted_margin": round(margin, 4),
            "predicted_roi": round(margin / cost, 4) if cost > 0 else None,
            "profitable": margin > 0,
            "warnings": warnings,
        })

    logger.info("Predicted %d planned buys", len(results))
    return results
