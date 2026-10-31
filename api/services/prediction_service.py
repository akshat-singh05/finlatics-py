"""
Prediction service: model loading, input preparation, and inference.

Centralizes all ML prediction logic for use by both the /predict
route and potential future consumers.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from analysis.metrics import compute_kpis
from models.feature_engineering import (
    create_derived_features, create_interaction_features, create_time_features,
)
from utils.logger import get_logger

logger = get_logger("prediction_service")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

_pipeline_cache = {}


def get_pipeline(target: str = "high_roi"):
    """Load a saved prediction pipeline with caching."""
    if target not in _pipeline_cache:
        for pattern in [f"best_pipeline_{target}.joblib", f"best_model_{target}.joblib"]:
            path = MODEL_DIR / pattern
            if path.exists():
                _pipeline_cache[target] = joblib.load(path)
                logger.info("Model loaded: %s", path.name)
                return _pipeline_cache[target]
        raise FileNotFoundError(f"No saved model for target '{target}'")
    return _pipeline_cache[target]


def get_model_metadata(target: str = "high_roi") -> dict:
    """Load model training metadata if available."""
    import json
    for pattern in [f"best_pipeline_{target}.meta.json", f"best_model_{target}.meta.json"]:
        path = MODEL_DIR / pattern
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def prepare_input(data: dict) -> pd.DataFrame:
    """Prepare raw input dict for prediction through the full pipeline."""
    row = {k: data[k] for k in [
        "campaign", "channel", "device", "impressions", "clicks",
        "spend", "conversions", "bounce_rate", "avg_session_duration",
    ]}
    row["revenue"] = data.get("revenue", 0.0)
    row["date"] = pd.Timestamp.now()

    df = pd.DataFrame([row])
    df = compute_kpis(df)
    df = create_derived_features(df)
    df = create_interaction_features(df)
    df = create_time_features(df)

    drop_cols = ["date", "roi", "roas", "revenue", "revenue_per_conversion", "target"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


def predict(data: dict, target: str = "high_roi") -> dict:
    """Run prediction and return structured result with interpretation."""
    input_df = prepare_input(data)
    pipeline = get_pipeline(target)

    prediction = int(pipeline.predict(input_df)[0])

    result = {
        "prediction": prediction,
        "target_type": target,
    }

    if target in ("high_roi", "high_conversion"):
        result["label"] = "Profitable" if prediction == 1 else "Unprofitable"

        confidence = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0]
            confidence = round(float(max(proba)), 4)
            result["confidence"] = confidence
            result["probabilities"] = {
                "class_0": round(float(proba[0]), 4),
                "class_1": round(float(proba[1]), 4),
            }

        result["interpretation"] = _generate_interpretation(prediction, confidence, data)
    else:
        result["predicted_revenue"] = round(float(prediction), 2)

    logger.info(
        "Prediction: %s | target=%s | campaign=%s | channel=%s",
        result.get("label", prediction), target,
        data.get("campaign"), data.get("channel"),
    )

    return result


def _generate_interpretation(prediction, confidence, data) -> dict:
    """Generate actionable business interpretation for a prediction."""
    spend = data.get("spend", 0)
    clicks = data.get("clicks", 0)
    impressions = data.get("impressions", 0)

    ctr = (clicks / impressions * 100) if impressions > 0 else 0
    cpc = (spend / clicks) if clicks > 0 else 0

    interpretation = {
        "ctr_estimate": round(ctr, 2),
        "cpc_estimate": round(cpc, 2),
    }

    if prediction == 1:
        interpretation["summary"] = "This campaign configuration is predicted to be PROFITABLE."
        if confidence and confidence > 0.85:
            interpretation["recommendation"] = (
                "Strong signal. Consider scaling spend by 15-25% "
                "while monitoring actual ROAS."
            )
        else:
            interpretation["recommendation"] = (
                "Moderate confidence. Run a small test budget first "
                "before committing larger spend."
            )
    else:
        interpretation["summary"] = "This campaign configuration is predicted to be UNPROFITABLE."
        if cpc > 3.0:
            interpretation["recommendation"] = (
                f"High CPC (${cpc:.2f}). Consider switching channels or "
                "refining audience targeting to reduce cost per click."
            )
        elif ctr < 1.0:
            interpretation["recommendation"] = (
                f"Low CTR ({ctr:.2f}%). Improve ad creative or targeting "
                "to increase engagement before scaling."
            )
        else:
            interpretation["recommendation"] = (
                "Review conversion funnel. Consider A/B testing landing pages "
                "and adjusting bid strategy."
            )

    return interpretation
