"""
Tests for ML feature preparation, baselines and business metrics.

These need only pandas/numpy/matplotlib, so they run without scikit-learn.
Training itself is exercised by ``python main.py``.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.data_loader import DEFAULT_RAW_PATH, load_data
from analysis.data_cleaning import clean_data
from analysis.metrics import compute_kpis
from models.feature_engineering import (
    FEATURE_COLUMNS, LEAKAGE_COLUMNS, build_features, prepare_features, target_to_revenue,
)
from models.business_evaluation import (
    classification_metrics, margin_captured, regression_metrics, roc_auc,
    segment_rate_baseline,
)


def _frame(n_days=10, rows_per_day=4):
    """Small cleaned-looking frame with a known revenue-per-display rate."""
    rows = []
    for d in range(n_days):
        for k in range(rows_per_day):
            displays = 1000 * (k + 1)
            rate = 0.01 if k % 2 == 0 else 0.02          # two segments, two rates
            revenue = displays * rate * (1 - 0.05 * d)    # gentle downward drift
            cost = displays * 0.012
            rows.append({
                "date": pd.Timestamp("2020-04-01") + pd.Timedelta(days=d),
                "campaign_number": "camp 1", "user_engagement": "High",
                "banner": "160 x 600" if k % 2 == 0 else "240 x 400",
                "placement": "abc", "displays": displays, "cost": cost,
                "revenue": revenue, "clicks": 10, "firm_margin": revenue - cost,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def test_build_features_has_no_leakage_columns():
    X = build_features(_frame())
    assert list(X.columns) == FEATURE_COLUMNS
    assert not set(X.columns) & set(LEAKAGE_COLUMNS)


def test_build_features_is_identical_for_training_and_requests():
    df = _frame()
    request = df.drop(columns=["revenue", "clicks", "firm_margin"]).iloc[[3]]
    pd.testing.assert_frame_equal(build_features(df).iloc[[3]], build_features(request))


def test_build_features_rejects_missing_inputs():
    with pytest.raises(ValueError, match="Missing columns"):
        build_features(_frame().drop(columns=["cost"]))


def test_prepare_features_splits_by_time():
    data = prepare_features(_frame(), test_start=pd.Timestamp("2020-04-08"))
    assert data["eval_train"]["date"].max() < pd.Timestamp("2020-04-08")
    assert data["eval_test"]["date"].min() >= pd.Timestamp("2020-04-08")
    assert data["X_train"]["day_index"].is_monotonic_increasing
    assert len(data["X_train"]) + len(data["X_test"]) == len(_frame())
    np.testing.assert_allclose(target_to_revenue(data["y_test"]),
                               data["eval_test"]["revenue"], rtol=1e-9)


def test_prepare_features_rejects_empty_split():
    with pytest.raises(ValueError, match="empty"):
        prepare_features(_frame(), test_start=pd.Timestamp("2021-01-01"))


# ---------------------------------------------------------------------
# Baselines and metrics
# ---------------------------------------------------------------------

def test_segment_rate_baseline_recovers_known_rates():
    df = _frame(n_days=1)
    pred = segment_rate_baseline(df, df)
    # Within one day each segment has a single rate, so prediction is exact
    np.testing.assert_allclose(pred, df["revenue"], rtol=1e-9)


def test_recent_baseline_tracks_drift_better_than_full_history():
    df = _frame(n_days=10)
    train, test = df[df["date"] < "2020-04-08"], df[df["date"] >= "2020-04-08"]
    full = segment_rate_baseline(train, test)
    recent = segment_rate_baseline(train, test, recent_days=2)
    err = lambda p: np.abs(p - test["revenue"]).mean()
    assert err(recent) < err(full)


def test_roc_auc_matches_known_values():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))


def test_classification_metrics_confusion():
    m = classification_metrics([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(0.6)


def test_regression_metrics_perfect_prediction():
    m = regression_metrics([0, 1, 10, 100], [0, 1, 10, 100])
    assert m["r2_log"] == pytest.approx(1.0)
    assert m["mae_usd"] == 0
    assert m["total_predicted_vs_actual"] == pytest.approx(1.0)


def test_margin_captured():
    m = margin_captured([5, -3, 2, -1], [True, False, True, True])
    assert m["buy_predicted_profitable"] == 6
    assert m["buy_everything"] == 3
    assert m["oracle"] == 7
    assert m["share_of_oracle"] == pytest.approx(6 / 7)


# ---------------------------------------------------------------------
# Real dataset (integration)
# ---------------------------------------------------------------------

@pytest.mark.skipif(not DEFAULT_RAW_PATH.exists(), reason="raw dataset not present")
def test_real_dataset_split_and_baseline():
    df = compute_kpis(clean_data(load_data()))
    data = prepare_features(df)
    assert len(data["X_train"]) == 12373
    assert len(data["X_test"]) == 3030
    assert data["X_train"].isna().sum().sum() == 0

    pred = segment_rate_baseline(data["eval_train"], data["eval_test"], recent_days=21)
    assert regression_metrics(data["eval_test"]["revenue"], pred)["r2_log"] > 0.9
