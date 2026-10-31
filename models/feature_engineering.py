"""
Feature engineering for the revenue prediction model.

Business question (Advert Firm A): "If we buy this inventory, how much
revenue will it earn, and will it make a margin?"

    target      log1p(revenue) of one row
                (one day × campaign × audience × banner × placement)
    inputs      only what is known BEFORE the ads run:
                campaign, engagement, banner, placement, weekday,
                day index, log(displays), log(cost)
    derived     predicted margin = expm1(prediction) - cost;
                profitable if predicted margin > 0

Assumption: displays and cost are fixed at purchase (inventory bought at a
CPM). Within a segment the CPM varies by ~17%, consistent with that.

Leakage prevention:
    - clicks, conversions, sales and every KPI derived from them are
      outcomes of the delivery and are never used as inputs
    - the split is by TIME (train on earlier days, test on later days);
      the data has a strong downward trend, so a random split would let
      the model learn from the future
    - encoding and scaling happen inside the sklearn Pipeline (trainer.py)
"""

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("feature_engineering")


TARGET_TYPE = "revenue"
DEFAULT_TEST_START = pd.Timestamp("2020-06-10")  # last 3 weeks held out
PERIOD_START = pd.Timestamp("2020-04-01")

CATEGORICAL_FEATURES = ["campaign_number", "user_engagement", "banner", "placement", "weekday"]
NUMERIC_FEATURES = ["log_displays", "log_cost", "day_index"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns that are outcomes of the ad delivery: never valid inputs
LEAKAGE_COLUMNS = [
    "clicks", "revenue", "post_click_conversions", "post_click_sales_amount",
    "firm_margin", "ctr", "conversion_rate", "firm_cpc", "firm_roi",
    "advertiser_cpc", "advertiser_cpa", "advertiser_roas", "avg_order_value",
    "flag_clicks_gt_displays", "flag_conv_gt_clicks", "flag_conv_without_clicks",
]

# Kept alongside the split for business evaluation (not model inputs)
EVAL_COLUMNS = ["date", "displays", "cost", "revenue", "firm_margin"] + CATEGORICAL_FEATURES[:4]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model input features from a cleaned frame (or a prediction request).

    Required columns: date, campaign_number, user_engagement, banner,
    placement, displays, cost. Used identically at training and at
    prediction time so the API/dashboard cannot drift from training.

    Parameters
    ----------
    df : pd.DataFrame
        Rows with the required columns.

    Returns
    -------
    pd.DataFrame
        Exactly FEATURE_COLUMNS, in order.
    """
    missing = {"date", "displays", "cost", *CATEGORICAL_FEATURES[:4]} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for feature building: {sorted(missing)}")

    dates = pd.to_datetime(df["date"])
    X = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_FEATURES[:4]:
        X[col] = df[col].astype(str)
    X["weekday"] = dates.dt.day_name()
    X["log_displays"] = np.log1p(df["displays"].astype(float))
    X["log_cost"] = np.log1p(df["cost"].astype(float))
    X["day_index"] = (dates - PERIOD_START).dt.days.astype(float)

    leaked = set(X.columns) & set(LEAKAGE_COLUMNS)
    assert not leaked, f"Leakage columns in features: {leaked}"
    return X[FEATURE_COLUMNS]


def target_to_revenue(y_log) -> np.ndarray:
    """Invert the log1p target transform, clipping at 0."""
    return np.clip(np.expm1(np.asarray(y_log, dtype=float)), 0, None)


def prepare_features(
    df: pd.DataFrame,
    test_start: pd.Timestamp = DEFAULT_TEST_START,
) -> dict:
    """
    Build features and a time-based train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with KPI columns (needs firm_margin).
    test_start : pd.Timestamp
        First day of the test period; everything earlier is training.

    Returns
    -------
    dict
        - X_train, X_test: raw (unencoded) features, sorted by date
        - y_train, y_test: log1p(revenue)
        - eval_train, eval_test: date/displays/cost/revenue/margin for
          baselines and business evaluation
        - numeric_features, categorical_features, feature_names
        - task_type ("regression"), target_type, time_ordered (True)
        - test_start, feature_count
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    print("\n[1/3] Building pre-delivery features...")
    X = build_features(df)
    y = np.log1p(df["revenue"].astype(float))
    print(f"  Categorical ({len(CATEGORICAL_FEATURES)}): {', '.join(CATEGORICAL_FEATURES)}")
    print(f"  Numeric ({len(NUMERIC_FEATURES)}): {', '.join(NUMERIC_FEATURES)}")
    print(f"  Excluded as leakage ({len(LEAKAGE_COLUMNS)}): {', '.join(LEAKAGE_COLUMNS)}")

    print("\n[2/3] Target: log1p(revenue)")
    print(f"  Revenue skewness {df['revenue'].skew():.1f} -> log target skewness {y.skew():.1f}")

    print(f"\n[3/3] Time-based split at {test_start.date()}...")
    train_mask = df["date"] < test_start
    if train_mask.all() or not train_mask.any():
        raise ValueError(f"test_start {test_start.date()} leaves an empty train or test set")

    split = {
        "X_train": X[train_mask], "X_test": X[~train_mask],
        "y_train": y[train_mask], "y_test": y[~train_mask],
        "eval_train": df.loc[train_mask, EVAL_COLUMNS],
        "eval_test": df.loc[~train_mask, EVAL_COLUMNS],
    }
    print(f"  Train: {len(split['X_train']):,} rows "
          f"({df.loc[train_mask, 'date'].min().date()} to {df.loc[train_mask, 'date'].max().date()})")
    print(f"  Test:  {len(split['X_test']):,} rows "
          f"({df.loc[~train_mask, 'date'].min().date()} to {df.loc[~train_mask, 'date'].max().date()})")
    print("=" * 60)

    logger.info("Feature prep complete: %d train, %d test, %d features",
                len(split["X_train"]), len(split["X_test"]), len(FEATURE_COLUMNS))

    return {
        **split,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names": FEATURE_COLUMNS,
        "task_type": "regression",
        "target_type": TARGET_TYPE,
        "time_ordered": True,
        "test_start": str(test_start.date()),
        "feature_count": len(FEATURE_COLUMNS),
    }
