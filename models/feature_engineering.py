"""
Advanced feature engineering for machine learning.

Provides a reusable, pipeline-oriented approach to feature creation:
  - Derived KPI features (CTR, CPC, Conversion Rate, Cost per Conversion)
  - Interaction features (campaign x channel, channel x device)
  - Time-based features (day, month, weekday, is_weekend)
  - Leakage-safe target creation and column management

NOTE: Categorical encoding and feature scaling are handled INSIDE
the sklearn Pipeline (in trainer.py) to prevent data leakage during
cross-validation and hyperparameter search.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# =====================================================================
# LEAKAGE PREVENTION CONFIG
# =====================================================================

LEAKAGE_MAP = {
    "high_conversion": [
        "conversions", "conversion_rate", "revenue",
        "revenue_per_conversion", "roas", "roi",
        "cost_per_conversion",
    ],
    "revenue": [
        "roas", "roi", "revenue_per_conversion",
    ],
    "high_roi": [
        "roi", "roas", "revenue", "revenue_per_conversion",
    ],
}

DROP_ALWAYS = ["date"]


# =====================================================================
# 1. TARGET VARIABLE CREATION
# =====================================================================

def create_target(df: pd.DataFrame, target_type: str = "high_roi") -> pd.DataFrame:
    """
    Create the target variable based on the chosen prediction task.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with KPI columns already computed.
    target_type : str
        One of:
        - "high_conversion": binary, 1 if conversions > median
        - "revenue": continuous revenue value (regression)
        - "high_roi": binary, 1 if ROI > 0 (profitable)

    Returns
    -------
    pd.DataFrame
        DataFrame with a 'target' column added.
    """
    df = df.copy()

    if target_type == "high_conversion":
        median_conv = df["conversions"].median()
        df["target"] = (df["conversions"] > median_conv).astype(int)
        print(f"  Target: high_conversion (conversions > {median_conv})")
        print(f"  Class distribution: {df['target'].value_counts().to_dict()}")

    elif target_type == "revenue":
        df["target"] = df["revenue"]
        print(f"  Target: revenue (continuous)")
        print(f"  Range: ${df['target'].min():.2f} - ${df['target'].max():.2f}")
        print(f"  Mean:  ${df['target'].mean():.2f}")

    elif target_type == "high_roi":
        df["target"] = (df["roi"] > 0).astype(int)
        print(f"  Target: high_roi (ROI > 0 = profitable)")
        print(f"  Class distribution: {df['target'].value_counts().to_dict()}")

    else:
        raise ValueError(f"Unknown target_type: {target_type}. "
                         f"Use 'high_conversion', 'revenue', or 'high_roi'.")

    return df


# =====================================================================
# 2. DERIVED FEATURES
# =====================================================================

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived KPI features from raw advertising metrics.

    New columns (added only if not already present):
        - ctr, cpc, cpm, conversion_rate
        - cost_per_conversion, revenue_per_click
        - bounce_click_ratio, impression_conversion_rate

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with raw metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame with new derived columns.
    """
    df = df.copy()
    count = 0

    if "ctr" not in df.columns:
        df["ctr"] = np.where(df["impressions"] > 0,
                             df["clicks"] / df["impressions"], 0).round(6)
        count += 1

    if "cpc" not in df.columns:
        df["cpc"] = np.where(df["clicks"] > 0,
                             df["spend"] / df["clicks"], 0).round(4)
        count += 1

    if "cpm" not in df.columns:
        df["cpm"] = np.where(df["impressions"] > 0,
                             (df["spend"] / df["impressions"]) * 1000, 0).round(4)
        count += 1

    if "conversion_rate" not in df.columns:
        df["conversion_rate"] = np.where(df["clicks"] > 0,
                                         df["conversions"] / df["clicks"], 0).round(6)
        count += 1

    if "cost_per_conversion" not in df.columns:
        df["cost_per_conversion"] = np.where(
            df["conversions"] > 0, df["spend"] / df["conversions"], 0
        ).round(4)
        count += 1

    if "revenue_per_click" not in df.columns and "revenue" in df.columns:
        df["revenue_per_click"] = np.where(
            df["clicks"] > 0, df["revenue"] / df["clicks"], 0
        ).round(4)
        count += 1

    if "bounce_rate" in df.columns and "clicks" in df.columns:
        df["bounce_click_ratio"] = (df["bounce_rate"] * np.log1p(df["clicks"])).round(4)
        count += 1

    df["impression_conversion_rate"] = np.where(
        df["impressions"] > 0, df["conversions"] / df["impressions"], 0
    ).round(8)
    count += 1

    print(f"  Created {count} derived features")
    return df


# =====================================================================
# 3. INTERACTION FEATURES
# =====================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create categorical interaction features.

    New columns: campaign_channel, channel_device, campaign_device

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with categorical columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with interaction columns.
    """
    df = df.copy()
    count = 0

    if "campaign" in df.columns and "channel" in df.columns:
        df["campaign_channel"] = df["campaign"] + "_" + df["channel"]
        count += 1

    if "channel" in df.columns and "device" in df.columns:
        df["channel_device"] = df["channel"] + "_" + df["device"]
        count += 1

    if "campaign" in df.columns and "device" in df.columns:
        df["campaign_device"] = df["campaign"] + "_" + df["device"]
        count += 1

    print(f"  Created {count} interaction features")
    return df


# =====================================================================
# 4. TIME-BASED FEATURES
# =====================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the date column.

    New columns: day, month, weekday, is_weekend, day_of_year, week_of_year

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with time-based features.
    """
    df = df.copy()

    if "date" not in df.columns:
        print("  [SKIP] No 'date' column found")
        return df

    dt = pd.to_datetime(df["date"])

    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["weekday"] = dt.dt.weekday
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    print(f"  Created 6 time-based features")
    return df


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def prepare_features(
    df: pd.DataFrame,
    target_type: str = "high_roi",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Feature preparation pipeline (encoding/scaling deferred to sklearn Pipeline).

    Steps:
        1. Create target variable
        2. Create derived features (KPIs)
        3. Create interaction features
        4. Create time-based features
        5. Drop leakage and identifier columns
        6. Identify column types for Pipeline preprocessing
        7. Train/test split (on raw features — no leakage)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with KPIs computed.
    target_type : str
        Target variable type.
    test_size : float
        Fraction of data for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        - X_train, X_test, y_train, y_test: raw (unscaled) split data
        - numeric_features, categorical_features: column lists for Pipeline
        - task_type, target_type: metadata
        - feature_count: total features
    """
    print("\n" + "=" * 60)
    print("ADVANCED FEATURE ENGINEERING")
    print("=" * 60)

    # Step 1: Create target
    print("\n[1/7] Creating target variable...")
    df = create_target(df, target_type)

    # Step 2: Derived features
    print("\n[2/7] Creating derived features...")
    df = create_derived_features(df)

    # Step 3: Interaction features
    print("\n[3/7] Creating interaction features...")
    df = create_interaction_features(df)

    # Step 4: Time-based features
    print("\n[4/7] Extracting time-based features...")
    df = create_time_features(df)

    # Step 5: Drop leakage & identifier columns
    print("\n[5/7] Removing leakage & identifier columns...")
    leakage_cols = LEAKAGE_MAP.get(target_type, [])
    drop_cols = list(set(DROP_ALWAYS + leakage_cols + ["target"]))
    available_drop = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    print(f"  Dropped ({len(available_drop)}): {', '.join(sorted(available_drop))}")

    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Step 6: Identify column types for the sklearn Pipeline
    print("\n[6/7] Identifying column types for Pipeline...")
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"  Numeric ({len(numeric_features)}): {', '.join(numeric_features)}")
    print(f"  Categorical ({len(categorical_features)}): {', '.join(categorical_features)}")

    # Step 7: Train/test split (BEFORE encoding/scaling to prevent leakage)
    print(f"\n[7/7] Splitting data ({int((1-test_size)*100)}/{int(test_size*100)} "
          f"train/test, seed={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if target_type != "revenue" else None,
    )

    task_type = "regression" if target_type == "revenue" else "classification"
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}")
    print(f"  Task type: {task_type}")
    print("=" * 60)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names": feature_cols,
        "task_type": task_type,
        "target_type": target_type,
        "feature_count": len(feature_cols),
    }
