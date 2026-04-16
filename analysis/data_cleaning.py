"""
Data cleaning and preprocessing pipeline.

Handles missing values, duplicates, type conversions,
and inconsistent data formats.
"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full cleaning pipeline to raw advertising data.

    Steps:
        1. Remove duplicate rows
        2. Standardize categorical values (campaign names, etc.)
        3. Handle missing values
        4. Fix data types
        5. Remove invalid records

    Parameters
    ----------
    df : pd.DataFrame
        Raw advertising data.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()
    initial_rows = len(df)

    print("\n" + "=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)

    # Step 1: Remove duplicates
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[1/5] Removed {n_dupes} duplicate rows")

    # Step 2: Standardize categorical values
    df = _standardize_categories(df)
    print("[2/5] Standardized categorical values")

    # Step 3: Handle missing values
    df = _handle_missing_values(df)
    print("[3/5] Handled missing values")

    # Step 4: Fix data types
    df = _fix_dtypes(df)
    print("[4/5] Fixed data types")

    # Step 5: Remove invalid records
    df = _remove_invalid_records(df)
    print("[5/5] Removed invalid records")

    final_rows = len(df)
    print(f"\nCleaning complete: {initial_rows} -> {final_rows} rows "
          f"({initial_rows - final_rows} removed)")
    print("=" * 60)

    return df


def _standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize campaign names, channel names, and device types."""
    # Fix inconsistent casing (e.g., "BRAND AWARENESS" → "Brand Awareness")
    df["campaign"] = df["campaign"].str.strip().str.title()
    df["channel"] = df["channel"].str.strip().str.title()
    df["device"] = df["device"].str.strip().str.title()

    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using appropriate strategies.

    - Numeric performance metrics: fill with median (robust to outliers)
    - Categorical: fill with mode
    """
    # Numeric columns: fill with median
    numeric_cols = ["impressions", "clicks", "spend", "conversions",
                    "revenue", "bounce_rate", "avg_session_duration"]
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            n_filled = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"    -> {col}: filled {n_filled} nulls with median ({median_val:.2f})")

    # Categorical columns: fill with mode
    cat_cols = ["campaign", "channel", "device"]
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            n_filled = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            print(f"    -> {col}: filled {n_filled} nulls with mode ({mode_val})")

    return df


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types for all columns."""
    # Integer columns
    int_cols = ["impressions", "clicks", "conversions"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Float columns
    float_cols = ["spend", "revenue", "bounce_rate", "avg_session_duration"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def _remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with logically impossible values."""
    initial = len(df)

    # Clicks cannot exceed impressions
    df = df[df["clicks"] <= df["impressions"]]

    # Spend, impressions, clicks must be non-negative
    df = df[(df["spend"] >= 0) & (df["impressions"] >= 0) & (df["clicks"] >= 0)]

    # Bounce rate must be between 0 and 1
    df = df[(df["bounce_rate"] >= 0) & (df["bounce_rate"] <= 1)]

    removed = initial - len(df)
    if removed > 0:
        print(f"    -> Removed {removed} logically invalid rows")

    return df.reset_index(drop=True)
