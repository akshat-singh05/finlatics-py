"""
Data cleaning and preprocessing pipeline.

Handles missing values, duplicates, type conversions, date construction
and data-quality flags for the advertising performance dataset.

Design notes:
    - Rows are never dropped for being extreme: the metrics are heavily
      right-skewed by nature (a few placements carry most of the volume).
    - Logically suspicious rows (e.g. more conversions than clicks) are
      FLAGGED, not removed. Conversions use a post-click attribution window,
      so they are not guaranteed to be bounded by same-day clicks.
"""

import pandas as pd

from analysis.data_loader import DATA_YEAR, NUMERIC_COLUMNS


# Each row is one day × campaign × audience × banner × placement combination
GRAIN_COLUMNS = ["date", "campaign_number", "user_engagement", "banner", "placement"]

ENGAGEMENT_ORDER = ["Low", "Medium", "High"]
UNKNOWN_PLACEMENT = "Unknown"


def clean_data(df: pd.DataFrame, return_report: bool = False):
    """
    Apply the full cleaning pipeline to raw advertising data.

    Steps:
        1. Standardize categorical values
        2. Handle missing values
        3. Remove exact duplicate rows
        4. Build a proper date column from month + day
        5. Validate numeric columns
        6. Add data-quality flags

    Parameters
    ----------
    df : pd.DataFrame
        Raw advertising data from ``load_data``.
    return_report : bool
        If True, also return a dict describing every change made.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, dict)
        Cleaned DataFrame (and the cleaning report if requested).

    Raises
    ------
    ValueError
        If dates cannot be built or numeric columns contain negatives.
    """
    df = df.copy()
    initial_rows = len(df)
    report = {"initial_rows": initial_rows}

    print("\n" + "=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)

    # Step 1: Standardize categorical values
    df = _standardize_categories(df)
    print("[1/6] Standardized categorical values")

    # Step 2: Handle missing values
    df, report["missing_placement_filled"] = _handle_missing_values(df)
    print("[2/6] Handled missing values")

    # Step 3: Remove exact duplicates
    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = n_dupes
    print(f"[3/6] Removed {n_dupes} exact duplicate rows")

    # Step 4: Build date column
    df = _build_date(df)
    print(f"[4/6] Built date column ({df['date'].min().date()} to {df['date'].max().date()})")

    # Step 5: Validate numeric columns
    df = _validate_numeric(df)
    print("[5/6] Validated numeric columns")

    # Step 6: Data-quality flags
    df, report["flags"] = _add_quality_flags(df)
    print("[6/6] Added data-quality flags")

    # Duplicates on the grain (after exact dedup) would mean conflicting records
    report["grain_duplicates"] = int(df.duplicated(GRAIN_COLUMNS).sum())
    if report["grain_duplicates"]:
        print(f"    -> [WARN] {report['grain_duplicates']} rows share a grain key "
              f"with different values")

    report["final_rows"] = len(df)
    print(f"\nCleaning complete: {initial_rows} -> {len(df)} rows "
          f"({initial_rows - len(df)} removed)")
    print("=" * 60)

    if return_report:
        return df, report
    return df


def _standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalize casing/spacing of categorical columns."""
    for col in ["month", "campaign_number", "user_engagement", "banner", "placement"]:
        df[col] = df[col].astype("string").str.strip()

    df["month"] = df["month"].str.title()
    df["user_engagement"] = df["user_engagement"].str.title()
    df["campaign_number"] = df["campaign_number"].str.lower()
    df["placement"] = df["placement"].str.lower()
    # "160x600" / "160 X 600" -> "160 x 600"
    df["banner"] = (df["banner"].str.lower()
                    .str.replace(r"\s*x\s*", " x ", regex=True))

    return df


def _handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Handle missing values.

    - placement: filled with the explicit level "Unknown". These rows carry
      real displays/cost, so dropping them would understate totals, and
      imputing the mode would invent a placement that was never recorded.
    - numeric columns: the source has none missing; any NaN is an error.
    """
    n_placement = int(df["placement"].isna().sum())
    if n_placement:
        df["placement"] = df["placement"].fillna(UNKNOWN_PLACEMENT)
        print(f"    -> placement: filled {n_placement} nulls with '{UNKNOWN_PLACEMENT}'")

    other_nulls = df.drop(columns=["placement"]).isna().sum()
    other_nulls = other_nulls[other_nulls > 0]
    if not other_nulls.empty:
        raise ValueError(f"Unexpected missing values: {other_nulls.to_dict()}")

    return df, n_placement


def _build_date(df: pd.DataFrame) -> pd.DataFrame:
    """Combine month name + day (+ DATA_YEAR) into a datetime column."""
    date_str = df["month"] + " " + df["day"].astype(str) + " " + str(DATA_YEAR)
    df.insert(0, "date", pd.to_datetime(date_str, format="%B %d %Y", errors="coerce"))

    n_bad = int(df["date"].isna().sum())
    if n_bad:
        raise ValueError(f"{n_bad} rows have an invalid month/day combination")

    df["weekday"] = df["date"].dt.day_name()
    df["user_engagement"] = pd.Categorical(
        df["user_engagement"], categories=ENGAGEMENT_ORDER, ordered=True
    )
    return df


def _validate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce numeric dtypes and reject negative values."""
    for col in ["displays", "clicks", "post_click_conversions"]:
        df[col] = df[col].astype("int64")
    for col in ["cost", "revenue", "post_click_sales_amount"]:
        df[col] = df[col].astype(float)

    negatives = (df[NUMERIC_COLUMNS] < 0).sum()
    negatives = negatives[negatives > 0]
    if not negatives.empty:
        raise ValueError(f"Negative values found: {negatives.to_dict()}")

    return df


def _add_quality_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Flag rows that are logically suspicious but kept for analysis.

    - flag_clicks_gt_displays: more clicks than ad displays
    - flag_conv_gt_clicks: more post-click conversions than clicks
    - flag_conv_without_clicks: conversions recorded on a row with 0 clicks
    """
    df["flag_clicks_gt_displays"] = df["clicks"] > df["displays"]
    df["flag_conv_gt_clicks"] = df["post_click_conversions"] > df["clicks"]
    df["flag_conv_without_clicks"] = (df["post_click_conversions"] > 0) & (df["clicks"] == 0)

    flags = {col: int(df[col].sum()) for col in df.columns if col.startswith("flag_")}
    for col, count in flags.items():
        if count:
            print(f"    -> {col}: {count} rows (kept, flagged)")

    return df, flags
