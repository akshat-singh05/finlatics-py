"""
Data loading and validation utilities.

Handles CSV ingestion, schema validation, and initial data inspection.
"""

import pandas as pd
from pathlib import Path


# Expected schema for validation
EXPECTED_COLUMNS = [
    "date", "campaign", "channel", "device",
    "impressions", "clicks", "spend",
    "conversions", "revenue", "bounce_rate", "avg_session_duration"
]


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load advertising performance data from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with parsed dates.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["date"])

    # Validate schema
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Loaded {len(df)} rows × {len(df.columns)} columns from {filepath.name}")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Perform initial data inspection and return a summary.

    Parameters
    ----------
    df : pd.DataFrame
        Raw advertising data.

    Returns
    -------
    dict
        Summary statistics including shape, dtypes, missing values, duplicates.
    """
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_summary": df.describe().to_dict(),
    }

    print("\n" + "=" * 60)
    print("DATA INSPECTION REPORT")
    print("=" * 60)
    print(f"Shape: {summary['shape'][0]} rows × {summary['shape'][1]} columns")
    print(f"Duplicates: {summary['duplicates']}")
    print(f"\nMissing Values:")
    for col, count in summary["missing_values"].items():
        if count > 0:
            pct = summary["missing_pct"][col]
            print(f"  {col}: {count} ({pct}%)")
    print(f"\nData Types:")
    for col, dtype in summary["dtypes"].items():
        print(f"  {col}: {dtype}")
    print("=" * 60)

    return summary
