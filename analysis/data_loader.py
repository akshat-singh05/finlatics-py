"""
Data loading and validation utilities.

Handles CSV ingestion, schema validation, and initial data inspection
for the Finlatics online advertising performance dataset
(Company X campaigns run through Advert Firm A, April-June 2020).
"""

import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "online_advertising_performance_data.csv"

# The file stores month names and day numbers only; the dataset description
# places the campaigns in 2020.
DATA_YEAR = 2020

CATEGORICAL_COLUMNS = ["month", "campaign_number", "user_engagement", "banner", "placement"]
NUMERIC_COLUMNS = [
    "displays", "cost", "clicks", "revenue",
    "post_click_conversions", "post_click_sales_amount",
]

# Expected schema for validation
EXPECTED_COLUMNS = ["month", "day"] + CATEGORICAL_COLUMNS[1:] + NUMERIC_COLUMNS


def load_data(filepath: str | Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Load the raw advertising performance data from a CSV file.

    The source export ends every line with two trailing commas, which pandas
    reads as two fully empty ``Unnamed`` columns; those are dropped here.
    No other transformation is applied.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file. Defaults to ``data/raw/``.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame restricted to the expected columns.

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

    df = pd.read_csv(filepath)

    # Drop export artifacts: unnamed columns with no values at all
    empty_unnamed = [c for c in df.columns
                     if str(c).startswith("Unnamed") and df[c].isna().all()]
    df = df.drop(columns=empty_unnamed)

    # Validate schema
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra_cols:
        print(f"  [WARN] Ignoring unexpected columns: {extra_cols}")

    df = df[EXPECTED_COLUMNS]

    print(f"Loaded {len(df)} rows × {len(df.columns)} columns from {filepath.name}"
          + (f" (dropped {len(empty_unnamed)} empty columns)" if empty_unnamed else ""))
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
        Summary including shape, dtypes, missing values, duplicates,
        category levels and numeric statistics.
    """
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "category_levels": {
            col: df[col].value_counts(dropna=False).to_dict()
            for col in CATEGORICAL_COLUMNS if col in df.columns
        },
        "numeric_summary": df.describe().to_dict(),
    }

    print("\n" + "=" * 60)
    print("DATA INSPECTION REPORT")
    print("=" * 60)
    print(f"Shape: {summary['shape'][0]} rows × {summary['shape'][1]} columns")
    print(f"Exact duplicate rows: {summary['duplicates']}")
    print(f"\nMissing Values:")
    any_missing = False
    for col, count in summary["missing_values"].items():
        if count > 0:
            any_missing = True
            pct = summary["missing_pct"][col]
            print(f"  {col}: {count} ({pct}%)")
    if not any_missing:
        print("  none")
    print(f"\nCategory Levels:")
    for col, levels in summary["category_levels"].items():
        print(f"  {col} ({len(levels)}): {', '.join(str(k) for k in levels)}")
    print(f"\nData Types:")
    for col, dtype in summary["dtypes"].items():
        print(f"  {col}: {dtype}")
    print("=" * 60)

    return summary
