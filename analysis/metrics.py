"""
Advertising performance metrics and KPI computations.

Derives CTR, CPC, CPM, Conversion Rate, ROAS, ROI and provides
aggregation utilities for campaign/channel/device analysis.
"""

import pandas as pd
import numpy as np


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived KPI columns to the DataFrame.

    New columns:
        - ctr: Click-Through Rate (clicks / impressions)
        - cpc: Cost Per Click (spend / clicks)
        - cpm: Cost Per Mille (spend / impressions × 1000)
        - conversion_rate: conversions / clicks
        - roas: Return on Ad Spend (revenue / spend)
        - roi: Return on Investment ((revenue - spend) / spend)
        - revenue_per_conversion: revenue / conversions

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned advertising data.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional KPI columns.
    """
    df = df.copy()

    # Avoid division by zero with np.where
    df["ctr"] = np.where(df["impressions"] > 0,
                         df["clicks"] / df["impressions"], 0).round(6)

    df["cpc"] = np.where(df["clicks"] > 0,
                         df["spend"] / df["clicks"], 0).round(4)

    df["cpm"] = np.where(df["impressions"] > 0,
                         (df["spend"] / df["impressions"]) * 1000, 0).round(4)

    df["conversion_rate"] = np.where(df["clicks"] > 0,
                                     df["conversions"] / df["clicks"], 0).round(6)

    df["roas"] = np.where(df["spend"] > 0,
                          df["revenue"] / df["spend"], 0).round(4)

    df["roi"] = np.where(df["spend"] > 0,
                         (df["revenue"] - df["spend"]) / df["spend"], 0).round(4)

    df["revenue_per_conversion"] = np.where(df["conversions"] > 0,
                                            df["revenue"] / df["conversions"], 0).round(2)

    print(f"\nComputed 7 KPI columns: ctr, cpc, cpm, conversion_rate, roas, roi, revenue_per_conversion")
    return df


def summarize_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Aggregate performance metrics grouped by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with KPI columns.
    group_col : str
        Column to group by (e.g., 'campaign', 'channel', 'device').

    Returns
    -------
    pd.DataFrame
        Aggregated summary table sorted by total spend descending.
    """
    agg = df.groupby(group_col).agg(
        total_impressions=("impressions", "sum"),
        total_clicks=("clicks", "sum"),
        total_spend=("spend", "sum"),
        total_conversions=("conversions", "sum"),
        total_revenue=("revenue", "sum"),
        avg_ctr=("ctr", "mean"),
        avg_cpc=("cpc", "mean"),
        avg_conversion_rate=("conversion_rate", "mean"),
        avg_roas=("roas", "mean"),
        avg_roi=("roi", "mean"),
        avg_bounce_rate=("bounce_rate", "mean"),
        record_count=("impressions", "count"),
    ).round(4)

    agg = agg.sort_values("total_spend", ascending=False)
    return agg


def get_overall_summary(df: pd.DataFrame) -> dict:
    """
    Compute high-level summary statistics across the entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with KPI columns.

    Returns
    -------
    dict
        Dictionary of overall performance metrics.
    """
    summary = {
        "total_records": len(df),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "total_impressions": int(df["impressions"].sum()),
        "total_clicks": int(df["clicks"].sum()),
        "total_spend": round(df["spend"].sum(), 2),
        "total_conversions": int(df["conversions"].sum()),
        "total_revenue": round(df["revenue"].sum(), 2),
        "overall_ctr": round(df["clicks"].sum() / df["impressions"].sum(), 6),
        "overall_cpc": round(df["spend"].sum() / df["clicks"].sum(), 4),
        "overall_roas": round(df["revenue"].sum() / df["spend"].sum(), 4),
        "overall_roi": round(
            (df["revenue"].sum() - df["spend"].sum()) / df["spend"].sum(), 4
        ),
        "unique_campaigns": df["campaign"].nunique(),
        "unique_channels": df["channel"].nunique(),
    }

    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 60)
    for key, val in summary.items():
        label = key.replace("_", " ").title()
        if isinstance(val, float):
            print(f"  {label}: {val:,.4f}")
        elif isinstance(val, int):
            print(f"  {label}: {val:,}")
        else:
            print(f"  {label}: {val}")
    print("=" * 60)

    return summary
