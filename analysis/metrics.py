"""
Advertising performance metrics and KPI computations.

Column semantics (from the dataset description):
    - displays: ad impressions served
    - clicks: clicks on the ad
    - cost: what Advert Firm A paid publishers for the placements
    - revenue: what Company X paid Advert Firm A — revenue for the firm,
      advertising SPEND for Company X
    - post_click_conversions / post_click_sales_amount: transactions and
      sales attributed to the ad within the post-click window

So there are two perspectives:
    - Advert Firm A: margin = revenue - cost, ROI = margin / cost
    - Company X:     ROAS = attributed sales / revenue (its ad spend)

Company X's profit ROI cannot be computed: product costs are not in the data.

All grouped rates divide total numerators by total denominators (never the
mean of row-level ratios), and a zero denominator yields NaN, not 0.
"""

import pandas as pd
import numpy as np


# name -> (numerator column, denominator column, multiplier)
RATIO_DEFINITIONS = {
    "ctr": ("clicks", "displays", 1),
    "cpm": ("cost", "displays", 1000),
    "conversion_rate": ("post_click_conversions", "clicks", 1),
    "firm_cpc": ("cost", "clicks", 1),
    "firm_roi": ("firm_margin", "cost", 1),
    "advertiser_cpc": ("revenue", "clicks", 1),
    "advertiser_cpa": ("revenue", "post_click_conversions", 1),
    "advertiser_roas": ("post_click_sales_amount", "revenue", 1),
    "avg_order_value": ("post_click_sales_amount", "post_click_conversions", 1),
}

SUM_COLUMNS = [
    "displays", "clicks", "cost", "revenue", "firm_margin",
    "post_click_conversions", "post_click_sales_amount",
]


def _safe_ratio(numerator, denominator, multiplier: float = 1):
    """Divide, returning NaN where the denominator is zero."""
    if isinstance(denominator, pd.Series):
        return numerator / denominator.where(denominator != 0) * multiplier
    return numerator / denominator * multiplier if denominator else np.nan


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add row-level KPI columns to the DataFrame.

    New columns:
        - firm_margin: revenue - cost (Advert Firm A)
        - ctr: clicks / displays
        - cpm: cost / displays × 1000
        - conversion_rate: post_click_conversions / clicks (can exceed 1,
          since conversions are attributed over a window)
        - firm_cpc: cost / clicks
        - firm_roi: firm_margin / cost
        - advertiser_cpc: revenue / clicks (Company X's cost per click)
        - advertiser_cpa: revenue / post_click_conversions
        - advertiser_roas: post_click_sales_amount / revenue
        - avg_order_value: post_click_sales_amount / post_click_conversions

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned advertising data.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional KPI columns (NaN where undefined).
    """
    df = df.copy()
    df["firm_margin"] = df["revenue"] - df["cost"]

    for name, (num, den, mult) in RATIO_DEFINITIONS.items():
        df[name] = _safe_ratio(df[num], df[den], mult)

    print(f"\nComputed {len(RATIO_DEFINITIONS) + 1} KPI columns: firm_margin, "
          f"{', '.join(RATIO_DEFINITIONS)}")
    return df


def _ratios_from_totals(totals: pd.DataFrame | pd.Series):
    """Compute every KPI ratio from summed columns."""
    out = totals.copy()
    for name, (num, den, mult) in RATIO_DEFINITIONS.items():
        out[name] = _safe_ratio(totals[num], totals[den], mult)
    return out


def summarize_by(df: pd.DataFrame, group_col: str | list[str]) -> pd.DataFrame:
    """
    Aggregate performance metrics grouped by one or more columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``firm_margin`` computed (see ``compute_kpis``).
    group_col : str or list of str
        Column(s) to group by (e.g. 'campaign_number', 'banner',
        'placement', 'user_engagement', 'date').

    Returns
    -------
    pd.DataFrame
        Totals, ratio-of-totals KPIs, record count, active days and share of
        total revenue, sorted by revenue descending.
    """
    grouped = df.groupby(group_col, observed=True)
    totals = grouped[SUM_COLUMNS].sum()
    totals["record_count"] = grouped.size()
    totals["active_days"] = grouped["date"].nunique()

    summary = _ratios_from_totals(totals)
    summary["revenue_share"] = summary["revenue"] / df["revenue"].sum()

    return summary.sort_values("revenue", ascending=False)


def get_overall_summary(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Compute high-level summary statistics across the entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``firm_margin`` computed.
    verbose : bool
        Print the summary to the console.

    Returns
    -------
    dict
        Dictionary of overall performance metrics.
    """
    totals = _ratios_from_totals(df[SUM_COLUMNS].sum())

    summary = {
        "total_records": len(df),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "days": int(df["date"].nunique()),
        "campaigns": int(df["campaign_number"].nunique()),
        "banners": int(df["banner"].nunique()),
        "placements": int(df["placement"].nunique()),
    }
    for col in SUM_COLUMNS:
        val = totals[col]
        summary[f"total_{col}"] = int(val) if col in ("displays", "clicks",
                                                    "post_click_conversions") else round(float(val), 2)
    for name in RATIO_DEFINITIONS:
        summary[name] = round(float(totals[name]), 6)

    if verbose:
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
