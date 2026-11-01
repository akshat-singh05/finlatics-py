"""
Business insights engine.

Turns the cleaned, KPI-enriched dataset into a structured set of
quantified findings (one block per EDA question). Every number is
computed from the data; nothing is hardcoded. The findings are printed
as an executive summary and saved as JSON so the report, dashboard and
API can all reuse the same figures.

Perspectives (see analysis/metrics.py):
    - Advert Firm A earns `revenue` and pays `cost` -> margin / ROI
    - Company X spends `revenue` and gets attributed sales -> ROAS
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.metrics import summarize_by, get_overall_summary


# Segments with fewer clicks than this are too thin to rank on rates
MIN_CLICKS_FOR_RATE_RANKING = 1000

KEY_COLUMNS = ["revenue", "cost", "firm_margin", "post_click_sales_amount",
               "displays", "clicks", "post_click_conversions",
               "ctr", "conversion_rate", "firm_roi", "advertiser_roas",
               "record_count", "active_days", "revenue_share"]


def generate_insights(df: pd.DataFrame, output_dir: str | Path = "reports") -> dict:
    """
    Compute all EDA findings, print an executive summary and save JSON.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with KPI columns.
    output_dir : str or Path
        Directory for ``eda_findings.json``.

    Returns
    -------
    dict
        Findings keyed by topic.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)

    findings = {
        "overall": get_overall_summary(df, verbose=False),
        "time": _time_trends(df),
        "campaigns": _campaigns(df),
        "engagement": _engagement(df),
        "banners": _dimension(df, "banner"),
        "placements": _dimension(df, "placement"),
        "segments": _segments(df),
        "relationships": _relationships(df),
        "distribution": _distribution(df),
        "weekday": _weekday(df),
        "data_quality": _data_quality(df),
    }

    _print_executive_summary(findings)

    path = output_dir / "eda_findings.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(findings), f, indent=2)
    print(f"\n  Findings saved: {path}")
    print("=" * 60)

    return findings


# =====================================================================
# FINDINGS
# =====================================================================

def _records(summary: pd.DataFrame) -> dict:
    """Summary table -> {group: {metric: value}} for the key columns."""
    cols = [c for c in KEY_COLUMNS if c in summary.columns]
    return summary[cols].round(6).to_dict(orient="index")


def _time_trends(df: pd.DataFrame) -> dict:
    """Q1-Q2: daily trend, monthly comparison and the campaign-mix effect."""
    daily = summarize_by(df, "date").sort_index()
    month_order = ["April", "May", "June"]
    monthly = summarize_by(df, "month").reindex(month_order)
    without_camp2 = summarize_by(df[df["campaign_number"] != "camp 2"],
                                 "month").reindex(month_order)

    first, last = daily.iloc[:14], daily.iloc[-14:]
    negative_days = daily[daily["firm_margin"] < 0]

    return {
        "first_14_days_avg_daily_revenue": first["revenue"].mean(),
        "last_14_days_avg_daily_revenue": last["revenue"].mean(),
        "first_14_days_avg_daily_clicks": first["clicks"].mean(),
        "last_14_days_avg_daily_clicks": last["clicks"].mean(),
        "revenue_change_first_to_last_14_days": last["revenue"].mean() / first["revenue"].mean() - 1,
        "monthly": _records(monthly),
        "monthly_excluding_camp2": _records(without_camp2),
        "april_to_june_revenue_change_excl_camp2":
            without_camp2.loc["June", "revenue"] / without_camp2.loc["April", "revenue"] - 1,
        "days_with_negative_firm_margin": len(negative_days),
        "first_negative_margin_day": (str(negative_days.index.min().date())
                                      if len(negative_days) else None),
    }


def _campaigns(df: pd.DataFrame) -> dict:
    """Q8: campaign comparison (camp 2 only ran in April -> compare rates)."""
    s = summarize_by(df, "campaign_number")
    dates = df.groupby("campaign_number")["date"].agg(["min", "max"])
    out = _records(s)
    for camp in out:
        out[camp]["first_date"] = str(dates.loc[camp, "min"].date())
        out[camp]["last_date"] = str(dates.loc[camp, "max"].date())
    return out


def _engagement(df: pd.DataFrame) -> dict:
    """Q3/Q8: engagement levels overall and within each campaign."""
    return {
        "overall": _records(summarize_by(df, "user_engagement")),
        "by_campaign": {f"{c} | {e}": v for (c, e), v in
                        _records(summarize_by(df, ["campaign_number", "user_engagement"])).items()},
    }


def _dimension(df: pd.DataFrame, dim: str) -> dict:
    """Q4/Q5: volume leaders, CTR leaders (with enough clicks) and loss-makers."""
    s = summarize_by(df, dim)
    rankable = s[s["clicks"] >= MIN_CLICKS_FOR_RATE_RANKING]
    return {
        "summary": _records(s),
        "most_displays": s["displays"].idxmax(),
        "most_clicks": s["clicks"].idxmax(),
        "highest_ctr": rankable["ctr"].idxmax(),
        "lowest_ctr": rankable["ctr"].idxmin(),
        "highest_firm_roi": rankable["firm_roi"].idxmax(),
        "loss_making": s.index[s["firm_margin"] < 0].tolist(),
    }


def _segments(df: pd.DataFrame) -> dict:
    """Q9: banner × placement combinations that drive or destroy firm margin."""
    s = summarize_by(df, ["banner", "placement"])
    s.index = [f"{b} @ {p}" for b, p in s.index]
    total_margin = s["firm_margin"].sum()
    top = s.sort_values("firm_margin", ascending=False)
    losses = s[s["firm_margin"] < 0].sort_values("firm_margin")

    return {
        "top_5_by_margin": _records(top.head(5)),
        "top_segment_margin_share": top["firm_margin"].iloc[0] / total_margin,
        "loss_making_count": len(losses),
        "loss_making_total_margin": losses["firm_margin"].sum(),
        "worst_5_by_margin": _records(losses.head(5)),
    }


def _relationships(df: pd.DataFrame) -> dict:
    """Q6/Q7/Q12: correlations (Pearson is skew-sensitive, Spearman is not)."""
    pairs = [("cost", "revenue"), ("clicks", "revenue"), ("displays", "clicks"),
             ("clicks", "post_click_conversions"),
             ("post_click_conversions", "post_click_sales_amount"),
             ("revenue", "post_click_sales_amount")]
    return {
        f"{a}~{b}": {
            "pearson": df[a].corr(df[b]),
            "spearman": df[a].corr(df[b], method="spearman"),
        }
        for a, b in pairs
    }


def _distribution(df: pd.DataFrame) -> dict:
    """Q10: skewness, zero-inflation and revenue concentration."""
    rev = df["revenue"].sort_values(ascending=False)
    share = rev.cumsum() / rev.sum()
    n = len(rev)
    return {
        "skewness": df[["displays", "clicks", "cost", "revenue"]].skew().to_dict(),
        "zero_share": {c: float((df[c] == 0).mean())
                       for c in ["clicks", "revenue", "post_click_conversions"]},
        "top_1pct_rows_revenue_share": share.iloc[int(n * 0.01) - 1],
        "top_10pct_rows_revenue_share": share.iloc[int(n * 0.10) - 1],
        "rows_with_negative_firm_margin_share": float((df["firm_margin"] < 0).mean()),
    }


def _weekday(df: pd.DataFrame) -> dict:
    """Q11: per-day averages by weekday (each weekday occurs 13 times)."""
    s = summarize_by(df, "weekday")
    per_day = s["revenue"] / s["active_days"]
    return {
        "avg_daily_revenue": per_day.to_dict(),
        "avg_daily_clicks": (s["clicks"] / s["active_days"]).to_dict(),
        "ctr": s["ctr"].to_dict(),
        "best_day": per_day.idxmax(),
        "worst_day": per_day.idxmin(),
        "best_to_worst_ratio": per_day.max() / per_day.min(),
    }


def _data_quality(df: pd.DataFrame) -> dict:
    """Counts of flagged rows kept in the analysis."""
    return {col: int(df[col].sum()) for col in df.columns if col.startswith("flag_")}


# =====================================================================
# OUTPUT
# =====================================================================

def _to_jsonable(obj):
    """Recursively convert numpy/pandas scalars and NaN for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 6)
    return obj


def _print_executive_summary(f: dict):
    """Print the headline findings."""
    o, t, seg, dist = f["overall"], f["time"], f["segments"], f["distribution"]
    camps = f["campaigns"]
    rel = f["relationships"]["cost~revenue"]

    lines = [
        f"Period {o['date_range']}: Company X paid ${o['total_revenue']:,.0f}; placements "
        f"cost Advert Firm A ${o['total_cost']:,.0f} (firm ROI {o['firm_roi']:.0%}).",
        f"Activity declined steadily: avg daily revenue fell from "
        f"${t['first_14_days_avg_daily_revenue']:,.0f} (first 14 days) to "
        f"${t['last_14_days_avg_daily_revenue']:,.0f} (last 14 days); even excluding "
        f"camp 2, April->June revenue changed {t['april_to_june_revenue_change_excl_camp2']:.0%}.",
        f"Firm margin turned negative on {t['days_with_negative_firm_margin']} days, "
        f"starting {t['first_negative_margin_day']}.",
        f"camp 1 generated {camps['camp 1']['revenue_share']:.0%} of revenue; camp 2 ran "
        f"{camps['camp 2']['first_date']} to {camps['camp 2']['last_date']} only.",
        f"Banner with highest CTR: {f['banners']['highest_ctr']}; placement with highest "
        f"CTR: {f['placements']['highest_ctr']}.",
        f"Top banner×placement segment ({next(iter(seg['top_5_by_margin']))}) produced "
        f"{seg['top_segment_margin_share']:.0%} of total firm margin; "
        f"{seg['loss_making_count']} segments lost ${-seg['loss_making_total_margin']:,.0f}.",
        f"Revenue is concentrated: top 1% of rows = {dist['top_1pct_rows_revenue_share']:.0%}, "
        f"top 10% = {dist['top_10pct_rows_revenue_share']:.0%}.",
        f"Cost vs revenue: Pearson r = {rel['pearson']:.2f}, Spearman rho = {rel['spearman']:.2f} "
        f"(strong monotonic link; Pearson is pulled down by skew).",
    ]
    print("\nEXECUTIVE SUMMARY")
    for i, line in enumerate(lines, 1):
        print(f"  {i}. {line}")
