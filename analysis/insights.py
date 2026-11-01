"""
Business insights engine.

Turns the cleaned, KPI-enriched dataset into a structured set of
quantified findings for the 20 official questions (Q1-Q20 in
reports/eda_findings.md). Every number is
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
               "firm_cpc", "advertiser_cpc", "advertiser_cpa",
               "record_count", "active_days", "revenue_share"]

# Segments with fewer conversions than this are too thin to rank on cost per conversion
MIN_CONVERSIONS_FOR_CPA_RANKING = 100

# Weekly / monthly ROI is ignored for segments with less placement cost than this
MIN_COST_FOR_ROI = 20

MONTH_ORDER = ["April", "May", "June"]
OUTLIER_COLUMNS = ["cost", "clicks", "revenue"]


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
        "revenue_per_click": _revenue_per_click(df),
        "sales_trend": _sales_trend(df),
        "engagement_by_banner": _engagement_by_banner(df),
        "engagement_revenue": _engagement_revenue(df),
        "placement_conversions": _placement_conversions(df),
        "outliers": _outliers(df),
        "campaign_banner_placement": _campaign_banner_placement(df),
        "roi_consistency": _roi_consistency(df),
        "weekend_engagement": _weekend_engagement(df),
        "cost_per_conversion": _cost_per_conversion(df),
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
    """Q1/Q10: daily trend, monthly comparison and the campaign-mix effect."""
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
    """Q6/Q17/Q18: campaign comparison (camp 2 only ran in April -> compare rates)."""
    s = summarize_by(df, "campaign_number")
    dates = df.groupby("campaign_number")["date"].agg(["min", "max"])
    out = _records(s)
    for camp in out:
        out[camp]["first_date"] = str(dates.loc[camp, "min"].date())
        out[camp]["last_date"] = str(dates.loc[camp, "max"].date())
    return out


def _engagement(df: pd.DataFrame) -> dict:
    """Q1/Q20: engagement levels overall and within each campaign."""
    return {
        "overall": _records(summarize_by(df, "user_engagement")),
        "by_campaign": {f"{c} | {e}": v for (c, e), v in
                        _records(summarize_by(df, ["campaign_number", "user_engagement"])).items()},
    }


def _dimension(df: pd.DataFrame, dim: str) -> dict:
    """Q2/Q3: volume leaders, CTR leaders (with enough clicks) and loss-makers."""
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
    """Q13: banner × placement combinations that drive or destroy firm margin."""
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
    """Q4: correlations (Pearson is skew-sensitive, Spearman is not)."""
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
    """Q12: skewness, zero-inflation and revenue concentration."""
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
    """Q10/Q19: per-day averages and conversion rate by weekday (13 of each)."""
    s = summarize_by(df, "weekday")
    per_day = s["revenue"] / s["active_days"]
    camp1 = summarize_by(df[df["campaign_number"] == "camp 1"], "weekday")
    return {
        "avg_daily_revenue": per_day.to_dict(),
        "avg_daily_clicks": (s["clicks"] / s["active_days"]).to_dict(),
        "avg_daily_displays": (s["displays"] / s["active_days"]).to_dict(),
        "ctr": s["ctr"].to_dict(),
        "conversion_rate": s["conversion_rate"].to_dict(),
        "conversion_rate_camp1": camp1["conversion_rate"].to_dict(),
        "best_day": per_day.idxmax(),
        "worst_day": per_day.idxmin(),
        "best_to_worst_ratio": per_day.max() / per_day.min(),
    }


def _data_quality(df: pd.DataFrame) -> dict:
    """Counts of flagged rows kept in the analysis."""
    return {col: int(df[col].sum()) for col in df.columns if col.startswith("flag_")}


def _revenue_per_click(df: pd.DataFrame) -> dict:
    """Q5: what Company X paid per click (revenue ÷ clicks), overall and by month."""
    monthly = summarize_by(df, "month").reindex(MONTH_ORDER)
    return {
        "overall": df["revenue"].sum() / df["clicks"].sum(),
        "by_month": monthly["advertiser_cpc"].to_dict(),
        "by_campaign": summarize_by(df, "campaign_number")["advertiser_cpc"].to_dict(),
        "sales_per_click": df["post_click_sales_amount"].sum() / df["clicks"].sum(),
    }


def _sales_trend(df: pd.DataFrame) -> dict:
    """Q7: post-click sales over time, and which campaign drives them."""
    daily = summarize_by(df, "date").sort_index()
    first, last = daily.iloc[:14], daily.iloc[-14:]
    monthly = summarize_by(df, ["month", "campaign_number"])["post_click_sales_amount"]
    by_campaign = df.groupby("campaign_number")["post_click_sales_amount"].sum()
    return {
        "monthly": summarize_by(df, "month").reindex(MONTH_ORDER)[
            ["post_click_sales_amount", "advertiser_roas"]].to_dict(orient="index"),
        "monthly_by_campaign": {f"{m} | {c}": v for (m, c), v in monthly.items()},
        "first_14_days_avg_daily_sales": first["post_click_sales_amount"].mean(),
        "last_14_days_avg_daily_sales": last["post_click_sales_amount"].mean(),
        "sales_change_first_to_last_14_days":
            last["post_click_sales_amount"].mean() / first["post_click_sales_amount"].mean() - 1,
        "camp1_sales_share": by_campaign["camp 1"] / by_campaign.sum(),
        "top_5_sales_days": {str(d.date()): v for d, v in
                             daily["post_click_sales_amount"].nlargest(5).items()},
    }


def _engagement_by_banner(df: pd.DataFrame) -> dict:
    """Q8: engagement mix (share of displays) and CTR within each banner size."""
    s = summarize_by(df, ["banner", "user_engagement"])
    displays = s["displays"].unstack("user_engagement")
    share = displays.div(displays.sum(axis=1), axis=0)
    ctr = s["ctr"].unstack("user_engagement")
    return {
        "display_share": share.to_dict(orient="index"),
        "ctr": ctr.to_dict(orient="index"),
        "most_high_engagement_banner": share["High"].idxmax(),
        "least_high_engagement_banner": share["High"].idxmin(),
        "banners_where_medium_has_best_ctr": int((ctr.idxmax(axis=1) == "Medium").sum()),
        "banner_count": len(ctr),
    }


def _engagement_revenue(df: pd.DataFrame) -> dict:
    """Q11: engagement level vs revenue (ordinal rank correlation and per-display rates)."""
    s = summarize_by(df, "user_engagement")
    level_rank = df["user_engagement"].map({"Low": 0, "Medium": 1, "High": 2})
    return {
        "spearman_engagement_revenue": level_rank.corr(df["revenue"], method="spearman"),
        "revenue_per_1000_displays": (s["revenue"] / s["displays"] * 1000).to_dict(),
        "revenue_share": s["revenue_share"].to_dict(),
        "advertiser_cpc": s["advertiser_cpc"].to_dict(),
    }


def _placement_conversions(df: pd.DataFrame) -> dict:
    """Q9/Q15: conversion rate by placement and the per-row conversion distribution."""
    s = summarize_by(df, "placement")
    rankable = s[s["clicks"] >= MIN_CLICKS_FOR_RATE_RANKING]
    per_row = df.groupby("placement")["post_click_conversions"]
    return {
        "conversion_rate": s["conversion_rate"].to_dict(),
        "conversion_share": (s["post_click_conversions"]
                             / s["post_click_conversions"].sum()).to_dict(),
        "highest_conversion_rate": rankable["conversion_rate"].idxmax(),
        "lowest_conversion_rate": rankable["conversion_rate"].idxmin(),
        "per_row": per_row.describe(percentiles=[0.5, 0.9, 0.99]).to_dict(orient="index"),
        "zero_conversion_row_share": (df["post_click_conversions"].eq(0)
                                      .groupby(df["placement"]).mean().to_dict()),
    }


def _outliers(df: pd.DataFrame) -> dict:
    """Q12: IQR outliers on raw and log scales, extreme rows and anomalies."""
    out = {}
    for col in OUTLIER_COLUMNS:
        raw, logged = df[col], np.log1p(df[col])
        q1, q3 = raw.quantile([0.25, 0.75])
        lq1, lq3 = logged.quantile([0.25, 0.75])
        out[col] = {
            "raw_iqr_upper_outliers": int((raw > q3 + 1.5 * (q3 - q1)).sum()),
            "log_iqr_upper_outliers": int((logged > lq3 + 1.5 * (lq3 - lq1)).sum()),
            "p99": raw.quantile(0.99),
            "max": raw.max(),
        }
    top = df.nlargest(5, "revenue")
    out["top_5_revenue_rows"] = [
        {"date": str(r.date.date()), "campaign_number": r.campaign_number,
         "user_engagement": r.user_engagement, "banner": r.banner,
         "placement": r.placement, "clicks": r.clicks, "revenue": r.revenue}
        for r in top.itertuples()
    ]
    out["clicks_but_zero_revenue_rows"] = int(((df["clicks"] > 0) & (df["revenue"] == 0)).sum())
    out["revenue_without_clicks_rows"] = int(((df["revenue"] > 0) & (df["clicks"] == 0)).sum())
    return out


def _campaign_banner_placement(df: pd.DataFrame) -> dict:
    """Q13: CTR, conversion rate and ROI by campaign × banner and banner × placement."""
    def rankable(group):
        s = summarize_by(df, group)
        s = s[s["clicks"] >= MIN_CLICKS_FOR_RATE_RANKING]
        s.index = [" | ".join(i) for i in s.index]
        return s

    cb, bp = rankable(["campaign_number", "banner"]), rankable(["banner", "placement"])
    cols = ["clicks", "ctr", "conversion_rate", "firm_roi"]
    return {
        "campaign_banner": cb[cols].to_dict(orient="index"),
        "banner_placement": bp[cols].to_dict(orient="index"),
        "best_banner_by_ctr_per_campaign": {
            camp: cb.loc[[i for i in cb.index if i.startswith(camp)], "ctr"].idxmax().split(" | ")[1]
            for camp in sorted(df["campaign_number"].unique())
        },
    }


def _roi_consistency(df: pd.DataFrame) -> dict:
    """Q14: does any campaign or banner lead on ROI every month / week?"""
    def roi_table(period_col, dim):
        s = summarize_by(df.assign(period=period_col), ["period", dim])
        # Periods with almost no spend give meaningless ROI (e.g. +2,500% on a few dollars)
        return s["firm_roi"].where(s["cost"] >= MIN_COST_FOR_ROI).unstack(dim).sort_index()

    weekly_banner = roi_table(df["date"].dt.to_period("W").dt.start_time, "banner")
    monthly_banner = roi_table(df["month"], "banner").reindex(MONTH_ORDER)
    monthly_campaign = roi_table(df["month"], "campaign_number").reindex(MONTH_ORDER)
    roas = summarize_by(df, ["month", "campaign_number"])["advertiser_roas"]
    return {
        "monthly_firm_roi_by_banner": monthly_banner.T.to_dict(orient="index"),
        "monthly_firm_roi_by_campaign": monthly_campaign.T.to_dict(orient="index"),
        "monthly_roas_by_campaign": {f"{m} | {c}": v for (m, c), v in roas.items()},
        "weeks_as_top_banner_by_roi": weekly_banner.idxmax(axis=1).value_counts().to_dict(),
        "weeks": len(weekly_banner),
        "banners_profitable_every_month": monthly_banner.columns[
            (monthly_banner > 0).all()].tolist(),
    }


def _weekend_engagement(df: pd.DataFrame) -> dict:
    """Q16: per-day volume and CTR by engagement level, weekdays vs weekends."""
    t = df.assign(day_type=np.where(df["date"].dt.day_of_week >= 5, "weekend", "weekday"))
    s = summarize_by(t, ["user_engagement", "day_type"])
    days = t.groupby("day_type")["date"].nunique()
    per_day = s[["displays", "clicks", "revenue"]].div(
        s.index.get_level_values("day_type").map(days).values, axis=0)
    clicks = s["clicks"].unstack("user_engagement")
    out = {"days": days.to_dict()}
    for (level, day_type), row in per_day.iterrows():
        out[f"{level} | {day_type}"] = {
            "avg_daily_displays": row["displays"], "avg_daily_clicks": row["clicks"],
            "avg_daily_revenue": row["revenue"], "ctr": s.loc[(level, day_type), "ctr"],
        }
    out["click_mix"] = clicks.div(clicks.sum(axis=1), axis=0).to_dict(orient="index")
    return out


def _cost_per_conversion(df: pd.DataFrame) -> dict:
    """Q18: Company X's cost per conversion (revenue ÷ conversions) by segment."""
    s = summarize_by(df, ["campaign_number", "placement"])
    s = s[s["post_click_conversions"] >= MIN_CONVERSIONS_FOR_CPA_RANKING]
    s.index = [f"{c} | {p}" for c, p in s.index]
    by_placement = summarize_by(df, "placement")
    by_placement = by_placement[by_placement["post_click_conversions"]
                                >= MIN_CONVERSIONS_FOR_CPA_RANKING]
    return {
        "by_campaign": summarize_by(df, "campaign_number")["advertiser_cpa"].to_dict(),
        "by_placement": by_placement["advertiser_cpa"].sort_values().to_dict(),
        "by_campaign_placement": s["advertiser_cpa"].sort_values().to_dict(),
    }


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
