"""
Exploratory data analysis charts for the advertising performance dataset.

Each chart answers one or more of the 20 official questions (Q1-Q20 in
reports/eda_findings.md) and is saved as a PNG to reports/figures/. All
grouped rates are ratio-of-totals via ``analysis.metrics.summarize_by``.

Conventions:
    - One y-axis per panel (two measures of different scale -> two panels)
    - Heavily skewed volumes are shown on log scales
    - Categorical colours follow a fixed order: campaigns and engagement
      levels always get the same colour in every chart
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.metrics import summarize_by
from analysis.data_cleaning import ENGAGEMENT_ORDER
from utils.theme import CATEGORICAL, SEQUENTIAL_BLUE, DIVERGING, INK


CAMPAIGN_ORDER = ["camp 1", "camp 2", "camp 3"]
CAMPAIGN_COLORS = dict(zip(CAMPAIGN_ORDER, CATEGORICAL))
ENGAGEMENT_COLORS = dict(zip(ENGAGEMENT_ORDER, CATEGORICAL))
PRIMARY = CATEGORICAL[0]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
MONTH_ORDER = ["April", "May", "June"]
# Segments below these volumes are left out of rate charts (rates too noisy)
MIN_CLICKS = 1000
MIN_DISPLAYS = 10000
MIN_CONVERSIONS = 100
MIN_WEEKLY_COST = 20

def fmt_money(x: float) -> str:
    """$1,234 / −$1,234; values under a dollar shown as <$1."""
    if abs(x) < 0.5:
        return "<$1" if x > 0 else ("−<$1" if x < 0 else "$0")
    return f"{'−' if x < 0 else ''}${abs(x):,.0f}"


def _fmt_compact(x: float) -> str:
    """1.2k / 3.4M style labels for log axes."""
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:.3g}{suffix}"
    return f"{x:.3g}"


_money = mticker.FuncFormatter(lambda x, _: fmt_money(x))
_cents = mticker.FuncFormatter(lambda x, _: f"${x:,.2f}")
_us_cents = mticker.FuncFormatter(lambda x, _: f"{x * 100:g}¢")
_pct = mticker.PercentFormatter(1.0, decimals=2)
_pct0 = mticker.PercentFormatter(1.0)
_compact = mticker.FuncFormatter(lambda x, _: _fmt_compact(x))


def setup_style():
    """Apply a clean, recessive chart style."""
    plt.rcParams.update({
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "axes.edgecolor": INK["grid"],
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.grid.axis": "y",
        "grid.color": INK["grid"],
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK["primary"],
        "axes.labelcolor": INK["secondary"],
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"],
        "legend.frameon": False,
        "font.family": "sans-serif",
        "font.size": 10,
        "lines.linewidth": 2,
        "figure.dpi": 150,
    })


def _save(fig, filepath: Path, title: str):
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _label_bars(ax, bars, fmt, horizontal=False):
    """Direct value labels in secondary ink."""
    for bar in bars:
        val = bar.get_width() if horizontal else bar.get_height()
        if np.isnan(val):
            continue
        if horizontal:
            ax.annotate(fmt(val), (val, bar.get_y() + bar.get_height() / 2),
                        xytext=(4 if val >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if val >= 0 else "right", va="center",
                        fontsize=8.5, color=INK["secondary"])
        else:
            ax.annotate(fmt(val), (bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK["secondary"])


def generate_all_charts(df: pd.DataFrame, output_dir: str | Path = "reports/figures") -> list[str]:
    """
    Generate all EDA charts and save them to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with KPI columns (see ``compute_kpis``).
    output_dir : str or Path
        Directory to save chart images.

    Returns
    -------
    list[str]
        Paths to all saved chart files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    saved = []
    chart_funcs = [
        ("01_daily_trends.png", _chart_daily_trends),
        ("02_monthly_by_campaign.png", _chart_monthly_by_campaign),
        ("03_engagement_trends.png", _chart_engagement_trends),
        ("04_banner_performance.png", _chart_banner_performance),
        ("05_placement_performance.png", _chart_placement_performance),
        ("06_cost_vs_revenue.png", _chart_cost_vs_revenue),
        ("07_correlation_heatmap.png", _chart_correlation_heatmap),
        ("08_campaign_engagement_rates.png", _chart_campaign_engagement_rates),
        ("09_banner_placement_margin.png", _chart_banner_placement_margin),
        ("10_distributions_concentration.png", _chart_distributions),
        ("11_weekday_patterns.png", _chart_weekday_patterns),
        ("12_conversions_vs_clicks.png", _chart_conversions_vs_clicks),
        ("13_sales_over_time.png", _chart_sales_over_time),
        ("14_engagement_by_banner.png", _chart_engagement_by_banner),
        ("15_placement_conversions.png", _chart_placement_conversions),
        ("16_cost_per_click_and_conversion.png", _chart_cost_efficiency),
        ("17_segment_effectiveness.png", _chart_segment_effectiveness),
        ("18_roi_consistency.png", _chart_roi_consistency),
        ("19_weekday_vs_weekend_engagement.png", _chart_weekend_engagement),
    ]

    for filename, func in chart_funcs:
        filepath = output_dir / filename
        try:
            func(df, filepath)
            saved.append(str(filepath))
            print(f"  [OK] Saved: {filename}")
        except Exception as e:
            print(f"  [FAIL] Failed: {filename} - {e}")

    return saved


# =====================================================================
# TIME
# =====================================================================

def _chart_daily_trends(df: pd.DataFrame, filepath: Path):
    """Q1/Q10: How did displays, clicks, revenue, cost and margin evolve?"""
    daily = summarize_by(df, "date").sort_index()
    roll = daily.rolling(7, min_periods=1).mean()
    camp2_end = df.loc[df["campaign_number"] == "camp 2", "date"].max()

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    for ax, col, title, fmt in [
        (axes[0], "displays", "Daily displays", _compact),
        (axes[1], "clicks", "Daily clicks",
         mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")),
    ]:
        ax.plot(daily.index, daily[col], color=PRIMARY, alpha=0.25, linewidth=1)
        ax.plot(roll.index, roll[col], color=PRIMARY)
        ax.set_title(f"{title} (faint) and 7-day average")
        ax.yaxis.set_major_formatter(fmt)

    ax = axes[2]
    ax.plot(roll.index, roll["revenue"], color=CATEGORICAL[0], label="Revenue (Company X spend)")
    ax.plot(roll.index, roll["cost"], color=CATEGORICAL[1], label="Cost (placement cost)")
    ax.set_title("Revenue vs cost, 7-day average")
    ax.yaxis.set_major_formatter(_money)
    ax.legend(loc="upper right")

    ax = axes[3]
    margin = roll["firm_margin"]
    ax.fill_between(margin.index, margin, 0, where=margin >= 0,
                    color=DIVERGING["positive"], alpha=0.35, linewidth=0)
    ax.fill_between(margin.index, margin, 0, where=margin < 0,
                    color=DIVERGING["negative"], alpha=0.35, linewidth=0)
    ax.plot(margin.index, margin, color=INK["secondary"], linewidth=1.5)
    ax.axhline(0, color=INK["muted"], linewidth=1)
    ax.set_title("Advert Firm A margin (revenue − cost), 7-day average")
    ax.yaxis.set_major_formatter(_money)

    for ax in axes:
        ax.axvline(camp2_end, color=INK["muted"], linestyle="--", linewidth=1)
    axes[0].annotate("camp 2 ends", (camp2_end, axes[0].get_ylim()[1]),
                     xytext=(4, -12), textcoords="offset points",
                     fontsize=8.5, color=INK["secondary"])
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    _save(fig, filepath, "Q1, Q10 · Performance over time (Apr–Jun 2020)")


def _chart_monthly_by_campaign(df: pd.DataFrame, filepath: Path):
    """Q1 context: How does each month compare, and how much is the campaign mix?"""
    monthly = summarize_by(df, ["month", "campaign_number"])
    month_order = ["April", "May", "June"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    pivot = monthly["revenue"].unstack("campaign_number").reindex(month_order)
    bottom = np.zeros(len(pivot))
    for camp in CAMPAIGN_ORDER:
        if camp not in pivot:
            continue
        vals = pivot[camp].fillna(0).values
        ax.bar(pivot.index, vals, bottom=bottom, color=CAMPAIGN_COLORS[camp],
               label=camp, width=0.6, edgecolor=INK["surface"], linewidth=2)
        bottom += vals
    ax.set_title("Revenue by month, stacked by campaign")
    ax.yaxis.set_major_formatter(_money)
    ax.legend()

    ax = axes[1]
    by_month = summarize_by(df, "month").reindex(month_order)
    colors = [DIVERGING["positive"] if v >= 0 else DIVERGING["negative"]
              for v in by_month["firm_roi"]]
    bars = ax.bar(by_month.index, by_month["firm_roi"], color=colors, width=0.6)
    ax.axhline(0, color=INK["muted"], linewidth=1)
    _label_bars(ax, bars, lambda v: f"{v:.0%}")
    ax.set_title("Advert Firm A ROI by month")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    _save(fig, filepath, "Monthly comparison by campaign")


def _chart_engagement_trends(df: pd.DataFrame, filepath: Path):
    """Q1: How did each user-engagement level trend week to week?"""
    t = df.copy()
    t["week"] = t["date"].dt.to_period("W").dt.start_time
    weekly = summarize_by(t, ["week", "user_engagement"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for level in ENGAGEMENT_ORDER:
        series = weekly.xs(level, level="user_engagement").sort_index()
        # First and last weeks are partial (Apr 1 is a Wednesday), so use daily averages
        axes[0].plot(series.index, series["clicks"] / series["active_days"],
                     color=ENGAGEMENT_COLORS[level], marker="o", markersize=4, label=level)
        axes[1].plot(series.index, series["ctr"], color=ENGAGEMENT_COLORS[level],
                     marker="o", markersize=4, label=level)

    axes[0].set_title("Average daily clicks, by week (log scale)")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[1].set_title("Weekly CTR")
    axes[1].yaxis.set_major_formatter(_pct)
    for ax in axes:
        ax.legend(title="Engagement")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    _save(fig, filepath, "Q1 · User-engagement levels over time")


# =====================================================================
# CREATIVE & INVENTORY
# =====================================================================

def _chart_dimension_panels(df, dim, filepath, title, order=None):
    """Displays and clicks (volume), CTR (efficiency) and firm ROI for one dimension."""
    s = summarize_by(df, dim)
    s = s.reindex(order) if order else s.sort_values("clicks")

    fig, axes = plt.subplots(1, 4, figsize=(19, 5), sharey=True)

    for ax, col in zip(axes[:2], ["displays", "clicks"]):
        bars = ax.barh(s.index, s[col], color=PRIMARY, height=0.6)
        _label_bars(ax, bars, _fmt_compact, horizontal=True)
        ax.set_title(f"Total {col} (log scale)")
        ax.set_xscale("log")
        ax.set_xlim(left=1)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=100))
        ax.xaxis.set_major_formatter(_compact)

    bars = axes[2].barh(s.index, s["ctr"], color=PRIMARY, height=0.6)
    _label_bars(axes[2], bars, lambda v: f"{v:.2%}", horizontal=True)
    axes[2].set_title("CTR (clicks ÷ displays)")
    axes[2].xaxis.set_major_formatter(_pct)

    colors = [DIVERGING["positive"] if v >= 0 else DIVERGING["negative"] for v in s["firm_roi"]]
    bars = axes[3].barh(s.index, s["firm_roi"], color=colors, height=0.6)
    _label_bars(axes[3], bars, lambda v: f"{v:.0%}", horizontal=True)
    axes[3].axvline(0, color=INK["muted"], linewidth=1)
    axes[3].set_title("Advert Firm A ROI")
    axes[3].xaxis.set_major_formatter(_pct0)

    for ax in axes:
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
    _save(fig, filepath, title)


def _chart_banner_performance(df: pd.DataFrame, filepath: Path):
    """Q2: How does banner size affect clicks and click-through?"""
    _chart_dimension_panels(df, "banner", filepath, "Q2 · Banner size: volume, CTR and margin")


def _chart_placement_performance(df: pd.DataFrame, filepath: Path):
    """Q3: Which placements yield the most displays, clicks and margin?"""
    _chart_dimension_panels(df, "placement", filepath,
                            "Q3 · Placement: volume, CTR and margin")


def _chart_banner_placement_margin(df: pd.DataFrame, filepath: Path):
    """Q13: Which banner × placement combinations make or lose money for the firm?"""
    s = summarize_by(df, ["banner", "placement"])["firm_margin"].unstack("placement")
    s = s.loc[s.sum(axis=1).sort_values(ascending=False).index]
    s = s[s.sum().sort_values(ascending=False).index]

    limit = np.nanmax(np.abs(s.values))
    cmap = LinearSegmentedColormap.from_list(
        "div", [DIVERGING["negative"], DIVERGING["neutral"], DIVERGING["positive"]])
    # Symmetric-log style compression so small losses are still visible
    signed_log = np.sign(s) * np.log10(1 + np.abs(s))
    lim = np.log10(1 + limit)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(signed_log.values, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(s.shape[1]), s.columns)
    ax.set_yticks(range(s.shape[0]), s.index)
    ax.grid(False)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            v = s.iat[i, j]
            if np.isnan(v):
                continue
            strong = abs(signed_log.iat[i, j]) > lim * 0.6
            ax.text(j, i, fmt_money(v), ha="center", va="center", fontsize=8,
                    color="white" if strong else INK["primary"])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    ticks = np.array([-1e4, -1e3, -1e2, 0, 1e2, 1e3, 1e4])
    ticks = ticks[np.abs(ticks) <= limit]
    cbar.set_ticks(np.sign(ticks) * np.log10(1 + np.abs(ticks)))
    cbar.set_ticklabels([fmt_money(t) for t in ticks])
    cbar.set_label("Firm margin (log-compressed)")
    ax.set_xlabel("Placement")
    ax.set_ylabel("Banner")

    _save(fig, filepath, "Q13 · Firm margin by banner × placement")


# =====================================================================
# RELATIONSHIPS
# =====================================================================

def _chart_cost_vs_revenue(df: pd.DataFrame, filepath: Path):
    """Q4: Is there a relationship between cost and revenue?"""
    d = df[(df["cost"] > 0) & (df["revenue"] > 0)]
    pearson = df["cost"].corr(df["revenue"])
    spearman = df["cost"].corr(df["revenue"], method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))
    for camp in CAMPAIGN_ORDER:
        sub = d[d["campaign_number"] == camp]
        ax.scatter(sub["cost"], sub["revenue"], s=10, alpha=0.35,
                   color=CAMPAIGN_COLORS[camp], label=camp, linewidths=0)
    lo, hi = d[["cost", "revenue"]].min().min(), d[["cost", "revenue"]].max().max()
    ax.plot([lo, hi], [lo, hi], "--", color=INK["muted"], linewidth=1, label="Break-even")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cost ($, log)")
    ax.set_ylabel("Revenue ($, log)")
    ax.grid(axis="x")
    ax.legend(loc="upper left", markerscale=2)
    ax.text(0.98, 0.03,
            f"All rows: Pearson r = {pearson:.2f} · Spearman ρ = {spearman:.2f}\n"
            f"Plotted: {len(d):,} rows with cost and revenue > 0",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=INK["secondary"])

    _save(fig, filepath, "Q4 · Cost vs revenue (each dot = one row)")


def _chart_correlation_heatmap(df: pd.DataFrame, filepath: Path):
    """Q4: How are the raw metrics correlated (rank-based, robust to skew)?"""
    cols = ["displays", "clicks", "cost", "revenue",
            "post_click_conversions", "post_click_sales_amount"]
    labels = ["displays", "clicks", "cost", "revenue", "conversions", "sales"]
    corr = df[cols].corr(method="spearman")

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    masked = np.ma.masked_where(np.triu(np.ones_like(corr, dtype=bool), k=1), corr.values)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(i + 1):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.5 else INK["primary"])
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")

    _save(fig, filepath, "Q4 · Rank correlation between metrics")


def _chart_conversions_vs_clicks(df: pd.DataFrame, filepath: Path):
    """Q6: Which campaigns convert clicks into post-click conversions best?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    d = df[(df["clicks"] > 0) & (df["post_click_conversions"] > 0)]
    for camp in CAMPAIGN_ORDER:
        sub = d[d["campaign_number"] == camp]
        axes[0].scatter(sub["clicks"], sub["post_click_conversions"], s=10, alpha=0.35,
                        color=CAMPAIGN_COLORS[camp], label=camp, linewidths=0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Clicks (log)")
    axes[0].set_ylabel("Post-click conversions (log)")
    axes[0].set_title("Clicks vs conversions (rows with both > 0)")
    axes[0].grid(axis="x")
    axes[0].legend(markerscale=2)

    s = summarize_by(df, "campaign_number").reindex(CAMPAIGN_ORDER)
    bars = axes[1].bar(s.index, s["conversion_rate"],
                       color=[CAMPAIGN_COLORS[c] for c in s.index], width=0.6)
    _label_bars(axes[1], bars, lambda v: f"{v:.1%}")
    axes[1].set_title("Conversions per click by campaign")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    _save(fig, filepath, "Q6 · From clicks to conversions")


# =====================================================================
# SEGMENTS, DISTRIBUTION, CALENDAR
# =====================================================================

def _chart_campaign_engagement_rates(df: pd.DataFrame, filepath: Path):
    """Q11/Q20: How do campaigns and engagement levels compare on rates?"""
    s = summarize_by(df, ["campaign_number", "user_engagement"])
    s["revenue_per_1000_displays"] = s["revenue"] / s["displays"] * 1000
    metrics = [("ctr", "CTR", _pct),
               ("conversion_rate", "Post-click conversions per click", _pct0),
               ("advertiser_cpa", "Company X cost per conversion", _cents),
               ("revenue_per_1000_displays", "Revenue per 1,000 displays", _cents),
               ("firm_roi", "Advert Firm A ROI", _pct0),
               ("advertiser_roas", "Company X ROAS (sales ÷ spend)",
                mticker.FuncFormatter(lambda x, _: f"{x:,.0f}×"))]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    x = np.arange(len(CAMPAIGN_ORDER))
    width = 0.26
    for ax, (col, title, fmt) in zip(axes.flat, metrics):
        for k, level in enumerate(ENGAGEMENT_ORDER):
            vals = [s[col].get((camp, level), np.nan) for camp in CAMPAIGN_ORDER]
            ax.bar(x + (k - 1) * width, vals, width=width - 0.03,
                   color=ENGAGEMENT_COLORS[level], label=level)
            for xi, v in zip(x + (k - 1) * width, vals):
                if np.isnan(v):
                    ax.annotate("no rows", (xi, 0), xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", rotation=90, fontsize=8,
                                color=INK["muted"])
        ax.set_xticks(x, CAMPAIGN_ORDER)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(fmt)
        ax.axhline(0, color=INK["muted"], linewidth=1)
    axes[0, 0].legend(title="Engagement")

    _save(fig, filepath, "Q11, Q20 · Campaign × engagement level")


def _chart_distributions(df: pd.DataFrame, filepath: Path):
    """Q12: How skewed are the metrics, how concentrated is revenue, which rows are outliers?"""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    ax = axes[0]
    for col, color in zip(["displays", "clicks", "revenue"], CATEGORICAL):
        vals = df.loc[df[col] > 0, col]
        ax.hist(np.log10(vals), bins=50, histtype="step", linewidth=2,
                color=color, label=f"{col} ({(df[col] == 0).mean():.0%} zeros)")
    ax.set_xlabel("log10(value), non-zero rows")
    ax.set_ylabel("Rows")
    ax.set_title("Distributions on a log scale")
    ax.legend()

    ax = axes[1]
    rev = np.sort(df["revenue"].values)[::-1]
    share = np.cumsum(rev) / rev.sum()
    rows = np.arange(1, len(rev) + 1) / len(rev)
    ax.plot(rows, share, color=PRIMARY)
    for frac in (0.01, 0.10):
        idx = int(len(rev) * frac) - 1
        ax.scatter([rows[idx]], [share[idx]], s=40, color=PRIMARY,
                   edgecolors=INK["surface"], linewidths=2, zorder=3)
        ax.annotate(f"top {frac:.0%} of rows → {share[idx]:.0%} of revenue",
                    (rows[idx], share[idx]), xytext=(-10, 6), textcoords="offset points",
                    ha="right", fontsize=9, color=INK["secondary"])
    ax.set_xscale("log")
    ax.set_xlabel("Share of rows, ranked by revenue (log)")
    ax.set_ylabel("Cumulative share of revenue")
    ax.set_title("Revenue concentration")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:g}%"))
    ax.yaxis.set_major_formatter(_pct0)
    ax.grid(axis="x")

    ax = axes[2]
    cols = ["cost", "clicks", "revenue"]
    logged = [np.log1p(df[c]) for c in cols]
    ax.boxplot(logged, whis=1.5, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=SEQUENTIAL_BLUE[0], edgecolor=PRIMARY),
               medianprops=dict(color=PRIMARY, linewidth=2),
               whiskerprops=dict(color=INK["secondary"]), capprops=dict(color=INK["secondary"]),
               flierprops=dict(marker="o", markersize=2, alpha=0.3,
                               markerfacecolor=INK["muted"], markeredgecolor="none"))
    ax.set_xticks(range(1, len(cols) + 1), cols)
    ticks = np.array([0, 1, 10, 100, 1000, 10000])
    ax.set_yticks(np.log1p(ticks), [_fmt_compact(t) for t in ticks])
    for i, vals in enumerate(logged, 1):
        q1, q3 = vals.quantile([0.25, 0.75])
        n_out = int((vals > q3 + 1.5 * (q3 - q1)).sum())
        ax.annotate(f"{n_out:,} high outliers", (i, vals.max()), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK["secondary"])
    ax.set_title("Outliers: 1.5 × IQR on a log scale (all rows)")

    _save(fig, filepath, "Q12 · Skew, concentration and outliers")


def _chart_weekday_patterns(df: pd.DataFrame, filepath: Path):
    """Q10/Q19: Is there a day-of-week pattern in volume and conversion rate?"""
    s = summarize_by(df, "weekday").reindex(WEEKDAY_ORDER)
    per_day = s[["revenue", "clicks"]].div(s["active_days"], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    short = [d[:3] for d in WEEKDAY_ORDER]

    panels = [
        (axes[0, 0], per_day["revenue"], "Average revenue per day", lambda v: f"${v:,.0f}", _money),
        (axes[0, 1], per_day["clicks"], "Average clicks per day", lambda v: f"{v/1000:,.1f}k",
         mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")),
        (axes[1, 0], s["ctr"], "CTR", lambda v: f"{v:.2%}", _pct),
        (axes[1, 1], s["conversion_rate"], "Post-click conversion rate (conversions ÷ clicks)",
         lambda v: f"{v:.1%}", _pct0),
    ]
    for ax, vals, title, label, fmt in panels:
        bars = ax.bar(short, vals, color=PRIMARY, width=0.6)
        _label_bars(ax, bars, label)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(fmt)

    lo, hi = int(s["active_days"].min()), int(s["active_days"].max())
    fig.text(0.01, -0.02, f"Each weekday occurs {lo if lo == hi else f'{lo}–{hi}'} times "
             "in the period, so the overall downward trend is spread evenly across them.",
             fontsize=9, color=INK["secondary"])
    _save(fig, filepath, "Q10, Q19 · Day-of-week pattern")


# =====================================================================
# OFFICIAL QUESTIONS NOT COVERED ABOVE
# =====================================================================

def _heatmap(ax, table: pd.DataFrame, fmt, title: str):
    """Sequential heatmap with value labels; NaN cells (too thin or never served) stay blank."""
    table = table.dropna(how="all").dropna(axis=1, how="all")
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL_BLUE)
    vmax = np.nanmax(table.values)
    ax.imshow(np.ma.masked_invalid(table.values.astype(float)), cmap=cmap,
              vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                    color="white" if v > vmax * 0.55 else INK["primary"])
    ax.set_title(title)


def _chart_sales_over_time(df: pd.DataFrame, filepath: Path):
    """Q7: Are there trends or patterns in post-click sales amounts over time?"""
    daily = summarize_by(df, "date").sort_index()
    roll = daily[["post_click_sales_amount", "revenue"]].rolling(7, min_periods=1).sum()
    camp2_end = df.loc[df["campaign_number"] == "camp 2", "date"].max()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(daily.index, daily["post_click_sales_amount"], color=PRIMARY, alpha=0.25, linewidth=1)
    ax.plot(daily.index, daily["post_click_sales_amount"].rolling(7, min_periods=1).mean(),
            color=PRIMARY)
    ax.set_title("Daily post-click sales (faint) and 7-day average")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${_fmt_compact(x)}"))
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(roll.index, roll["post_click_sales_amount"] / roll["revenue"], color=PRIMARY)
    ax.set_title("Company X ROAS (sales ÷ ad spend), 7-day window")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}×"))
    ax.set_ylim(bottom=0)

    for ax in axes:
        ax.axvline(camp2_end, color=INK["muted"], linestyle="--", linewidth=1)
    axes[0].annotate("camp 2 ends", (camp2_end, axes[0].get_ylim()[1]),
                     xytext=(4, -12), textcoords="offset points",
                     fontsize=8.5, color=INK["secondary"])
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    _save(fig, filepath, "Q7 · Post-click sales over time")


def _chart_engagement_by_banner(df: pd.DataFrame, filepath: Path):
    """Q8: How does the level of user engagement vary across banner sizes?"""
    s = summarize_by(df, ["banner", "user_engagement"])
    displays = s["displays"].unstack("user_engagement")[ENGAGEMENT_ORDER]
    order = displays.sum(axis=1).sort_values().index
    share = displays.div(displays.sum(axis=1), axis=0).loc[order]
    ctr = s["ctr"].unstack("user_engagement")[ENGAGEMENT_ORDER].loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    ax = axes[0]
    left = np.zeros(len(share))
    for level in ENGAGEMENT_ORDER:
        vals = share[level].values
        ax.barh(share.index, vals, left=left, color=ENGAGEMENT_COLORS[level], label=level,
                height=0.6, edgecolor=INK["surface"], linewidth=1.5)
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 0.08:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white")
        left += vals
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(_pct0)
    ax.set_title("Share of each banner's displays by engagement level")
    ax.legend(title="Engagement", loc="lower right", bbox_to_anchor=(1, 1.06), ncol=3)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    for k, level in enumerate(ENGAGEMENT_ORDER):
        ax.scatter(ctr[level], np.arange(len(ctr)) + (k - 1) * 0.18, s=45,
                   color=ENGAGEMENT_COLORS[level], label=level, zorder=3)
    ax.xaxis.set_major_formatter(_pct)
    ax.set_title("CTR by engagement level within each banner")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    _save(fig, filepath, "Q8 · User engagement across banner sizes (ordered by displays)")


def _chart_placement_conversions(df: pd.DataFrame, filepath: Path):
    """Q9/Q15: Which placements convert best, and how are conversions distributed?"""
    s = summarize_by(df, "placement").sort_values("post_click_conversions", ascending=False)
    order = s.index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    colors = [PRIMARY if c >= MIN_CLICKS else INK["muted"] for c in s["clicks"]]
    bars = ax.bar(order, s["conversion_rate"], color=colors, width=0.6)
    _label_bars(ax, bars, lambda v: f"{v:.1%}")
    ax.set_title("Post-click conversion rate (conversions ÷ clicks)")
    ax.yaxis.set_major_formatter(_pct0)
    ax.margins(y=0.12)

    ax = axes[1]
    share = s["post_click_conversions"] / s["post_click_conversions"].sum()
    bars = ax.bar(order, share, color=PRIMARY, width=0.6)
    _label_bars(ax, bars, lambda v: f"{v:.1%}")
    ax.set_title("Share of all post-click conversions")
    ax.yaxis.set_major_formatter(_pct0)

    ax = axes[2]
    logged = [np.log1p(df.loc[df["placement"] == p, "post_click_conversions"]) for p in order]
    ax.boxplot(logged, whis=1.5, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=SEQUENTIAL_BLUE[0], edgecolor=PRIMARY),
               medianprops=dict(color=PRIMARY, linewidth=2),
               flierprops=dict(marker="o", markersize=2, alpha=0.3,
                               markerfacecolor=INK["muted"], markeredgecolor="none"))
    zero_share = df["post_click_conversions"].eq(0).groupby(df["placement"]).mean()
    ax.set_xticks(range(1, len(order) + 1),
                  [f"{p}\n{zero_share[p]:.0%} zero" for p in order])
    ticks = np.array([0, 1, 10, 100, 1000])
    ax.set_yticks(np.log1p(ticks), [_fmt_compact(t) for t in ticks])
    ax.set_title("Conversions per row (log scale; most rows have none)")

    fig.text(0.01, -0.03, f"Grey bar: fewer than {MIN_CLICKS:,} clicks, rate unreliable. "
             f"abc's rate rests on {int(s.loc['abc', 'clicks']):,} clicks.",
             fontsize=9, color=INK["secondary"])
    _save(fig, filepath, "Q9, Q15 · Post-click conversions by placement")


def _chart_cost_efficiency(df: pd.DataFrame, filepath: Path):
    """Q5/Q17/Q18: revenue per click over time, CPC by campaign and banner, cost per conversion."""
    t = df.assign(week=df["date"].dt.to_period("W").dt.start_time)
    weekly = summarize_by(t, ["week", "campaign_number"])
    overall_weekly = summarize_by(t, "week").sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    for camp in CAMPAIGN_ORDER:
        series = weekly.xs(camp, level="campaign_number").sort_index()
        ax.plot(series.index, series["advertiser_cpc"], color=CAMPAIGN_COLORS[camp],
                marker="o", markersize=4, label=camp)
    ax.plot(overall_weekly.index, overall_weekly["advertiser_cpc"], color=INK["secondary"],
            linestyle="--", linewidth=1.5, label="All campaigns")
    ax.set_title("Revenue per click (what Company X pays), by week")
    ax.yaxis.set_major_formatter(_us_cents)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylim(bottom=0)
    ax.legend()

    def paired_bars(ax, s, title):
        x = np.arange(len(s))
        for k, (col, label, color) in enumerate([
            ("advertiser_cpc", "Company X pays (revenue ÷ clicks)", PRIMARY),
            ("firm_cpc", "Firm A pays publisher (cost ÷ clicks)", INK["muted"]),
        ]):
            bars = ax.bar(x + (k - 0.5) * 0.36, s[col], width=0.34, color=color, label=label)
            _label_bars(ax, bars, lambda v: f"{v * 100:.1f}¢")
        ax.set_xticks(x, s.index)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(_us_cents)

    paired_bars(axes[0, 1], summarize_by(df, "campaign_number").reindex(CAMPAIGN_ORDER),
                "Cost per click by campaign")
    axes[0, 1].legend(loc="upper right")
    banners = summarize_by(df, "banner")
    banners = banners[banners["clicks"] >= MIN_CLICKS].sort_values("clicks", ascending=False)
    paired_bars(axes[1, 0], banners,
                f"Cost per click by banner (banners with ≥ {MIN_CLICKS:,} clicks)")
    axes[1, 0].tick_params(axis="x", labelsize=8.5)

    ax = axes[1, 1]
    s = summarize_by(df, ["campaign_number", "placement"])
    s = s[s["post_click_conversions"] >= MIN_CONVERSIONS].sort_values("advertiser_cpa",
                                                                      ascending=False)
    labels = [f"{c} · {p}" for c, p in s.index]
    bars = ax.barh(labels, s["advertiser_cpa"], height=0.6,
                   color=[CAMPAIGN_COLORS[c] for c, _ in s.index])
    _label_bars(ax, bars, lambda v: f"${v:.2f}", horizontal=True)
    ax.set_title(f"Company X cost per conversion, campaign · placement "
                 f"(≥ {MIN_CONVERSIONS} conversions)")
    ax.xaxis.set_major_formatter(_cents)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)

    _save(fig, filepath, "Q5, Q17, Q18 · What Company X pays per click and per conversion")


def _chart_segment_effectiveness(df: pd.DataFrame, filepath: Path):
    """Q13: How does effectiveness vary with ad size and placement, by campaign?"""
    def rate_table(rows, cols, rate, min_col, min_val):
        s = summarize_by(df, [rows, cols])
        return s[rate].where(s[min_col] >= min_val).unstack(cols)

    bp = rate_table("banner", "placement", "ctr", "displays", MIN_DISPLAYS)
    cb = rate_table("campaign_number", "banner", "ctr", "displays", MIN_DISPLAYS)
    cp = rate_table("campaign_number", "placement", "conversion_rate", "clicks", MIN_CLICKS)
    bp = bp.loc[bp.mean(axis=1).sort_values(ascending=False).index]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6),
                             gridspec_kw={"width_ratios": [1.1, 1.5, 1]})
    _heatmap(axes[0], bp, lambda v: f"{v:.2%}", "CTR: banner × placement")
    _heatmap(axes[1], cb, lambda v: f"{v:.2%}", "CTR: campaign × banner")
    _heatmap(axes[2], cp, lambda v: f"{v:.1%}", "Conversion rate: campaign × placement")
    fig.text(0.01, -0.03, f"Blank cells: never served, or fewer than {MIN_DISPLAYS:,} displays "
             f"(CTR) / {MIN_CLICKS:,} clicks (conversion rate).",
             fontsize=9, color=INK["secondary"])

    _save(fig, filepath, "Q13 · Effectiveness by campaign, ad size and placement")


def _chart_roi_consistency(df: pd.DataFrame, filepath: Path):
    """Q14: Do any campaigns or banner sizes consistently outperform on ROI?"""
    t = df.assign(week=df["date"].dt.to_period("W").dt.start_time)
    s = summarize_by(t, ["banner", "week"])
    # Weeks with almost no spend give meaningless ROI (e.g. +2,500% on a few dollars)
    weekly = s["firm_roi"].where(s["cost"] >= MIN_WEEKLY_COST).unstack("week").sort_index(axis=1)
    banners = summarize_by(df, "banner")
    banners = banners[banners["clicks"] >= MIN_CLICKS].sort_values("firm_roi", ascending=False)
    weekly = weekly.loc[banners.index]
    monthly = summarize_by(df, ["campaign_number", "month"])["firm_roi"].unstack("month")
    monthly = monthly.reindex(index=CAMPAIGN_ORDER, columns=MONTH_ORDER)

    cmap = LinearSegmentedColormap.from_list(
        "div", [DIVERGING["negative"], DIVERGING["neutral"], DIVERGING["positive"]])
    norm = TwoSlopeNorm(0, -1, 2)   # colours saturate at +200%

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.8), gridspec_kw={"width_ratios": [3.2, 1]})
    for ax, table, xlabels, title in [
        (axes[0], weekly, [d.strftime("%d %b") for d in weekly.columns],
         "Weekly Advert Firm A ROI by banner (week starting)"),
        (axes[1], monthly, monthly.columns, "Monthly Advert Firm A ROI by campaign"),
    ]:
        ax.imshow(np.ma.masked_invalid(table.values.astype(float)), cmap=cmap, norm=norm,
                  aspect="auto")
        ax.set_xticks(range(table.shape[1]), xlabels, rotation=45, ha="right")
        ax.set_yticks(range(table.shape[0]), table.index)
        ax.grid(False)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.iat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(np.clip(norm(v), 0, 1) - 0.5) > 0.35
                        else INK["primary"])
        ax.set_title(title)
    fig.text(0.01, -0.03, f"Blank: not served, or under ${MIN_WEEKLY_COST} of cost that week "
             "(camp 2 ran in April only). Banners with fewer than "
             f"{MIN_CLICKS:,} clicks left out; ordered by overall ROI. "
             "First and last weeks are partial.",
             fontsize=9, color=INK["secondary"])

    _save(fig, filepath, "Q14 · ROI consistency over time")


def _chart_weekend_engagement(df: pd.DataFrame, filepath: Path):
    """Q16: Do user-engagement levels differ between weekdays and weekends?"""
    t = df.assign(day_type=np.where(df["date"].dt.day_of_week >= 5, "Weekend", "Weekday"))
    s = summarize_by(t, ["user_engagement", "day_type"])
    days = t.groupby("day_type")["date"].nunique()
    clicks_per_day = s["clicks"].unstack("day_type").div(days, axis=1).reindex(ENGAGEMENT_ORDER)
    ctr = s["ctr"].unstack("day_type").reindex(ENGAGEMENT_ORDER)
    mix = s["clicks"].unstack("user_engagement")[ENGAGEMENT_ORDER]
    mix = mix.div(mix.sum(axis=1), axis=0)

    day_colors = {"Weekday": SEQUENTIAL_BLUE[2], "Weekend": SEQUENTIAL_BLUE[4]}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(ENGAGEMENT_ORDER))
    for ax, table, title, label, fmt in [
        (axes[0], clicks_per_day, "Average clicks per day", lambda v: f"{v/1000:,.1f}k",
         mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")),
        (axes[1], ctr, "CTR", lambda v: f"{v:.2%}", _pct),
    ]:
        for k, day_type in enumerate(["Weekday", "Weekend"]):
            bars = ax.bar(x + (k - 0.5) * 0.36, table[day_type], width=0.34,
                          color=day_colors[day_type],
                          label=f"{day_type} ({days[day_type]} days)")
            _label_bars(ax, bars, label)
        ax.set_xticks(x, ENGAGEMENT_ORDER)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(fmt)
    axes[0].legend()

    ax = axes[2]
    bottom = np.zeros(len(mix))
    for level in ENGAGEMENT_ORDER:
        vals = mix[level].values
        ax.bar(mix.index, vals, bottom=bottom, color=ENGAGEMENT_COLORS[level], label=level,
               width=0.55, edgecolor=INK["surface"], linewidth=1.5)
        for xi, (b, v) in enumerate(zip(bottom, vals)):
            if v >= 0.04:
                ax.text(xi, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white")
        bottom += vals
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(_pct0)
    ax.set_title("Click mix by engagement level")
    ax.legend(title="Engagement", loc="upper left", bbox_to_anchor=(1, 1))

    _save(fig, filepath, "Q16 · User engagement on weekdays vs weekends")
