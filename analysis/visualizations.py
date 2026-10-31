"""
Exploratory data analysis charts for the advertising performance dataset.

Each chart answers one EDA question (see reports/eda_findings.md) and is
saved as a PNG to reports/figures/. All grouped rates are ratio-of-totals via
``analysis.metrics.summarize_by``.

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

def _fmt_money(x: float) -> str:
    """$1,234 / −$1,234; values under a dollar shown as <$1."""
    if abs(x) < 0.5:
        return "<$1" if x > 0 else ("−<$1" if x < 0 else "$0")
    return f"{'−' if x < 0 else ''}${abs(x):,.0f}"


def _fmt_compact(x: float) -> str:
    """1.2k / 3.4M style labels for log axes."""
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:g}{suffix}"
    return f"{x:g}"


_money = mticker.FuncFormatter(lambda x, _: _fmt_money(x))
_pct = mticker.PercentFormatter(1.0, decimals=2)
_compact = mticker.FuncFormatter(lambda x, _: _fmt_compact(x))


def _setup_style():
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

    _setup_style()

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
    """Q1: How did volume, revenue, cost and margin evolve over the period?"""
    daily = summarize_by(df, "date").sort_index()
    roll = daily.rolling(7, min_periods=1).mean()
    camp2_end = df.loc[df["campaign_number"] == "camp 2", "date"].max()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(daily.index, daily["clicks"], color=PRIMARY, alpha=0.25, linewidth=1)
    ax.plot(roll.index, roll["clicks"], color=PRIMARY, label="Clicks (7-day avg)")
    ax.set_title("Daily clicks")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax = axes[1]
    ax.plot(roll.index, roll["revenue"], color=CATEGORICAL[0], label="Revenue (Company X spend)")
    ax.plot(roll.index, roll["cost"], color=CATEGORICAL[1], label="Cost (placement cost)")
    ax.set_title("Revenue vs cost, 7-day average")
    ax.yaxis.set_major_formatter(_money)
    ax.legend(loc="upper right")

    ax = axes[2]
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

    _save(fig, filepath, "Q1 · Performance over time (Apr–Jun 2020)")


def _chart_monthly_by_campaign(df: pd.DataFrame, filepath: Path):
    """Q2: How does each month compare, and how much is the campaign mix?"""
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

    _save(fig, filepath, "Q2 · Monthly comparison")


def _chart_engagement_trends(df: pd.DataFrame, filepath: Path):
    """Q3: How did each user-engagement segment trend week to week?"""
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

    _save(fig, filepath, "Q3 · User-engagement segments over time")


# =====================================================================
# CREATIVE & INVENTORY
# =====================================================================

def _chart_dimension_panels(df, dim, filepath, title, order=None):
    """Clicks (volume), CTR (efficiency) and firm ROI for one dimension."""
    s = summarize_by(df, dim)
    s = s.reindex(order) if order else s.sort_values("clicks")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    bars = axes[0].barh(s.index, s["clicks"], color=PRIMARY, height=0.6)
    _label_bars(axes[0], bars, lambda v: f"{v:,.0f}", horizontal=True)
    axes[0].set_title("Total clicks (log scale)")
    axes[0].set_xscale("log")
    axes[0].set_xlim(left=1)
    axes[0].xaxis.set_major_locator(mticker.LogLocator(base=100))
    axes[0].xaxis.set_major_formatter(_compact)

    bars = axes[1].barh(s.index, s["ctr"], color=PRIMARY, height=0.6)
    _label_bars(axes[1], bars, lambda v: f"{v:.2%}", horizontal=True)
    axes[1].set_title("CTR (clicks ÷ displays)")
    axes[1].xaxis.set_major_formatter(_pct)

    colors = [DIVERGING["positive"] if v >= 0 else DIVERGING["negative"] for v in s["firm_roi"]]
    bars = axes[2].barh(s.index, s["firm_roi"], color=colors, height=0.6)
    _label_bars(axes[2], bars, lambda v: f"{v:.0%}", horizontal=True)
    axes[2].axvline(0, color=INK["muted"], linewidth=1)
    axes[2].set_title("Advert Firm A ROI")
    axes[2].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    for ax in axes:
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
    _save(fig, filepath, title)


def _chart_banner_performance(df: pd.DataFrame, filepath: Path):
    """Q4: How does banner size affect clicks and click-through?"""
    _chart_dimension_panels(df, "banner", filepath, "Q4 · Banner size: volume, CTR and margin")


def _chart_placement_performance(df: pd.DataFrame, filepath: Path):
    """Q5: Which placements yield the most displays, clicks and margin?"""
    _chart_dimension_panels(df, "placement", filepath,
                            "Q5 · Placement: volume, CTR and margin")


def _chart_banner_placement_margin(df: pd.DataFrame, filepath: Path):
    """Q9: Which banner × placement combinations make or lose money for the firm?"""
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
            ax.text(j, i, _fmt_money(v), ha="center", va="center", fontsize=8,
                    color="white" if strong else INK["primary"])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    ticks = np.array([-1e4, -1e3, -1e2, 0, 1e2, 1e3, 1e4])
    ticks = ticks[np.abs(ticks) <= limit]
    cbar.set_ticks(np.sign(ticks) * np.log10(1 + np.abs(ticks)))
    cbar.set_ticklabels([_fmt_money(t) for t in ticks])
    cbar.set_label("Firm margin (log-compressed)")
    ax.set_xlabel("Placement")
    ax.set_ylabel("Banner")

    _save(fig, filepath, "Q9 · Firm margin by banner × placement")


# =====================================================================
# RELATIONSHIPS
# =====================================================================

def _chart_cost_vs_revenue(df: pd.DataFrame, filepath: Path):
    """Q6: Is there a relationship between cost and revenue?"""
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

    _save(fig, filepath, "Q6 · Cost vs revenue (each dot = one row)")


def _chart_correlation_heatmap(df: pd.DataFrame, filepath: Path):
    """Q7: How are the raw metrics correlated (rank-based, robust to skew)?"""
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

    _save(fig, filepath, "Q7 · Rank correlation between metrics")


def _chart_conversions_vs_clicks(df: pd.DataFrame, filepath: Path):
    """Q12: Do clicks translate into post-click conversions and sales?"""
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

    _save(fig, filepath, "Q12 · From clicks to conversions")


# =====================================================================
# SEGMENTS, DISTRIBUTION, CALENDAR
# =====================================================================

def _chart_campaign_engagement_rates(df: pd.DataFrame, filepath: Path):
    """Q8: How do campaigns and engagement segments compare on rates?"""
    s = summarize_by(df, ["campaign_number", "user_engagement"])
    metrics = [("ctr", "CTR", _pct),
               ("firm_roi", "Advert Firm A ROI", mticker.PercentFormatter(1.0)),
               ("advertiser_roas", "Company X ROAS (sales ÷ spend)",
                mticker.FuncFormatter(lambda x, _: f"{x:,.0f}×"))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(CAMPAIGN_ORDER))
    width = 0.26
    for ax, (col, title, fmt) in zip(axes, metrics):
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
    axes[0].legend(title="Engagement")

    _save(fig, filepath, "Q8 · Campaign × engagement segment rates")


def _chart_distributions(df: pd.DataFrame, filepath: Path):
    """Q10: How skewed are the metrics, and how concentrated is revenue?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

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
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="x")

    _save(fig, filepath, "Q10 · Skew and concentration")


def _chart_weekday_patterns(df: pd.DataFrame, filepath: Path):
    """Q11: Is there a day-of-week pattern (using daily averages)?"""
    s = summarize_by(df, "weekday").reindex(WEEKDAY_ORDER)
    per_day = s[["revenue", "clicks"]].div(s["active_days"], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    short = [d[:3] for d in WEEKDAY_ORDER]

    bars = axes[0].bar(short, per_day["revenue"], color=PRIMARY, width=0.6)
    _label_bars(axes[0], bars, lambda v: f"${v:,.0f}")
    axes[0].set_title("Average revenue per day")
    axes[0].yaxis.set_major_formatter(_money)

    bars = axes[1].bar(short, per_day["clicks"], color=PRIMARY, width=0.6)
    _label_bars(axes[1], bars, lambda v: f"{v/1000:,.1f}k")
    axes[1].set_title("Average clicks per day")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    bars = axes[2].bar(short, s["ctr"], color=PRIMARY, width=0.6)
    _label_bars(axes[2], bars, lambda v: f"{v:.2%}")
    axes[2].set_title("CTR")
    axes[2].yaxis.set_major_formatter(_pct)

    lo, hi = int(s["active_days"].min()), int(s["active_days"].max())
    fig.text(0.01, -0.02, f"Each weekday occurs {lo if lo == hi else f'{lo}–{hi}'} times "
             "in the period, so the overall downward trend is spread evenly across them.",
             fontsize=9, color=INK["secondary"])
    _save(fig, filepath, "Q11 · Day-of-week pattern")
