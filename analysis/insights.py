"""
Business insights engine.

Converts raw metrics into actionable, quantified business recommendations:
  - Budget allocation optimization
  - Channel/campaign performance rankings
  - ROI opportunity identification
  - Efficiency analysis with annotations
  - Enhanced charts with highlights and callouts
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path


PALETTE = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6", "#8B5CF6"]
COLORS = {
    "primary": "#6366F1",
    "success": "#10B981",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "info": "#3B82F6",
    "gold": "#D97706",
}


def generate_insights(df: pd.DataFrame, output_dir: str | Path = "output") -> dict:
    """
    Generate comprehensive business insights with enhanced visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with KPI columns.
    output_dir : str or Path
        Directory to save charts.

    Returns
    -------
    dict
        Structured insights with recommendations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)

    insights = {}

    # 1. Channel efficiency analysis
    print("\n[1/5] Analyzing channel efficiency...")
    insights["channel"] = _analyze_channels(df, output_dir)

    # 2. Campaign performance ranking
    print("\n[2/5] Ranking campaign performance...")
    insights["campaign"] = _analyze_campaigns(df, output_dir)

    # 3. Budget allocation recommendation
    print("\n[3/5] Computing budget allocation...")
    insights["budget"] = _recommend_budget(df, output_dir)

    # 4. Device performance analysis
    print("\n[4/5] Analyzing device performance...")
    insights["device"] = _analyze_devices(df, output_dir)

    # 5. Time-based opportunities
    print("\n[5/5] Identifying temporal opportunities...")
    insights["time"] = _analyze_time_patterns(df, output_dir)

    # Print executive summary
    _print_executive_summary(insights, df)

    print("=" * 60)
    return insights


# =====================================================================
# 1. CHANNEL ANALYSIS
# =====================================================================

def _analyze_channels(df: pd.DataFrame, output_dir: Path) -> dict:
    """Rank channels by ROAS and identify top/bottom performers."""
    channel_stats = df.groupby("channel").agg(
        total_spend=("spend", "sum"),
        total_revenue=("revenue", "sum"),
        total_clicks=("clicks", "sum"),
        total_conversions=("conversions", "sum"),
        avg_ctr=("ctr", "mean"),
        avg_cpc=("cpc", "mean"),
        avg_roas=("roas", "mean"),
        count=("spend", "count"),
    ).sort_values("avg_roas", ascending=False)

    channel_stats["roas_actual"] = (
        channel_stats["total_revenue"] / channel_stats["total_spend"]
    ).round(4)
    channel_stats["profit"] = (
        channel_stats["total_revenue"] - channel_stats["total_spend"]
    ).round(2)

    best = channel_stats.index[0]
    worst = channel_stats.index[-1]

    # Enhanced chart: Channel ROAS with profit annotations
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(channel_stats))
    roas_vals = channel_stats["roas_actual"]
    colors = [COLORS["success"] if v >= 1.5 else COLORS["warning"] if v >= 1.0 else COLORS["danger"]
              for v in roas_vals]

    bars = ax.bar(x, roas_vals, color=colors, alpha=0.9, edgecolor="white", linewidth=1.5)

    # Break-even line
    ax.axhline(y=1.0, color="#9CA3AF", linewidth=1.5, linestyle="--", label="Break-even (1.0x)")

    # Annotations with profit amounts
    for i, (bar, (ch, row)) in enumerate(zip(bars, channel_stats.iterrows())):
        height = bar.get_height()
        profit = row["profit"]
        color = COLORS["success"] if profit > 0 else COLORS["danger"]
        sign = "+" if profit > 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f"{height:.2f}x\n{sign}${profit:,.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)

    # Highlight best performer
    ax.annotate("BEST", xy=(0, roas_vals.iloc[0]), fontsize=9, fontweight="bold",
                color=COLORS["gold"], ha="center",
                xytext=(0, roas_vals.iloc[0] + 1.5),
                arrowprops=dict(arrowstyle="->", color=COLORS["gold"], lw=2))

    ax.set_xticks(list(x))
    ax.set_xticklabels(channel_stats.index, fontsize=10)
    ax.set_ylabel("ROAS (Return on Ad Spend)")
    ax.set_title("Channel ROAS with Profit/Loss", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(roas_vals) + 3)

    plt.tight_layout()
    fig.savefig(output_dir / "16_channel_efficiency.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: 16_channel_efficiency.png")

    result = {
        "best_channel": best,
        "worst_channel": worst,
        "best_roas": round(roas_vals.iloc[0], 2),
        "worst_roas": round(roas_vals.iloc[-1], 2),
        "stats": channel_stats,
    }

    print(f"  Best channel: {best} (ROAS: {result['best_roas']}x)")
    print(f"  Worst channel: {worst} (ROAS: {result['worst_roas']}x)")
    return result


# =====================================================================
# 2. CAMPAIGN ANALYSIS
# =====================================================================

def _analyze_campaigns(df: pd.DataFrame, output_dir: Path) -> dict:
    """Rank campaigns by efficiency score (composite metric)."""
    camp_stats = df.groupby("campaign").agg(
        total_spend=("spend", "sum"),
        total_revenue=("revenue", "sum"),
        total_conversions=("conversions", "sum"),
        avg_ctr=("ctr", "mean"),
        avg_conversion_rate=("conversion_rate", "mean"),
        avg_roas=("roas", "mean"),
    )

    # Composite efficiency score (normalized)
    for col in ["avg_ctr", "avg_conversion_rate", "avg_roas"]:
        max_val = camp_stats[col].max()
        if max_val > 0:
            camp_stats[f"{col}_norm"] = camp_stats[col] / max_val

    camp_stats["efficiency_score"] = (
        camp_stats["avg_ctr_norm"] * 0.2 +
        camp_stats["avg_conversion_rate_norm"] * 0.3 +
        camp_stats["avg_roas_norm"] * 0.5
    ).round(4)

    camp_stats = camp_stats.sort_values("efficiency_score", ascending=False)

    # Enhanced chart: Campaign scorecard
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Efficiency score
    ax = axes[0]
    scores = camp_stats["efficiency_score"]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(scores))]
    bars = ax.barh(scores.index[::-1], scores.values[::-1], color=colors[::-1], alpha=0.9, edgecolor="white")

    for bar, score in zip(bars, scores.values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
                f"{score:.2f}", va="center", fontsize=10, fontweight="bold")

    # Gold star on best
    ax.text(scores.values[-1] + 0.04, len(scores) - 1,
            "BEST", fontsize=9, fontweight="bold", color=COLORS["gold"], va="center")

    ax.set_xlabel("Efficiency Score")
    ax.set_title("Campaign Efficiency Ranking", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.15)

    # Right: Spend vs Revenue with profit labels
    ax = axes[1]
    camps = camp_stats.index
    spend = camp_stats["total_spend"]
    revenue = camp_stats["total_revenue"]

    y = range(len(camps))
    ax.barh([i + 0.15 for i in y], revenue, height=0.3, color=COLORS["success"], label="Revenue", alpha=0.9)
    ax.barh([i - 0.15 for i in y], spend, height=0.3, color=COLORS["danger"], label="Spend", alpha=0.9)

    for i, (s, r) in enumerate(zip(spend, revenue)):
        profit = r - s
        sign = "+" if profit > 0 else ""
        col = COLORS["success"] if profit > 0 else COLORS["danger"]
        ax.text(max(s, r) + 200, i, f"{sign}${profit:,.0f}", va="center", fontsize=9, fontweight="bold", color=col)

    ax.set_yticks(list(y))
    ax.set_yticklabels(camps)
    ax.set_xlabel("Amount ($)")
    ax.set_title("Revenue vs Spend (with Profit)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    fig.savefig(output_dir / "17_campaign_scorecard.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: 17_campaign_scorecard.png")

    best = camp_stats.index[0]
    print(f"  Top campaign: {best} (score: {camp_stats.loc[best, 'efficiency_score']:.2f})")
    return {"ranking": camp_stats, "best_campaign": best}


# =====================================================================
# 3. BUDGET RECOMMENDATION
# =====================================================================

def _recommend_budget(df: pd.DataFrame, output_dir: Path) -> dict:
    """Compute optimal budget allocation based on channel ROAS."""
    channel_perf = df.groupby("channel").agg(
        total_spend=("spend", "sum"),
        total_revenue=("revenue", "sum"),
    )
    channel_perf["roas"] = (channel_perf["total_revenue"] / channel_perf["total_spend"]).round(4)
    channel_perf["profit"] = channel_perf["total_revenue"] - channel_perf["total_spend"]

    total_budget = channel_perf["total_spend"].sum()

    # Current allocation
    channel_perf["current_pct"] = (channel_perf["total_spend"] / total_budget * 100).round(1)

    # Recommended allocation: weight by ROAS (normalize so sum = 100%)
    roas_positive = channel_perf["roas"].clip(lower=0)
    channel_perf["recommended_pct"] = (roas_positive / roas_positive.sum() * 100).round(1)

    # Budget shift
    channel_perf["shift_pct"] = (channel_perf["recommended_pct"] - channel_perf["current_pct"]).round(1)
    channel_perf["shift_amount"] = (channel_perf["shift_pct"] / 100 * total_budget).round(0)

    channel_perf = channel_perf.sort_values("roas", ascending=False)

    # Enhanced chart: Current vs Recommended allocation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = channel_perf.index
    current = channel_perf["current_pct"]
    recommended = channel_perf["recommended_pct"]

    # Donut: Current
    axes[0].pie(current, labels=labels, autopct="%1.1f%%", colors=PALETTE[:len(labels)],
                startangle=90, pctdistance=0.75,
                wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2))
    axes[0].set_title("Current Budget Allocation", fontsize=13, fontweight="bold")

    # Donut: Recommended
    axes[1].pie(recommended, labels=labels, autopct="%1.1f%%", colors=PALETTE[:len(labels)],
                startangle=90, pctdistance=0.75,
                wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2))
    axes[1].set_title("Recommended Allocation (by ROAS)", fontsize=13, fontweight="bold")

    fig.suptitle("Budget Optimization: Current vs Recommended",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "18_budget_allocation.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: 18_budget_allocation.png")

    # Print recommendations
    print(f"\n  Budget Reallocation Recommendations (total: ${total_budget:,.0f}):")
    for ch, row in channel_perf.iterrows():
        direction = "INCREASE" if row["shift_pct"] > 0 else "DECREASE" if row["shift_pct"] < 0 else "KEEP"
        sign = "+" if row["shift_amount"] > 0 else ""
        print(f"    {ch:<15s} {row['current_pct']:5.1f}% -> {row['recommended_pct']:5.1f}%  "
              f"({direction} {sign}${row['shift_amount']:,.0f})")

    return {"allocation": channel_perf, "total_budget": total_budget}


# =====================================================================
# 4. DEVICE ANALYSIS
# =====================================================================

def _analyze_devices(df: pd.DataFrame, output_dir: Path) -> dict:
    """Analyze performance differences across device types."""
    device_stats = df.groupby("device").agg(
        total_spend=("spend", "sum"),
        total_revenue=("revenue", "sum"),
        avg_ctr=("ctr", "mean"),
        avg_conversion_rate=("conversion_rate", "mean"),
        avg_roas=("roas", "mean"),
        avg_bounce=("bounce_rate", "mean"),
        count=("spend", "count"),
    ).sort_values("avg_roas", ascending=False)

    device_stats["roas_actual"] = (device_stats["total_revenue"] / device_stats["total_spend"]).round(4)

    best_device = device_stats.index[0]
    print(f"  Best device: {best_device} (ROAS: {device_stats.loc[best_device, 'roas_actual']:.2f}x)")

    return {"stats": device_stats, "best_device": best_device}


# =====================================================================
# 5. TIME PATTERNS
# =====================================================================

def _analyze_time_patterns(df: pd.DataFrame, output_dir: Path) -> dict:
    """Identify best-performing days and trends."""
    if "date" not in df.columns:
        print("  [SKIP] No date column for time analysis")
        return {}

    df_t = df.copy()
    df_t["weekday"] = pd.to_datetime(df_t["date"]).dt.day_name()

    weekday_stats = df_t.groupby("weekday").agg(
        avg_roas=("roas", "mean"),
        avg_ctr=("ctr", "mean"),
        total_conversions=("conversions", "sum"),
    )

    # Order by day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_stats = weekday_stats.reindex([d for d in day_order if d in weekday_stats.index])

    best_day = weekday_stats["avg_roas"].idxmax()
    print(f"  Best day for ROAS: {best_day} ({weekday_stats.loc[best_day, 'avg_roas']:.2f}x)")

    return {"weekday_stats": weekday_stats, "best_day": best_day}


# =====================================================================
# EXECUTIVE SUMMARY
# =====================================================================

def _print_executive_summary(insights: dict, df: pd.DataFrame):
    """Print a concise, actionable executive summary."""
    total_spend = df["spend"].sum()
    total_revenue = df["revenue"].sum()
    overall_roi = (total_revenue - total_spend) / total_spend

    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    print(f"\n  Overall Performance:")
    print(f"    Total Spend:    ${total_spend:>12,.2f}")
    print(f"    Total Revenue:  ${total_revenue:>12,.2f}")
    print(f"    Net Profit:     ${total_revenue - total_spend:>12,.2f}")
    print(f"    Overall ROI:    {overall_roi:>11.1%}")

    ch = insights.get("channel", {})
    if ch:
        print(f"\n  Channel Insights:")
        print(f"    Best channel:   {ch['best_channel']} ({ch['best_roas']}x ROAS)")
        print(f"    Worst channel:  {ch['worst_channel']} ({ch['worst_roas']}x ROAS)")

    camp = insights.get("campaign", {})
    if camp:
        print(f"\n  Campaign Insights:")
        print(f"    Top campaign:   {camp['best_campaign']}")

    budget = insights.get("budget", {})
    if budget:
        alloc = budget["allocation"]
        top_increase = alloc["shift_pct"].idxmax()
        top_decrease = alloc["shift_pct"].idxmin()
        print(f"\n  Key Recommendations:")
        print(f"    -> INCREASE budget for: {top_increase} "
              f"(+{alloc.loc[top_increase, 'shift_pct']:.1f}%)")
        print(f"    -> DECREASE budget for: {top_decrease} "
              f"({alloc.loc[top_decrease, 'shift_pct']:.1f}%)")

    time_data = insights.get("time", {})
    if time_data:
        print(f"    -> Best day to advertise: {time_data.get('best_day', 'N/A')}")

    print()
