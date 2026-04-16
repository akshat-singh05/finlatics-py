"""
Professional visualization suite for advertising performance data.

Generates publication-quality charts using matplotlib and seaborn,
saving all outputs to the output/ directory.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


# --- Style Configuration ---
COLORS = {
    "primary": "#6366F1",    # Indigo
    "secondary": "#EC4899",  # Pink
    "success": "#10B981",    # Emerald
    "warning": "#F59E0B",    # Amber
    "danger": "#EF4444",     # Red
    "info": "#3B82F6",       # Blue
}

PALETTE = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6", "#8B5CF6"]

def _setup_style():
    """Apply a clean, modern chart style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#E5E7EB",
        "grid.color": "#F3F4F6",
        "text.color": "#1F2937",
        "axes.labelcolor": "#374151",
        "xtick.color": "#6B7280",
        "ytick.color": "#6B7280",
        "font.family": "sans-serif",
        "figure.dpi": 150,
    })


def generate_all_charts(df: pd.DataFrame, output_dir: str | Path = "output") -> list[str]:
    """
    Generate all visualization charts and save them to disk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with KPI columns computed.
    output_dir : str or Path
        Directory to save chart images.

    Returns
    -------
    list[str]
        Paths to all saved chart files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    _setup_style()

    saved = []
    chart_funcs = [
        ("01_campaign_performance.png", _chart_campaign_performance),
        ("02_channel_breakdown.png", _chart_channel_breakdown),
        ("03_spend_vs_revenue.png", _chart_spend_vs_revenue),
        ("04_ctr_trends.png", _chart_ctr_trends),
        ("05_correlation_heatmap.png", _chart_correlation_heatmap),
        ("06_device_distribution.png", _chart_device_distribution),
        ("07_roi_comparison.png", _chart_roi_comparison),
        ("08_monthly_trends.png", _chart_monthly_trends),
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


def _chart_campaign_performance(df: pd.DataFrame, filepath: Path):
    """Bar chart comparing total spend and revenue by campaign."""
    fig, ax = plt.subplots(figsize=(12, 6))

    campaign_data = df.groupby("campaign").agg(
        total_spend=("spend", "sum"),
        total_revenue=("revenue", "sum")
    ).sort_values("total_revenue", ascending=True)

    y = range(len(campaign_data))
    bar_height = 0.35

    ax.barh([i + bar_height / 2 for i in y], campaign_data["total_revenue"],
            height=bar_height, color=COLORS["success"], label="Revenue", alpha=0.9)
    ax.barh([i - bar_height / 2 for i in y], campaign_data["total_spend"],
            height=bar_height, color=COLORS["danger"], label="Spend", alpha=0.9)

    ax.set_yticks(list(y))
    ax.set_yticklabels(campaign_data.index)
    ax.set_xlabel("Amount ($)")
    ax.set_title("Campaign Performance: Spend vs Revenue", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _chart_channel_breakdown(df: pd.DataFrame, filepath: Path):
    """Donut chart showing spend distribution by channel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, title in zip(
        axes,
        ["spend", "revenue"],
        ["Spend by Channel", "Revenue by Channel"]
    ):
        channel_data = df.groupby("channel")[metric].sum().sort_values(ascending=False)
        wedges, texts, autotexts = ax.pie(
            channel_data, labels=channel_data.index, autopct="%1.1f%%",
            colors=PALETTE, startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2)
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    fig.suptitle("Channel Breakdown", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _chart_spend_vs_revenue(df: pd.DataFrame, filepath: Path):
    """Scatter plot of spend vs revenue colored by campaign."""
    fig, ax = plt.subplots(figsize=(10, 7))

    campaigns = df["campaign"].unique()
    for i, campaign in enumerate(campaigns):
        mask = df["campaign"] == campaign
        ax.scatter(df.loc[mask, "spend"], df.loc[mask, "revenue"],
                   alpha=0.6, s=40, color=PALETTE[i % len(PALETTE)],
                   label=campaign, edgecolors="white", linewidth=0.5)

    # Break-even line
    max_val = max(df["spend"].max(), df["revenue"].max())
    ax.plot([0, max_val], [0, max_val], "--", color="#9CA3AF", linewidth=1, alpha=0.7, label="Break-even")

    ax.set_xlabel("Spend ($)")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Spend vs Revenue by Campaign", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _chart_ctr_trends(df: pd.DataFrame, filepath: Path):
    """Line chart showing CTR trends over time by channel."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Weekly aggregation for smoother trends
    df_ts = df.copy()
    df_ts["week"] = df_ts["date"].dt.to_period("W").apply(lambda r: r.start_time)

    for i, channel in enumerate(df["channel"].unique()):
        channel_data = df_ts[df_ts["channel"] == channel].groupby("week")["ctr"].mean()
        ax.plot(channel_data.index, channel_data.values * 100,
                color=PALETTE[i % len(PALETTE)], label=channel,
                linewidth=2, alpha=0.85, marker="o", markersize=3)

    ax.set_xlabel("Week")
    ax.set_ylabel("CTR (%)")
    ax.set_title("Click-Through Rate Trends by Channel", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _chart_correlation_heatmap(df: pd.DataFrame, filepath: Path):
    """Heatmap of correlations between numeric features."""
    fig, ax = plt.subplots(figsize=(10, 8))

    numeric_cols = ["impressions", "clicks", "spend", "conversions",
                    "revenue", "bounce_rate", "ctr", "cpc", "roas", "roi"]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8})

    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _chart_device_distribution(df: pd.DataFrame, filepath: Path):
    """Grouped bar chart showing performance metrics by device."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    device_data = df.groupby("device").agg(
        avg_ctr=("ctr", "mean"),
        avg_conversion_rate=("conversion_rate", "mean"),
        avg_roas=("roas", "mean"),
    )

    for ax, col, title, fmt in zip(
        axes,
        ["avg_ctr", "avg_conversion_rate", "avg_roas"],
        ["Avg CTR", "Avg Conversion Rate", "Avg ROAS"],
        ["{:.2%}", "{:.2%}", "{:.2f}x"]
    ):
        bars = ax.bar(device_data.index, device_data[col],
                      color=PALETTE[:len(device_data)], alpha=0.9, edgecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(title)

        # Value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    fmt.format(height), ha="center", va="bottom", fontsize=10)

    fig.suptitle("Performance by Device Type", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _chart_roi_comparison(df: pd.DataFrame, filepath: Path):
    """Horizontal bar chart comparing ROI across campaigns and channels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, group_col, title in zip(
        axes,
        ["campaign", "channel"],
        ["ROI by Campaign", "ROI by Channel"]
    ):
        roi_data = df.groupby(group_col)["roi"].mean().sort_values()
        colors = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in roi_data]

        ax.barh(roi_data.index, roi_data.values, color=colors, alpha=0.9, edgecolor="white")
        ax.axvline(x=0, color="#9CA3AF", linewidth=1, linestyle="--")
        ax.set_xlabel("Average ROI")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _chart_monthly_trends(df: pd.DataFrame, filepath: Path):
    """Multi-panel line chart showing monthly spend, revenue, and conversions."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    df_monthly = df.copy()
    df_monthly["month"] = df_monthly["date"].dt.to_period("M").apply(lambda r: r.start_time)

    monthly = df_monthly.groupby("month").agg(
        total_spend=("spend", "sum"),
        total_revenue=("revenue", "sum"),
        total_conversions=("conversions", "sum"),
    )

    configs = [
        ("total_spend", "Monthly Spend", COLORS["danger"], "$"),
        ("total_revenue", "Monthly Revenue", COLORS["success"], "$"),
        ("total_conversions", "Monthly Conversions", COLORS["info"], ""),
    ]

    for ax, (col, title, color, prefix) in zip(axes, configs):
        ax.fill_between(monthly.index, monthly[col], alpha=0.15, color=color)
        ax.plot(monthly.index, monthly[col], color=color, linewidth=2.5, marker="o", markersize=6)
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if prefix == "$":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    axes[-1].set_xlabel("Month")
    fig.suptitle("Monthly Performance Trends", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
