"""
Generate synthetic online advertising performance data.

Creates a realistic dataset with campaigns, channels, devices,
and performance metrics for analysis and ML modeling.

Usage:
    python scripts/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_advertising_data(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic advertising performance dataset.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with advertising performance data.
    """
    rng = np.random.default_rng(seed)

    # --- Categorical dimensions ---
    campaigns = [
        "Brand Awareness", "Product Launch", "Holiday Sale",
        "Retargeting", "Lead Generation", "App Install"
    ]
    channels = ["Google Ads", "Facebook", "Instagram", "YouTube", "Email", "LinkedIn"]
    devices = ["Mobile", "Desktop", "Tablet"]

    # --- Date range: 6 months of daily data ---
    start_date = pd.Timestamp("2025-07-01")
    dates = pd.date_range(start=start_date, periods=180, freq="D")

    rows = []
    for _ in range(n_rows):
        campaign = rng.choice(campaigns)
        channel = rng.choice(channels)
        device = rng.choice(devices)
        date = rng.choice(dates)

        # Performance metrics with realistic distributions
        # Impressions: 500 – 100,000 (log-normal)
        impressions = int(np.clip(rng.lognormal(mean=8.5, sigma=1.2), 500, 100_000))

        # CTR varies by channel (search > social > display)
        base_ctr = {
            "Google Ads": 0.035, "Facebook": 0.012, "Instagram": 0.010,
            "YouTube": 0.008, "Email": 0.025, "LinkedIn": 0.015
        }[channel]
        ctr = np.clip(rng.normal(base_ctr, base_ctr * 0.3), 0.001, 0.15)
        clicks = max(1, int(impressions * ctr))

        # Cost: CPC varies by channel
        base_cpc = {
            "Google Ads": 2.50, "Facebook": 1.20, "Instagram": 1.40,
            "YouTube": 0.80, "Email": 0.30, "LinkedIn": 3.50
        }[channel]
        cpc = max(0.05, rng.normal(base_cpc, base_cpc * 0.25))
        spend = round(clicks * cpc, 2)

        # Conversions: conversion rate 1%–8%
        conv_rate = np.clip(rng.normal(0.04, 0.02), 0.005, 0.12)
        conversions = max(0, int(clicks * conv_rate))

        # Revenue per conversion: $10–$200
        rev_per_conv = max(5, rng.normal(60, 30))
        revenue = round(conversions * rev_per_conv, 2)

        # Bounce rate: 30%–80%
        bounce_rate = round(np.clip(rng.normal(0.50, 0.12), 0.20, 0.90), 4)

        # Session duration: 15–600 seconds
        avg_session_duration = round(np.clip(rng.normal(120, 60), 15, 600), 1)

        rows.append({
            "date": date,
            "campaign": campaign,
            "channel": channel,
            "device": device,
            "impressions": impressions,
            "clicks": clicks,
            "spend": spend,
            "conversions": conversions,
            "revenue": revenue,
            "bounce_rate": bounce_rate,
            "avg_session_duration": avg_session_duration,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    # --- Inject realistic data quality issues for cleaning practice ---
    # Sprinkle some missing values (~3%)
    for col in ["impressions", "clicks", "spend", "conversions", "revenue", "bounce_rate"]:
        mask = rng.random(n_rows) < 0.03
        df.loc[mask, col] = np.nan

    # Add a few duplicate rows (~1%)
    n_dupes = max(1, int(n_rows * 0.01))
    dupes = df.sample(n=n_dupes, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    # Introduce some inconsistent campaign names (case variations)
    inconsistent_idx = rng.choice(len(df), size=8, replace=False)
    for idx in inconsistent_idx:
        df.loc[idx, "campaign"] = df.loc[idx, "campaign"].upper()

    return df


def main():
    """Generate and save the advertising dataset."""
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "online_advertising_performance_data.csv"

    print("Generating synthetic advertising performance data...")
    df = generate_advertising_data(n_rows=500, seed=42)

    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")


if __name__ == "__main__":
    main()
