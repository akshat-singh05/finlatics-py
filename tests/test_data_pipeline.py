"""
Tests for data loading, cleaning and KPI computation.

Run:  pytest
"""

import numpy as np
import pandas as pd
import pytest

from analysis.data_loader import DEFAULT_RAW_PATH, EXPECTED_COLUMNS, load_data
from analysis.data_cleaning import UNKNOWN_PLACEMENT, clean_data
from analysis.metrics import compute_kpis, get_overall_summary, summarize_by


RAW_HEADER = ",".join(EXPECTED_COLUMNS) + ",,"


def _raw_frame(rows):
    """Build a raw-schema DataFrame from (month, day, ..., sales) tuples."""
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


@pytest.fixture
def raw_df():
    return _raw_frame([
        ("April", 1, "camp 1", "High", "160 x 600", "abc", 1000, 10.0, 20, 15.0, 5, 250.0),
        ("April", 1, "camp 1", "High", "160 x 600", "abc", 1000, 10.0, 20, 15.0, 5, 250.0),  # dup
        ("April", 2, " Camp 2 ", "low", "728X90", None, 500, 2.0, 0, 0.0, 1, 40.0),
        ("June", 30, "camp 3", "Medium", "300 x 250", "def", 0, 0.5, 0, 0.0, 0, 0.0),
    ])


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def test_load_data_drops_empty_trailing_columns(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(RAW_HEADER + "\nApril,1,camp 1,High,160 x 600,abc,4,0.006,0,0,0,0,,\n")
    df = load_data(csv)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 1


def test_load_data_missing_column_raises(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("month,day\nApril,1\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_data(csv)


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")


# ---------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------

def test_clean_data_report_and_rules(raw_df):
    df, report = clean_data(raw_df, return_report=True)

    assert report["duplicates_removed"] == 1
    assert report["missing_placement_filled"] == 1
    assert len(df) == 3

    # Categorical normalization
    assert set(df["campaign_number"]) == {"camp 1", "camp 2", "camp 3"}
    assert "728 x 90" in set(df["banner"])
    assert UNKNOWN_PLACEMENT in set(df["placement"])
    assert list(df["user_engagement"].cat.categories) == ["Low", "Medium", "High"]

    # Date construction
    assert df["date"].min() == pd.Timestamp("2020-04-01")
    assert df["date"].max() == pd.Timestamp("2020-06-30")

    # Conversions with zero clicks are flagged, not dropped
    assert report["flags"]["flag_conv_without_clicks"] == 1
    assert report["flags"]["flag_conv_gt_clicks"] == 1


def test_clean_data_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_data_rejects_negative_values(raw_df):
    raw_df.loc[0, "cost"] = -1.0
    with pytest.raises(ValueError, match="Negative values"):
        clean_data(raw_df)


def test_clean_data_rejects_invalid_date(raw_df):
    raw_df.loc[3, "day"] = 31  # June 31 does not exist
    with pytest.raises(ValueError, match="invalid month/day"):
        clean_data(raw_df)


# ---------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------

def test_row_kpis_nan_on_zero_denominator(raw_df):
    df = compute_kpis(clean_data(raw_df))
    first = df.iloc[0]
    assert first["ctr"] == pytest.approx(0.02)
    assert first["firm_margin"] == pytest.approx(5.0)
    assert first["firm_roi"] == pytest.approx(0.5)
    assert first["advertiser_roas"] == pytest.approx(250 / 15)

    zero_displays = df[df["displays"] == 0].iloc[0]
    assert np.isnan(zero_displays["ctr"])
    assert np.isnan(zero_displays["firm_cpc"])


def test_summaries_use_ratio_of_totals():
    df = _raw_frame([
        ("April", 1, "camp 1", "High", "160 x 600", "abc", 100, 1.0, 10, 2.0, 1, 10.0),
        ("April", 2, "camp 1", "High", "160 x 600", "abc", 900, 9.0, 10, 18.0, 1, 10.0),
    ])
    df = compute_kpis(clean_data(df))

    summary = summarize_by(df, "campaign_number").loc["camp 1"]
    # Ratio of totals: 20 / 1000, not the mean of 0.1 and 0.0111
    assert summary["ctr"] == pytest.approx(0.02)
    assert summary["active_days"] == 2
    assert summary["revenue_share"] == pytest.approx(1.0)

    overall = get_overall_summary(df, verbose=False)
    assert overall["ctr"] == pytest.approx(0.02)
    assert overall["total_revenue"] == pytest.approx(20.0)


# ---------------------------------------------------------------------
# Real dataset (integration)
# ---------------------------------------------------------------------

@pytest.mark.skipif(not DEFAULT_RAW_PATH.exists(), reason="raw dataset not present")
def test_real_dataset_end_to_end():
    df, report = clean_data(load_data(), return_report=True)
    df = compute_kpis(df)

    assert report["initial_rows"] == 15408
    assert report["duplicates_removed"] == 5
    assert report["missing_placement_filled"] == 413
    assert report["grain_duplicates"] == 0
    assert df["date"].nunique() == 91
    assert set(df["campaign_number"]) == {"camp 1", "camp 2", "camp 3"}
    assert df["banner"].nunique() == 8


@pytest.mark.skipif(not DEFAULT_RAW_PATH.exists(), reason="raw dataset not present")
def test_real_dataset_official_question_findings(tmp_path):
    """Spot-check answers to the 20 official questions (reports/eda_findings.md)."""
    from analysis.insights import generate_insights
    from analysis.visualizations import generate_all_charts

    df = compute_kpis(clean_data(load_data()))
    f = generate_insights(df, tmp_path)

    assert (tmp_path / "eda_findings.json").exists()
    assert f["revenue_per_click"]["overall"] == pytest.approx(0.1108, abs=1e-4)          # Q5
    assert f["revenue_per_click"]["by_campaign"]["camp 2"] == pytest.approx(0.0396, abs=1e-4)
    assert f["sales_trend"]["camp1_sales_share"] == pytest.approx(0.948, abs=1e-3)      # Q7
    assert f["engagement_by_banner"]["banners_where_medium_has_best_ctr"] == 8          # Q8
    assert f["placement_conversions"]["lowest_conversion_rate"] == "def"                # Q9
    assert f["outliers"]["clicks_but_zero_revenue_rows"] == 206                         # Q12
    assert f["outliers"]["clicks"]["log_iqr_upper_outliers"] == 0
    assert f["roi_consistency"]["weeks_as_top_banner_by_roi"] == {"240 x 400": 14}      # Q14
    assert f["roi_consistency"]["banners_profitable_every_month"] == ["240 x 400"]
    assert f["weekend_engagement"]["days"] == {"weekday": 65, "weekend": 26}            # Q16
    cpa = f["cost_per_conversion"]["by_campaign"]                                       # Q18
    assert min(cpa, key=cpa.get) == "camp 1"

    saved = generate_all_charts(df, tmp_path / "figures")
    assert len(saved) == 19
