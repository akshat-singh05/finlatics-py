"""
Finlatics-Py: Interactive Analytics Dashboard

Run:  streamlit run streamlit_app.py

Every number comes from the same modules as main.py and the API:
analysis.* for data and KPIs, reports/*.json for findings and model
evaluation, and models.predictor for predictions.
"""

import io
import json
import contextlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.data_loader import DEFAULT_RAW_PATH, load_data
from analysis.data_cleaning import ENGAGEMENT_ORDER, clean_data
from analysis.metrics import compute_kpis, get_overall_summary, summarize_by
from analysis.visualizations import fmt_money as report_money
from models import predictor
from utils.logger import get_logger
from utils.theme import CATEGORICAL, DIVERGING, SEQUENTIAL_BLUE

logger = get_logger("dashboard")

REPORTS_DIR = PROJECT_ROOT / "reports"
SHAP_CSV = PROJECT_ROOT / "output" / "ml" / "shap_importance.csv"

FILTERS = {"campaign_number": "Campaign", "user_engagement": "Engagement",
           "banner": "Banner", "placement": "Placement"}
DIMENSIONS = {**FILTERS, "month": "Month", "weekday": "Weekday"}
ORDERS = {"user_engagement": ENGAGEMENT_ORDER, "month": ["April", "May", "June"],
          "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]}
POSITIVE, NEGATIVE, PRIMARY = DIVERGING["positive"], DIVERGING["negative"], CATEGORICAL[0]

st.set_page_config(page_title="Finlatics Ad Analytics", page_icon="📊", layout="wide")


# =====================================================================
# DATA
# =====================================================================

@st.cache_data(show_spinner="Loading and cleaning data...")
def load_dataset() -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):  # pipeline prints reports
        df = compute_kpis(clean_data(load_data(DEFAULT_RAW_PATH)))
    df["user_engagement"] = df["user_engagement"].astype(str)
    logger.info("Dashboard data loaded: %d rows", len(df))
    return df


@st.cache_data
def load_report(name: str) -> dict | None:
    path = REPORTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_money(x: float, decimals: int = 0) -> str:
    """Money label; whole dollars reuse the report formatter (handles <$1)."""
    if pd.isna(x):
        return ""
    if decimals == 0:
        return report_money(x)
    return f"{'−' if x < 0 else ''}${abs(x):,.{decimals}f}"


def md_money(x: float) -> str:
    """Money for markdown text: Streamlit would render $...$ pairs as LaTeX."""
    return fmt_money(x).replace("$", r"\$")


def hbar(df: pd.DataFrame, category: str, value: str, title: str,
         tickformat: str, prefix: str = "", signed: bool = False):
    """Single-series horizontal bar chart; red/blue by sign when ``signed``."""
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in df[value]] if signed else PRIMARY
    label = f"{prefix}%{{x:{tickformat}}}"
    fig = go.Figure(go.Bar(x=df[value], y=df[category], orientation="h", marker_color=colors,
                           texttemplate=label, textposition="outside", textangle=0,
                           cliponaxis=False, hovertemplate=f"%{{y}}: {label}<extra></extra>"))
    fig.update_layout(title=title, height=360, margin=dict(l=0, r=40, t=40, b=0),
                      xaxis=dict(tickformat=tickformat, tickprefix=prefix))
    return fig


if not DEFAULT_RAW_PATH.exists():
    st.error(f"Dataset not found at `{DEFAULT_RAW_PATH}`.")
    st.stop()

df_all = load_dataset()


# =====================================================================
# SIDEBAR FILTERS
# =====================================================================

with st.sidebar:
    st.header("Filters")
    selections = {}
    for col, label in FILTERS.items():
        options = ORDERS.get(col) or sorted(df_all[col].astype(str).unique())
        selections[col] = st.multiselect(label, options, placeholder="All")

    d_min, d_max = df_all["date"].min().date(), df_all["date"].max().date()
    date_range = st.date_input("Date range", value=(d_min, d_max), min_value=d_min, max_value=d_max)
    st.caption("Company X's campaigns run by Advert Firm A, Apr–Jun 2020 "
               "(Finlatics online advertising dataset).")

df = df_all
for col, chosen in selections.items():
    if chosen:
        df = df[df[col].isin(chosen)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()


# =====================================================================
# HEADER + KPIs
# =====================================================================

st.title("Online Advertising Performance")
s = get_overall_summary(df, verbose=False)
st.caption(f"{s['total_records']:,} rows · {s['date_range']} · "
           "**revenue** = what Company X paid Advert Firm A · **cost** = what the firm paid publishers")

k = st.columns(6)
k[0].metric("Revenue", fmt_money(s["total_revenue"]),
            help="What Company X paid Advert Firm A (Company X's ad spend)")
k[1].metric("Placement cost", fmt_money(s["total_cost"]))
k[2].metric("Firm margin", fmt_money(s["total_firm_margin"]), f"ROI {s['firm_roi']:.0%}",
            delta_color="normal" if s["firm_roi"] >= 0 else "inverse")
k[3].metric("CTR", f"{s['ctr']:.2%}", f"{s['total_clicks']:,} clicks", delta_color="off")
k[4].metric("Conversions / click", f"{s['conversion_rate']:.1%}")
k[5].metric("Company X ROAS", f"{s['advertiser_roas']:,.0f}×",
            "attributed sales ÷ spend", delta_color="off")

tabs = st.tabs(["Overview", "Breakdown", "Segments", "Insights", "Model & Predict", "Data"])


# ---------- OVERVIEW ----------
with tabs[0]:
    daily = summarize_by(df, "date").sort_index()
    roll = daily[["revenue", "cost", "firm_margin", "clicks"]].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_scatter(x=roll.index, y=roll["revenue"], name="Revenue", line=dict(color=CATEGORICAL[0], width=2))
    fig.add_scatter(x=roll.index, y=roll["cost"], name="Cost", line=dict(color=CATEGORICAL[1], width=2))
    fig.update_layout(title="Revenue vs cost (7-day average)", height=360, hovermode="x unified",
                      yaxis=dict(tickprefix="$"), margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.12))
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    margin = roll["firm_margin"]
    fig = go.Figure(go.Bar(x=margin.index, y=margin,
                           marker_color=[POSITIVE if v >= 0 else NEGATIVE for v in margin],
                           hovertemplate="%{x|%d %b}: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(title="Firm margin (7-day average)", height=320,
                      yaxis=dict(tickprefix="$"), margin=dict(l=0, r=0, t=40, b=0))
    c1.plotly_chart(fig)

    fig = go.Figure(go.Scatter(x=roll.index, y=roll["clicks"], line=dict(color=PRIMARY, width=2),
                               hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>"))
    fig.update_layout(title="Clicks per day (7-day average)", height=320,
                      margin=dict(l=0, r=0, t=40, b=0))
    c2.plotly_chart(fig)


# ---------- BREAKDOWN ----------
with tabs[1]:
    dim = st.radio("Break down by", list(DIMENSIONS), format_func=DIMENSIONS.get, horizontal=True)
    summary = summarize_by(df, dim)
    order = [o for o in ORDERS.get(dim, []) if o in summary.index]
    summary = (summary.reindex(order) if order else summary.sort_values("revenue")).reset_index()
    summary[dim] = summary[dim].astype(str)

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(hbar(summary, dim, "revenue", "Revenue", ",.0f", prefix="$"))
    c2.plotly_chart(hbar(summary, dim, "ctr", "CTR", ".2%"))
    c3.plotly_chart(hbar(summary, dim, "firm_roi", "Firm ROI", ".0%", signed=True))

    if dim == "campaign_number":
        st.info("camp 2 ran in April only — compare campaigns on rates, not totals.")
    st.dataframe(
        summary[[dim, "revenue", "cost", "firm_margin", "firm_roi", "displays", "clicks", "ctr",
                 "conversion_rate", "advertiser_roas", "revenue_share", "active_days"]],
        hide_index=True,
        column_config={
            "revenue": st.column_config.NumberColumn("Revenue", format="$%.0f"),
            "cost": st.column_config.NumberColumn("Cost", format="$%.0f"),
            "firm_margin": st.column_config.NumberColumn("Firm margin", format="$%.0f"),
            "firm_roi": st.column_config.NumberColumn("Firm ROI", format="percent"),
            "ctr": st.column_config.NumberColumn("CTR", format="%.4f"),
            "conversion_rate": st.column_config.NumberColumn("Conv./click", format="%.3f"),
            "advertiser_roas": st.column_config.NumberColumn("ROAS", format="%.1f×"),
            "revenue_share": st.column_config.NumberColumn("Revenue share", format="percent"),
        },
    )


# ---------- SEGMENTS ----------
with tabs[2]:
    metric = st.radio("Metric", ["firm_margin", "revenue", "ctr", "firm_roi"], horizontal=True,
                      format_func={"firm_margin": "Firm margin", "revenue": "Revenue",
                                   "ctr": "CTR", "firm_roi": "Firm ROI"}.get)
    grid = summarize_by(df, ["banner", "placement"])[metric].unstack("placement")
    grid = grid.loc[grid.abs().sum(axis=1).sort_values(ascending=False).index]

    diverging = metric in ("firm_margin", "firm_roi")
    values = grid.values.astype(float)
    if metric == "firm_margin":  # signed log so small losses stay visible
        z = np.sign(values) * np.log10(1 + np.abs(values))
    else:
        z = values
    if diverging:
        lim = np.nanmax(np.abs(z)) or 1
        colorscale = [[0, NEGATIVE], [0.5, DIVERGING["neutral"]], [1, POSITIVE]]
        zrange = dict(zmin=-lim, zmax=lim)
    else:
        colorscale = [[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)]
        zrange = {}
    text = {"firm_margin": [[fmt_money(v) for v in row] for row in values],
            "revenue": [[fmt_money(v) for v in row] for row in values],
            "ctr": [[f"{v:.2%}" if pd.notna(v) else "" for v in row] for row in values],
            "firm_roi": [[f"{v:.0%}" if pd.notna(v) else "" for v in row] for row in values]}[metric]
    fig = go.Figure(go.Heatmap(z=z, x=grid.columns, y=grid.index, text=text, texttemplate="%{text}",
                               colorscale=colorscale, showscale=False, hoverongaps=False,
                               hovertemplate="%{y} @ %{x}: %{text}<extra></extra>", **zrange))
    fig.update_layout(title="Banner × placement", height=480, margin=dict(l=0, r=0, t=40, b=0),
                      xaxis_title="Placement", yaxis=dict(title="Banner", autorange="reversed"))
    st.plotly_chart(fig)
    st.caption("Empty cells: the combination was never served. "
               "Firm margin colours use a log scale so small losses remain visible.")


# ---------- INSIGHTS ----------
with tabs[3]:
    findings = load_report("eda_findings")
    if findings is None:
        st.info("Run `python main.py` to generate reports/eda_findings.json.")
    else:
        t, seg, dist = findings["time"], findings["segments"], findings["distribution"]
        rel = findings["relationships"]["cost~revenue"]
        st.subheader("Key findings (full dataset)")
        st.markdown(f"""
- **Steady decline:** average daily revenue fell from {md_money(t['first_14_days_avg_daily_revenue'])}
  (first 14 days) to {md_money(t['last_14_days_avg_daily_revenue'])} (last 14 days); excluding camp 2,
  April → June revenue changed {t['april_to_june_revenue_change_excl_camp2']:.0%}.
- **Margin turned negative** on {t['days_with_negative_firm_margin']} days, starting {t['first_negative_margin_day']}.
- **One segment carries the firm:** {next(iter(seg['top_5_by_margin']))} produced
  {seg['top_segment_margin_share']:.0%} of all firm margin; {seg['loss_making_count']} banner × placement
  combinations lost {md_money(-seg['loss_making_total_margin'])}.
- **Best CTR:** banner {findings['banners']['highest_ctr']}, placement {findings['placements']['highest_ctr']}.
- **Concentration:** the top 1% of rows produce {dist['top_1pct_rows_revenue_share']:.0%} of revenue,
  the top 10% produce {dist['top_10pct_rows_revenue_share']:.0%}.
- **Cost vs revenue:** Pearson r = {rel['pearson']:.2f}, Spearman ρ = {rel['spearman']:.2f}.
""")
        st.caption("Full write-up with 12 charts: reports/eda_findings.md")


# ---------- MODEL & PREDICT ----------
with tabs[4]:
    evaluation = load_report("model_evaluation")
    if evaluation:
        st.subheader(f"Revenue model: {evaluation['selected_model']}")
        st.caption(f"Trained on {evaluation['train_rows']:,} rows before {evaluation['test_start']}, "
                   f"evaluated on the following {evaluation['test_rows']:,} rows (time-based split).")
        rows = []
        for name, r in evaluation["results"].items():
            rows.append({"Approach": name, "R² (log)": r["revenue"]["r2_log"],
                         "MAE ($/row)": r["revenue"]["mae_usd"],
                         "Total pred ÷ actual": r["revenue"]["total_predicted_vs_actual"],
                         "Profitable F1": r["profitability"]["f1"],
                         "ROC-AUC": r["profitability"]["roc_auc"],
                         "Margin captured": r["margin"]["buy_predicted_profitable"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            "Margin captured": st.column_config.NumberColumn(format="$%.0f"),
            "MAE ($/row)": st.column_config.NumberColumn(format="$%.2f"),
        })
        ref = next(iter(evaluation["results"].values()))["margin"]
        st.caption(f"Buying everything in the test period: {md_money(ref['buy_everything'])} · "
                   f"perfect foresight: {md_money(ref['oracle'])}. Details: reports/model_report.md")

        if SHAP_CSV.exists():
            shap_df = pd.read_csv(SHAP_CSV).head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=shap_df["importance_pct"], y=shap_df["feature"], orientation="h",
                                   marker_color=PRIMARY, texttemplate="%{x:.1f}%"))
            fig.update_layout(title="What drives predictions (share of mean |SHAP|)", height=380,
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig)

    st.subheader("Predict a planned buy")
    if not predictor.model_available():
        st.info("No trained model yet. Run `python main.py` to train it.")
    else:
        categories = predictor.valid_categories()
        # Display order: engagement Low->High, "Unknown" placement last
        options = {col: (ORDERS[col] if col in ORDERS else
                         sorted(levels, key=lambda v: (v == "Unknown", v)))
                   for col, levels in categories.items()}
        with st.form("predict"):
            c1, c2, c3, c4 = st.columns(4)
            date = c1.date_input("Date", value=pd.Timestamp("2020-06-15"),
                                 min_value=predictor.PERIOD_START.date())
            campaign = c2.selectbox("Campaign", options["campaign_number"])
            engagement = c3.selectbox("Engagement", options["user_engagement"])
            banner_sel = c4.selectbox("Banner", options["banner"])
            c5, c6, c7 = st.columns(3)
            placement = c5.selectbox("Placement", options["placement"])
            displays = c6.number_input("Displays", min_value=0, value=20000, step=1000)
            cost = c7.number_input("Cost ($)", min_value=0.0, value=20.0, step=1.0)
            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            request = {"date": str(date), "campaign_number": campaign, "user_engagement": engagement,
                       "banner": banner_sel, "placement": placement,
                       "displays": int(displays), "cost": float(cost)}
            errors = predictor.validate_request(request, categories)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                r = predictor.predict([request])[0]
                m = st.columns(4)
                m[0].metric("Predicted revenue", fmt_money(r["predicted_revenue"], 2))
                m[1].metric("Cost", fmt_money(r["cost"], 2))
                m[2].metric("Predicted margin", fmt_money(r["predicted_margin"], 2),
                            "profitable" if r["profitable"] else "loss",
                            delta_color="normal" if r["profitable"] else "inverse")
                m[3].metric("Predicted ROI", "—" if r["predicted_roi"] is None else f"{r['predicted_roi']:.0%}")
                for w in r["warnings"]:
                    st.warning(w)

        with st.expander("Batch prediction from CSV"):
            st.caption(f"Columns: {', '.join(predictor.REQUEST_FIELDS)} · up to {predictor.MAX_BATCH} rows")
            upload = st.file_uploader("CSV of planned buys", type=["csv"])
            if upload is not None:
                try:
                    batch = pd.read_csv(upload)
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
                    batch = None
                if batch is not None:
                    items = batch.to_dict(orient="records")[:predictor.MAX_BATCH]
                    problems = {i + 1: errs for i, item in enumerate(items)
                                if (errs := predictor.validate_request(item, categories))}
                    if problems:
                        st.error(f"{len(problems)} invalid rows (first: row {next(iter(problems))}: "
                                 f"{problems[next(iter(problems))][0]})")
                    else:
                        result = pd.concat([batch.iloc[:len(items)].reset_index(drop=True),
                                            pd.DataFrame(predictor.predict(items))
                                            .drop(columns=["cost"])], axis=1)
                        st.dataframe(result, hide_index=True)
                        st.download_button("Download predictions", result.to_csv(index=False),
                                           "predictions.csv", "text/csv")


# ---------- DATA ----------
with tabs[5]:
    st.caption(f"{len(df):,} rows after filters. Flag columns mark rows kept despite "
               "suspicious values (e.g. more conversions than clicks).")
    st.dataframe(df, hide_index=True)
    st.download_button("Download filtered CSV", df.to_csv(index=False),
                       "advertising_filtered.csv", "text/csv")
