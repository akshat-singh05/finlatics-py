"""
Finlatics-Py: Production-Ready Interactive Analytics Dashboard

Run:  streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from analysis.dashboard_charts import (
    plot_feature_importance_interactive,
    plot_confidence_gauge,
    plot_prediction_distribution,
    plot_shap_waterfall_interactive,
    plot_confusion_matrix_interactive,
    plot_scenario_comparison,
)
from utils.logger import get_logger

logger = get_logger("dashboard")


# =====================================================================
# CONFIG
# =====================================================================

st.set_page_config(
    page_title="Finlatics Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHART_TEMPLATE = "plotly_dark"
COLORS = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6", "#8B5CF6",
          "#14B8A6", "#F97316", "#EF4444", "#06B6D4"]

# =====================================================================
# CSS
# =====================================================================

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    border-right: 1px solid #334155;
}
.kpi-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(51,65,85,0.6) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.2); border-radius: 16px;
    padding: 20px 24px; text-align: center;
    transition: transform 0.3s cubic-bezier(0.4,0,0.2,1), box-shadow 0.3s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(99, 102, 241, 0.2);
    border-color: rgba(99,102,241,0.4);
}
.kpi-value {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #818CF8, #6366F1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 4px 0;
}
.kpi-label {
    font-size: 0.85rem; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
}
.kpi-delta-up { color: #10B981; font-size: 0.8rem; font-weight: 600; }
.kpi-delta-down { color: #EF4444; font-size: 0.8rem; font-weight: 600; }
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #E2E8F0;
    margin: 2rem 0 1rem 0; padding-bottom: 8px; border-bottom: 2px solid #6366F1;
}
.prediction-box {
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(51,65,85,0.7));
    backdrop-filter: blur(12px);
    border: 2px solid #6366F1; border-radius: 16px;
    padding: 24px; text-align: center;
    animation: fadeIn 0.5s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(99,102,241,0.3); }
    50% { box-shadow: 0 0 20px rgba(99,102,241,0.5); }
}
.pred-profitable { color: #10B981; font-size: 1.5rem; font-weight: 800; }
.pred-unprofitable { color: #EF4444; font-size: 1.5rem; font-weight: 800; }
.stTabs [data-baseweb="tab"] {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 8px 8px 0 0; color: #94A3B8; padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #6366F1 !important; color: white !important;
}
.data-quality-banner {
    background: linear-gradient(90deg, rgba(99,102,241,0.1), rgba(59,130,246,0.1));
    border: 1px solid rgba(99,102,241,0.3); border-radius: 12px;
    padding: 12px 20px; margin-bottom: 1rem;
    color: #94A3B8; font-size: 0.85rem;
}
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SESSION STATE
# =====================================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =====================================================================
# HELPERS
# =====================================================================

def render_kpi(label, value, delta="", up=True):
    dc = "kpi-delta-up" if up else "kpi-delta-down"
    dh = f'<div class="{dc}">{delta}</div>' if delta else ""
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>{dh}</div>',
                unsafe_allow_html=True)


def safe_div(a, b):
    return a / b if b != 0 else 0


@st.cache_data(show_spinner="Loading data...")
def load_and_process(path):
    from analysis.data_loader import load_data
    from analysis.data_cleaning import clean_data
    from analysis.metrics import compute_kpis
    df = load_data(path)
    df = clean_data(df)
    df = compute_kpis(df)
    logger.info("Dashboard data loaded: %d rows from %s", len(df), path)
    return df


@st.cache_resource(show_spinner="Loading model...")
def load_pipeline(target="high_roi"):
    for pattern in [f"best_pipeline_{target}.joblib", f"best_model_{target}.joblib"]:
        p = PROJECT_ROOT / "models" / "saved" / pattern
        if p.exists():
            logger.info("Dashboard model loaded: %s", p.name)
            return joblib.load(p)
    return None


def apply_filters(df, campaign, channel, device, date_range):
    f = df.copy()
    if campaign != "All":
        f = f[f["campaign"] == campaign]
    if channel != "All":
        f = f[f["channel"] == channel]
    if device != "All":
        f = f[f["device"] == device]
    if "date" in f.columns and date_range and len(date_range) == 2:
        f = f[(pd.to_datetime(f["date"]).dt.date >= date_range[0]) &
              (pd.to_datetime(f["date"]).dt.date <= date_range[1])]
    return f


def get_data_quality_summary(df):
    """Return a data quality summary string."""
    nulls = df.isnull().sum().sum()
    dupes = df.duplicated().sum()
    return (f"📊 <b>{len(df):,}</b> records · "
            f"<b>{df['campaign'].nunique()}</b> campaigns · "
            f"<b>{df['channel'].nunique()}</b> channels · "
            f"<b>{nulls}</b> nulls · "
            f"<b>{dupes}</b> duplicates")


# =====================================================================
# SIDEBAR
# =====================================================================

with st.sidebar:
    st.markdown("## 📊 Finlatics Dashboard")
    st.markdown("---")

    st.markdown("### Data Source")
    source = st.radio("Source:", ["Default Dataset", "Upload CSV"], label_visibility="collapsed")

    df = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                from analysis.data_cleaning import clean_data
                from analysis.metrics import compute_kpis
                raw = pd.read_csv(uploaded)
                raw = clean_data(raw)
                df = compute_kpis(raw)
                st.success(f"Loaded {len(df)} rows")
                logger.info("Dashboard: uploaded CSV loaded, %d rows", len(df))
            except Exception as e:
                st.error(f"Invalid CSV: {e}")
                logger.error("Dashboard: CSV upload failed: %s", e)

    if df is None:
        data_path = PROJECT_ROOT / "data" / "synthetic" / "synthetic_advertising_data.csv"
        if data_path.exists():
            try:
                df = load_and_process(str(data_path))
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                logger.error("Dashboard: data load failed: %s", e)
                st.stop()
        else:
            st.error("No data found. Run `python main.py` first.")
            st.stop()

    st.markdown("---")
    st.markdown("### Filters")

    sel_campaign = st.selectbox("Campaign", ["All"] + sorted(df["campaign"].unique().tolist()))
    sel_channel = st.selectbox("Channel", ["All"] + sorted(df["channel"].unique().tolist()))
    sel_device = st.selectbox("Device", ["All"] + sorted(df["device"].unique().tolist()))

    date_range = None
    if "date" in df.columns:
        mn = pd.to_datetime(df["date"]).min().date()
        mx = pd.to_datetime(df["date"]).max().date()
        date_range = st.date_input("Date Range", value=(mn, mx), min_value=mn, max_value=mx)

    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("v2.0.0 · Built with Streamlit · Finlatics-Py")

fdf = apply_filters(df, sel_campaign, sel_channel, sel_device, date_range)

if len(fdf) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()


# =====================================================================
# HEADER + DATA QUALITY + KPIs
# =====================================================================

st.markdown('<h1 style="text-align:center;color:#E2E8F0;font-weight:800;">'
            '📊 Advertising Performance Dashboard</h1>', unsafe_allow_html=True)

st.markdown(f'<div class="data-quality-banner">{get_data_quality_summary(fdf)}</div>',
            unsafe_allow_html=True)

spend = fdf["spend"].sum()
rev = fdf["revenue"].sum()
clicks = fdf["clicks"].sum()
imps = fdf["impressions"].sum()
convs = fdf["conversions"].sum()
ctr = safe_div(clicks, imps) * 100
cpc = safe_div(spend, clicks)
roas = safe_div(rev, spend)
roi = safe_div(rev - spend, spend) * 100

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: render_kpi("Total Spend", f"${spend:,.0f}")
with c2: render_kpi("Revenue", f"${rev:,.0f}",
                    f"{'+'if rev>spend else ''}{rev-spend:,.0f}", rev>spend)
with c3: render_kpi("CTR", f"{ctr:.2f}%")
with c4: render_kpi("CPC", f"${cpc:.2f}")
with c5: render_kpi("ROAS", f"{roas:.2f}x", "Profitable" if roas>1 else "Loss", roas>1)
with c6: render_kpi("ROI", f"{roi:.1f}%", "Positive" if roi>0 else "Negative", roi>0)


# =====================================================================
# TABS
# =====================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Performance", "📊 Breakdown", "💡 Insights",
    "🔮 Predictions", "📋 Data", "🔍 Explainability",
])


# ---------- TAB 1: PERFORMANCE ----------
with tab1:
    try:
        st.markdown('<div class="section-header">Campaign Performance</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            camp = fdf.groupby("campaign").agg(spend=("spend","sum"),
                   revenue=("revenue","sum")).reset_index().sort_values("revenue")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=camp["campaign"], x=camp["spend"], name="Spend",
                                 orientation="h", marker_color="#EF4444", opacity=0.85))
            fig.add_trace(go.Bar(y=camp["campaign"], x=camp["revenue"], name="Revenue",
                                 orientation="h", marker_color="#10B981", opacity=0.85))
            fig.update_layout(template=CHART_TEMPLATE, barmode="group",
                              title="Revenue vs Spend", height=400,
                              margin=dict(l=0,r=0,t=40,b=0),
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            camp_roi = fdf.groupby("campaign")["roi"].mean().reset_index().sort_values("roi")
            colors = ["#10B981" if v>0 else "#EF4444" for v in camp_roi["roi"]]
            fig = go.Figure(go.Bar(y=camp_roi["campaign"], x=camp_roi["roi"],
                                   orientation="h", marker_color=colors, opacity=0.9))
            fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
            fig.update_layout(template=CHART_TEMPLATE, title="Average ROI by Campaign",
                              height=400, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        if "date" in fdf.columns:
            st.markdown('<div class="section-header">Trends Over Time</div>', unsafe_allow_html=True)
            t = fdf.copy()
            t["week"] = pd.to_datetime(t["date"]).dt.to_period("W").astype(str)
            w = t.groupby("week").agg(spend=("spend","sum"), revenue=("revenue","sum"),
                                       conversions=("conversions","sum")).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w["week"], y=w["revenue"], name="Revenue",
                                     line=dict(color="#10B981", width=3),
                                     fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"))
            fig.add_trace(go.Scatter(x=w["week"], y=w["spend"], name="Spend",
                                     line=dict(color="#EF4444", width=3),
                                     fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"))
            fig.update_layout(template=CHART_TEMPLATE, title="Weekly Revenue vs Spend",
                              height=350, margin=dict(l=0,r=0,t=40,b=0),
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Performance tab error: {e}")
        logger.error("Dashboard tab1 error: %s", e)


# ---------- TAB 2: BREAKDOWN ----------
with tab2:
    try:
        st.markdown('<div class="section-header">Channel Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            ch = fdf.groupby("channel")["spend"].sum().reset_index()
            fig = px.pie(ch, values="spend", names="channel", title="Spend by Channel",
                         color_discrete_sequence=COLORS, hole=0.45)
            fig.update_layout(template=CHART_TEMPLATE, height=400,
                              margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            ch2 = fdf.groupby("channel").agg(spend=("spend","sum"),
                  revenue=("revenue","sum")).reset_index()
            ch2["roas"] = (ch2["revenue"]/ch2["spend"]).round(2)
            ch2 = ch2.sort_values("roas")
            colors = ["#10B981" if v>=1 else "#EF4444" for v in ch2["roas"]]
            fig = go.Figure(go.Bar(y=ch2["channel"], x=ch2["roas"], orientation="h",
                                   marker_color=colors, opacity=0.9,
                                   text=ch2["roas"].apply(lambda v: f"{v:.2f}x"),
                                   textposition="outside"))
            fig.add_vline(x=1, line_dash="dash", line_color="#F59E0B",
                          annotation_text="Break-even")
            fig.update_layout(template=CHART_TEMPLATE, title="ROAS by Channel",
                              height=400, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Device Performance</div>', unsafe_allow_html=True)
        device_metrics = []
        for metric, label in [("ctr","CTR"), ("conversion_rate","Conversion Rate"), ("roas","ROAS")]:
            if metric in fdf.columns:
                device_metrics.append((metric, label))
        if device_metrics:
            cols = st.columns(len(device_metrics))
            for col, (metric, label) in zip(cols, device_metrics):
                with col:
                    d = fdf.groupby("device")[metric].mean().reset_index()
                    fig = px.bar(d, x="device", y=metric, title=f"Avg {label} by Device",
                                 color="device", color_discrete_sequence=COLORS)
                    fig.update_layout(template=CHART_TEMPLATE, height=300,
                                      showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Correlations</div>', unsafe_allow_html=True)
        num_cols = [c for c in ["impressions","clicks","spend","conversions","revenue",
                    "bounce_rate","ctr","cpc","roas","roi"] if c in fdf.columns]
        corr = fdf[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Correlation Matrix", aspect="auto")
        fig.update_layout(template=CHART_TEMPLATE, height=500,
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Breakdown tab error: {e}")
        logger.error("Dashboard tab2 error: %s", e)


# ---------- TAB 3: INSIGHTS ----------
with tab3:
    try:
        st.markdown('<div class="section-header">Business Insights</div>', unsafe_allow_html=True)

        ch_perf = fdf.groupby("channel").agg(
            spend=("spend","sum"), revenue=("revenue","sum"),
            conversions=("conversions","sum")).reset_index()
        ch_perf["roas"] = (ch_perf["revenue"]/ch_perf["spend"]).round(2)
        ch_perf["profit"] = ch_perf["revenue"] - ch_perf["spend"]
        ch_perf = ch_perf.sort_values("roas", ascending=False)

        best_ch = ch_perf.iloc[0]
        worst_ch = ch_perf.iloc[-1]

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Best Channel:** {best_ch['channel']} "
                       f"({best_ch['roas']:.2f}x ROAS, +${best_ch['profit']:,.0f} profit)")
        with col2:
            if worst_ch["profit"] < 0:
                st.error(f"**Worst Channel:** {worst_ch['channel']} "
                         f"({worst_ch['roas']:.2f}x ROAS, ${worst_ch['profit']:,.0f} loss)")
            else:
                st.warning(f"**Lowest Channel:** {worst_ch['channel']} "
                           f"({worst_ch['roas']:.2f}x ROAS)")

        st.markdown('<div class="section-header">Budget Optimization</div>', unsafe_allow_html=True)

        total_budget = ch_perf["spend"].sum()
        ch_perf["current_pct"] = (ch_perf["spend"]/total_budget*100).round(1)
        roas_pos = ch_perf["roas"].clip(lower=0)
        ch_perf["recommended_pct"] = (roas_pos/roas_pos.sum()*100).round(1)
        ch_perf["shift"] = ch_perf["recommended_pct"] - ch_perf["current_pct"]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(ch_perf, values="current_pct", names="channel",
                         title="Current Allocation", color_discrete_sequence=COLORS, hole=0.45)
            fig.update_layout(template=CHART_TEMPLATE, height=350,
                              margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(ch_perf, values="recommended_pct", names="channel",
                         title="Recommended (ROAS-weighted)", color_discrete_sequence=COLORS, hole=0.45)
            fig.update_layout(template=CHART_TEMPLATE, height=350,
                              margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Reallocation Recommendations")
        for _, row in ch_perf.iterrows():
            direction = "INCREASE" if row["shift"]>2 else "DECREASE" if row["shift"]<-2 else "KEEP"
            icon = "🟢" if direction=="INCREASE" else "🔴" if direction=="DECREASE" else "🟡"
            sign = "+" if row["shift"]>0 else ""
            st.markdown(f"{icon} **{row['channel']}**: "
                        f"{row['current_pct']:.1f}% -> {row['recommended_pct']:.1f}% "
                        f"({sign}{row['shift']:.1f}pp, {sign}${row['shift']/100*total_budget:,.0f})")

        st.markdown('<div class="section-header">Campaign Ranking</div>', unsafe_allow_html=True)
        camp_agg = {"spend": ("spend","sum"), "revenue": ("revenue","sum"), "avg_ctr": ("ctr","mean")}
        if "conversion_rate" in fdf.columns:
            camp_agg["avg_conv"] = ("conversion_rate","mean")
        if "roas" in fdf.columns:
            camp_agg["avg_roas"] = ("roas","mean")
        camp_rank = fdf.groupby("campaign").agg(**camp_agg).reset_index()
        for c in ["avg_ctr","avg_conv","avg_roas"]:
            if c in camp_rank.columns:
                mx = camp_rank[c].max()
                camp_rank[f"{c}_n"] = camp_rank[c]/mx if mx>0 else 0
        if all(c in camp_rank.columns for c in ["avg_ctr_n","avg_conv_n","avg_roas_n"]):
            camp_rank["score"] = (camp_rank["avg_ctr_n"]*0.2 +
                                  camp_rank["avg_conv_n"]*0.3 +
                                  camp_rank["avg_roas_n"]*0.5).round(3)
            camp_rank = camp_rank.sort_values("score", ascending=True)

            fig = go.Figure(go.Bar(y=camp_rank["campaign"], x=camp_rank["score"],
                                   orientation="h", marker_color=COLORS[:len(camp_rank)], opacity=0.9,
                                   text=camp_rank["score"].apply(lambda v: f"{v:.2f}"),
                                   textposition="outside"))
            fig.update_layout(template=CHART_TEMPLATE, title="Campaign Efficiency Score",
                              height=350, margin=dict(l=0,r=0,t=40,b=0), xaxis=dict(range=[0,1.15]))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Insights tab error: {e}")
        logger.error("Dashboard tab3 error: %s", e)


# ---------- TAB 4: PREDICTIONS ----------
with tab4:
    try:
        st.markdown('<div class="section-header">ML Predictions</div>', unsafe_allow_html=True)

        pipeline = load_pipeline("high_roi")

        if pipeline is None:
            st.warning("No trained model found. Run `python main.py --target high_roi` to train.")
        else:
            pred_mode = st.radio("Mode:", ["🎯 Single Prediction", "📁 Batch Prediction"],
                                 horizontal=True, label_visibility="collapsed")

            if pred_mode == "🎯 Single Prediction":
                st.markdown("Enter campaign parameters to predict **profitability (ROI > 0)**:")

                with st.form("predict"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        p_camp = st.selectbox("Campaign", sorted(df["campaign"].unique()))
                        p_chan = st.selectbox("Channel", sorted(df["channel"].unique()))
                        p_dev = st.selectbox("Device", sorted(df["device"].unique()))
                    with c2:
                        p_imp = st.number_input("Impressions", 100, 100000, 5000, 500)
                        p_clk = st.number_input("Clicks", 1, 10000, 80, 10)
                        p_spd = st.number_input("Spend ($)", 1.0, 50000.0, 150.0, 10.0)
                    with c3:
                        p_con = st.number_input("Conversions", 0, 1000, 5, 1)
                        p_bnc = st.slider("Bounce Rate", 0.0, 1.0, 0.5, 0.05)
                        p_ses = st.number_input("Avg Session (s)", 1.0, 600.0, 120.0, 10.0)

                    go_btn = st.form_submit_button("🔮 Predict Profitability", use_container_width=True)

                if go_btn:
                    validation_errors = []
                    if p_clk > p_imp:
                        validation_errors.append("Clicks cannot exceed Impressions.")
                    if p_spd <= 0:
                        validation_errors.append("Spend must be positive.")
                    if p_con > p_clk:
                        validation_errors.append("Conversions cannot exceed Clicks.")
                    if p_bnc > 0.95:
                        st.warning("⚠️ Bounce rate above 95% is unusual — results may be unreliable.")

                    if validation_errors:
                        for err in validation_errors:
                            st.error(err)
                    else:
                        try:
                            from analysis.metrics import compute_kpis as ckpi
                            from models.feature_engineering import (
                                create_derived_features, create_interaction_features,
                                create_time_features,
                            )

                            inp = pd.DataFrame([{
                                "campaign": p_camp, "channel": p_chan, "device": p_dev,
                                "impressions": p_imp, "clicks": p_clk, "spend": p_spd,
                                "conversions": p_con, "revenue": 0.0,
                                "bounce_rate": p_bnc, "avg_session_duration": p_ses,
                                "date": pd.Timestamp.now(),
                            }])
                            inp = ckpi(inp)
                            inp = create_derived_features(inp)
                            inp = create_interaction_features(inp)
                            inp = create_time_features(inp)

                            drop = ["date","roi","roas","revenue","revenue_per_conversion","target"]
                            inp_clean = inp.drop(columns=[c for c in drop if c in inp.columns], errors="ignore")

                            pred = pipeline.predict(inp_clean)[0]
                            prob = None
                            if hasattr(pipeline, "predict_proba"):
                                prob = pipeline.predict_proba(inp_clean)[0]

                            profitable = pred == 1
                            confidence = float(max(prob)) if prob is not None else None

                            logger.info("Dashboard prediction: %s (conf=%.3f)",
                                        "profitable" if profitable else "unprofitable",
                                        confidence or 0)

                            c1, c2 = st.columns([2, 1])
                            with c1:
                                label = "PROFITABLE" if profitable else "UNPROFITABLE"
                                cls = "pred-profitable" if profitable else "pred-unprofitable"
                                emoji = "✅" if profitable else "❌"
                                conf_html = (f'<p style="color:#94A3B8;">'
                                             f'Confidence: <b>{confidence*100:.1f}%</b></p>'
                                             if confidence else "")
                                st.markdown(
                                    f'<div class="prediction-box">'
                                    f'<div class="kpi-label">Prediction</div>'
                                    f'<div class="{cls}">{emoji} {label}</div>'
                                    f'{conf_html}</div>',
                                    unsafe_allow_html=True,
                                )
                            with c2:
                                if confidence:
                                    st.plotly_chart(
                                        plot_confidence_gauge(confidence, label),
                                        use_container_width=True,
                                    )

                            # Business interpretation
                            st.markdown('<div class="section-header">Business Interpretation</div>',
                                        unsafe_allow_html=True)
                            ctr_est = safe_div(p_clk, p_imp) * 100
                            cpc_est = safe_div(p_spd, p_clk)
                            ic1, ic2, ic3, ic4 = st.columns(4)
                            with ic1: render_kpi("Est. CTR", f"{ctr_est:.2f}%")
                            with ic2: render_kpi("Est. CPC", f"${cpc_est:.2f}")
                            with ic3: render_kpi("Campaign", p_camp)
                            with ic4: render_kpi("Channel", p_chan)

                            if profitable:
                                if confidence and confidence > 0.85:
                                    st.success(
                                        "**Strong signal.** Consider scaling spend by 15-25% "
                                        "while monitoring actual ROAS."
                                    )
                                else:
                                    st.info(
                                        "**Moderate confidence.** Run a small test budget first "
                                        "before committing larger spend."
                                    )
                            else:
                                if cpc_est > 3.0:
                                    st.warning(
                                        f"**High CPC (${cpc_est:.2f}).** Consider switching "
                                        "channels or refining audience targeting."
                                    )
                                elif ctr_est < 1.0:
                                    st.warning(
                                        f"**Low CTR ({ctr_est:.2f}%).** Improve ad creative "
                                        "or targeting to increase engagement."
                                    )
                                else:
                                    st.warning(
                                        "**Review conversion funnel.** Consider A/B testing "
                                        "landing pages and adjusting bid strategy."
                                    )

                            # Per-prediction SHAP explanation
                            st.markdown('<div class="section-header">Why This Prediction?</div>',
                                        unsafe_allow_html=True)
                            try:
                                from models.explainability import explain_single_prediction
                                shap_result = explain_single_prediction(
                                    pipeline, inp_clean, task_type="classification"
                                )
                                shap_vals = np.array(shap_result["shap_values"])
                                feat_names = shap_result["feature_names"]
                                st.plotly_chart(
                                    plot_shap_waterfall_interactive(
                                        [shap_vals.reshape(1, -1)],
                                        feat_names, sample_idx=0, top_n=10
                                    ),
                                    use_container_width=True,
                                )

                                pos_col, neg_col = st.columns(2)
                                with pos_col:
                                    st.markdown("**🟢 Pushing prediction UP:**")
                                    for name, val in shap_result["top_positive"][:3]:
                                        st.markdown(f"- `{name}`: +{val:.4f}")
                                with neg_col:
                                    st.markdown("**🔴 Pushing prediction DOWN:**")
                                    for name, val in shap_result["top_negative"][:3]:
                                        st.markdown(f"- `{name}`: {val:.4f}")
                            except Exception as shap_err:
                                st.info(f"SHAP explanation unavailable: {shap_err}")

                            # Save to prediction history
                            st.session_state.prediction_history.append({
                                "label": f"{p_camp[:10]}|{p_chan[:6]}",
                                "Campaign": p_camp, "Channel": p_chan,
                                "Device": p_dev, "Spend": p_spd,
                                "Clicks": p_clk, "Impressions": p_imp,
                                "Prediction": label,
                                "Confidence": f"{confidence*100:.1f}%" if confidence else "N/A",
                            })

                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                            logger.error("Dashboard prediction error: %s", e)

            # --- Batch Prediction ---
            else:
                st.markdown("Upload a CSV with columns: `campaign, channel, device, impressions, "
                            "clicks, spend, conversions, bounce_rate, avg_session_duration`")
                batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_csv")

                if batch_file:
                    try:
                        from analysis.metrics import compute_kpis as ckpi
                        from models.feature_engineering import (
                            create_derived_features, create_interaction_features,
                            create_time_features,
                        )

                        batch_df = pd.read_csv(batch_file)
                        required = ["campaign","channel","device","impressions","clicks",
                                    "spend","conversions","bounce_rate","avg_session_duration"]
                        missing = [c for c in required if c not in batch_df.columns]
                        if missing:
                            st.error(f"Missing columns: {missing}")
                        elif len(batch_df) > 1000:
                            st.error("Batch limited to 1,000 rows. Please split your file.")
                        else:
                            with st.spinner(f"Processing {len(batch_df)} predictions..."):
                                batch_df["revenue"] = batch_df.get("revenue", 0.0)
                                batch_df["date"] = pd.Timestamp.now()
                                processed = ckpi(batch_df.copy())
                                processed = create_derived_features(processed)
                                processed = create_interaction_features(processed)
                                processed = create_time_features(processed)

                                drop = ["date","roi","roas","revenue","revenue_per_conversion","target"]
                                X_batch = processed.drop(
                                    columns=[c for c in drop if c in processed.columns], errors="ignore"
                                )

                                preds = pipeline.predict(X_batch)
                                batch_df["Prediction"] = ["Profitable" if p == 1 else "Unprofitable" for p in preds]

                                if hasattr(pipeline, "predict_proba"):
                                    proba = pipeline.predict_proba(X_batch)
                                    batch_df["Confidence"] = [f"{max(p)*100:.1f}%" for p in proba]

                            profitable_count = (preds == 1).sum()
                            bc1, bc2, bc3 = st.columns(3)
                            with bc1: render_kpi("Total Rows", f"{len(batch_df):,}")
                            with bc2: render_kpi("Profitable", f"{profitable_count}", up=True)
                            with bc3: render_kpi("Unprofitable", f"{len(batch_df)-profitable_count}", up=False)

                            st.dataframe(batch_df, use_container_width=True, height=400)

                            csv_out = batch_df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Results", csv_out,
                                               "batch_predictions.csv", "text/csv",
                                               use_container_width=True)
                            logger.info("Batch prediction: %d rows, %d profitable",
                                        len(batch_df), profitable_count)

                    except Exception as e:
                        st.error(f"Batch prediction error: {e}")
                        logger.error("Dashboard batch prediction error: %s", e)

            # --- Prediction History & Scenario Comparison ---
            if st.session_state.prediction_history:
                st.markdown('<div class="section-header">Prediction History</div>',
                            unsafe_allow_html=True)
                hist_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(hist_df.drop(columns=["label"], errors="ignore"),
                             use_container_width=True, height=200)

                col_dl, col_clear = st.columns(2)
                with col_dl:
                    csv_hist = hist_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download History", csv_hist,
                                       "prediction_history.csv", "text/csv",
                                       use_container_width=True)
                with col_clear:
                    if st.button("🗑️ Clear History", use_container_width=True):
                        st.session_state.prediction_history = []
                        st.rerun()

                if len(st.session_state.prediction_history) >= 2:
                    st.markdown('<div class="section-header">Scenario Comparison</div>',
                                unsafe_allow_html=True)
                    recent = st.session_state.prediction_history[-5:]
                    st.plotly_chart(plot_scenario_comparison(recent), use_container_width=True)

    except Exception as e:
        st.error(f"Predictions tab error: {e}")
        logger.error("Dashboard tab4 error: %s", e)


# ---------- TAB 5: DATA ----------
with tab5:
    try:
        st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(fdf):,}")
        c2.metric("Columns", len(fdf.columns))
        c3.metric("Campaigns", fdf["campaign"].nunique())
        c4.metric("Channels", fdf["channel"].nunique())

        st.dataframe(fdf, use_container_width=True, height=500)

        csv = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Filtered CSV", csv, "filtered_data.csv",
                           "text/csv", use_container_width=True)
    except Exception as e:
        st.error(f"Data tab error: {e}")
        logger.error("Dashboard tab5 error: %s", e)


# ---------- TAB 6: EXPLAINABILITY ----------
with tab6:
    try:
        st.markdown('<div class="section-header">Model Explainability</div>',
                    unsafe_allow_html=True)

        ex_pipeline = load_pipeline("high_roi")

        if ex_pipeline is None:
            st.warning("No trained model found. Run `python main.py` to train.")
        else:
            from models.trainer import get_feature_importance

            feature_names = [c for c in df.columns
                             if c not in ["date", "roi", "roas", "revenue",
                                          "revenue_per_conversion", "target"]]
            importance_df = get_feature_importance(ex_pipeline, feature_names)

            if importance_df["importance"].sum() > 0:
                st.markdown('<div class="section-header">Feature Importance</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(
                    plot_feature_importance_interactive(importance_df, "Best Model"),
                    use_container_width=True,
                )

                st.markdown('<div class="section-header">Top Feature Insights</div>',
                            unsafe_allow_html=True)
                top5 = importance_df.head(5)
                cols = st.columns(len(top5))
                for col, (_, row) in zip(cols, top5.iterrows()):
                    with col:
                        render_kpi(row["feature"][:20], f"{row['importance_pct']:.1f}%")

            # SHAP section — interactive charts first, static fallback
            st.markdown('<div class="section-header">SHAP Analysis</div>',
                        unsafe_allow_html=True)

            shap_csv = PROJECT_ROOT / "output" / "shap_importance.csv"
            if shap_csv.exists():
                shap_df = pd.read_csv(shap_csv)
                st.plotly_chart(
                    plot_feature_importance_interactive(shap_df, "SHAP Importance"),
                    use_container_width=True,
                )

                pos_col, neg_col = st.columns(2)
                with pos_col:
                    st.markdown("**🟢 Top Positive Impact Features**")
                    top_pos = shap_df.head(5)
                    for _, r in top_pos.iterrows():
                        st.markdown(f"- `{r['feature']}`: {r['importance_pct']:.1f}%")
                with neg_col:
                    st.markdown("**📊 Least Impactful Features**")
                    bottom = shap_df.tail(5)
                    for _, r in bottom.iterrows():
                        st.markdown(f"- `{r['feature']}`: {r['importance_pct']:.1f}%")
            else:
                st.info("SHAP analysis not yet generated. Run `python main.py` to compute SHAP values.")

            # Static SHAP plots fallback
            shap_files = {
                "SHAP Summary (Beeswarm)": "13_shap_summary.png",
                "SHAP Feature Importance": "14_shap_importance.png",
                "SHAP Impact Breakdown": "15_shap_waterfall.png",
            }
            for title, fname in shap_files.items():
                img_path = PROJECT_ROOT / "output" / fname
                if img_path.exists():
                    with st.expander(f"📷 {title} (static)"):
                        st.image(str(img_path), use_container_width=True)

    except Exception as e:
        st.error(f"Explainability error: {e}")
        logger.error("Dashboard tab6 error: %s", e)
