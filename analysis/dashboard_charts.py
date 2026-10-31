"""
Reusable interactive Plotly chart functions for the Streamlit dashboard.

Provides dashboard-optimized visualizations that complement the existing
matplotlib charts in visualizations.py and insights.py.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CHART_TEMPLATE = "plotly_dark"
COLORS = [
    "#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6",
    "#8B5CF6", "#14B8A6", "#F97316", "#EF4444", "#06B6D4",
]


def plot_feature_importance_interactive(
    importance_df: pd.DataFrame,
    model_name: str = "Best Model",
    top_n: int = 15,
) -> go.Figure:
    """
    Interactive horizontal bar chart of feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must have 'feature' and 'importance_pct' columns.
    model_name : str
        Model name for the title.
    top_n : int
        Number of top features to display.
    """
    plot_data = importance_df.head(top_n).iloc[::-1]

    fig = go.Figure(go.Bar(
        y=plot_data["feature"],
        x=plot_data["importance_pct"],
        orientation="h",
        marker_color="#6366F1",
        opacity=0.9,
        text=plot_data["importance_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=f"Top {len(plot_data)} Feature Importance — {model_name}",
        xaxis_title="Importance (%)",
        height=max(350, top_n * 28),
        margin=dict(l=0, r=60, t=40, b=0),
    )

    return fig


def plot_prediction_distribution(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> go.Figure:
    """
    Histogram of prediction probabilities for the positive class.

    Parameters
    ----------
    probabilities : np.ndarray
        2D array of shape (n_samples, 2) from predict_proba.
    threshold : float
        Decision threshold for the vertical line.
    """
    pos_probs = probabilities[:, 1]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=pos_probs,
        nbinsx=30,
        marker_color="#6366F1",
        opacity=0.85,
        name="Predictions",
    ))

    fig.add_vline(
        x=threshold, line_dash="dash", line_color="#F59E0B",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top right",
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        title="Prediction Probability Distribution",
        xaxis_title="P(Profitable)",
        yaxis_title="Count",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_shap_waterfall_interactive(
    shap_values: np.ndarray,
    feature_names: list,
    sample_idx: int = 0,
    top_n: int = 12,
) -> go.Figure:
    """
    Interactive SHAP waterfall for a single prediction.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array (for the positive class if classification).
    feature_names : list
        Feature names corresponding to SHAP columns.
    sample_idx : int
        Index of the sample to explain.
    top_n : int
        Number of top features to show.
    """
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1]) if len(shap_values) == 2 else np.array(shap_values[0])
    else:
        sv = np.array(shap_values)

    n = min(sv.shape[1], len(feature_names))
    sample_shap = sv[sample_idx, :n]

    sorted_idx = np.argsort(np.abs(sample_shap))[-top_n:]
    sorted_shap = sample_shap[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    colors = ["#10B981" if v > 0 else "#EF4444" for v in sorted_shap]

    fig = go.Figure(go.Bar(
        y=sorted_names,
        x=sorted_shap,
        orientation="h",
        marker_color=colors,
        opacity=0.9,
        text=[f"{v:+.3f}" for v in sorted_shap],
        textposition="outside",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8")

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=f"SHAP Impact Breakdown (Sample #{sample_idx})",
        xaxis_title="SHAP Value",
        height=max(350, top_n * 30),
        margin=dict(l=0, r=80, t=40, b=0),
    )

    return fig


def plot_confusion_matrix_interactive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """
    Interactive heatmap confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    model_name : str
        Model name for the title.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Positive"]

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=f"Confusion Matrix — {model_name}",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_actual_vs_predicted_interactive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """
    Scatter plot of actual vs predicted values with perfect-prediction line.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Actual and predicted values.
    model_name : str
        Model name for the title.
    """
    from sklearn.metrics import r2_score, mean_squared_error

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(color="#6366F1", size=6, opacity=0.6),
        name="Predictions",
    ))

    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    fig.add_trace(go.Scatter(
        x=lims, y=lims,
        mode="lines",
        line=dict(color="#EF4444", dash="dash", width=2),
        name="Perfect Prediction",
    ))

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=f"Actual vs Predicted — {model_name} (R²={r2:.4f}, RMSE={rmse:.2f})",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_confidence_gauge(confidence: float, label: str) -> go.Figure:
    """
    Gauge chart showing prediction confidence.

    Parameters
    ----------
    confidence : float
        Confidence score between 0 and 1.
    label : str
        Prediction label (e.g., "Profitable").
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#E2E8F0"}},
        title={"text": f"Confidence: {label}", "font": {"size": 14, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar": {"color": "#6366F1"},
            "bgcolor": "#1E293B",
            "bordercolor": "#475569",
            "steps": [
                {"range": [0, 50], "color": "#7F1D1D"},
                {"range": [50, 75], "color": "#78350F"},
                {"range": [75, 100], "color": "#064E3B"},
            ],
            "threshold": {
                "line": {"color": "#F59E0B", "width": 3},
                "thickness": 0.8,
                "value": 75,
            },
        },
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"},
    )

    return fig


def plot_roc_curve_interactive(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """
    Interactive ROC curve with AUC annotation.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Model name for the title.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(color="#6366F1", width=3),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.1)",
        name=f"{model_name} (AUC={roc_auc:.4f})",
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#94A3B8", dash="dash", width=2),
        name="Random (AUC=0.5)",
    ))

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=f"ROC Curve — {model_name} (AUC={roc_auc:.4f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.55, y=0.05),
    )

    return fig


def plot_precision_recall_interactive(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """
    Interactive Precision-Recall curve with average precision annotation.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Model name for the title.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        line=dict(color="#EC4899", width=3),
        fill="tozeroy",
        fillcolor="rgba(236, 72, 153, 0.1)",
        name=f"{model_name} (AP={ap:.4f})",
    ))

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=f"Precision-Recall Curve — {model_name} (AP={ap:.4f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.05, y=0.05),
        yaxis=dict(range=[0, 1.05]),
    )

    return fig


def plot_residual_interactive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """
    Interactive residuals vs predicted scatter plot for regression diagnostics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Actual and predicted values.
    model_name : str
        Model name for the title.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    std_r = residuals.std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(color="#3B82F6", size=6, opacity=0.6),
        name="Residuals",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444", line_width=2)
    fig.add_hline(y=2 * std_r, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"+2σ ({2*std_r:.1f})")
    fig.add_hline(y=-2 * std_r, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"-2σ ({-2*std_r:.1f})")

    fig.update_layout(
        template=CHART_TEMPLATE,
        title=f"Residual Plot — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Residual (Actual − Predicted)",
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_scenario_comparison(scenarios: list[dict]) -> go.Figure:
    """
    Grouped bar chart comparing KPIs across saved prediction scenarios.

    Parameters
    ----------
    scenarios : list[dict]
        Each dict must have keys: 'label', 'Spend', 'Clicks', 'Impressions', 'Prediction'.
    """
    if not scenarios:
        return go.Figure()

    labels = [s.get("label", f"Scenario {i+1}") for i, s in enumerate(scenarios)]
    spends = [s.get("Spend", 0) for s in scenarios]
    clicks = [s.get("Clicks", 0) for s in scenarios]
    impressions = [s.get("Impressions", 0) for s in scenarios]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Spend ($)", x=labels, y=spends,
        marker_color="#EF4444", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Clicks", x=labels, y=clicks,
        marker_color="#6366F1", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Impressions (÷100)", x=labels, y=[i / 100 for i in impressions],
        marker_color="#10B981", opacity=0.85,
    ))

    fig.update_layout(
        template=CHART_TEMPLATE,
        title="Scenario Comparison",
        barmode="group",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.12),
    )

    return fig
