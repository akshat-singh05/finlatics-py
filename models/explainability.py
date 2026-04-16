"""
Model explainability module using SHAP (SHapley Additive exPlanations).

Provides:
  - Tree-based feature importance extraction
  - SHAP value computation for the best model
  - SHAP summary plot (feature impact direction + magnitude)
  - SHAP bar plot (mean absolute impact)
  - Feature importance comparison chart
  - Top positive/negative contributors report
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from pathlib import Path


# Chart style
PALETTE = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6", "#8B5CF6"]
COLORS = {
    "primary": "#6366F1",
    "success": "#10B981",
    "danger": "#EF4444",
    "info": "#3B82F6",
    "warning": "#F59E0B",
}


def explain_model(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list,
    model_name: str,
    task_type: str,
    output_dir: str | Path = "output",
) -> dict:
    """
    Generate full model explainability report using SHAP.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Fitted Pipeline (preprocessor + model).
    X_train : pd.DataFrame
        Training features (raw, before preprocessing).
    X_test : pd.DataFrame
        Test features (raw, before preprocessing).
    feature_names : list
        Original feature column names.
    model_name : str
        Name of the best model.
    task_type : str
        "classification" or "regression".
    output_dir : str or Path
        Directory to save charts.

    Returns
    -------
    dict
        - shap_values: computed SHAP values
        - feature_importance: DataFrame of mean |SHAP| per feature
        - top_positive: features with highest positive impact
        - top_negative: features with highest negative impact
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print(f"MODEL EXPLAINABILITY (SHAP) - {model_name}")
    print("=" * 60)

    # Get the model and preprocessor from the pipeline
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Transform features through the preprocessor
    print("\n  Preprocessing features for SHAP...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get transformed feature names
    try:
        transformed_names = list(preprocessor.get_feature_names_out())
    except Exception:
        n_features = X_train_transformed.shape[1]
        transformed_names = [f"feature_{i}" for i in range(n_features)]

    # Convert to DataFrame for SHAP
    if not isinstance(X_train_transformed, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train_transformed, columns=transformed_names)
        X_test_df = pd.DataFrame(X_test_transformed, columns=transformed_names)
    else:
        X_train_df = X_train_transformed
        X_test_df = X_test_transformed

    # Compute SHAP values
    print("  Computing SHAP values (this may take a moment)...")
    shap_values, explainer = _compute_shap_values(model, X_train_df, X_test_df, task_type)

    # Build importance table
    print("  Building feature importance table...")
    importance_df = _build_importance_table(shap_values, transformed_names, task_type)

    # Identify positive and negative contributors
    pos_neg = _analyze_impact_direction(shap_values, transformed_names, task_type)

    # Print summary
    _print_summary(importance_df, pos_neg)

    # Generate charts
    print("\n  Generating explainability charts...")
    _plot_shap_summary(shap_values, X_test_df, transformed_names, task_type, model_name, output_dir)
    _plot_shap_bar(importance_df, model_name, output_dir)
    _plot_shap_waterfall(shap_values, X_test_df, transformed_names, task_type, model_name, output_dir)

    print("=" * 60)

    return {
        "shap_values": shap_values,
        "feature_importance": importance_df,
        "top_positive": pos_neg["top_positive"],
        "top_negative": pos_neg["top_negative"],
    }


def _compute_shap_values(model, X_train_df, X_test_df, task_type):
    """Compute SHAP values using the appropriate explainer."""
    model_type = type(model).__name__

    # Use TreeExplainer for tree-based models (fast and exact)
    if hasattr(model, "estimators_") or "Boosting" in model_type or "Forest" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)
    else:
        # Use KernelExplainer for linear / other models (sample background)
        background = shap.sample(X_train_df, min(100, len(X_train_df)))
        if task_type == "classification":
            explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_test_df)

    return shap_values, explainer


def _build_importance_table(shap_values, feature_names, task_type):
    """Build a sorted DataFrame of mean absolute SHAP values per feature."""
    # Handle multi-class: shap_values might be a list of arrays
    if isinstance(shap_values, list):
        # For binary classification, use positive class
        sv = np.array(shap_values[1]) if len(shap_values) == 2 else np.mean([np.abs(s) for s in shap_values], axis=0)
    else:
        sv = np.array(shap_values)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(sv).mean(axis=0)

    # Ensure lengths match
    n = min(len(feature_names), len(mean_abs_shap))
    df = pd.DataFrame({
        "feature": feature_names[:n],
        "mean_abs_shap": mean_abs_shap[:n],
    })

    total = df["mean_abs_shap"].sum()
    if total > 0:
        df["importance_pct"] = (df["mean_abs_shap"] / total * 100).round(2)
    else:
        df["importance_pct"] = 0.0

    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def _analyze_impact_direction(shap_values, feature_names, task_type):
    """Identify features with strongest positive and negative impacts."""
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1]) if len(shap_values) == 2 else np.array(shap_values[0])
    else:
        sv = np.array(shap_values)

    n = min(len(feature_names), sv.shape[1])
    mean_shap = sv[:, :n].mean(axis=0)

    impact_df = pd.DataFrame({
        "feature": feature_names[:n],
        "mean_shap": mean_shap,
        "direction": ["Positive" if v > 0 else "Negative" for v in mean_shap],
    }).sort_values("mean_shap", ascending=False)

    top_positive = impact_df[impact_df["mean_shap"] > 0].head(5)
    top_negative = impact_df[impact_df["mean_shap"] < 0].tail(5).sort_values("mean_shap")

    return {
        "impact_df": impact_df,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


def _print_summary(importance_df, pos_neg):
    """Print a human-readable explainability summary."""
    print("\n--- Top 10 Most Important Features (by SHAP) ---")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:<35s} {row['importance_pct']:6.2f}%")

    print("\n--- Top Positive Contributors (push prediction higher) ---")
    for _, row in pos_neg["top_positive"].iterrows():
        print(f"  [+] {row['feature']:<35s} avg SHAP: {row['mean_shap']:+.4f}")

    print("\n--- Top Negative Contributors (push prediction lower) ---")
    for _, row in pos_neg["top_negative"].iterrows():
        print(f"  [-] {row['feature']:<35s} avg SHAP: {row['mean_shap']:+.4f}")


# =====================================================================
# VISUALIZATION
# =====================================================================

def _plot_shap_summary(shap_values, X_test_df, feature_names, task_type, model_name, output_dir):
    """SHAP beeswarm/summary plot showing feature impact direction and magnitude."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get values for the positive class (classification) or raw (regression)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    else:
        sv = shap_values

    sv_array = np.array(sv)
    n = min(sv_array.shape[1], len(feature_names), X_test_df.shape[1])

    # Limit to top 15 features by mean absolute SHAP
    mean_abs = np.abs(sv_array[:, :n]).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:][::-1]

    sv_top = sv_array[:, top_idx]
    names_top = [feature_names[i] for i in top_idx]
    X_top = X_test_df.iloc[:, top_idx].values

    # Normalize feature values for coloring
    X_norm = np.zeros_like(X_top, dtype=float)
    for j in range(X_top.shape[1]):
        col = X_top[:, j].astype(float)
        vmin, vmax = col.min(), col.max()
        if vmax > vmin:
            X_norm[:, j] = (col - vmin) / (vmax - vmin)
        else:
            X_norm[:, j] = 0.5

    # Plot as scatter (beeswarm-style)
    for i in range(len(names_top)):
        y_positions = np.full(sv_top.shape[0], i)
        jitter = np.random.normal(0, 0.1, size=len(y_positions))

        scatter = ax.scatter(
            sv_top[:, i],
            y_positions + jitter,
            c=X_norm[:, i],
            cmap="RdBu_r",
            alpha=0.6,
            s=12,
            edgecolors="none",
        )

    ax.set_yticks(range(len(names_top)))
    ax.set_yticklabels(names_top, fontsize=9)
    ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
    ax.set_title(f"SHAP Summary Plot - {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.axvline(x=0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Feature Value", fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    plt.tight_layout()
    filepath = output_dir / "13_shap_summary.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_shap_bar(importance_df, model_name, output_dir):
    """Bar chart of mean absolute SHAP values (global importance)."""
    top_n = min(15, len(importance_df))
    plot_data = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by importance level
    colors = []
    for pct in plot_data["importance_pct"]:
        if pct >= 10:
            colors.append(COLORS["danger"])
        elif pct >= 5:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["primary"])

    ax.barh(
        plot_data["feature"], plot_data["mean_abs_shap"],
        color=colors, alpha=0.9, edgecolor="white"
    )

    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"SHAP Feature Importance - {model_name}",
                 fontsize=14, fontweight="bold", pad=15)

    # Value labels
    for i, (_, row) in enumerate(plot_data.iterrows()):
        ax.text(row["mean_abs_shap"] + 0.001, i,
                f"{row['importance_pct']:.1f}%",
                va="center", fontsize=9)

    plt.tight_layout()
    filepath = output_dir / "14_shap_importance.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_shap_waterfall(shap_values, X_test_df, feature_names, task_type, model_name, output_dir):
    """Waterfall plot showing positive vs negative impact for a sample prediction."""
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1]) if len(shap_values) == 2 else np.array(shap_values[0])
    else:
        sv = np.array(shap_values)

    n = min(sv.shape[1], len(feature_names))

    # Pick a representative sample (one with highest absolute total SHAP)
    sample_idx = np.abs(sv).sum(axis=1).argmax()
    sample_shap = sv[sample_idx, :n]

    # Sort by absolute value, take top 12
    sorted_idx = np.argsort(np.abs(sample_shap))[-12:]
    sorted_shap = sample_shap[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors_bar = [COLORS["success"] if v > 0 else COLORS["danger"] for v in sorted_shap]

    ax.barh(sorted_names, sorted_shap, color=colors_bar, alpha=0.9, edgecolor="white")
    ax.axvline(x=0, color="#9CA3AF", linewidth=1, linestyle="--")

    ax.set_xlabel("SHAP Value", fontsize=11)
    ax.set_title(f"SHAP Impact Breakdown (Sample #{sample_idx}) - {model_name}",
                 fontsize=13, fontweight="bold", pad=15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["success"], label="Pushes prediction UP"),
        Patch(facecolor=COLORS["danger"], label="Pushes prediction DOWN"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Value labels
    for i, v in enumerate(sorted_shap):
        offset = 0.002 if v >= 0 else -0.002
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, i, f"{v:+.3f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    filepath = output_dir / "15_shap_waterfall.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")
