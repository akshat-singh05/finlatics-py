"""
ML-focused visualizations for model diagnostics.

Generates prediction analysis charts:
  - Predicted vs Actual scatter plot
  - Residual distribution plot
  - Residuals vs Predicted plot
  - Prediction error by category
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error


PALETTE = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6", "#8B5CF6"]
COLORS = {
    "primary": "#6366F1",
    "success": "#10B981",
    "danger": "#EF4444",
    "info": "#3B82F6",
}


def generate_ml_diagnostics(
    trained_models: dict,
    prepared_data: dict,
    output_dir: str | Path = "output",
):
    """
    Generate ML diagnostic plots for the best model.

    Parameters
    ----------
    trained_models : dict
        Output from trainer.train_all_models().
    prepared_data : dict
        Output from feature_engineering.prepare_features().
    output_dir : str or Path
        Save directory.
    """
    task_type = prepared_data["task_type"]
    y_test = prepared_data["y_test"]
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("ML DIAGNOSTIC VISUALIZATIONS")
    print("=" * 60)

    # Find best model
    if task_type == "classification":
        from sklearn.metrics import f1_score as f1
        best_name = max(trained_models, key=lambda n: f1(y_test, trained_models[n]["predictions"], average="weighted"))
    else:
        best_name = max(trained_models, key=lambda n: r2_score(y_test, trained_models[n]["predictions"]))

    best = trained_models[best_name]
    preds = best["predictions"]

    if task_type == "regression":
        _plot_predicted_vs_actual(y_test, preds, best_name, output_dir)
        _plot_residuals(y_test, preds, best_name, output_dir)
        _plot_residual_distribution(y_test, preds, best_name, output_dir)
    else:
        _plot_classification_analysis(y_test, preds, best_name, trained_models, output_dir)

    # Prediction confidence for all models
    _plot_model_accuracy_bars(trained_models, y_test, task_type, output_dir)

    print("=" * 60)


def _plot_predicted_vs_actual(y_test, preds, model_name, output_dir):
    """Scatter plot: predicted vs actual values with perfect prediction line."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, preds, alpha=0.6, s=40, color=COLORS["primary"],
               edgecolors="white", linewidth=0.5, label="Predictions")

    # Perfect prediction line
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "--", color=COLORS["danger"], linewidth=2, alpha=0.7, label="Perfect prediction")

    # Metrics annotation
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    ax.text(0.05, 0.95, f"R2 = {r2:.4f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor="#E5E7EB"))

    ax.set_xlabel("Actual Values", fontsize=11)
    ax.set_ylabel("Predicted Values", fontsize=11)
    ax.set_title(f"Predicted vs Actual - {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right")

    plt.tight_layout()
    filepath = output_dir / "19_predicted_vs_actual.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_residuals(y_test, preds, model_name, output_dir):
    """Residuals vs predicted values (check for heteroscedasticity)."""
    residuals = np.array(y_test) - np.array(preds)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(preds, residuals, alpha=0.6, s=30, color=COLORS["info"],
               edgecolors="white", linewidth=0.5)
    ax.axhline(y=0, color=COLORS["danger"], linewidth=2, linestyle="--", alpha=0.7)

    # Add std bands
    std = residuals.std()
    ax.axhline(y=2*std, color="#9CA3AF", linewidth=1, linestyle=":", alpha=0.5, label=f"+/- 2 std ({2*std:.1f})")
    ax.axhline(y=-2*std, color="#9CA3AF", linewidth=1, linestyle=":", alpha=0.5)
    ax.fill_between(ax.get_xlim(), -2*std, 2*std, alpha=0.05, color=COLORS["info"])

    # Count outliers
    outliers = np.sum(np.abs(residuals) > 2 * std)
    ax.text(0.95, 0.95, f"Outliers (>2 std): {outliers}/{len(residuals)}",
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#F59E0B"))

    ax.set_xlabel("Predicted Values", fontsize=11)
    ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
    ax.set_title(f"Residual Plot - {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left")

    plt.tight_layout()
    filepath = output_dir / "20_residual_plot.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_residual_distribution(y_test, preds, model_name, output_dir):
    """Histogram of residuals (check for normality)."""
    residuals = np.array(y_test) - np.array(preds)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(residuals, bins=30, color=COLORS["primary"], alpha=0.8,
            edgecolor="white", linewidth=1)

    # Stats
    mean_r = residuals.mean()
    std_r = residuals.std()
    ax.axvline(x=mean_r, color=COLORS["danger"], linewidth=2, linestyle="--",
               label=f"Mean: {mean_r:.2f}")
    ax.axvline(x=mean_r + std_r, color=COLORS["warning"], linewidth=1.5, linestyle=":",
               label=f"Std: {std_r:.2f}")
    ax.axvline(x=mean_r - std_r, color=COLORS["warning"], linewidth=1.5, linestyle=":")

    ax.set_xlabel("Residual Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Residual Distribution - {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right")

    plt.tight_layout()
    filepath = output_dir / "21_residual_distribution.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_classification_analysis(y_test, preds, model_name, trained_models, output_dir):
    """Classification: error analysis by predicted class."""
    from sklearn.metrics import classification_report

    correct = np.array(y_test) == np.array(preds)
    n_correct = correct.sum()
    n_wrong = len(correct) - n_correct

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Correct vs Wrong
    ax = axes[0]
    bars = ax.bar(["Correct", "Incorrect"], [n_correct, n_wrong],
                  color=[COLORS["success"], COLORS["danger"]], alpha=0.9, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f"{h} ({h/len(correct)*100:.1f}%)",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction Accuracy - {model_name}", fontsize=13, fontweight="bold")

    # Right: Per-class accuracy comparison across models
    ax = axes[1]
    from sklearn.metrics import accuracy_score
    model_accuracies = {}
    for name, result in trained_models.items():
        model_accuracies[name] = accuracy_score(y_test, result["predictions"])

    names = list(model_accuracies.keys())
    accs = list(model_accuracies.values())
    bars = ax.bar(range(len(names)), accs, color=PALETTE[:len(names)], alpha=0.9, edgecolor="white")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f"{h:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    filepath = output_dir / "19_classification_analysis.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_model_accuracy_bars(trained_models, y_test, task_type, output_dir):
    """Horizontal accuracy/R2 bars with train time annotations."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(trained_models.keys())
    if task_type == "classification":
        from sklearn.metrics import f1_score as f1
        scores = [f1(y_test, trained_models[n]["predictions"], average="weighted") for n in names]
        metric_name = "F1 Score"
    else:
        scores = [r2_score(y_test, trained_models[n]["predictions"]) for n in names]
        metric_name = "R2 Score"

    times = [trained_models[n]["train_time"] for n in names]

    # Sort by score
    sorted_idx = np.argsort(scores)
    names = [names[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    times = [times[i] for i in sorted_idx]

    bars = ax.barh(names, scores, color=PALETTE[:len(names)], alpha=0.9, edgecolor="white")

    for i, (bar, score, t) in enumerate(zip(bars, scores, times)):
        ax.text(bar.get_width() + 0.005, i,
                f"{score:.3f} ({t:.1f}s)",
                va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel(metric_name)
    ax.set_title(f"Model {metric_name} (with training time)",
                 fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    filepath = output_dir / "22_model_diagnostics.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")
