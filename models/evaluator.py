"""
Model evaluation and comparison module.

Evaluates Pipeline-based models with:
  - Test set metrics (accuracy, F1, RMSE, R2, etc.)
  - Cross-validation scores
  - Best hyperparameters
  - Comparison table & charts
  - Confusion matrices
  - Feature importance visualization
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

from models.trainer import get_feature_importance
from utils.logger import get_logger

logger = get_logger("evaluator")


# Chart style
PALETTE = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6", "#8B5CF6"]
COLORS = {
    "primary": "#6366F1",
    "success": "#10B981",
    "danger": "#EF4444",
    "info": "#3B82F6",
}


def evaluate_all_models(
    trained_models: dict,
    prepared_data: dict,
    output_dir: str | Path = "output",
) -> tuple[pd.DataFrame, str]:
    """
    Evaluate all trained models and generate comparison reports + charts.

    Parameters
    ----------
    trained_models : dict
        Output from trainer.train_all_models().
    prepared_data : dict
        Output from feature_engineering.prepare_features().
    output_dir : str or Path
        Directory for evaluation charts.

    Returns
    -------
    tuple[pd.DataFrame, str]
        - Comparison table of all models
        - Name of the best model
    """
    task_type = prepared_data["task_type"]
    y_test = prepared_data["y_test"]
    feature_names = prepared_data["feature_names"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"MODEL EVALUATION ({task_type.upper()})")
    print("=" * 60)

    if task_type == "classification":
        comparison = _evaluate_classifiers(trained_models, y_test)
    else:
        comparison = _evaluate_regressors(trained_models, y_test)

    # Print comparison table
    print("\n--- Model Comparison ---")
    print(comparison.to_string(index=True))

    if task_type == "regression":
        print("  (regression metrics are on the model's target scale)")

    # Identify best model by cross-validation score: choosing by test score
    # would tune the selection to the test set and overstate performance
    best_metric = "f1_score" if task_type == "classification" else "r2_score"
    best_model_name = comparison["cv_mean"].idxmax()
    best_score = comparison.loc[best_model_name, best_metric]
    cv_mean = comparison.loc[best_model_name, "cv_mean"]
    print(f"\n  >> Best model (by CV mean): {best_model_name}")
    print(f"     Test {best_metric}: {best_score:.4f} | CV mean: {cv_mean:.4f}")

    # Print best params for each model
    print("\n--- Best Hyperparameters ---")
    for name, result in trained_models.items():
        print(f"  {name}: {result['best_params']}")

    # Generate evaluation charts
    print(f"\n  Generating evaluation charts...")
    _plot_model_comparison(comparison, task_type, output_dir)
    _plot_cv_scores(trained_models, task_type, output_dir)
    _plot_feature_importance(
        trained_models[best_model_name]["pipeline"],
        feature_names, best_model_name, output_dir
    )

    if task_type == "classification":
        _plot_confusion_matrices(trained_models, y_test, output_dir)

    # Export comparison table
    csv_path = output_dir / "model_comparison.csv"
    comparison.to_csv(csv_path)
    logger.info("Model comparison saved to %s", csv_path.name)

    print("=" * 60)
    logger.info("Best model: %s (%s=%.4f, CV=%.4f)",
                best_model_name, best_metric, best_score, cv_mean)

    return comparison, best_model_name


def _evaluate_classifiers(trained_models: dict, y_test) -> pd.DataFrame:
    """Evaluate classification models with test + CV metrics."""
    rows = []
    for name, result in trained_models.items():
        preds = result["predictions"]
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
            "f1_score": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
            "cv_mean": result["cv_mean"],
            "cv_std": result["cv_std"],
            "train_time_s": result["train_time"],
        }

        if "probabilities" in result:
            try:
                metrics["roc_auc"] = round(
                    roc_auc_score(y_test, result["probabilities"][:, 1]), 4
                )
            except Exception:
                metrics["roc_auc"] = None

        rows.append(metrics)

        print(f"\n  {name}:")
        for k, v in metrics.items():
            if v is not None:
                print(f"    {k}: {v}")

    return pd.DataFrame(rows, index=trained_models.keys())


def _evaluate_regressors(trained_models: dict, y_test) -> pd.DataFrame:
    """Evaluate regression models with test + CV metrics."""
    rows = []
    for name, result in trained_models.items():
        preds = result["predictions"]
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        metrics = {
            "rmse": round(rmse, 4),
            "mae": round(mean_absolute_error(y_test, preds), 4),
            "r2_score": round(r2_score(y_test, preds), 4),
            "cv_mean": result["cv_mean"],
            "cv_std": result["cv_std"],
            "train_time_s": result["train_time"],
        }
        rows.append(metrics)

        print(f"\n  {name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    return pd.DataFrame(rows, index=trained_models.keys())


# =====================================================================
# VISUALIZATION
# =====================================================================

def _plot_model_comparison(comparison: pd.DataFrame, task_type: str, output_dir: Path):
    """Bar chart comparing model performance metrics."""
    if task_type == "classification":
        metrics = ["accuracy", "f1_score", "cv_mean"]
        titles = ["Test Accuracy", "Test F1 Score", "CV Mean (F1)"]
        if "roc_auc" in comparison.columns and comparison["roc_auc"].notna().all():
            metrics.append("roc_auc")
            titles.append("ROC AUC")
    else:
        metrics = ["r2_score", "rmse", "cv_mean"]
        titles = ["Test R-squared", "Test RMSE", "CV Mean (R2)"]

    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        values = comparison[metric]
        bars = ax.bar(
            range(len(values)), values,
            color=PALETTE[:len(values)], alpha=0.9, edgecolor="white", linewidth=1.5
        )
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.index, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(title)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Model Performance Comparison (Test + CV)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    filepath = output_dir / "09_model_comparison.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_cv_scores(trained_models: dict, task_type: str, output_dir: Path):
    """Box plot of cross-validation score distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(trained_models.keys())
    cv_data = [trained_models[n]["cv_scores"] for n in names]

    # Tick labels set separately: boxplot's `labels` kwarg was renamed in
    # matplotlib 3.9 and removed later, so avoid it for version compatibility
    bp = ax.boxplot(
        cv_data, patch_artist=True, widths=0.5,
        medianprops=dict(color="white", linewidth=2),
    )
    ax.set_xticks(range(1, len(names) + 1), names)

    for patch, color in zip(bp["boxes"], PALETTE[:len(names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Overlay individual points
    for i, scores in enumerate(cv_data):
        x = np.random.normal(i + 1, 0.04, size=len(scores))
        ax.scatter(x, scores, alpha=0.6, color="white", edgecolors="black", s=30, zorder=5)

    metric_name = "F1 Score" if task_type == "classification" else "R2 Score"
    ax.set_ylabel(f"CV {metric_name}")
    cv_label = trained_models[names[0]].get("cv_label", f"{len(cv_data[0])}-fold CV")
    ax.set_title(f"Cross-Validation Score Distribution ({cv_label})",
                 fontsize=14, fontweight="bold", pad=15)

    # Add mean annotations
    for i, scores in enumerate(cv_data):
        ax.text(i + 1, scores.mean(), f"  {scores.mean():.3f}",
                va="center", ha="left", fontsize=10, fontweight="bold", color=PALETTE[i])

    plt.tight_layout()
    filepath = output_dir / "12_cv_scores.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_feature_importance(pipeline, feature_names: list, model_name: str, output_dir: Path):
    """Horizontal bar chart of feature importance for the best model."""
    importance_df = get_feature_importance(pipeline, feature_names)

    if importance_df["importance"].sum() == 0:
        print(f"  [SKIP] No feature importance available for {model_name}")
        return

    top_n = min(15, len(importance_df))
    plot_data = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        plot_data["feature"], plot_data["importance_pct"],
        color=COLORS["primary"], alpha=0.9, edgecolor="white"
    )
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Top {top_n} Feature Importance ({model_name})",
                 fontsize=14, fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    for i, (_, row) in enumerate(plot_data.iterrows()):
        ax.text(row["importance_pct"] + 0.3, i, f"{row['importance_pct']:.1f}%",
                va="center", fontsize=9)

    plt.tight_layout()
    filepath = output_dir / "10_feature_importance.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_confusion_matrices(trained_models: dict, y_test, output_dir: Path):
    """Plot confusion matrices for all classification models."""
    n_models = len(trained_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    cmap = plt.cm.Blues

    for ax, (name, result) in zip(axes, trained_models.items()):
        preds = result["predictions"]
        cm = confusion_matrix(y_test, preds)

        ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(name, fontsize=12, fontweight="bold")

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight="bold")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])

    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    filepath = output_dir / "11_confusion_matrices.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")
