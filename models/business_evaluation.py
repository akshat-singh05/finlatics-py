"""
Business evaluation of the revenue model against simple baselines.

The model predicts log1p(revenue) for a planned inventory buy. This module
answers the questions that matter to Advert Firm A:

    1. How close are the predicted dollars?        (MAE $, total revenue)
    2. Does it beat a simple rate-card rule?        (segment-rate baselines)
    3. Can it tell profitable buys from losses?     (predicted margin > 0)
    4. How much margin would following it capture?  (margin captured)

Metrics are implemented with numpy (no scikit-learn) so they are unit
testable anywhere and reusable by the API.
"""

import json
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.visualizations import fmt_money, setup_style
from models.feature_engineering import target_to_revenue
from utils.theme import CATEGORICAL, INK, SEQUENTIAL_BLUE
from utils.logger import get_logger

logger = get_logger("business_evaluation")


SEGMENT_COLUMNS = ["campaign_number", "user_engagement", "banner", "placement"]
RECENT_DAYS = 21
MATERIAL_COST = 1.0  # rows costing at least $1 are reported separately


# =====================================================================
# BASELINES
# =====================================================================

def _rates(train: pd.DataFrame, by: list[str] | None) -> pd.Series | float:
    """Revenue per display, overall or per group (NaN where no displays)."""
    if not by:
        return train["revenue"].sum() / train["displays"].sum()
    g = train.groupby(by, observed=True)[["revenue", "displays"]].sum()
    return g["revenue"] / g["displays"].where(g["displays"] > 0)


def segment_rate_baseline(eval_train: pd.DataFrame, eval_test: pd.DataFrame,
                          recent_days: int | None = None) -> np.ndarray:
    """
    Predict revenue = displays × the segment's historical revenue per display.

    Parameters
    ----------
    eval_train, eval_test : pd.DataFrame
        Frames with date, displays, revenue and SEGMENT_COLUMNS.
    recent_days : int or None
        Use only the last ``recent_days`` of training (a "current rate card");
        segments unseen in that window fall back to the full history, then
        to the global rate.

    Returns
    -------
    np.ndarray
        Predicted revenue in dollars.
    """
    global_rate = _rates(eval_train, None)
    rate = eval_test[SEGMENT_COLUMNS].join(
        _rates(eval_train, SEGMENT_COLUMNS).rename("full"), on=SEGMENT_COLUMNS)["full"]

    if recent_days:
        cutoff = eval_train["date"].max() - pd.Timedelta(days=recent_days - 1)
        recent = eval_train[eval_train["date"] >= cutoff]
        recent_rate = eval_test[SEGMENT_COLUMNS].join(
            _rates(recent, SEGMENT_COLUMNS).rename("recent"), on=SEGMENT_COLUMNS)["recent"]
        rate = recent_rate.fillna(rate)

    rate = rate.fillna(global_rate)
    return (eval_test["displays"] * rate).to_numpy(dtype=float)


def global_rate_baseline(eval_train: pd.DataFrame, eval_test: pd.DataFrame) -> np.ndarray:
    """Predict revenue = displays × the overall revenue per display."""
    return (eval_test["displays"] * _rates(eval_train, None)).to_numpy(dtype=float)


# =====================================================================
# METRICS
# =====================================================================

def regression_metrics(actual_revenue, predicted_revenue) -> dict:
    """Log-scale fit plus dollar error and total-revenue bias."""
    a = np.asarray(actual_revenue, dtype=float)
    p = np.asarray(predicted_revenue, dtype=float)
    la, lp = np.log1p(a), np.log1p(p)
    ss_res = ((la - lp) ** 2).sum()
    ss_tot = ((la - la.mean()) ** 2).sum()
    return {
        "r2_log": 1 - ss_res / ss_tot if ss_tot else np.nan,
        "mae_log": np.abs(la - lp).mean(),
        "rmse_log": np.sqrt(((la - lp) ** 2).mean()),
        "mae_usd": np.abs(a - p).mean(),
        "total_predicted_vs_actual": p.sum() / a.sum() if a.sum() else np.nan,
    }


def roc_auc(y_true, score) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (ties get average rank)."""
    y = np.asarray(y_true, dtype=bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def classification_metrics(y_true, y_pred, score=None) -> dict:
    """Binary metrics with 'profitable' as the positive class."""
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(t),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": roc_auc(t, score) if score is not None else np.nan,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def margin_captured(actual_margin, predicted_profitable) -> dict:
    """Margin earned by buying only rows predicted profitable vs alternatives."""
    m = np.asarray(actual_margin, dtype=float)
    buy = np.asarray(predicted_profitable, dtype=bool)
    oracle = m[m > 0].sum()
    return {
        "buy_predicted_profitable": m[buy].sum(),
        "buy_everything": m.sum(),
        "oracle": oracle,
        "share_of_oracle": m[buy].sum() / oracle if oracle else np.nan,
        "rows_bought_share": buy.mean(),
    }


def evaluate_predictions(eval_test: pd.DataFrame, predicted_revenue) -> dict:
    """All business metrics for one set of revenue predictions."""
    pred = np.asarray(predicted_revenue, dtype=float)
    actual_profitable = eval_test["firm_margin"].to_numpy() > 0
    pred_margin = pred - eval_test["cost"].to_numpy()

    out = {
        "revenue": regression_metrics(eval_test["revenue"], pred),
        "profitability": classification_metrics(actual_profitable, pred_margin > 0, pred_margin),
        "margin": margin_captured(eval_test["firm_margin"], pred_margin > 0),
    }

    material = (eval_test["cost"] >= MATERIAL_COST).to_numpy()
    if material.any():
        sub = eval_test[material]
        out["material_rows"] = {
            "rows": int(material.sum()),
            "revenue": regression_metrics(sub["revenue"], pred[material]),
            "profitability": classification_metrics(
                actual_profitable[material], pred_margin[material] > 0, pred_margin[material]),
        }
    return out


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def evaluate_against_baselines(
    pipeline,
    model_name: str,
    prepared_data: dict,
    output_dir: str | Path = "reports",
) -> dict:
    """
    Compare the selected model with the baselines on the held-out period.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Fitted best pipeline (predicts log1p revenue).
    model_name : str
        Name of the selected model.
    prepared_data : dict
        Output of ``prepare_features``.
    output_dir : str or Path
        ``model_evaluation.json`` goes here, charts to ``figures/ml/``.

    Returns
    -------
    dict
        Metrics per approach plus the split description.
    """
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures" / "ml"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tr, te = prepared_data["eval_train"], prepared_data["eval_test"]
    predictions = {
        "Baseline: global rate": global_rate_baseline(tr, te),
        "Baseline: segment rate (all history)": segment_rate_baseline(tr, te),
        f"Baseline: segment rate (last {RECENT_DAYS} days)": segment_rate_baseline(tr, te, RECENT_DAYS),
        f"Model: {model_name}": target_to_revenue(pipeline.predict(prepared_data["X_test"])),
    }

    results = {name: evaluate_predictions(te, pred) for name, pred in predictions.items()}

    print("\n" + "=" * 60)
    print("BUSINESS EVALUATION (held-out period from "
          f"{prepared_data['test_start']}, {len(te):,} rows)")
    print("=" * 60)
    table = pd.DataFrame({
        name: {
            "R2 (log)": r["revenue"]["r2_log"],
            "MAE ($)": r["revenue"]["mae_usd"],
            "Total pred/actual": r["revenue"]["total_predicted_vs_actual"],
            "Profit F1": r["profitability"]["f1"],
            "Profit ROC-AUC": r["profitability"]["roc_auc"],
            "Margin captured ($)": r["margin"]["buy_predicted_profitable"],
        } for name, r in results.items()
    }).T
    print(table.round(3).to_string())
    m = next(iter(results.values()))["margin"]
    print(f"\n  Reference: buying everything = ${m['buy_everything']:,.0f}; "
          f"perfect foresight = ${m['oracle']:,.0f}")

    _plot_comparison(table, m, fig_dir / "01_model_vs_baselines.png")
    model_key = f"Model: {model_name}"
    _plot_predicted_vs_actual(te, predictions[model_key], model_name,
                              fig_dir / "02_predicted_vs_actual.png")
    _plot_confusion(results[model_key]["profitability"]["confusion"], model_name,
                    fig_dir / "03_profitability_confusion.png")

    report = {
        "question": "Revenue (and derived margin) for a planned inventory buy",
        "target": "log1p(revenue)",
        "features": prepared_data["feature_names"],
        "train_rows": len(prepared_data["X_train"]),
        "test_rows": len(te),
        "test_start": prepared_data["test_start"],
        "selected_model": model_name,
        "results": results,
    }
    path = output_dir / "model_evaluation.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2)
    print(f"\n  Evaluation saved: {path}")
    print("=" * 60)
    logger.info("Business evaluation saved: %s", path)
    return report


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 6)
    return obj


# =====================================================================
# CHARTS
# =====================================================================

def _short(name: str) -> str:
    return (name.replace("Baseline: ", "").replace("Model: ", "")
            .replace("segment rate ", "segment\n").replace("(", "").replace(")", ""))


def _plot_comparison(table: pd.DataFrame, margin_ref: dict, filepath: Path):
    """R², dollar error and margin captured: model vs baselines."""
    setup_style()
    labels = [_short(n) for n in table.index]
    colors = [CATEGORICAL[1] if n.startswith("Model") else CATEGORICAL[0] for n in table.index]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = [("R2 (log)", "Fit on log revenue (R², higher is better)", lambda v: f"{v:.3f}"),
              ("MAE ($)", "Mean absolute error, $ (lower is better)", lambda v: f"${v:,.2f}"),
              ("Margin captured ($)", "Firm margin if buying only predicted-profitable rows",
               fmt_money)]
    for ax, (col, title, fmt) in zip(axes, panels):
        bars = ax.bar(labels, table[col], color=colors, width=0.6)
        for bar, v in zip(bars, table[col]):
            ax.annotate(fmt(v), (bar.get_x() + bar.get_width() / 2, max(v, 0)),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK["secondary"])
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.axhline(0, color=INK["muted"], linewidth=1)
    axes[2].axhline(margin_ref["buy_everything"], color=INK["muted"], linestyle="--", linewidth=1)
    axes[2].annotate(f"buy everything {fmt_money(margin_ref['buy_everything'])}",
                     (0, margin_ref["buy_everything"]), xytext=(0, 4),
                     textcoords="offset points", fontsize=8.5, color=INK["secondary"])
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_money(x)))

    fig.suptitle("Model vs baselines on the held-out period", fontsize=14,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_predicted_vs_actual(te: pd.DataFrame, pred, model_name: str, filepath: Path):
    """Log-log predicted vs actual revenue for the selected model."""
    setup_style()
    mask = (te["revenue"].to_numpy() > 0) & (np.asarray(pred) > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(te["revenue"].to_numpy()[mask], np.asarray(pred)[mask], s=10, alpha=0.35,
               color=CATEGORICAL[0], linewidths=0)
    lo = min(te["revenue"][te["revenue"] > 0].min(), np.asarray(pred)[mask].min())
    hi = max(te["revenue"].max(), np.asarray(pred).max())
    ax.plot([lo, hi], [lo, hi], "--", color=INK["muted"], linewidth=1, label="Perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual revenue ($, log)")
    ax.set_ylabel("Predicted revenue ($, log)")
    ax.grid(axis="x")
    ax.legend(loc="upper left")
    fig.suptitle(f"Predicted vs actual revenue — {model_name}", fontsize=14,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


def _plot_confusion(cm: dict, model_name: str, filepath: Path):
    """Profitability confusion matrix (positive = profitable)."""
    setup_style()
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(grid, cmap=cmap)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if grid[i, j] > grid.max() * 0.6 else INK["primary"])
    ax.set_xticks([0, 1], ["Loss", "Profitable"])
    ax.set_yticks([0, 1], ["Loss", "Profitable"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.suptitle(f"Profitability calls — {model_name}", fontsize=13,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")
