#!/usr/bin/env python3
"""
Finlatics-Py: Online Advertising Performance Analysis & ML Pipeline

End-to-end data science pipeline that loads, cleans, analyzes,
visualizes, and builds a predictive model on advertising performance data.

Data:
    Reads data/raw/online_advertising_performance_data.csv (never modified)
    and writes the cleaned, KPI-enriched table to data/processed/.

ML task:
    Predict revenue (log1p) for a planned inventory buy from pre-delivery
    features, then derive margin and profitability. Time-based split; see
    models/feature_engineering.py.

Usage:
    python main.py                            # Full pipeline (analysis + ML)
    python main.py --no-ml                    # Analysis only, skip ML
    python main.py --test-start 2020-06-01    # Change the held-out period
    N_JOBS=2 python main.py                   # Fewer workers on low-memory machines

Outputs:
    - reports/figures/: 19 EDA charts; reports/eda_findings.json
    - reports/figures/ml/: model vs baselines; reports/model_evaluation.json
    - output/ml/: CV, feature importance, SHAP and residual charts
    - models/saved/revenue_model.joblib (+ .meta.json)
"""

import sys
import time
import argparse
import joblib
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from analysis.data_loader import DEFAULT_RAW_PATH, load_data, inspect_data
from analysis.data_cleaning import clean_data
from analysis.metrics import compute_kpis, summarize_by, get_overall_summary
from analysis.visualizations import generate_all_charts
from analysis.insights import generate_insights

logger = get_logger("pipeline")

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "advertising_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
ML_OUTPUT_DIR = PROJECT_ROOT / "output" / "ml"
MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "revenue_model.joblib"

SUMMARY_DIMENSIONS = ["campaign_number", "user_engagement", "banner", "placement"]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finlatics-Py: Advertising Performance Analysis & ML Pipeline"
    )
    parser.add_argument(
        "--no-ml", action="store_true",
        help="Skip machine learning, run analysis only"
    )
    parser.add_argument(
        "--test-start", type=str, default="2020-06-10",
        help="First day of the held-out test period (default: 2020-06-10)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of time-series CV folds (default: 5)"
    )
    return parser.parse_args()


def run_analysis(total_steps: int) -> pd.DataFrame | None:
    """Steps 1-6: load, inspect, clean, KPIs, EDA charts, insights."""
    # Step 1: Load Data
    print(f"\n[STEP 1/{total_steps}] Loading data...")
    try:
        df = load_data(DEFAULT_RAW_PATH)
        logger.info("Data loaded: %d rows from %s", len(df), DEFAULT_RAW_PATH.name)
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        print(f"\n[ERROR] {e}")
        return None

    # Step 2: Inspect Raw Data
    print(f"\n[STEP 2/{total_steps}] Inspecting raw data...")
    inspect_data(df)

    # Step 3: Clean Data
    print(f"\n[STEP 3/{total_steps}] Cleaning data...")
    try:
        df = clean_data(df)
        logger.info("Data cleaned: %d rows remaining", len(df))
    except Exception as e:
        logger.error("Data cleaning failed: %s", e)
        print(f"\n[ERROR] Cleaning failed: {e}")
        return None

    # Step 4: Compute KPIs & Summarize
    print(f"\n[STEP 4/{total_steps}] Computing KPIs...")
    df = compute_kpis(df)
    get_overall_summary(df)

    summary_cols = ["revenue", "cost", "firm_margin", "post_click_sales_amount",
                    "ctr", "conversion_rate", "firm_roi", "advertiser_roas",
                    "record_count", "active_days"]
    for dim in SUMMARY_DIMENSIONS:
        print(f"\n--- Summary by {dim} ---")
        print(summarize_by(df, dim)[summary_cols].round(4).to_string())

    # Save processed dataset (raw file is never modified)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n  Processed dataset saved: {PROCESSED_PATH}")
    logger.info("Processed dataset saved: %s", PROCESSED_PATH)

    # Step 5: Generate Visualizations
    fig_dir = REPORTS_DIR / "figures"
    print(f"\n[STEP 5/{total_steps}] Generating EDA charts to {fig_dir}/...")
    saved_charts = generate_all_charts(df, fig_dir)
    print(f"\nSaved {len(saved_charts)} charts to {fig_dir}/")
    logger.info("Generated %d analysis charts", len(saved_charts))

    # Step 6: Business Insights
    print(f"\n[STEP 6/{total_steps}] Generating business insights...")
    generate_insights(df, REPORTS_DIR)

    return df


def run_ml(df: pd.DataFrame, args, total_steps: int) -> str | None:
    """Steps 7-11: features, training, evaluation, SHAP, diagnostics."""
    # Imported here so `--no-ml` works without scikit-learn/SHAP installed
    try:
        from models.feature_engineering import prepare_features
        from models.trainer import train_all_models, save_training_metadata
        from models.evaluator import evaluate_all_models
        from models.business_evaluation import evaluate_against_baselines
        from models.explainability import explain_model
        from models.ml_visualizations import generate_ml_diagnostics
    except ImportError as e:
        logger.error("ML dependencies unavailable: %s", e)
        print(f"\n[ERROR] ML dependencies could not be loaded: {e}\n"
              "        Install requirements.txt, use the Dockerfile, or run with --no-ml.")
        return None

    # Step 7: Feature Engineering
    print(f"\n[STEP 7/{total_steps}] Preparing features for ML...")
    try:
        prepared_data = prepare_features(df, test_start=pd.Timestamp(args.test_start))
    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        print(f"\n[ERROR] Feature engineering failed: {e}")
        return None

    # Step 8: Train Models
    print(f"\n[STEP 8/{total_steps}] Training models...")
    try:
        trained_models = train_all_models(prepared_data, cv_folds=args.cv_folds)
        logger.info("Trained %d models", len(trained_models))
    except Exception as e:
        logger.error("Model training failed: %s", e)
        print(f"\n[ERROR] Model training failed: {e}")
        return None

    # Step 9: Evaluate (model selection by CV) & compare with baselines
    print(f"\n[STEP 9/{total_steps}] Evaluating models...")
    comparison, best_model_name = evaluate_all_models(
        trained_models, prepared_data, ML_OUTPUT_DIR
    )
    best_pipeline = trained_models[best_model_name]["pipeline"]
    evaluate_against_baselines(best_pipeline, best_model_name, prepared_data, REPORTS_DIR)

    # Save the best pipeline (preprocessor + model) and its metadata
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    meta_path = save_training_metadata(MODEL_PATH, best_model_name, comparison, prepared_data)
    print(f"\n  Best pipeline saved: {MODEL_PATH}")
    logger.info("Best pipeline saved: %s (%s), metadata: %s",
                MODEL_PATH.name, best_model_name, meta_path)

    # Step 10: Model Explainability (SHAP)
    print(f"\n[STEP 10/{total_steps}] Explaining best model with SHAP...")
    try:
        explain_model(
            pipeline=best_pipeline,
            X_train=prepared_data["X_train"],
            X_test=prepared_data["X_test"],
            feature_names=prepared_data["feature_names"],
            model_name=best_model_name,
            task_type=prepared_data["task_type"],
            output_dir=ML_OUTPUT_DIR,
        )
        logger.info("SHAP explainability completed for %s", best_model_name)
    except Exception as e:
        logger.warning("SHAP explainability failed (non-critical): %s", e)
        print(f"\n  [WARN] SHAP failed: {e}")

    # Step 11: ML Diagnostic Visualizations
    print(f"\n[STEP 11/{total_steps}] Generating ML diagnostic charts...")
    try:
        generate_ml_diagnostics(trained_models, prepared_data, ML_OUTPUT_DIR, best_model_name)
    except Exception as e:
        logger.warning("ML diagnostics failed (non-critical): %s", e)
        print(f"\n  [WARN] ML diagnostics failed: {e}")

    return best_model_name


def main():
    """Run the full analysis and ML pipeline."""
    args = parse_args()
    total_steps = 6 if args.no_ml else 11
    pipeline_start = time.time()

    print("=" * 60)
    print("  FINLATICS-PY: Advertising Performance Analysis")
    print("=" * 60)
    logger.info("Pipeline started | ML=%s", not args.no_ml)

    df = run_analysis(total_steps)
    if df is None:
        return None

    best_model_name = None if args.no_ml else run_ml(df, args, total_steps)

    elapsed = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Records analyzed: {len(df):,}")
    if best_model_name:
        print(f"  Best model:       {best_model_name}")
        print(f"  Model saved to:   {MODEL_PATH}")
    print(f"  Reports:          {REPORTS_DIR}")
    print(f"  Total time:       {elapsed:.1f}s")
    print("=" * 60)

    logger.info("Pipeline complete in %.1fs | %d records | model=%s",
                elapsed, len(df), best_model_name)
    return df


if __name__ == "__main__":
    main()
