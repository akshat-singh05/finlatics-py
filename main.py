#!/usr/bin/env python3
"""
Finlatics-Py: Online Advertising Performance Analysis & ML Pipeline

End-to-end data science pipeline that loads, cleans, analyzes,
visualizes, and builds predictive models on advertising performance data.

Usage:
    python main.py                          # Full pipeline (analysis + ML)
    python main.py --target high_roi        # ML with ROI classification
    python main.py --target revenue         # ML with revenue regression
    python main.py --target high_conversion # ML with conversion classification
    python main.py --no-ml                  # Analysis only, skip ML

Outputs:
    - Console: data inspection, cleaning report, KPI summary, model comparison
    - output/: visualization charts + ML evaluation charts (PNG)
"""

import sys
import argparse
import joblib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.data_loader import load_data, inspect_data
from analysis.data_cleaning import clean_data
from analysis.metrics import compute_kpis, summarize_by, get_overall_summary
from analysis.visualizations import generate_all_charts
from analysis.insights import generate_insights
from models.feature_engineering import prepare_features
from models.trainer import train_all_models
from models.evaluator import evaluate_all_models
from models.explainability import explain_model
from models.ml_visualizations import generate_ml_diagnostics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finlatics-Py: Advertising Performance Analysis & ML Pipeline"
    )
    parser.add_argument(
        "--target", type=str, default="high_roi",
        choices=["high_roi", "high_conversion", "revenue"],
        help="ML target variable (default: high_roi)"
    )
    parser.add_argument(
        "--no-ml", action="store_true",
        help="Skip machine learning, run analysis only"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test set fraction (default: 0.2)"
    )
    return parser.parse_args()


def main():
    """Run the full analysis and ML pipeline."""
    args = parse_args()
    data_path = PROJECT_ROOT / "data" / "online_advertising_performance_data.csv"
    output_dir = PROJECT_ROOT / "output"

    total_steps = 6 if args.no_ml else 11

    print("=" * 60)
    print("  FINLATICS-PY: Advertising Performance Analysis")
    if not args.no_ml:
        print(f"  ML Target: {args.target}")
    print("=" * 60)

    # ==================================================================
    # PHASE 1: DATA ANALYSIS
    # ==================================================================

    # Step 1: Load Data
    print(f"\n[STEP 1/{total_steps}] Loading data...")
    df = load_data(data_path)

    # Step 2: Inspect Raw Data
    print(f"\n[STEP 2/{total_steps}] Inspecting raw data...")
    inspect_data(df)

    # Step 3: Clean Data
    print(f"\n[STEP 3/{total_steps}] Cleaning data...")
    df = clean_data(df)

    # Step 4: Compute KPIs & Summarize
    print(f"\n[STEP 4/{total_steps}] Computing KPIs...")
    df = compute_kpis(df)
    get_overall_summary(df)

    # Print campaign summary table
    print("\n--- Campaign Summary ---")
    campaign_summary = summarize_by(df, "campaign")
    print(campaign_summary.to_string())

    # Print channel summary table
    print("\n--- Channel Summary ---")
    channel_summary = summarize_by(df, "channel")
    print(channel_summary.to_string())

    # Step 5: Generate Visualizations
    print(f"\n[STEP 5/{total_steps}] Generating visualizations to {output_dir}/...")
    saved_charts = generate_all_charts(df, output_dir)
    print(f"\nSaved {len(saved_charts)} charts to {output_dir}/")

    # Step 6: Business Insights
    print(f"\n[STEP 6/{total_steps}] Generating business insights...")
    insights = generate_insights(df, output_dir)

    # ==================================================================
    # PHASE 2: MACHINE LEARNING
    # ==================================================================

    if not args.no_ml:
        # Step 7: Feature Engineering
        print(f"\n[STEP 7/{total_steps}] Preparing features for ML...")
        prepared_data = prepare_features(
            df, target_type=args.target, test_size=args.test_size
        )

        # Step 8: Train Models
        print(f"\n[STEP 8/{total_steps}] Training models...")
        trained_models = train_all_models(prepared_data)

        # Step 9: Evaluate & Compare Models
        print(f"\n[STEP 9/{total_steps}] Evaluating models...")
        comparison, best_model_name = evaluate_all_models(
            trained_models, prepared_data, output_dir
        )

        # Save the best pipeline (preprocessor + model)
        model_dir = PROJECT_ROOT / "models" / "saved"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"best_pipeline_{args.target}.joblib"
        joblib.dump(trained_models[best_model_name]["pipeline"], model_path)
        print(f"\n  Best pipeline saved: {model_path}")

        # Step 10: Model Explainability (SHAP)
        print(f"\n[STEP 10/{total_steps}] Explaining best model with SHAP...")
        explain_results = explain_model(
            pipeline=trained_models[best_model_name]["pipeline"],
            X_train=prepared_data["X_train"],
            X_test=prepared_data["X_test"],
            feature_names=prepared_data["feature_names"],
            model_name=best_model_name,
            task_type=prepared_data["task_type"],
            output_dir=output_dir,
        )

        # Step 11: ML Diagnostic Visualizations
        print(f"\n[STEP 11/{total_steps}] Generating ML diagnostic charts...")
        generate_ml_diagnostics(trained_models, prepared_data, output_dir)

    # ==================================================================
    # DONE
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Records analyzed: {len(df):,}")
    print(f"  Charts generated: {len(saved_charts)}")
    if not args.no_ml:
        print(f"  ML target:        {args.target}")
        print(f"  Best model:       {best_model_name}")
        print(f"  Model saved to:   {model_path}")
    print(f"  Output directory:  {output_dir}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
