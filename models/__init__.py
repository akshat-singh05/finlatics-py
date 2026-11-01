"""
Machine learning package: revenue prediction for planned inventory buys.

Modules:
    - feature_engineering: Pre-delivery features, leakage list, time-based split
    - trainer: sklearn Pipelines + RandomizedSearchCV with time-series CV
    - evaluator: Model comparison and selection by CV score
    - business_evaluation: Rate-card baselines, $ error, profitability, margin captured
    - explainability: SHAP analysis of the selected model
    - ml_visualizations: Residual and prediction diagnostics
    - predictor: Shared prediction service for the API and dashboard
"""
