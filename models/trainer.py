"""
Model training with sklearn Pipelines, hyperparameter tuning, and cross-validation.

Architecture:
    ColumnTransformer (preprocessing)
        -> StandardScaler for numeric features
        -> OneHotEncoder for categorical features
    Pipeline (preprocessing + model)
    RandomizedSearchCV (hyperparameter optimization + CV)

Models:
    Classification: Logistic Regression, Random Forest, Gradient Boosting
    Regression: Ridge, Random Forest, Gradient Boosting
"""

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)

from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import randint, uniform, loguniform

warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================================
# PREPROCESSOR BUILDER
# =====================================================================

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles both numeric and categorical features.

    - Numeric: StandardScaler (z-score normalization)
    - Categorical: OneHotEncoder (with unknown handling for robustness)

    Parameters
    ----------
    numeric_features : list
        Column names for numeric features.
    categorical_features : list
        Column names for categorical features.

    Returns
    -------
    ColumnTransformer
        Fitted-ready preprocessing transformer.
    """
    transformers = []

    if numeric_features:
        transformers.append(
            ("num", StandardScaler(), numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="infrequent_if_exist",
                min_frequency=2,
            ), categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# =====================================================================
# MODEL + HYPERPARAMETER CONFIGS
# =====================================================================

def _get_classification_configs():
    """Return model constructors and hyperparameter search spaces for classification."""
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs"),
            "params": {
                "model__C": loguniform(0.01, 100),
                "model__l1_ratio": [0],  # Equivalent to L2 penalty
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 20,
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "model__n_estimators": randint(100, 500),
                "model__max_depth": [5, 8, 12, 15, None],
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 30,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": randint(3, 10),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__subsample": uniform(0.6, 0.4),
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 10),
            },
            "n_iter": 30,
        },
    }


def _get_regression_configs():
    """Return model constructors and hyperparameter search spaces for regression."""
    return {
        "Ridge Regression": {
            "model": Ridge(),
            "params": {
                "model__alpha": loguniform(0.001, 1000),
            },
            "n_iter": 20,
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "model__n_estimators": randint(100, 500),
                "model__max_depth": [5, 8, 12, 15, None],
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", None],
            },
            "n_iter": 30,
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": randint(3, 10),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__subsample": uniform(0.6, 0.4),
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 10),
            },
            "n_iter": 30,
        },
    }


# =====================================================================
# TRAINING PIPELINE
# =====================================================================

def train_all_models(prepared_data: dict, cv_folds: int = 5) -> dict:
    """
    Train all models using sklearn Pipelines with hyperparameter tuning.

    Each model is wrapped in:
        Pipeline([preprocessor, model])
            -> RandomizedSearchCV(pipeline, param_grid, cv=cv_folds)

    Parameters
    ----------
    prepared_data : dict
        Output from feature_engineering.prepare_features().
    cv_folds : int
        Number of cross-validation folds (default 5).

    Returns
    -------
    dict
        For each model name:
        - pipeline: fitted Pipeline (best estimator from search)
        - search: fitted RandomizedSearchCV object
        - best_params: best hyperparameters found
        - cv_scores: cross-validation scores on training data
        - cv_mean: mean CV score
        - cv_std: CV score standard deviation
        - predictions: test set predictions
        - train_time: total training time
    """
    task_type = prepared_data["task_type"]
    X_train = prepared_data["X_train"]
    y_train = prepared_data["y_train"]
    X_test = prepared_data["X_test"]
    numeric_features = prepared_data["numeric_features"]
    categorical_features = prepared_data["categorical_features"]

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Get model configs
    if task_type == "classification":
        configs = _get_classification_configs()
        scoring = "f1_weighted"
    else:
        configs = _get_regression_configs()
        scoring = "r2"

    results = {}

    print("\n" + "=" * 60)
    print(f"MODEL TRAINING WITH HYPERPARAMETER TUNING ({task_type.upper()})")
    print(f"  Strategy: RandomizedSearchCV | CV folds: {cv_folds} | Scoring: {scoring}")
    print("=" * 60)

    for name, config in configs.items():
        print(f"\n  [{name}]")

        # Build pipeline: preprocessor -> model
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", config["model"]),
        ])

        # RandomizedSearchCV for hyperparameter tuning
        start_time = time.time()

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            cv=cv_folds,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            return_train_score=True,
            error_score="raise",
        )

        search.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Best pipeline
        best_pipeline = search.best_estimator_
        predictions = best_pipeline.predict(X_test)

        # Cross-validation scores on training data with best model
        cv_scores = cross_val_score(
            best_pipeline, X_train, y_train, cv=cv_folds, scoring=scoring
        )

        # Extract best params (clean names)
        best_params = {
            k.replace("model__", ""): v
            for k, v in search.best_params_.items()
        }

        results[name] = {
            "pipeline": best_pipeline,
            "search": search,
            "best_params": best_params,
            "cv_scores": cv_scores,
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "predictions": predictions,
            "train_time": round(train_time, 2),
        }

        # Get probabilities for classification
        if task_type == "classification" and hasattr(best_pipeline, "predict_proba"):
            try:
                results[name]["probabilities"] = best_pipeline.predict_proba(X_test)
            except Exception:
                pass

        # Report
        print(f"    Time: {train_time:.1f}s ({config['n_iter']} iterations x {cv_folds} folds)")
        print(f"    Best CV score: {search.best_score_:.4f}")
        print(f"    CV mean +/- std: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"    Best params: {best_params}")

    print(f"\n  Trained {len(results)} optimized models.")
    print("=" * 60)

    return results


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

def get_feature_importance(pipeline, feature_names_in: list) -> pd.DataFrame:
    """
    Extract feature importance from a trained Pipeline.

    Handles the feature name transformation done by ColumnTransformer
    (one-hot encoded columns have different names from input).

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Fitted pipeline with preprocessor and model steps.
    feature_names_in : list
        Original input feature names.

    Returns
    -------
    pd.DataFrame
        Feature importance table sorted descending.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get transformed feature names
    try:
        transformed_names = preprocessor.get_feature_names_out()
    except Exception:
        transformed_names = [f"feature_{i}" for i in range(len(feature_names_in))]

    # Get importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
        if len(importances) != len(transformed_names):
            importances = np.abs(model.coef_).mean(axis=0)
    else:
        return pd.DataFrame({
            "feature": list(transformed_names)[:20],
            "importance": [0] * min(20, len(transformed_names))
        })

    # Align lengths
    n = min(len(transformed_names), len(importances))
    df = pd.DataFrame({
        "feature": list(transformed_names)[:n],
        "importance": importances[:n],
    })

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    total = df["importance"].sum()
    if total > 0:
        df["importance_pct"] = (df["importance"] / total * 100).round(2)

    return df
