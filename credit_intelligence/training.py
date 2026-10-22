from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_sample_weight

from credit_intelligence.constants import (
    CATEGORICAL_FEATURES,
    EDA_DIR,
    EXPERIMENTS_PATH,
    FEATURE_COLUMNS,
    MLRUNS_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET_COLUMN,
)
from credit_intelligence.data import load_dataset, write_json
from credit_intelligence.spark_processing import run_spark_analysis

MODEL_NAMES = ("Logistic Regression", "Random Forest", "Gradient Boosting")


def make_pipeline(model_name: str) -> Pipeline:
    scale_numeric = model_name == "Logistic Regression"
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    if model_name == "Logistic Regression":
        estimator = LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=180,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "Gradient Boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[("preprocess", preprocessing), ("model", estimator)]
    )


def fit_pipeline(model_name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    if model_name == "Gradient Boosting":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        pipeline.fit(X, y, model__sample_weight=weights)
    else:
        pipeline.fit(X, y)
    return pipeline


def evaluate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, object], np.ndarray]:
    probability = pipeline.predict_proba(X)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metrics: dict[str, object] = {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, predicted)),
        "brier_score": float(brier_score_loss(y, probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics, probability


def calibration_points(y: pd.Series, probability: np.ndarray) -> list[dict[str, float | int]]:
    observed, predicted = calibration_curve(
        y, probability, n_bins=10, strategy="quantile"
    )
    return [
        {
            "mean_predicted_probability": float(predicted_value),
            "fraction_positive": float(observed_value),
        }
        for observed_value, predicted_value in zip(observed, predicted, strict=True)
    ]


def run_training() -> dict[str, object]:
    frame, dataset_profile = load_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    write_json(EDA_DIR / "dataset_profile.json", dataset_profile)

    spark_summary = run_spark_analysis(frame, EDA_DIR)
    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].astype(int)

    # Reserve the final test set before fitting any preprocessing. The remaining
    # development partition is split into training and validation for selection.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    validation_results: dict[str, dict[str, object]] = {}
    for model_name in MODEL_NAMES:
        candidate = fit_pipeline(
            model_name,
            make_pipeline(model_name),
            X_train,
            y_train,
        )
        metrics, _ = evaluate(candidate, X_validation, y_validation)
        validation_results[model_name] = metrics
        print(
            f"{model_name}: validation F1={metrics['f1']:.4f}, "
            f"recall={metrics['recall']:.4f}"
        )

    selected_name = max(
        MODEL_NAMES,
        key=lambda name: (
            float(validation_results[name]["f1"]),
            float(validation_results[name]["recall"]),
            float(validation_results[name]["roc_auc"]),
        ),
    )
    development_rows = len(X_development)
    experiments: list[dict[str, object]] = []
    test_probabilities: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Pipeline] = {}
    for model_name in MODEL_NAMES:
        fitted = fit_pipeline(
            model_name,
            make_pipeline(model_name),
            X_development,
            y_development,
        )
        metrics, probability = evaluate(fitted, X_test, y_test)
        fitted_models[model_name] = fitted
        test_probabilities[model_name] = probability
        experiments.append(
            {
                "model": model_name,
                **metrics,
                "validation_f1": float(validation_results[model_name]["f1"]),
                "validation_recall": float(validation_results[model_name]["recall"]),
                "selected": model_name == selected_name,
                "calibration": calibration_points(y_test, probability),
            }
        )

    selected_pipeline = fitted_models[selected_name]
    joblib.dump(selected_pipeline, MODEL_PATH)
    selected_validation_model = fit_pipeline(
        selected_name,
        make_pipeline(selected_name),
        X_train,
        y_train,
    )
    importance = permutation_importance(
        selected_validation_model,
        X_validation,
        y_validation,
        scoring="f1",
        n_repeats=3,
        random_state=RANDOM_STATE,
        max_samples=0.25,
        n_jobs=-1,
    )
    feature_importance = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(
            zip(FEATURE_COLUMNS, importance.importances_mean, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    results: dict[str, object] = {
        "status": "ready",
        "dataset": dataset_profile,
        "selected_model": selected_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "split": {
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
            "selection_criterion": (
                "Highest validation F1 for the default class; ties are broken by "
                "validation recall, then validation ROC-AUC. Test results are not "
                "used to select the model."
            ),
            "test_threshold": 0.5,
        },
        "experiments": experiments,
        "feature_importance": feature_importance,
        "spark_sql": spark_summary,
        "limitations": [
            "The source is a historical Taiwan credit-card dataset, not current Brazilian credit data.",
            "The model is an educational portfolio example and must not be used for real credit decisions.",
            "The reported probabilities are not post-hoc calibrated; Brier scores and reliability bins are diagnostics, not a calibration guarantee.",
            "A single random holdout does not establish performance across populations or time.",
        ],
    }

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("credit-intelligence-uci")
    with mlflow.start_run(run_name="credit-default-model-comparison") as run:
        mlflow.log_params(
            {
                "dataset": dataset_profile["name"],
                "random_state": RANDOM_STATE,
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "test_rows": len(X_test),
                "selected_model": selected_name,
                "selection_metric": "validation_f1_then_recall_then_roc_auc",
            }
        )
        for experiment in experiments:
            prefix = str(experiment["model"]).lower().replace(" ", "_")
            for metric_name in (
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "accuracy",
                "brier_score",
            ):
                mlflow.log_metric(
                    f"{prefix}.{metric_name}", float(experiment[metric_name])
                )
        mlflow.log_metric(
            "selected.validation_f1",
            float(validation_results[selected_name]["f1"]),
        )
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="selected_model")
        results["mlflow_run_id"] = run.info.run_id

    write_json(RESULTS_PATH, results)
    write_json(EXPERIMENTS_PATH, {"experiments": experiments})
    print(f"Training complete. Selected by validation F1: {selected_name}")
    print(f"Real holdout results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_training()
