#!/usr/bin/env python
"""
CLI training script for the Telco Churn project.

Trains a model, evaluates it, and logs everything to MLflow.

Usage examples:
    python src/train.py --model-type logistic
    python src/train.py --model-type rf --params '{"n_estimators": 200, "max_depth": 15}'
    python src/train.py --model-type xgb --params '{"n_estimators": 200, "learning_rate": 0.05}'
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from src.data_preprocessing import load_data, preprocess, split_data


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "logistic": {
        "class": LogisticRegression,
        "defaults": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        "flavour": "sklearn",
    },
    "rf": {
        "class": RandomForestClassifier,
        "defaults": {"n_estimators": 100, "max_depth": 10},
        "flavour": "sklearn",
    },
    "xgb": {
        "class": XGBClassifier,
        "defaults": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        },
        "flavour": "xgboost",
    },
}


def build_model(model_type: str, user_params: dict):
    """Instantiate a model, merging user params over defaults."""
    entry = MODEL_REGISTRY[model_type]
    params = {**entry["defaults"], **user_params}
    params["random_state"] = params.get("random_state", 42)
    model = entry["class"](**params)
    return model, params, entry["flavour"]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_plots(model, X_test, y_test):
    """Create and log confusion matrix and ROC curve plots to MLflow."""
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train a Telco Churn model with MLflow tracking.")
    parser.add_argument(
        "--model-type",
        choices=list(MODEL_REGISTRY.keys()),
        default="rf",
        help="Model type to train (default: rf)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help='JSON string of hyperparameters, e.g. \'{"n_estimators": 200}\'',
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="telco-churn-cli",
        help="MLflow experiment name (default: telco-churn-cli)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV data file (default: data/telco_churn_full.csv)",
    )
    args = parser.parse_args()

    user_params = json.loads(args.params)

    # ---- Data ----
    print("Loading and preprocessing data...")
    df = load_data(args.data_path)
    X, y, scaler, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ---- Model ----
    model, params, flavour = build_model(args.model_type, user_params)
    print(f"Training {args.model_type} with params: {params}")
    model.fit(X_train, y_train)

    # ---- Evaluate ----
    metrics = evaluate(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- MLflow logging ----
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"{args.model_type}-cli"):
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("run_source", "cli")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log model
        log_fn = mlflow.sklearn.log_model if flavour == "sklearn" else mlflow.xgboost.log_model
        log_fn(model, artifact_path="model", input_example=X_test.iloc[:5])

        # Log plots
        log_plots(model, X_test, y_test)

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("Done!")


if __name__ == "__main__":
    main()
