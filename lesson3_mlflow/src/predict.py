#!/usr/bin/env python
"""
Load a registered model from the MLflow Model Registry and score new data.

Usage examples:
    python src/predict.py
    python src/predict.py --model-name TelcoChurnModel --alias champion
    python src/predict.py --model-name TelcoChurnModel --version 1
    python src/predict.py --data-path data/telco_churn_full.csv --output predictions.csv
"""

import argparse
import os
import sys

import mlflow
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from src.data_preprocessing import load_data, preprocess


def main():
    parser = argparse.ArgumentParser(
        description="Score data using a registered MLflow model.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="TelcoChurnModel",
        help="Registered model name (default: TelcoChurnModel)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--alias",
        type=str,
        default="champion",
        help="Model alias to load (default: champion)",
    )
    group.add_argument(
        "--version",
        type=int,
        default=None,
        help="Specific model version to load",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV file to score (default: data/telco_churn_full.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path for predictions (default: print to stdout)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit scoring to first N rows (useful for testing)",
    )
    args = parser.parse_args()

    # Treat empty strings as None for optional params
    if args.data_path.strip() == "" or args.data_path == "''":
        args.data_path = None
    if args.output.strip() == "" or args.output == "''":
        args.output = None

    # ---- Build model URI ----
    if args.version is not None:
        model_uri = f"models:/{args.model_name}/{args.version}"
    else:
        model_uri = f"models:/{args.model_name}@{args.alias}"

    print(f"Loading model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # ---- Load & preprocess data ----
    print("Loading and preprocessing data...")
    df = load_data(args.data_path)

    if args.limit:
        df = df.head(args.limit)

    X, y, scaler, feature_names = preprocess(df)
    print(f"  Scoring {len(X)} samples with {len(feature_names)} features")

    # ---- Predict ----
    predictions = model.predict(X)

    results = pd.DataFrame({
        "actual": y.values,
        "predicted": predictions,
    })

    # ---- Output ----
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print("\nPrediction results (first 20 rows):")
        print(results.head(20).to_string(index=False))

    # Quick accuracy summary
    correct = (results["actual"] == results["predicted"]).sum()
    total = len(results)
    print(f"\nAccuracy: {correct}/{total} = {correct / total:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
