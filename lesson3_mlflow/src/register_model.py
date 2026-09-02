#!/usr/bin/env python
"""
Register the best model from an MLflow experiment in the Model Registry.

Finds the run with the highest value of a chosen metric, registers the model,
and assigns the 'champion' alias.

Usage examples:
    python src/register_model.py
    python src/register_model.py --experiment-name telco-churn --metric roc_auc
    python src/register_model.py --model-name TelcoChurnModel --alias champion
"""

import argparse
import os
import mlflow
from mlflow import MlflowClient


def main():
    parser = argparse.ArgumentParser(
        description="Register the best model from an MLflow experiment.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="telco-churn-cli",
        help="Name of the MLflow experiment to search (default: telco-churn-cli)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1_score",
        help="Metric to rank runs by (default: f1_score)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="TelcoChurnModel",
        help="Name to register the model under (default: TelcoChurnModel)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="champion",
        help="Alias to assign to the registered version (default: champion)",
    )
    args = parser.parse_args()

    # ---- Find best run ----
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        print(f"ERROR: Experiment '{args.experiment_name}' not found.")
        print("Run some training first (see notebooks or src/train.py).")
        return

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{args.metric} DESC"],
    )

    if runs_df.empty:
        print(f"ERROR: No runs found in experiment '{args.experiment_name}'.")
        return

    best_run = runs_df.iloc[0]
    best_run_id = best_run["run_id"]
    best_metric = best_run[f"metrics.{args.metric}"]
    run_name = best_run.get("tags.mlflow.runName", "unknown")

    print(f"Best run: {run_name}")
    print(f"  Run ID: {best_run_id}")
    print(f"  {args.metric}: {best_metric:.4f}")

    # ---- Register model ----
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    print(f"\nRegistered: {result.name} v{result.version}")

    # ---- Add description & alias ----
    client = MlflowClient()

    client.update_model_version(
        name=args.model_name,
        version=result.version,
        description=(f"Auto-registered from run '{run_name}' "
                     f"({args.metric}={best_metric:.4f})."),
    )

    client.set_registered_model_alias(
        name=args.model_name,
        alias=args.alias,
        version=result.version,
    )

    print(f"Alias '{args.alias}' -> {args.model_name} v{result.version}")
    print("Done!")


if __name__ == "__main__":
    main()
