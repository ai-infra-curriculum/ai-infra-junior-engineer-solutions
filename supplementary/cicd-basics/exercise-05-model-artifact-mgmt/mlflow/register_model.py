#!/usr/bin/env python3
"""
Register trained model in MLflow Model Registry.

This script registers a model from an MLflow run into the Model Registry.
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient


def register_model(run_id: str, model_name: str, stage: str = "None"):
    """Register model from run ID."""
    client = MlflowClient()

    # Get run info
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)

    print(f"Registering model from run: {run_id}")
    print(f"Experiment: {experiment.name}")
    print(f"Model name: {model_name}")

    # Check if registered model exists
    try:
        registered_model = client.get_registered_model(model_name)
        print(f"Model '{model_name}' already exists in registry")
    except Exception:
        # Create registered model
        print(f"Creating new registered model: {model_name}")
        client.create_registered_model(
            name=model_name,
            tags={
                "team": "ml-platform",
                "framework": "scikit-learn"
            },
            description=f"Machine learning model: {model_name}"
        )

    # Create model version
    model_uri = f"runs:/{run_id}/model"

    print(f"Creating model version from: {model_uri}")
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        tags={
            "training_date": run.data.tags.get("training_date", "unknown"),
            "git_commit": run.data.tags.get("git_commit", "unknown"),
            "data_hash": run.data.tags.get("data_hash", "unknown"),
        }
    )

    print(f"✓ Model version created: {model_version.version}")

    # Transition to stage if specified
    if stage and stage != "None":
        print(f"Transitioning to stage: {stage}")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage,
            archive_existing_versions=False
        )

    # Get metrics from run
    metrics = run.data.metrics
    print("\nModel Metrics:")
    for metric_name in ["test_accuracy", "test_f1", "test_auc"]:
        if metric_name in metrics:
            print(f"  {metric_name}: {metrics[metric_name]:.4f}")

    print(f"\n✓ Model registered successfully!")
    print(f"Model: {model_name}")
    print(f"Version: {model_version.version}")
    print(f"Stage: {model_version.current_stage}")

    return model_version.version


def main():
    parser = argparse.ArgumentParser(description="Register model in MLflow Registry")

    parser.add_argument("--run-id", type=str, required=True,
                        help="MLflow run ID")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Registered model name")
    parser.add_argument("--stage", type=str, default="None",
                        choices=["None", "Staging", "Production", "Archived"],
                        help="Initial stage")

    args = parser.parse_args()

    version = register_model(args.run_id, args.model_name, args.stage)

    print(f"\nMODEL_VERSION={version}")


if __name__ == "__main__":
    main()
