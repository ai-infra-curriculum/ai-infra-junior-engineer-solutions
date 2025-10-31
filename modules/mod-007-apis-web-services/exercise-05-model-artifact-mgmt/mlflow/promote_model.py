#!/usr/bin/env python3
"""
Promote model to a new stage in MLflow Model Registry.

This script handles model promotion workflows with validation.
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient


def validate_model_metrics(client: MlflowClient, model_name: str, version: int, thresholds: dict):
    """Validate model metrics meet promotion thresholds."""
    # Get model version
    model_version = client.get_model_version(model_name, version)

    # Get run metrics
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    print("\nValidating model metrics...")
    validation_passed = True

    for metric_name, threshold in thresholds.items():
        if metric_name in metrics:
            metric_value = metrics[metric_name]
            passed = metric_value >= threshold

            status = "✓" if passed else "✗"
            print(f"  {status} {metric_name}: {metric_value:.4f} (threshold: {threshold})")

            if not passed:
                validation_passed = False
        else:
            print(f"  ⚠ {metric_name}: Not found in run metrics")

    return validation_passed


def promote_model(
    model_name: str,
    version: int,
    stage: str,
    archive_existing: bool = True,
    validate: bool = True
):
    """Promote model to new stage."""
    client = MlflowClient()

    # Get model version
    model_version = client.get_model_version(model_name, version)

    print(f"Model: {model_name}")
    print(f"Version: {version}")
    print(f"Current stage: {model_version.current_stage}")
    print(f"Target stage: {stage}")

    # Validation thresholds
    thresholds = {
        "test_accuracy": 0.80,
        "test_f1": 0.75,
    }

    if stage == "Production":
        thresholds["test_accuracy"] = 0.85
        thresholds["test_f1"] = 0.80

    # Validate metrics
    if validate:
        if not validate_model_metrics(client, model_name, version, thresholds):
            raise ValueError(
                f"Model validation failed. "
                f"Metrics do not meet thresholds for {stage} promotion."
            )
        print("\n✓ Validation passed")

    # Get current production model if promoting to production
    if stage == "Production" and archive_existing:
        current_prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if current_prod_versions:
            for pv in current_prod_versions:
                print(f"\nArchiving existing production model v{pv.version}...")
                client.transition_model_version_stage(
                    name=model_name,
                    version=pv.version,
                    stage="Archived"
                )

    # Transition to new stage
    print(f"\nPromoting model to {stage}...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=False  # Already handled above
    )

    # Add promotion tags
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promoted_to",
        value=stage
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promoted_from",
        value=model_version.current_stage
    )

    # Update description
    new_description = f"Promoted to {stage}"
    if model_version.description:
        new_description = f"{model_version.description}\n{new_description}"

    client.update_model_version(
        name=model_name,
        version=version,
        description=new_description
    )

    print(f"\n✓ Model promoted successfully!")
    print(f"Model: {model_name} v{version}")
    print(f"Stage: {stage}")

    # Show current versions by stage
    print(f"\nCurrent versions of {model_name}:")
    for stage_name in ["Staging", "Production"]:
        versions = client.get_latest_versions(model_name, stages=[stage_name])
        if versions:
            for v in versions:
                print(f"  {stage_name}: v{v.version}")
        else:
            print(f"  {stage_name}: (none)")


def main():
    parser = argparse.ArgumentParser(description="Promote model in MLflow Registry")

    parser.add_argument("--model-name", type=str, required=True,
                        help="Registered model name")
    parser.add_argument("--version", type=int, required=True,
                        help="Model version to promote")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["Staging", "Production", "Archived"],
                        help="Target stage")
    parser.add_argument("--no-archive", action="store_true",
                        help="Don't archive existing production models")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip metric validation")

    args = parser.parse_args()

    promote_model(
        model_name=args.model_name,
        version=args.version,
        stage=args.stage,
        archive_existing=not args.no_archive,
        validate=not args.skip_validation
    )


if __name__ == "__main__":
    main()
