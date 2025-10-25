#!/usr/bin/env python3
"""
Train ML model with MLflow tracking.

This script demonstrates:
- MLflow experiment tracking
- Parameter and metric logging
- Model artifact logging
- Metadata tagging
"""

import argparse
import datetime
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def get_git_info():
    """Get current Git commit and branch."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('utf-8').strip()

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode('utf-8').strip()

        return commit, branch
    except Exception as e:
        print(f"Warning: Could not get Git info: {e}")
        return "unknown", "unknown"


def compute_data_hash(data):
    """Compute hash of dataset for reproducibility."""
    data_str = pd.util.hash_pandas_object(data).values
    return hashlib.sha256(str(data_str).encode()).hexdigest()[:16]


def load_data(data_path: str):
    """Load and validate dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path)
    print(f"Loaded data: {data.shape}")

    return data


def prepare_data(data, test_size, random_state):
    """Prepare train/test splits."""
    # Assume last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_model(model_type: str, **params):
    """Create model instance based on type."""
    models = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = models[model_type]

    # Filter params relevant to this model
    import inspect
    valid_params = inspect.signature(model_class.__init__).parameters
    filtered_params = {k: v for k, v in params.items() if k in valid_params}

    return model_class(**filtered_params)


def plot_confusion_matrix(y_true, y_pred, output_path="confusion_matrix.png"):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_feature_importance(model, feature_names, output_path="feature_importance.png"):
    """Generate and save feature importance plot."""
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20

    plt.figure(figsize=(10, 6))
    plt.title("Top 20 Feature Importances")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train ML model with MLflow")

    # Data parameters
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set size")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed")

    # Model parameters
    parser.add_argument("--model-type", type=str, default="random_forest",
                        choices=["random_forest", "gradient_boosting", "logistic_regression"],
                        help="Model type")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of estimators")
    parser.add_argument("--max-depth", type=int, default=10,
                        help="Maximum depth")
    parser.add_argument("--min-samples-split", type=int, default=2,
                        help="Min samples for split")

    # MLflow parameters
    parser.add_argument("--experiment-name", type=str, default="default",
                        help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="MLflow run name")

    args = parser.parse_args()

    # Set experiment
    mlflow.set_experiment(args.experiment_name)

    # Start MLflow run
    run_name = args.run_name or f"{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment: {args.experiment_name}")

        # Get Git info
        git_commit, git_branch = get_git_info()

        # Load data
        data = load_data(args.data_path)
        data_hash = compute_data_hash(data)

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(
            data, args.test_size, args.random_state
        )

        # Log parameters
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)

        # Log dataset info
        mlflow.log_param("n_samples", len(data))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        # Log tags
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("git_branch", git_branch)
        mlflow.set_tag("data_hash", data_hash)
        mlflow.set_tag("python_version", sys.version.split()[0])
        mlflow.set_tag("training_date", datetime.datetime.now().isoformat())

        # Create and train model
        print(f"Training {args.model_type} model...")
        model = create_model(
            args.model_type,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
        )

        start_time = datetime.datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.datetime.now() - start_time).total_seconds()

        mlflow.log_metric("training_time_seconds", training_time)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            # Training metrics
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_precision": precision_score(y_train, y_pred_train, average='weighted'),
            "train_recall": recall_score(y_train, y_pred_train, average='weighted'),
            "train_f1": f1_score(y_train, y_pred_train, average='weighted'),
            "train_auc": roc_auc_score(y_train, y_prob_train),

            # Test metrics
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test, average='weighted'),
            "test_recall": recall_score(y_test, y_pred_test, average='weighted'),
            "test_f1": f1_score(y_test, y_pred_test, average='weighted'),
            "test_auc": roc_auc_score(y_test, y_prob_test),
        }

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Calculate overfitting indicator
        overfit_score = metrics["train_accuracy"] - metrics["test_accuracy"]
        mlflow.log_metric("overfit_score", overfit_score)

        # Print results
        print("\n=== Training Results ===")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"Test F1 Score:  {metrics['test_f1']:.4f}")
        print(f"Test AUC:       {metrics['test_auc']:.4f}")
        print(f"Training Time:  {training_time:.2f}s")

        # Generate and log artifacts
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # Confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred_test, cm_path)
        mlflow.log_artifact(str(cm_path))

        # Feature importance
        fi_path = output_dir / "feature_importance.png"
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        if plot_feature_importance(model, feature_names, fi_path):
            mlflow.log_artifact(str(fi_path))

        # Log model
        print("\nLogging model to MLflow...")
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=None,  # Register separately
            input_example=X_train[:5],
            signature=mlflow.models.infer_signature(X_train, y_train),
        )

        print(f"\n✓ Model training complete!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")

        return run.info.run_id


if __name__ == "__main__":
    run_id = main()
    print(f"\nRUN_ID={run_id}")
