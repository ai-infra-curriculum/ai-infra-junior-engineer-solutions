"""Model evaluation module."""

import logging
import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate ML model performance."""

    def __init__(self):
        """Initialize model evaluator."""
        logger.info("ModelEvaluator initialized")

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        log_to_mlflow: bool = True
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            log_to_mlflow: Whether to log metrics to MLflow

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating model on {len(X_test)} test samples")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        # Try to calculate ROC AUC if model supports predict_proba
        try:
            y_proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:  # Binary classification
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        except (AttributeError, ValueError):
            pass

        # Log to MLflow
        if log_to_mlflow:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"eval_{metric_name}", metric_value)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")

        return metrics

    def get_confusion_matrix(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> np.ndarray:
        """Get confusion matrix.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Confusion matrix
        """
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def get_classification_report(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> str:
        """Get detailed classification report.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Classification report string
        """
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred)

    def check_production_readiness(
        self,
        metrics: Dict[str, float],
        accuracy_threshold: float = 0.85,
        precision_threshold: float = 0.80,
        recall_threshold: float = 0.80
    ) -> tuple[bool, str]:
        """Check if model meets production criteria.

        Args:
            metrics: Dictionary of metrics
            accuracy_threshold: Minimum required accuracy
            precision_threshold: Minimum required precision
            recall_threshold: Minimum required recall

        Returns:
            Tuple of (is_ready, message)
        """
        checks = {
            "accuracy": (metrics.get("accuracy", 0), accuracy_threshold),
            "precision": (metrics.get("precision", 0), precision_threshold),
            "recall": (metrics.get("recall", 0), recall_threshold)
        }

        failed_checks = []
        for metric_name, (value, threshold) in checks.items():
            if value < threshold:
                failed_checks.append(
                    f"{metric_name}: {value:.4f} < {threshold:.4f}"
                )

        if failed_checks:
            message = "Model NOT ready for production. Failed checks:\n" + "\n".join(
                [f"  - {check}" for check in failed_checks]
            )
            return False, message
        else:
            message = "Model ready for production. All checks passed."
            return True, message
