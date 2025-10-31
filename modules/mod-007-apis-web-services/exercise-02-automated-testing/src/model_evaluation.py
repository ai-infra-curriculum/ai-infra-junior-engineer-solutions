"""Model Evaluation Module"""

import logging
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, float]:
    """Evaluate classification model."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except ValueError:
            logger.warning("Could not calculate ROC AUC")

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Get classification report."""
    report = classification_report(y_true, y_pred)
    return report


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Get confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    return cm


def calculate_feature_importance(model: Any) -> Dict[str, float]:
    """Calculate feature importance if available."""
    if hasattr(model, "feature_importances_"):
        return {"importance": model.feature_importances_.tolist()}
    elif hasattr(model, "coef_"):
        return {"coefficients": model.coef_.tolist()}
    else:
        logger.warning("Model does not have feature importance")
        return {}
