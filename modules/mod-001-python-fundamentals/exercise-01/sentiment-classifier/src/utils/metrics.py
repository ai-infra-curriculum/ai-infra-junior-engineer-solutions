"""
Metrics computation utilities for sentiment classifier.
"""

from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: List of predicted labels
        labels: List of true labels

    Returns:
        Dictionary with accuracy, precision, recall, and F1 score
    """
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="binary", zero_division=0),
        "recall": recall_score(labels, predictions, average="binary", zero_division=0),
        "f1": f1_score(labels, predictions, average="binary", zero_division=0),
    }


def plot_confusion_matrix(
    labels: List[int],
    predictions: List[int],
    output_path: Path,
    class_names: List[str] = None,
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        labels: True labels
        predictions: Predicted labels
        output_path: Path to save plot
        class_names: Names of classes (default: ["Negative", "Positive"])
    """
    if class_names is None:
        class_names = ["Negative", "Positive"]

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
