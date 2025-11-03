"""
Machine Learning Metrics Module

Provides common evaluation metrics for classification and regression tasks.
"""

from typing import List, Dict, Tuple
import math


def accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: Predicted class labels
        labels: Ground truth labels

    Returns:
        Accuracy as float between 0 and 1

    Raises:
        ValueError: If lengths don't match or empty

    Examples:
        >>> accuracy([1, 0, 1, 1], [1, 0, 0, 1])
        0.75
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        return 0.0

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)


def precision(predictions: List[int],
             labels: List[int],
             positive_class: int = 1) -> float:
    """
    Calculate precision for binary classification.

    Precision = TP / (TP + FP)

    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        positive_class: Which class is considered positive (default: 1)

    Returns:
        Precision score

    Examples:
        >>> precision([1, 0, 1, 1], [1, 0, 0, 1])
        0.6666...
    """
    true_positives = sum(
        1 for p, l in zip(predictions, labels)
        if p == positive_class and l == positive_class
    )

    predicted_positives = sum(1 for p in predictions if p == positive_class)

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def recall(predictions: List[int],
          labels: List[int],
          positive_class: int = 1) -> float:
    """
    Calculate recall for binary classification.

    Recall = TP / (TP + FN)

    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        positive_class: Which class is considered positive (default: 1)

    Returns:
        Recall score

    Examples:
        >>> recall([1, 0, 1, 1], [1, 0, 0, 1])
        1.0
    """
    true_positives = sum(
        1 for p, l in zip(predictions, labels)
        if p == positive_class and l == positive_class
    )

    actual_positives = sum(1 for l in labels if l == positive_class)

    if actual_positives == 0:
        return 0.0

    return true_positives / actual_positives


def f1_score(predictions: List[int],
            labels: List[int],
            positive_class: int = 1) -> float:
    """
    Calculate F1 score (harmonic mean of precision and recall).

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        positive_class: Which class is considered positive (default: 1)

    Returns:
        F1 score

    Examples:
        >>> f1_score([1, 0, 1, 1], [1, 0, 0, 1])
        0.8
    """
    prec = precision(predictions, labels, positive_class)
    rec = recall(predictions, labels, positive_class)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def confusion_matrix(predictions: List[int],
                    labels: List[int],
                    num_classes: int) -> List[List[int]]:
    """
    Calculate confusion matrix.

    Matrix[i][j] = count of samples with true label i and predicted label j

    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        num_classes: Number of classes

    Returns:
        Confusion matrix as 2D list

    Examples:
        >>> cm = confusion_matrix([0, 1, 0, 1], [0, 1, 1, 1], 2)
        >>> cm
        [[1, 1], [0, 2]]
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]

    for pred, label in zip(predictions, labels):
        if 0 <= pred < num_classes and 0 <= label < num_classes:
            matrix[label][pred] += 1

    return matrix


def classification_report(predictions: List[int],
                         labels: List[int],
                         class_names: List[str] = None) -> Dict[str, Dict]:
    """
    Generate comprehensive classification report.

    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        class_names: Optional names for classes

    Returns:
        Dictionary with per-class metrics and overall accuracy

    Examples:
        >>> report = classification_report([1, 0, 1, 1], [1, 0, 0, 1],
        ...                               ["neg", "pos"])
        >>> report["accuracy"]
        0.75
    """
    if not predictions:
        return {"accuracy": 0.0}

    num_classes = max(max(predictions), max(labels)) + 1

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    report = {}

    # Per-class metrics
    for class_id, class_name in enumerate(class_names):
        if class_id < num_classes:
            report[class_name] = {
                "precision": precision(predictions, labels, class_id),
                "recall": recall(predictions, labels, class_id),
                "f1_score": f1_score(predictions, labels, class_id)
            }

    # Overall accuracy
    report["accuracy"] = accuracy(predictions, labels)

    return report


def mean_squared_error(predictions: List[float],
                      labels: List[float]) -> float:
    """
    Calculate mean squared error for regression.

    MSE = mean((predictions - labels)^2)

    Args:
        predictions: Predicted values
        labels: Ground truth values

    Returns:
        Mean squared error

    Examples:
        >>> mean_squared_error([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
        0.01
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        return 0.0

    squared_errors = [(p - l) ** 2 for p, l in zip(predictions, labels)]
    return sum(squared_errors) / len(squared_errors)


def root_mean_squared_error(predictions: List[float],
                           labels: List[float]) -> float:
    """
    Calculate root mean squared error for regression.

    RMSE = sqrt(MSE)

    Args:
        predictions: Predicted values
        labels: Ground truth values

    Returns:
        Root mean squared error

    Examples:
        >>> root_mean_squared_error([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
        0.1
    """
    mse = mean_squared_error(predictions, labels)
    return math.sqrt(mse)


def mean_absolute_error(predictions: List[float],
                       labels: List[float]) -> float:
    """
    Calculate mean absolute error for regression.

    MAE = mean(|predictions - labels|)

    Args:
        predictions: Predicted values
        labels: Ground truth values

    Returns:
        Mean absolute error

    Examples:
        >>> mean_absolute_error([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
        0.1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        return 0.0

    absolute_errors = [abs(p - l) for p, l in zip(predictions, labels)]
    return sum(absolute_errors) / len(absolute_errors)


def r_squared(predictions: List[float],
             labels: List[float]) -> float:
    """
    Calculate R² (coefficient of determination) for regression.

    R² = 1 - (SS_res / SS_tot)

    Args:
        predictions: Predicted values
        labels: Ground truth values

    Returns:
        R² score (1.0 = perfect, 0.0 = baseline, < 0 = worse than baseline)

    Examples:
        >>> r_squared([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 2.9, 4.1])
        0.99...
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        return 0.0

    # Calculate mean of labels
    mean_label = sum(labels) / len(labels)

    # Total sum of squares
    ss_tot = sum((l - mean_label) ** 2 for l in labels)

    # Residual sum of squares
    ss_res = sum((l - p) ** 2 for l, p in zip(labels, predictions))

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def mean_absolute_percentage_error(predictions: List[float],
                                   labels: List[float]) -> float:
    """
    Calculate mean absolute percentage error for regression.

    MAPE = mean(|predictions - labels| / |labels|) * 100

    Args:
        predictions: Predicted values
        labels: Ground truth values

    Returns:
        Mean absolute percentage error (as percentage)

    Examples:
        >>> mean_absolute_percentage_error([100.0, 200.0], [105.0, 210.0])
        5.0
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        return 0.0

    # Filter out zero labels to avoid division by zero
    valid_pairs = [(p, l) for p, l in zip(predictions, labels) if l != 0]

    if not valid_pairs:
        return 0.0

    percentage_errors = [
        abs((p - l) / l) * 100
        for p, l in valid_pairs
    ]

    return sum(percentage_errors) / len(percentage_errors)


def regression_report(predictions: List[float],
                     labels: List[float]) -> Dict[str, float]:
    """
    Generate comprehensive regression report.

    Args:
        predictions: Predicted values
        labels: Ground truth values

    Returns:
        Dictionary with all regression metrics

    Examples:
        >>> report = regression_report([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
        >>> report["mse"]
        0.01
    """
    return {
        "mse": mean_squared_error(predictions, labels),
        "rmse": root_mean_squared_error(predictions, labels),
        "mae": mean_absolute_error(predictions, labels),
        "r_squared": r_squared(predictions, labels),
        "mape": mean_absolute_percentage_error(predictions, labels)
    }
