"""ML metrics calculation utilities."""

from typing import List


def calculate_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: List of predicted labels
        labels: List of true labels

    Returns:
        Accuracy score between 0 and 1

    Raises:
        ValueError: If predictions and labels have different lengths
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        return 0.0

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)


def calculate_precision(predictions: List[int], labels: List[int], positive_class: int = 1) -> float:
    """
    Calculate precision for binary classification.

    Args:
        predictions: List of predicted labels
        labels: List of true labels
        positive_class: Label considered as positive class

    Returns:
        Precision score between 0 and 1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    true_positives = sum(1 for p, l in zip(predictions, labels)
                        if p == positive_class and l == positive_class)
    predicted_positives = sum(1 for p in predictions if p == positive_class)

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def calculate_recall(predictions: List[int], labels: List[int], positive_class: int = 1) -> float:
    """
    Calculate recall for binary classification.

    Args:
        predictions: List of predicted labels
        labels: List of true labels
        positive_class: Label considered as positive class

    Returns:
        Recall score between 0 and 1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    true_positives = sum(1 for p, l in zip(predictions, labels)
                        if p == positive_class and l == positive_class)
    actual_positives = sum(1 for l in labels if l == positive_class)

    if actual_positives == 0:
        return 0.0

    return true_positives / actual_positives


def calculate_f1_score(predictions: List[int], labels: List[int], positive_class: int = 1) -> float:
    """
    Calculate F1 score for binary classification.

    Args:
        predictions: List of predicted labels
        labels: List of true labels
        positive_class: Label considered as positive class

    Returns:
        F1 score between 0 and 1
    """
    precision = calculate_precision(predictions, labels, positive_class)
    recall = calculate_recall(predictions, labels, positive_class)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
