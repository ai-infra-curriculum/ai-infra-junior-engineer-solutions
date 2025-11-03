"""Classification metrics for ML model evaluation."""

from typing import List


def accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    Calculate accuracy score.

    Args:
        predictions: Model predictions
        labels: True labels

    Returns:
        Accuracy as float between 0 and 1

    Raises:
        ValueError: If predictions and labels have different lengths

    Examples:
        >>> accuracy([1, 0, 1, 1], [1, 0, 1, 0])
        0.75
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)


def precision(predictions: List[int], labels: List[int], positive_class: int = 1) -> float:
    """
    Calculate precision score.

    Precision = True Positives / (True Positives + False Positives)

    Args:
        predictions: Model predictions
        labels: True labels
        positive_class: Which class to consider positive (default: 1)

    Returns:
        Precision as float between 0 and 1

    Raises:
        ValueError: If predictions and labels have different lengths

    Examples:
        >>> precision([1, 0, 1, 1], [1, 0, 1, 0], positive_class=1)
        0.6666666666666666
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    true_positives = sum(
        p == positive_class and l == positive_class
        for p, l in zip(predictions, labels)
    )
    predicted_positives = sum(p == positive_class for p in predictions)

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def recall(predictions: List[int], labels: List[int], positive_class: int = 1) -> float:
    """
    Calculate recall score.

    Recall = True Positives / (True Positives + False Negatives)

    Args:
        predictions: Model predictions
        labels: True labels
        positive_class: Which class to consider positive (default: 1)

    Returns:
        Recall as float between 0 and 1

    Raises:
        ValueError: If predictions and labels have different lengths

    Examples:
        >>> recall([1, 0, 1, 1], [1, 0, 1, 0], positive_class=1)
        1.0
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    true_positives = sum(
        p == positive_class and l == positive_class
        for p, l in zip(predictions, labels)
    )
    actual_positives = sum(l == positive_class for l in labels)

    if actual_positives == 0:
        return 0.0

    return true_positives / actual_positives


def f1_score(predictions: List[int], labels: List[int], positive_class: int = 1) -> float:
    """
    Calculate F1 score.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        predictions: Model predictions
        labels: True labels
        positive_class: Which class to consider positive (default: 1)

    Returns:
        F1 score as float between 0 and 1

    Examples:
        >>> f1_score([1, 0, 1, 1], [1, 0, 1, 0], positive_class=1)
        0.8
    """
    prec = precision(predictions, labels, positive_class)
    rec = recall(predictions, labels, positive_class)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)
