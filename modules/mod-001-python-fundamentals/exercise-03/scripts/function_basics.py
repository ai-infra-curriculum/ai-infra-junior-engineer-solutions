#!/usr/bin/env python3
"""
Function Basics with Type Hints

Demonstrates clean, well-documented functions with comprehensive type hints
for ML infrastructure code.
"""

from typing import List, Dict, Tuple, Optional, Union
import random


def calculate_accuracy(predictions: List[int],
                      labels: List[int]) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: Model predictions as class indices
        labels: Ground truth labels as class indices

    Returns:
        Accuracy as a float between 0 and 1

    Raises:
        ValueError: If predictions and labels have different lengths

    Examples:
        >>> calculate_accuracy([1, 0, 1, 1], [1, 0, 0, 1])
        0.75
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Predictions and labels must have same length: "
            f"{len(predictions)} != {len(labels)}"
        )

    if len(predictions) == 0:
        return 0.0

    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(predictions)

    return accuracy


def normalize_features(data: List[float],
                      method: str = "minmax",
                      feature_range: Tuple[float, float] = (0.0, 1.0)
                      ) -> List[float]:
    """
    Normalize feature values.

    Args:
        data: List of feature values
        method: Normalization method ("minmax" or "zscore")
        feature_range: Target range for minmax normalization (default: 0-1)

    Returns:
        Normalized feature values

    Raises:
        ValueError: If method is not supported or data is invalid

    Examples:
        >>> normalize_features([1.0, 2.0, 3.0, 4.0, 5.0], method="minmax")
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    if not data:
        return []

    if method == "minmax":
        min_val = min(data)
        max_val = max(data)

        if min_val == max_val:
            # All values are the same - return target range minimum
            return [feature_range[0]] * len(data)

        # Scale to feature_range
        range_min, range_max = feature_range
        scale = (range_max - range_min) / (max_val - min_val)

        normalized = [
            range_min + (x - min_val) * scale
            for x in data
        ]

        return normalized

    elif method == "zscore":
        # Z-score normalization: (x - mean) / std
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = variance ** 0.5

        if std_dev == 0:
            # All values are same - return zeros
            return [0.0] * len(data)

        normalized = [(x - mean) / std_dev for x in data]
        return normalized

    else:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Supported methods: 'minmax', 'zscore'"
        )


def split_data(data: List,
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               shuffle: bool = True,
               random_seed: Optional[int] = None
               ) -> Tuple[List, List, List]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: Input data to split
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        shuffle: Whether to shuffle data before splitting (default: True)
        random_seed: Random seed for reproducibility (default: None)

    Returns:
        Tuple of (train_data, val_data, test_data)

    Raises:
        ValueError: If ratios don't sum to <= 1.0 or are negative

    Examples:
        >>> data = list(range(10))
        >>> train, val, test = split_data(data, 0.6, 0.2, shuffle=False)
        >>> len(train), len(val), len(test)
        (6, 2, 2)
    """
    # Validate ratios
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("Ratios must be non-negative")

    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be <= 1.0, got "
            f"{train_ratio + val_ratio:.2f}"
        )

    # Copy data to avoid modifying original
    data_copy = data.copy()

    # Shuffle if requested
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(data_copy)

    # Calculate split indices
    n = len(data_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split data
    train_data = data_copy[:train_end]
    val_data = data_copy[train_end:val_end]
    test_data = data_copy[val_end:]

    return train_data, val_data, test_data


def calculate_metrics(predictions: List[Union[int, float]],
                     labels: List[Union[int, float]],
                     metric_type: str = "classification"
                     ) -> Dict[str, float]:
    """
    Calculate evaluation metrics for ML models.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        metric_type: Type of metrics ("classification" or "regression")

    Returns:
        Dictionary containing relevant metrics

    Raises:
        ValueError: If metric_type is unknown or inputs are invalid

    Examples:
        >>> preds = [1, 0, 1, 1]
        >>> labels = [1, 0, 0, 1]
        >>> metrics = calculate_metrics(preds, labels, "classification")
        >>> metrics["accuracy"]
        0.75
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        raise ValueError("Cannot calculate metrics on empty data")

    if metric_type == "classification":
        # Classification metrics
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(predictions)

        # Simple precision/recall (assuming binary classification)
        true_positives = sum(
            1 for p, l in zip(predictions, labels)
            if p == 1 and l == 1
        )
        false_positives = sum(
            1 for p, l in zip(predictions, labels)
            if p == 1 and l == 0
        )
        false_negatives = sum(
            1 for p, l in zip(predictions, labels)
            if p == 0 and l == 1
        )

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    elif metric_type == "regression":
        # Regression metrics
        errors = [p - l for p, l in zip(predictions, labels)]

        # Mean Squared Error
        mse = sum(e ** 2 for e in errors) / len(errors)

        # Root Mean Squared Error
        rmse = mse ** 0.5

        # Mean Absolute Error
        mae = sum(abs(e) for e in errors) / len(errors)

        # R-squared
        mean_label = sum(labels) / len(labels)
        ss_tot = sum((l - mean_label) ** 2 for l in labels)
        ss_res = sum(e ** 2 for e in errors)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared
        }

    else:
        raise ValueError(
            f"Unknown metric_type: {metric_type}. "
            f"Supported types: 'classification', 'regression'"
        )


def batch_data(data: List,
              batch_size: int,
              drop_last: bool = False
              ) -> List[List]:
    """
    Split data into batches.

    Args:
        data: Input data to batch
        batch_size: Size of each batch
        drop_last: Whether to drop last incomplete batch (default: False)

    Returns:
        List of batches

    Raises:
        ValueError: If batch_size is invalid

    Examples:
        >>> data = list(range(10))
        >>> batches = batch_data(data, batch_size=3)
        >>> len(batches)
        4
        >>> batches[-1]
        [9]
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batches = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        # Only add batch if not dropping incomplete batches or batch is full
        if not drop_last or len(batch) == batch_size:
            batches.append(batch)

    return batches


def main():
    """Demonstrate function basics with type hints."""
    print("=" * 70)
    print("Function Basics with Type Hints")
    print("=" * 70)
    print()

    # Test 1: Accuracy calculation
    print("Test 1: Accuracy Calculation")
    print("-" * 70)
    preds = [1, 0, 1, 1, 0, 1, 0, 0]
    labels = [1, 0, 1, 0, 0, 1, 0, 1]
    acc = calculate_accuracy(preds, labels)
    print(f"Predictions: {preds}")
    print(f"Labels:      {labels}")
    print(f"Accuracy:    {acc:.2%} ({acc:.4f})")
    print()

    # Test 2: Feature normalization - MinMax
    print("Test 2: Feature Normalization (MinMax)")
    print("-" * 70)
    features = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized_minmax = normalize_features(features, method="minmax")
    print(f"Original:    {features}")
    print(f"Normalized:  {[f'{x:.2f}' for x in normalized_minmax]}")
    print()

    # Test 3: Feature normalization - Z-score
    print("Test 3: Feature Normalization (Z-score)")
    print("-" * 70)
    normalized_zscore = normalize_features(features, method="zscore")
    print(f"Original:    {features}")
    print(f"Z-score:     {[f'{x:.2f}' for x in normalized_zscore]}")
    print()

    # Test 4: Custom feature range
    print("Test 4: Custom Feature Range (-1, 1)")
    print("-" * 70)
    normalized_custom = normalize_features(features, method="minmax",
                                          feature_range=(-1.0, 1.0))
    print(f"Original:    {features}")
    print(f"Normalized:  {[f'{x:.2f}' for x in normalized_custom]}")
    print()

    # Test 5: Data splitting
    print("Test 5: Data Splitting")
    print("-" * 70)
    dataset = list(range(100))
    train, val, test = split_data(dataset, train_ratio=0.7, val_ratio=0.15,
                                   shuffle=True, random_seed=42)
    print(f"Total samples: {len(dataset)}")
    print(f"Train size:    {len(train)} ({len(train)/len(dataset):.1%})")
    print(f"Val size:      {len(val)} ({len(val)/len(dataset):.1%})")
    print(f"Test size:     {len(test)} ({len(test)/len(dataset):.1%})")
    print(f"Train samples: {train[:5]} ...")
    print()

    # Test 6: Classification metrics
    print("Test 6: Classification Metrics")
    print("-" * 70)
    preds = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    labels = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    metrics = calculate_metrics(preds, labels, metric_type="classification")
    print(f"Predictions: {preds}")
    print(f"Labels:      {labels}")
    print(f"\nMetrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name:12s}: {metric_value:.4f}")
    print()

    # Test 7: Regression metrics
    print("Test 7: Regression Metrics")
    print("-" * 70)
    preds_reg = [2.5, 3.1, 4.2, 5.8, 6.3]
    labels_reg = [2.0, 3.0, 4.0, 6.0, 6.5]
    metrics_reg = calculate_metrics(preds_reg, labels_reg,
                                    metric_type="regression")
    print(f"Predictions: {preds_reg}")
    print(f"Labels:      {labels_reg}")
    print(f"\nMetrics:")
    for metric_name, metric_value in metrics_reg.items():
        print(f"  {metric_name:12s}: {metric_value:.4f}")
    print()

    # Test 8: Batching
    print("Test 8: Data Batching")
    print("-" * 70)
    data = list(range(1, 21))
    batches = batch_data(data, batch_size=5)
    print(f"Data: {data}")
    print(f"Batch size: 5")
    print(f"Number of batches: {len(batches)}")
    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}: {batch}")
    print()

    # Test 9: Batching with drop_last
    print("Test 9: Data Batching (drop_last=True)")
    print("-" * 70)
    data_partial = list(range(1, 18))
    batches_dropped = batch_data(data_partial, batch_size=5, drop_last=True)
    print(f"Data: {data_partial}")
    print(f"Batch size: 5")
    print(f"Number of batches: {len(batches_dropped)}")
    for i, batch in enumerate(batches_dropped):
        print(f"  Batch {i+1}: {batch}")
    print()

    # Test 10: Error handling
    print("Test 10: Error Handling")
    print("-" * 70)
    try:
        # Try invalid ratios
        split_data(dataset, train_ratio=0.8, val_ratio=0.3)
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    try:
        # Try mismatched lengths
        calculate_accuracy([1, 0, 1], [1, 0])
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    try:
        # Try unknown normalization method
        normalize_features([1, 2, 3], method="unknown")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    print()
    print("=" * 70)
    print("✓ All function demonstrations completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
