"""Data preprocessing utilities for ML workflows."""

from typing import List, Optional


def remove_outliers(data: List[float], threshold: float = 1.5) -> List[float]:
    """
    Remove outliers using IQR method.

    Args:
        data: List of numeric values
        threshold: IQR multiplier for outlier detection

    Returns:
        List with outliers removed
    """
    if len(data) < 4:
        return data

    sorted_data = sorted(data)
    q1_idx = len(sorted_data) // 4
    q3_idx = 3 * len(sorted_data) // 4

    q1 = sorted_data[q1_idx]
    q3 = sorted_data[q3_idx]
    iqr = q3 - q1

    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr

    return [x for x in data if lower <= x <= upper]


def fill_missing(data: List[Optional[float]], strategy: str = "mean") -> List[float]:
    """
    Fill missing values in data.

    Args:
        data: List with optional None values
        strategy: Fill strategy ("mean" or "median")

    Returns:
        List with missing values filled

    Raises:
        ValueError: If strategy is unknown
    """
    valid = [x for x in data if x is not None]

    if not valid:
        return [0.0] * len(data)

    if strategy == "mean":
        fill_value = sum(valid) / len(valid)
    elif strategy == "median":
        sorted_valid = sorted(valid)
        mid = len(sorted_valid) // 2
        fill_value = sorted_valid[mid]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [x if x is not None else fill_value for x in data]


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize value to [0, 1] range.

    Args:
        value: Value to normalize
        min_val: Minimum value in range
        max_val: Maximum value in range

    Returns:
        Normalized value between 0 and 1
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)
