"""Data validation utilities for ML pipelines."""

from typing import List, Any


def check_missing_values(data: List[Any]) -> bool:
    """
    Check if data contains missing values (None or NaN-like values).

    Args:
        data: List of values to check

    Returns:
        True if missing values found, False otherwise

    Examples:
        >>> check_missing_values([1, 2, 3])
        False
        >>> check_missing_values([1, None, 3])
        True
    """
    return any(x is None or (isinstance(x, float) and x != x) for x in data)


def validate_range(data: List[float], min_val: float, max_val: float) -> bool:
    """
    Validate that all values are within specified range.

    Args:
        data: List of numerical values
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        True if all values within range, False otherwise

    Examples:
        >>> validate_range([1, 2, 3], 0, 5)
        True
        >>> validate_range([1, 2, 10], 0, 5)
        False
    """
    return all(min_val <= x <= max_val for x in data)


def check_data_types(data: List[Any], expected_type: type) -> bool:
    """
    Check if all elements match expected type.

    Args:
        data: List of values to check
        expected_type: Expected Python type

    Returns:
        True if all elements match expected type

    Examples:
        >>> check_data_types([1, 2, 3], int)
        True
        >>> check_data_types([1, "2", 3], int)
        False
    """
    return all(isinstance(x, expected_type) for x in data)
