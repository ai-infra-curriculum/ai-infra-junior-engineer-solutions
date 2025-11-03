"""Data normalization utilities for ML pipelines."""

from typing import List
import statistics


def normalize(data: List[float]) -> List[float]:
    """
    Normalize data to [0, 1] range using min-max scaling.

    Args:
        data: List of numerical values to normalize

    Returns:
        List of normalized values in range [0, 1]

    Raises:
        ValueError: If data is empty or all values are the same

    Examples:
        >>> normalize([1, 2, 3, 4, 5])
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    if not data:
        raise ValueError("Cannot normalize empty data")

    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        raise ValueError("All values are identical, cannot normalize")

    range_val = max_val - min_val
    return [(x - min_val) / range_val for x in data]


def standardize(data: List[float]) -> List[float]:
    """
    Standardize data to mean=0, std=1 (z-score normalization).

    Args:
        data: List of numerical values to standardize

    Returns:
        List of standardized values

    Raises:
        ValueError: If data is empty or has zero standard deviation
    """
    if not data:
        raise ValueError("Cannot standardize empty data")

    mean = statistics.mean(data)
    stdev = statistics.stdev(data)

    if stdev == 0:
        raise ValueError("Standard deviation is zero, cannot standardize")

    return [(x - mean) / stdev for x in data]


def clip_outliers(
    data: List[float], lower_percentile: float = 5, upper_percentile: float = 95
) -> List[float]:
    """
    Clip outliers based on percentiles.

    Args:
        data: List of numerical values
        lower_percentile: Lower percentile threshold (0-100)
        upper_percentile: Upper percentile threshold (0-100)

    Returns:
        List with outliers clipped to percentile values
    """
    if not data:
        return []

    sorted_data = sorted(data)
    n = len(sorted_data)

    lower_idx = int(n * lower_percentile / 100)
    upper_idx = int(n * upper_percentile / 100)

    lower_bound = sorted_data[lower_idx]
    upper_bound = sorted_data[upper_idx]

    return [max(lower_bound, min(x, upper_bound)) for x in data]
