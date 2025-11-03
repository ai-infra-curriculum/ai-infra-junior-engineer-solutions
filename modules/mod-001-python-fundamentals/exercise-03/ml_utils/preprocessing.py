"""
Data Preprocessing Module

Provides utilities for data cleaning, normalization, and preprocessing.
"""

from typing import List, Tuple, Optional, Dict, Any
import statistics
import random


def normalize_minmax(data: List[float],
                    feature_range: Tuple[float, float] = (0.0, 1.0)
                    ) -> List[float]:
    """
    Normalize data to specified range using min-max scaling.

    Args:
        data: List of values to normalize
        feature_range: Target range as (min, max) tuple

    Returns:
        Normalized values

    Examples:
        >>> normalize_minmax([1.0, 2.0, 3.0, 4.0, 5.0])
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    if not data:
        return []

    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        # All values are the same
        return [feature_range[0]] * len(data)

    range_min, range_max = feature_range
    scale = (range_max - range_min) / (max_val - min_val)

    return [range_min + (x - min_val) * scale for x in data]


def normalize_zscore(data: List[float]) -> List[float]:
    """
    Normalize data using z-score standardization.

    z = (x - mean) / std

    Args:
        data: List of values to normalize

    Returns:
        Standardized values (mean=0, std=1)

    Examples:
        >>> values = normalize_zscore([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> abs(sum(values)) < 0.01  # Mean ≈ 0
        True
    """
    if not data:
        return []

    if len(data) == 1:
        return [0.0]

    mean = statistics.mean(data)
    std_dev = statistics.stdev(data)

    if std_dev == 0:
        return [0.0] * len(data)

    return [(x - mean) / std_dev for x in data]


def remove_outliers(data: List[float],
                   method: str = "iqr",
                   threshold: float = 1.5) -> List[float]:
    """
    Remove outliers from data.

    Args:
        data: Input data
        method: Method to use ("iqr" or "zscore")
        threshold: Threshold for outlier detection
                  - For IQR: multiplier for IQR (default: 1.5)
                  - For z-score: max absolute z-score (default: 1.5)

    Returns:
        Data with outliers removed

    Examples:
        >>> data = [1, 2, 3, 4, 5, 100]
        >>> cleaned = remove_outliers(data, method="iqr")
        >>> 100 in cleaned
        False
    """
    if not data or len(data) < 4:
        return data

    if method == "iqr":
        # Interquartile range method
        sorted_data = sorted(data)
        q1_idx = len(sorted_data) // 4
        q3_idx = 3 * len(sorted_data) // 4

        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        return [x for x in data if lower_bound <= x <= upper_bound]

    elif method == "zscore":
        # Z-score method
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)

        if std_dev == 0:
            return data

        z_scores = [abs((x - mean) / std_dev) for x in data]
        return [x for x, z in zip(data, z_scores) if z <= threshold]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")


def fill_missing_values(data: List[Optional[float]],
                       strategy: str = "mean") -> List[float]:
    """
    Fill missing values in data.

    Args:
        data: Input data with possible None values
        strategy: Strategy to use ("mean", "median", "mode", "forward", "backward")

    Returns:
        Data with missing values filled

    Examples:
        >>> fill_missing_values([1.0, None, 3.0, None, 5.0], "mean")
        [1.0, 3.0, 3.0, 3.0, 5.0]
    """
    if not data:
        return []

    # Get non-missing values
    valid_values = [x for x in data if x is not None]

    if not valid_values:
        return [0.0] * len(data)

    if strategy == "mean":
        fill_value = statistics.mean(valid_values)
        return [x if x is not None else fill_value for x in data]

    elif strategy == "median":
        fill_value = statistics.median(valid_values)
        return [x if x is not None else fill_value for x in data]

    elif strategy == "mode":
        try:
            fill_value = statistics.mode(valid_values)
        except statistics.StatisticsError:
            # No unique mode, use first value
            fill_value = valid_values[0]
        return [x if x is not None else fill_value for x in data]

    elif strategy == "forward":
        # Forward fill: use last valid value
        result = []
        last_valid = valid_values[0]
        for x in data:
            if x is not None:
                last_valid = x
                result.append(x)
            else:
                result.append(last_valid)
        return result

    elif strategy == "backward":
        # Backward fill: use next valid value
        result = []
        data_reversed = list(reversed(data))
        valid_reversed = [x for x in data_reversed if x is not None]
        last_valid = valid_reversed[0]

        for x in data_reversed:
            if x is not None:
                last_valid = x
                result.append(x)
            else:
                result.append(last_valid)

        return list(reversed(result))

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Use 'mean', 'median', 'mode', 'forward', or 'backward'"
        )


def one_hot_encode(labels: List[int], num_classes: int) -> List[List[int]]:
    """
    Convert class labels to one-hot encoding.

    Args:
        labels: List of class labels (integers)
        num_classes: Total number of classes

    Returns:
        List of one-hot encoded vectors

    Examples:
        >>> one_hot_encode([0, 1, 2, 1], 3)
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
    """
    encoded = []

    for label in labels:
        one_hot = [0] * num_classes
        if 0 <= label < num_classes:
            one_hot[label] = 1
        encoded.append(one_hot)

    return encoded


def label_encode(labels: List[str]) -> Tuple[List[int], Dict[str, int]]:
    """
    Convert string labels to integer encoding.

    Args:
        labels: List of string labels

    Returns:
        Tuple of (encoded_labels, label_to_id_mapping)

    Examples:
        >>> encoded, mapping = label_encode(["cat", "dog", "cat", "bird"])
        >>> encoded
        [0, 1, 0, 2]
        >>> mapping["cat"]
        0
    """
    # Create label to ID mapping
    unique_labels = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

    # Encode labels
    encoded = [label_to_id[label] for label in labels]

    return encoded, label_to_id


def train_test_split(data: List[Any],
                    test_size: float = 0.2,
                    shuffle: bool = True,
                    random_seed: Optional[int] = None
                    ) -> Tuple[List[Any], List[Any]]:
    """
    Split data into training and test sets.

    Args:
        data: Input data to split
        test_size: Proportion of data for test set (0 < test_size < 1)
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)

    Examples:
        >>> train, test = train_test_split(list(range(10)), test_size=0.2,
        ...                                shuffle=False)
        >>> len(train), len(test)
        (8, 2)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    data_copy = data.copy()

    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(data_copy)

    split_idx = int(len(data_copy) * (1 - test_size))
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]

    return train_data, test_data


def stratified_split(data: List[Tuple[Any, int]],
                    test_size: float = 0.2,
                    random_seed: Optional[int] = None
                    ) -> Tuple[List[Tuple[Any, int]], List[Tuple[Any, int]]]:
    """
    Split data into train/test maintaining class distribution.

    Args:
        data: List of (sample, label) tuples
        test_size: Proportion of data for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)

    Examples:
        >>> data = [("a", 0), ("b", 1), ("c", 0), ("d", 1)]
        >>> train, test = stratified_split(data, test_size=0.5, random_seed=42)
        >>> len(train), len(test)
        (2, 2)
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Group data by label
    label_to_samples = {}
    for sample, label in data:
        if label not in label_to_samples:
            label_to_samples[label] = []
        label_to_samples[label].append((sample, label))

    train_data = []
    test_data = []

    # Split each class proportionally
    for label, samples in label_to_samples.items():
        samples_copy = samples.copy()
        random.shuffle(samples_copy)

        split_idx = int(len(samples_copy) * (1 - test_size))
        train_data.extend(samples_copy[:split_idx])
        test_data.extend(samples_copy[split_idx:])

    # Shuffle combined data
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data


def batch_normalize(batches: List[List[float]]) -> List[List[float]]:
    """
    Normalize each batch independently (batch normalization).

    Args:
        batches: List of batches (each batch is a list of values)

    Returns:
        Normalized batches

    Examples:
        >>> batches = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
        >>> normalized = batch_normalize(batches)
        >>> len(normalized)
        2
    """
    normalized_batches = []

    for batch in batches:
        if not batch:
            normalized_batches.append([])
            continue

        if len(batch) == 1:
            normalized_batches.append([0.0])
            continue

        mean = statistics.mean(batch)
        std_dev = statistics.stdev(batch)

        if std_dev == 0:
            normalized_batches.append([0.0] * len(batch))
        else:
            normalized = [(x - mean) / std_dev for x in batch]
            normalized_batches.append(normalized)

    return normalized_batches


def clip_values(data: List[float],
               min_value: Optional[float] = None,
               max_value: Optional[float] = None) -> List[float]:
    """
    Clip values to specified range.

    Args:
        data: Input data
        min_value: Minimum value (None = no minimum)
        max_value: Maximum value (None = no maximum)

    Returns:
        Clipped values

    Examples:
        >>> clip_values([1, 5, 10, 15, 20], min_value=5, max_value=15)
        [5, 5, 10, 15, 15]
    """
    result = []

    for x in data:
        value = x
        if min_value is not None:
            value = max(value, min_value)
        if max_value is not None:
            value = min(value, max_value)
        result.append(value)

    return result


def apply_log_transform(data: List[float], offset: float = 1.0) -> List[float]:
    """
    Apply logarithmic transformation to data.

    Useful for handling skewed distributions.

    Args:
        data: Input data (must be positive)
        offset: Offset to add before log (to handle zeros)

    Returns:
        Log-transformed values

    Examples:
        >>> import math
        >>> result = apply_log_transform([1.0, 10.0, 100.0], offset=0)
        >>> abs(result[1] - math.log(10.0)) < 0.01
        True
    """
    import math

    return [math.log(x + offset) for x in data]


def shuffle_data(data: List[Any],
                random_seed: Optional[int] = None) -> List[Any]:
    """
    Shuffle data randomly.

    Args:
        data: Input data to shuffle
        random_seed: Random seed for reproducibility

    Returns:
        Shuffled data

    Examples:
        >>> data = [1, 2, 3, 4, 5]
        >>> shuffled = shuffle_data(data, random_seed=42)
        >>> shuffled != data or len(shuffled) == len(data)
        True
    """
    if random_seed is not None:
        random.seed(random_seed)

    data_copy = data.copy()
    random.shuffle(data_copy)
    return data_copy
