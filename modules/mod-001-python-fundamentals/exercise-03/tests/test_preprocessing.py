"""
Tests for ml_utils.preprocessing module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_utils import preprocessing


class TestNormalization:
    """Test normalization functions."""

    def test_normalize_minmax_default_range(self):
        """Test min-max normalization to 0-1."""
        data = [0.0, 5.0, 10.0]
        normalized = preprocessing.normalize_minmax(data)
        assert normalized == [0.0, 0.5, 1.0]

    def test_normalize_minmax_custom_range(self):
        """Test min-max normalization to custom range."""
        data = [0.0, 5.0, 10.0]
        normalized = preprocessing.normalize_minmax(data, feature_range=(-1.0, 1.0))
        assert normalized == [-1.0, 0.0, 1.0]

    def test_normalize_minmax_empty(self):
        """Test min-max normalization with empty list."""
        assert preprocessing.normalize_minmax([]) == []

    def test_normalize_minmax_same_values(self):
        """Test min-max normalization with all same values."""
        data = [5.0, 5.0, 5.0]
        normalized = preprocessing.normalize_minmax(data)
        assert all(x == 0.0 for x in normalized)

    def test_normalize_zscore(self):
        """Test z-score normalization."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = preprocessing.normalize_zscore(data)
        # Check mean ≈ 0
        assert abs(sum(normalized) / len(normalized)) < 0.01
        # Check std ≈ 1
        mean = sum(normalized) / len(normalized)
        variance = sum((x - mean) ** 2 for x in normalized) / len(normalized)
        std = variance ** 0.5
        assert abs(std - 1.0) < 0.01

    def test_normalize_zscore_empty(self):
        """Test z-score normalization with empty list."""
        assert preprocessing.normalize_zscore([]) == []


class TestOutlierRemoval:
    """Test outlier removal."""

    def test_remove_outliers_iqr(self):
        """Test outlier removal with IQR method."""
        data = [1, 2, 3, 4, 5, 100]
        cleaned = preprocessing.remove_outliers(data, method="iqr")
        assert 100 not in cleaned
        assert len(cleaned) < len(data)

    def test_remove_outliers_zscore(self):
        """Test outlier removal with z-score method."""
        data = [1, 2, 3, 4, 5, 100]
        cleaned = preprocessing.remove_outliers(data, method="zscore", threshold=2.0)
        assert 100 not in cleaned

    def test_remove_outliers_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            preprocessing.remove_outliers([1, 2, 3], method="unknown")


class TestMissingValues:
    """Test missing value handling."""

    def test_fill_missing_mean(self):
        """Test filling missing values with mean."""
        data = [1.0, 2.0, None, 4.0]
        filled = preprocessing.fill_missing_values(data, "mean")
        assert None not in filled
        # Mean of [1, 2, 4] = 2.333...
        assert abs(filled[2] - 2.333) < 0.01

    def test_fill_missing_median(self):
        """Test filling missing values with median."""
        data = [1.0, 2.0, None, 4.0, 5.0]
        filled = preprocessing.fill_missing_values(data, "median")
        assert None not in filled

    def test_fill_missing_forward(self):
        """Test forward fill."""
        data = [1.0, None, None, 4.0]
        filled = preprocessing.fill_missing_values(data, "forward")
        assert filled == [1.0, 1.0, 1.0, 4.0]

    def test_fill_missing_backward(self):
        """Test backward fill."""
        data = [None, None, 3.0, 4.0]
        filled = preprocessing.fill_missing_values(data, "backward")
        assert filled == [3.0, 3.0, 3.0, 4.0]

    def test_fill_missing_empty(self):
        """Test filling missing values with empty list."""
        assert preprocessing.fill_missing_values([]) == []

    def test_fill_missing_all_none(self):
        """Test filling when all values are None."""
        data = [None, None, None]
        filled = preprocessing.fill_missing_values(data, "mean")
        assert all(x == 0.0 for x in filled)


class TestEncoding:
    """Test encoding functions."""

    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        labels = [0, 1, 2]
        encoded = preprocessing.one_hot_encode(labels, num_classes=3)
        assert encoded == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_label_encode(self):
        """Test label encoding."""
        labels = ["cat", "dog", "cat", "bird"]
        encoded, mapping = preprocessing.label_encode(labels)
        assert len(encoded) == len(labels)
        assert len(mapping) == 3  # bird, cat, dog
        assert encoded[0] == encoded[2]  # Both "cat"


class TestDataSplitting:
    """Test data splitting functions."""

    def test_train_test_split(self):
        """Test train-test split."""
        data = list(range(100))
        train, test = preprocessing.train_test_split(data, test_size=0.2, shuffle=False)
        assert len(train) == 80
        assert len(test) == 20
        assert set(train + test) == set(data)

    def test_train_test_split_with_seed(self):
        """Test train-test split with seed for reproducibility."""
        data = list(range(20))
        train1, test1 = preprocessing.train_test_split(data, test_size=0.2,
                                                       shuffle=True, random_seed=42)
        train2, test2 = preprocessing.train_test_split(data, test_size=0.2,
                                                       shuffle=True, random_seed=42)
        assert train1 == train2
        assert test1 == test2

    def test_train_test_split_invalid_size(self):
        """Test that invalid test_size raises error."""
        with pytest.raises(ValueError):
            preprocessing.train_test_split([1, 2, 3], test_size=1.5)

    def test_stratified_split(self):
        """Test stratified split."""
        data = [("a", 0), ("b", 1), ("c", 0), ("d", 1), ("e", 0), ("f", 1)]
        train, test = preprocessing.stratified_split(data, test_size=0.33, random_seed=42)

        # Check sizes
        assert len(train) + len(test) == len(data)

        # Check that both classes present
        train_labels = [label for _, label in train]
        test_labels = [label for _, label in test]
        assert 0 in train_labels or 0 in test_labels
        assert 1 in train_labels or 1 in test_labels


class TestOtherPreprocessing:
    """Test other preprocessing functions."""

    def test_clip_values(self):
        """Test value clipping."""
        data = [1, 5, 10, 15, 20]
        clipped = preprocessing.clip_values(data, min_value=5, max_value=15)
        assert clipped == [5, 5, 10, 15, 15]

    def test_batch_normalize(self):
        """Test batch normalization."""
        batches = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
        normalized = preprocessing.batch_normalize(batches)
        assert len(normalized) == 2
        # Each batch should be normalized independently
        assert len(normalized[0]) == 3
        assert len(normalized[1]) == 3

    def test_shuffle_data(self):
        """Test data shuffling."""
        data = list(range(10))
        shuffled = preprocessing.shuffle_data(data, random_seed=42)
        assert set(shuffled) == set(data)
        assert len(shuffled) == len(data)

    def test_shuffle_data_reproducible(self):
        """Test that shuffling with seed is reproducible."""
        data = list(range(10))
        shuffled1 = preprocessing.shuffle_data(data, random_seed=42)
        shuffled2 = preprocessing.shuffle_data(data, random_seed=42)
        assert shuffled1 == shuffled2
