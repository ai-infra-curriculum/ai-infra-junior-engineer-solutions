"""Tests for preprocessing utilities."""

import pytest
from ml_infra_utils.preprocessing import (
    normalize,
    standardize,
    clip_outliers,
    check_missing_values,
    validate_range,
    check_data_types,
)


class TestNormalization:
    """Tests for normalization functions."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        data = [1, 2, 3, 4, 5]
        result = normalize(data)
        assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_normalize_empty_raises(self):
        """Test normalization with empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot normalize empty data"):
            normalize([])

    def test_normalize_identical_raises(self):
        """Test normalization with identical values raises ValueError."""
        with pytest.raises(ValueError, match="All values are identical"):
            normalize([5, 5, 5, 5])

    def test_normalize_negative_values(self):
        """Test normalization with negative values."""
        data = [-2, -1, 0, 1, 2]
        result = normalize(data)
        assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_standardize_basic(self):
        """Test basic standardization."""
        data = [2, 4, 6, 8]
        result = standardize(data)
        # Mean = 5, Std = ~2.58
        # Check that mean is approximately 0 and std is approximately 1
        assert abs(sum(result) / len(result)) < 0.0001  # Mean ~0
        assert abs(
            sum((x - sum(result) / len(result)) ** 2 for x in result) / len(result)
        ) - 1.0 < 0.01  # Variance ~1

    def test_standardize_empty_raises(self):
        """Test standardization with empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot standardize empty data"):
            standardize([])

    def test_standardize_zero_std_raises(self):
        """Test standardization with zero std raises ValueError."""
        with pytest.raises(ValueError, match="Standard deviation is zero"):
            standardize([5, 5, 5, 5])

    def test_clip_outliers_basic(self):
        """Test basic outlier clipping."""
        data = list(range(1, 101))  # 1 to 100
        result = clip_outliers(data, lower_percentile=10, upper_percentile=90)
        # Values below 10th percentile (10) should be clipped to 10
        # Values above 90th percentile (90) should be clipped to 90
        assert result[0] >= 10
        assert result[-1] <= 90

    def test_clip_outliers_empty(self):
        """Test outlier clipping with empty data."""
        assert clip_outliers([]) == []


class TestValidation:
    """Tests for validation functions."""

    def test_check_missing_values_no_missing(self):
        """Test check for missing values when none present."""
        assert not check_missing_values([1, 2, 3, 4, 5])

    def test_check_missing_values_with_none(self):
        """Test check for missing values with None."""
        assert check_missing_values([1, None, 3])

    def test_check_missing_values_with_nan(self):
        """Test check for missing values with NaN."""
        assert check_missing_values([1, float("nan"), 3])

    def test_validate_range_all_valid(self):
        """Test range validation when all values are valid."""
        assert validate_range([1, 2, 3, 4, 5], 0, 10)

    def test_validate_range_out_of_bounds(self):
        """Test range validation when values are out of bounds."""
        assert not validate_range([1, 2, 15, 4, 5], 0, 10)

    def test_validate_range_edge_cases(self):
        """Test range validation with edge cases."""
        assert validate_range([0, 5, 10], 0, 10)
        assert not validate_range([-1, 5, 10], 0, 10)

    def test_check_data_types_all_match(self):
        """Test data type checking when all match."""
        assert check_data_types([1, 2, 3, 4], int)
        assert check_data_types([1.0, 2.0, 3.0], float)
        assert check_data_types(["a", "b", "c"], str)

    def test_check_data_types_mixed(self):
        """Test data type checking with mixed types."""
        assert not check_data_types([1, "2", 3], int)
        assert not check_data_types([1.0, 2, 3.0], float)
