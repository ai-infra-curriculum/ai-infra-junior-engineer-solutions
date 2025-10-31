"""
Unit Tests for Data Preprocessing

Tests for data cleaning, missing values, outliers, normalization, and encoding.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    clean_data,
    handle_missing_values,
    remove_outliers,
    normalize_data,
    encode_categorical,
    split_features_target,
    create_train_test_split,
)


@pytest.mark.unit
class TestDataCleaning:
    """Tests for data cleaning functions."""

    def test_clean_data_removes_duplicates(self, sample_data_with_duplicates):
        """Test that duplicates are removed."""
        cleaned = clean_data(sample_data_with_duplicates, remove_duplicates=True)
        assert cleaned.duplicated().sum() == 0

    def test_clean_data_preserves_unique_rows(self, sample_data):
        """Test that unique rows are preserved."""
        cleaned = clean_data(sample_data)
        assert len(cleaned) == len(sample_data)

    def test_clean_data_returns_dataframe(self, sample_data):
        """Test that clean_data returns DataFrame."""
        result = clean_data(sample_data)
        assert isinstance(result, pd.DataFrame)


@pytest.mark.unit
class TestMissingValues:
    """Tests for missing value handling."""

    @pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
    def test_handle_missing_values_strategies(self, sample_data_with_missing, strategy):
        """Test different imputation strategies."""
        filled = handle_missing_values(sample_data_with_missing, strategy=strategy)
        assert filled.isnull().sum().sum() == 0

    def test_handle_missing_values_no_missing(self, sample_data):
        """Test handling data without missing values."""
        filled = handle_missing_values(sample_data)
        assert filled.equals(sample_data)

    def test_handle_missing_values_forward_fill(self):
        """Test forward fill strategy."""
        df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})
        filled = handle_missing_values(df, strategy="forward_fill")
        expected = pd.Series([1.0, 1.0, 3.0, 3.0, 5.0], name='a')
        pd.testing.assert_series_equal(filled['a'], expected)

    def test_handle_missing_values_invalid_strategy(self, sample_data_with_missing):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_missing_values(sample_data_with_missing, strategy="invalid")


@pytest.mark.unit
class TestOutlierRemoval:
    """Tests for outlier detection and removal."""

    def test_remove_outliers_iqr_method(self, sample_data_with_outliers):
        """Test IQR method for outlier removal."""
        cleaned = remove_outliers(sample_data_with_outliers, method="iqr")
        assert len(cleaned) < len(sample_data_with_outliers)
        assert cleaned['feature1'].max() < 1000

    def test_remove_outliers_zscore_method(self, sample_data_with_outliers):
        """Test z-score method for outlier removal."""
        cleaned = remove_outliers(sample_data_with_outliers, method="zscore", threshold=2)
        assert len(cleaned) < len(sample_data_with_outliers)

    def test_remove_outliers_no_outliers(self, sample_data):
        """Test with data containing no outliers."""
        cleaned = remove_outliers(sample_data, method="iqr")
        assert len(cleaned) == len(sample_data)

    def test_remove_outliers_invalid_method(self, sample_data):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            remove_outliers(sample_data, method="invalid")


@pytest.mark.unit
class TestNormalization:
    """Tests for data normalization."""

    def test_normalize_standard_scaling(self, sample_data):
        """Test standard scaling normalization."""
        normalized, scaler = normalize_data(
            sample_data,
            columns=['feature1', 'feature2'],
            method="standard"
        )

        assert pytest.approx(normalized['feature1'].mean(), abs=1e-10) == 0
        assert pytest.approx(normalized['feature1'].std(), abs=1e-1) == 1

    def test_normalize_minmax_scaling(self, sample_data):
        """Test min-max scaling normalization."""
        normalized, scaler = normalize_data(
            sample_data,
            columns=['feature1'],
            method="minmax"
        )

        assert normalized['feature1'].min() == pytest.approx(0)
        assert normalized['feature1'].max() == pytest.approx(1)

    def test_normalize_returns_scaler(self, sample_data):
        """Test that normalize_data returns scaler object."""
        _, scaler = normalize_data(sample_data, method="standard")
        assert scaler is not None
        assert hasattr(scaler, 'transform')

    def test_normalize_invalid_method(self, sample_data):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            normalize_data(sample_data, method="invalid")


@pytest.mark.unit
class TestCategoricalEncoding:
    """Tests for categorical encoding."""

    def test_encode_categorical_onehot(self):
        """Test one-hot encoding."""
        df = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})
        encoded, encoders = encode_categorical(df, method="onehot")

        # Check that original column is removed
        assert 'category' not in encoded.columns

        # Check that dummy columns are created
        dummy_cols = [col for col in encoded.columns if col.startswith('category_')]
        assert len(dummy_cols) > 0

    def test_encode_categorical_label(self):
        """Test label encoding."""
        df = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})
        encoded, encoders = encode_categorical(df, method="label")

        # Check that column is encoded as integers
        assert encoded['category'].dtype in [np.int32, np.int64]
        assert set(encoded['category'].unique()) == {0, 1, 2}

    def test_encode_categorical_returns_encoders(self):
        """Test that encoders are returned."""
        df = pd.DataFrame({'category': ['A', 'B', 'A']})
        _, encoders = encode_categorical(df, method="label")

        assert 'category' in encoders
        assert encoders['category'] is not None


@pytest.mark.unit
class TestDataSplitting:
    """Tests for data splitting functions."""

    def test_split_features_target(self, sample_data):
        """Test splitting features and target."""
        X, y = split_features_target(sample_data, 'target')

        assert 'target' not in X.columns
        assert len(X) == len(y)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_split_features_target_invalid_column(self, sample_data):
        """Test error when target column doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            split_features_target(sample_data, 'nonexistent')

    def test_create_train_test_split(self, sample_data):
        """Test train/test split."""
        X, y = split_features_target(sample_data, 'target')
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        # Check sizes
        total = len(X_train) + len(X_test)
        assert total == len(X)

        # Check approximate test size
        test_ratio = len(X_test) / total
        assert 0.35 < test_ratio < 0.45

    def test_create_train_test_split_reproducible(self, sample_data):
        """Test that split is reproducible with same random_state."""
        X, y = split_features_target(sample_data, 'target')

        X_train1, _, _, _ = create_train_test_split(X, y, random_state=42)
        X_train2, _, _, _ = create_train_test_split(X, y, random_state=42)

        pd.testing.assert_frame_equal(X_train1, X_train2)


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_clean_data_empty_dataframe(self):
        """Test cleaning empty dataframe."""
        df = pd.DataFrame()
        result = clean_data(df)
        assert len(result) == 0

    def test_handle_missing_all_nan_column(self):
        """Test handling column with all NaN."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [np.nan, np.nan, np.nan]})
        result = handle_missing_values(df, strategy="mean")
        assert result['b'].isnull().all()

    def test_normalize_single_value_column(self):
        """Test normalizing column with single unique value."""
        df = pd.DataFrame({'a': [5, 5, 5, 5]})

        # Should handle gracefully (might result in NaN or 0)
        normalized, _ = normalize_data(df, method="standard")
        assert len(normalized) == len(df)
