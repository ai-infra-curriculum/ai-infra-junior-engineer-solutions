"""Tests for data processing modules."""

import pytest
import pandas as pd
import numpy as np
from src.data.ingestion import DataIngester
from src.data.validation import DataValidator, ValidationIssue
from src.data.preprocessing import DataPreprocessor


class TestDataIngester:
    """Test data ingestion."""

    def test_load_from_csv(self, sample_csv_file):
        """Test loading data from CSV."""
        ingester = DataIngester()
        df = ingester.load_from_csv(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        assert len(df.columns) == 21

    def test_load_from_csv_missing_file(self):
        """Test loading from non-existent file."""
        ingester = DataIngester()

        with pytest.raises(FileNotFoundError):
            ingester.load_from_csv("nonexistent.csv")

    def test_load_auto_detect_csv(self, sample_csv_file):
        """Test auto-detection of CSV files."""
        ingester = DataIngester()
        df = ingester.load(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestDataValidator:
    """Test data validation."""

    def test_check_schema_valid(self, sample_data):
        """Test schema validation with valid data."""
        validator = DataValidator(required_columns=["feature_0", "target"])
        issues = validator.check_schema(sample_data)

        assert len(issues) == 0

    def test_check_schema_missing_columns(self, sample_data):
        """Test schema validation with missing columns."""
        validator = DataValidator(required_columns=["feature_0", "missing_column"])
        issues = validator.check_schema(sample_data)

        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].category == "schema"

    def test_check_missing_values_none(self, sample_data):
        """Test missing value check with clean data."""
        validator = DataValidator()
        issues = validator.check_missing_values(sample_data)

        assert len(issues) == 0

    def test_check_missing_values_present(self, sample_data):
        """Test missing value check with missing data."""
        # Add missing values
        df = sample_data.copy()
        df.loc[0:100, "feature_0"] = np.nan

        validator = DataValidator(max_missing_ratio=0.05)
        issues = validator.check_missing_values(df)

        assert len(issues) > 0
        assert any(i.category == "missing_values" for i in issues)

    def test_check_duplicates_none(self, sample_data):
        """Test duplicate check with unique data."""
        validator = DataValidator()
        issues = validator.check_duplicates(sample_data)

        # Might have warnings but no errors
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_check_duplicates_present(self, sample_data):
        """Test duplicate check with duplicate data."""
        # Add duplicates
        df = pd.concat([sample_data, sample_data.head(200)], ignore_index=True)

        validator = DataValidator(max_duplicate_ratio=0.1)
        issues = validator.check_duplicates(df)

        assert len(issues) > 0
        assert any(i.category == "duplicates" for i in issues)

    def test_validate_all_checks(self, sample_data):
        """Test complete validation."""
        validator = DataValidator()
        is_valid, issues = validator.validate(sample_data)

        assert is_valid is True or is_valid is False  # Should return boolean
        assert isinstance(issues, list)

    def test_get_validation_report_no_issues(self):
        """Test validation report with no issues."""
        validator = DataValidator()
        report = validator.get_validation_report([])

        assert "All validation checks passed" in report

    def test_get_validation_report_with_issues(self):
        """Test validation report with issues."""
        issues = [
            ValidationIssue("error", "schema", "Test error"),
            ValidationIssue("warning", "missing_values", "Test warning")
        ]

        validator = DataValidator()
        report = validator.get_validation_report(issues)

        assert "ERRORS (1)" in report
        assert "WARNINGS (1)" in report


class TestDataPreprocessor:
    """Test data preprocessing."""

    def test_handle_missing_values_drop(self, sample_data):
        """Test dropping missing values."""
        # Add missing values
        df = sample_data.copy()
        df.loc[0:10, "feature_0"] = np.nan

        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(df, strategy="drop")

        assert len(result) < len(df)
        assert result.isnull().sum().sum() == 0

    def test_handle_missing_values_mean(self, sample_data):
        """Test filling missing values with mean."""
        # Add missing values
        df = sample_data.copy()
        df.loc[0:10, "feature_0"] = np.nan

        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(df, strategy="mean")

        assert len(result) == len(df)
        assert result["feature_0"].isnull().sum() == 0

    def test_remove_duplicates(self, sample_data):
        """Test removing duplicate rows."""
        # Add duplicates
        df = pd.concat([sample_data, sample_data.head(100)], ignore_index=True)

        preprocessor = DataPreprocessor()
        result = preprocessor.remove_duplicates(df)

        assert len(result) == len(sample_data)

    def test_encode_categorical(self):
        """Test categorical encoding."""
        df = pd.DataFrame({
            "cat1": ["A", "B", "C", "A", "B"],
            "cat2": ["X", "Y", "X", "Y", "X"],
            "num": [1, 2, 3, 4, 5]
        })

        preprocessor = DataPreprocessor()
        result = preprocessor.encode_categorical(df, columns=["cat1", "cat2"])

        assert result["cat1"].dtype in [np.int32, np.int64]
        assert result["cat2"].dtype in [np.int32, np.int64]

    def test_scale_features(self, sample_features):
        """Test feature scaling."""
        preprocessor = DataPreprocessor()
        result = preprocessor.scale_features(sample_features)

        # Check scaling (mean ~0, std ~1)
        assert abs(result.mean().mean()) < 0.1
        assert abs(result.std().mean() - 1.0) < 0.1

    def test_process_complete_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        result = preprocessor.process(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
