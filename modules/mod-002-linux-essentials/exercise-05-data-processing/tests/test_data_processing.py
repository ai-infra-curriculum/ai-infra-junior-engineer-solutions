#!/usr/bin/env python3
"""
Comprehensive Test Suite for Exercise 05: Data Processing

Tests for all data processing modules including CSV processing, data cleaning,
transformation, statistical analysis, and JSON/YAML processing.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import json
import yaml
from pathlib import Path

# Add solutions directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solutions'))

from csv_processor import CSVProcessor, CSVMerger, DataQualityReport
from data_cleaning import DataCleaner, DataValidator, CleaningStats
from data_transformation import DataTransformer, FeatureEngineer
from statistical_analysis import StatisticalAnalyzer, DataInsights, StatisticalSummary
from json_yaml_processor import JSONProcessor, YAMLProcessor, DataConverter


# ============= CSV Processing Tests =============

class TestCSVProcessor:
    """Test CSV processing functionality."""

    def setup_method(self):
        """Create sample CSV data for testing."""
        self.sample_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Alice'],
            'age': [25, 30, 35, 25],
            'city': ['NYC', 'LA', 'Chicago', 'NYC'],
            'salary': [100000, 80000, 120000, 100000]
        })

    def test_load_csv(self, tmp_path):
        """Test loading CSV file."""
        csv_file = tmp_path / "test.csv"
        self.sample_df.to_csv(csv_file, index=False)

        processor = CSVProcessor(str(csv_file))
        df = processor.load()

        assert len(df) == 4
        assert list(df.columns) == ['name', 'age', 'city', 'salary']

    def test_get_quality_report(self, tmp_path):
        """Test data quality report generation."""
        csv_file = tmp_path / "test.csv"
        self.sample_df.to_csv(csv_file, index=False)

        processor = CSVProcessor(str(csv_file))
        processor.load()
        report = processor.get_quality_report()

        assert report.total_rows == 4
        assert report.total_columns == 4
        assert report.duplicate_rows == 1  # Alice appears twice

    def test_clean_csv(self, tmp_path):
        """Test CSV cleaning."""
        csv_file = tmp_path / "test.csv"
        self.sample_df.to_csv(csv_file, index=False)

        processor = CSVProcessor(str(csv_file))
        processor.load()
        processor.clean(drop_duplicates=True)

        assert len(processor.df) == 3  # One duplicate removed

    def test_filter_csv(self, tmp_path):
        """Test CSV filtering."""
        csv_file = tmp_path / "test.csv"
        self.sample_df.to_csv(csv_file, index=False)

        processor = CSVProcessor(str(csv_file))
        processor.load()
        processor.filter(query_string='age > 25')

        assert len(processor.df) == 2  # Only Bob and Charlie

    def test_transform_csv(self, tmp_path):
        """Test CSV transformation."""
        csv_file = tmp_path / "test.csv"
        self.sample_df.to_csv(csv_file, index=False)

        processor = CSVProcessor(str(csv_file))
        processor.load()
        processor.transform(
            column_transformations={'salary': lambda x: x * 1.1}
        )

        assert processor.df['salary'].iloc[0] == 110000

    def test_aggregate_csv(self, tmp_path):
        """Test CSV aggregation."""
        csv_file = tmp_path / "test.csv"
        self.sample_df.to_csv(csv_file, index=False)

        processor = CSVProcessor(str(csv_file))
        processor.load()
        agg = processor.aggregate(
            group_by=['city'],
            aggregations={'salary': 'mean'}
        )

        assert len(agg) == 3  # 3 cities

    def test_save_csv(self, tmp_path):
        """Test saving CSV."""
        csv_file = tmp_path / "test.csv"
        output_file = tmp_path / "output.csv"

        self.sample_df.to_csv(csv_file, index=False)

        processor = CSVProcessor(str(csv_file))
        processor.load()
        processor.save(str(output_file))

        assert output_file.exists()

    def test_merge_files(self, tmp_path):
        """Test merging multiple CSV files."""
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"

        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        df2 = pd.DataFrame({'id': [2, 3], 'value': [30, 40]})

        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)

        merged = CSVMerger.merge_files([str(file1), str(file2)], how='outer', on='id')

        assert len(merged) == 3  # IDs 1, 2, 3


# ============= Data Cleaning Tests =============

class TestDataCleaning:
    """Test data cleaning functionality."""

    def setup_method(self):
        """Create sample data for testing."""
        self.df = pd.DataFrame({
            'name': ['Alice', 'Bob', None, 'David', 'Eve'],
            'age': [25, 30, 35, 40, 200],  # 200 is outlier
            'salary': [50000, 60000, None, 70000, 80000]
        })

    def test_handle_missing_values_drop(self):
        """Test handling missing values by dropping."""
        cleaner = DataCleaner(self.df)
        result = cleaner.handle_missing_values(strategy='drop').get_result()

        assert len(result) == 3  # Rows with None dropped

    def test_handle_missing_values_mean(self):
        """Test handling missing values with mean."""
        cleaner = DataCleaner(self.df)
        result = cleaner.handle_missing_values(strategy='mean', columns=['salary']).get_result()

        assert not result['salary'].isnull().any()

    def test_remove_duplicates(self):
        """Test removing duplicates."""
        df_with_dupes = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Alice'],
            'age': [25, 30, 25]
        })

        cleaner = DataCleaner(df_with_dupes)
        result = cleaner.remove_duplicates().get_result()

        assert len(result) == 2

    def test_remove_outliers_iqr(self):
        """Test removing outliers using IQR method."""
        cleaner = DataCleaner(self.df)
        result = cleaner.remove_outliers(['age'], method='iqr', threshold=1.5).get_result()

        assert 200 not in result['age'].values

    def test_normalize_text(self):
        """Test text normalization."""
        df = pd.DataFrame({'name': ['  Alice  ', 'BOB', 'Charlie']})
        cleaner = DataCleaner(df)
        result = cleaner.normalize_text(['name'], lowercase=True, remove_whitespace=True).get_result()

        assert result['name'].iloc[0] == 'alice'
        assert result['name'].iloc[1] == 'bob'

    def test_cap_values(self):
        """Test capping values."""
        cleaner = DataCleaner(self.df)
        result = cleaner.cap_values('age', lower=0, upper=100).get_result()

        assert result['age'].max() <= 100

    def test_get_cleaning_stats(self):
        """Test getting cleaning statistics."""
        cleaner = DataCleaner(self.df)
        cleaner.remove_duplicates()
        stats = cleaner.get_cleaning_stats()

        assert stats.rows_before == 5
        assert isinstance(stats, CleaningStats)

    def test_validate_not_null(self):
        """Test null validation."""
        valid, errors = DataValidator.validate_not_null(self.df, ['name', 'age'])

        assert not valid
        assert 'name' in errors

    def test_validate_unique(self):
        """Test uniqueness validation."""
        df = pd.DataFrame({'email': ['a@test.com', 'b@test.com', 'a@test.com']})
        valid, errors = DataValidator.validate_unique(df, ['email'])

        assert not valid
        assert 'email' in errors

    def test_validate_range(self):
        """Test range validation."""
        valid, count = DataValidator.validate_range(self.df, 'age', min_value=0, max_value=100)

        assert not valid
        assert count == 1  # One value (200) out of range


# ============= Data Transformation Tests =============

class TestDataTransformation:
    """Test data transformation functionality."""

    def setup_method(self):
        """Create sample data for testing."""
        self.df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'category': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
            'value': np.arange(10),
            'quantity': np.arange(1, 11)
        })

    def test_pivot_data(self):
        """Test data pivoting."""
        transformer = DataTransformer(self.df)
        pivoted = transformer.pivot_data(
            index='date',
            columns='category',
            values='value',
            aggfunc='mean'
        )

        assert 'A' in pivoted.columns
        assert 'B' in pivoted.columns

    def test_melt_data(self):
        """Test data melting."""
        df_wide = pd.DataFrame({
            'id': [1, 2],
            'metric1': [10, 20],
            'metric2': [30, 40]
        })

        transformer = DataTransformer(df_wide)
        melted = transformer.melt_data(id_vars=['id'])

        assert len(melted) == 4  # 2 rows × 2 metrics

    def test_bin_numeric(self):
        """Test numeric binning."""
        transformer = DataTransformer(self.df)
        result = transformer.bin_numeric(
            'value',
            bins=[0, 3, 6, 10],
            labels=['low', 'medium', 'high']
        )

        assert 'value_binned' in result.columns

    def test_encode_categorical_label(self):
        """Test label encoding."""
        transformer = DataTransformer(self.df)
        result = transformer.encode_categorical(['category'], method='label')

        assert 'category_encoded' in result.columns

    def test_encode_categorical_onehot(self):
        """Test one-hot encoding."""
        transformer = DataTransformer(self.df)
        result = transformer.encode_categorical(['category'], method='onehot')

        assert 'category_A' in result.columns or any('category' in col for col in result.columns)

    def test_create_lag_features(self):
        """Test lag feature creation."""
        transformer = DataTransformer(self.df)
        result = transformer.create_lag_features('value', lags=[1, 2])

        assert 'value_lag_1' in result.columns
        assert 'value_lag_2' in result.columns

    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        transformer = DataTransformer(self.df)
        result = transformer.create_rolling_features(
            'value',
            windows=[3],
            functions=['mean', 'std']
        )

        assert 'value_rolling_mean_3' in result.columns
        assert 'value_rolling_std_3' in result.columns

    def test_create_date_features(self):
        """Test date feature extraction."""
        transformer = DataTransformer(self.df)
        result = transformer.create_date_features('date')

        assert 'date_year' in result.columns
        assert 'date_month' in result.columns
        assert 'date_day' in result.columns

    def test_interaction_features(self):
        """Test interaction feature creation."""
        df = pd.DataFrame({'price': [10, 20, 30], 'quantity': [1, 2, 3]})
        result = FeatureEngineer.create_interaction_features(
            df,
            column_pairs=[('price', 'quantity')],
            operations=['multiply']
        )

        assert 'price_x_quantity' in result.columns
        assert result['price_x_quantity'].iloc[0] == 10

    def test_polynomial_features(self):
        """Test polynomial feature creation."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        result = FeatureEngineer.create_polynomial_features(df, columns=['value'], degree=2)

        assert 'value_pow_2' in result.columns

    def test_ratio_features(self):
        """Test ratio feature creation."""
        df = pd.DataFrame({'profit': [10, 20], 'revenue': [100, 200]})
        result = FeatureEngineer.create_ratio_features(
            df,
            numerator_cols=['profit'],
            denominator_cols=['revenue']
        )

        assert 'profit_per_revenue' in result.columns


# ============= Statistical Analysis Tests =============

class TestStatisticalAnalysis:
    """Test statistical analysis functionality."""

    def setup_method(self):
        """Create sample data for testing."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'value': np.random.normal(100, 15, 100),
            'group': np.random.choice(['A', 'B'], 100),
            'price': np.random.lognormal(4, 0.5, 100)
        })

    def test_get_summary_stats(self):
        """Test summary statistics calculation."""
        analyzer = StatisticalAnalyzer(self.df)
        summary = analyzer.get_summary_stats('value')

        assert isinstance(summary, StatisticalSummary)
        assert summary.count == 100
        assert summary.mean > 0

    def test_describe_all(self):
        """Test describe all columns."""
        analyzer = StatisticalAnalyzer(self.df)
        desc = analyzer.describe_all()

        assert 'value' in desc.columns
        assert 'mean' in desc.index

    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        analyzer = StatisticalAnalyzer(self.df)
        corr = analyzer.get_correlation_matrix(threshold=0.5)

        assert isinstance(corr.correlation_matrix, pd.DataFrame)
        assert isinstance(corr.strong_correlations, list)

    def test_detect_outliers_iqr(self):
        """Test outlier detection with IQR."""
        analyzer = StatisticalAnalyzer(self.df)
        outliers, stats = analyzer.detect_outliers('value', method='iqr')

        assert isinstance(outliers, pd.Series)
        assert 'n_outliers' in stats

    def test_detect_outliers_zscore(self):
        """Test outlier detection with Z-score."""
        analyzer = StatisticalAnalyzer(self.df)
        outliers, stats = analyzer.detect_outliers('value', method='zscore', threshold=3)

        assert 'n_outliers' in stats

    def test_test_normality(self):
        """Test normality testing."""
        analyzer = StatisticalAnalyzer(self.df)
        stat, p_value, is_normal = analyzer.test_normality('value', method='shapiro')

        assert isinstance(p_value, float)
        assert isinstance(is_normal, bool)

    def test_compare_groups(self):
        """Test group comparison."""
        analyzer = StatisticalAnalyzer(self.df)
        result = analyzer.compare_groups('value', 'group', test='ttest')

        assert 'p_value' in result
        assert 'significant' in result
        assert result['n_groups'] == 2

    def test_calculate_percentiles(self):
        """Test percentile calculation."""
        analyzer = StatisticalAnalyzer(self.df)
        percentiles = analyzer.calculate_percentiles('value', [25, 50, 75])

        assert 25 in percentiles
        assert 50 in percentiles
        assert 75 in percentiles

    def test_get_value_counts(self):
        """Test value counts."""
        analyzer = StatisticalAnalyzer(self.df)
        counts = analyzer.get_value_counts('group')

        assert len(counts) == 2  # Groups A and B

    def test_identify_trends(self):
        """Test trend identification."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'sales': np.arange(100) + np.random.randn(100) * 5
        })

        insights = DataInsights.identify_trends(df, 'date', 'sales')

        assert 'trend' in insights
        assert insights['trend'] in ['increasing', 'decreasing', 'stable']

    def test_find_top_contributors(self):
        """Test finding top contributors."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C', 'D', 'E'] * 20,
            'revenue': np.random.randint(100, 1000, 100)
        })

        top = DataInsights.find_top_contributors(df, 'product', 'revenue', n=3)

        assert len(top) == 3
        assert 'percentage' in top.columns


# ============= JSON/YAML Processing Tests =============

class TestJSONYAMLProcessing:
    """Test JSON and YAML processing."""

    def test_json_read_write(self, tmp_path):
        """Test JSON reading and writing."""
        json_file = tmp_path / "test.json"
        data = {'name': 'Alice', 'age': 30}

        JSONProcessor.write(str(json_file), data)
        loaded = JSONProcessor.read(str(json_file))

        assert loaded == data

    def test_jsonl_read_write(self, tmp_path):
        """Test JSON Lines reading and writing."""
        jsonl_file = tmp_path / "test.jsonl"
        data = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]

        JSONProcessor.write_jsonl(str(jsonl_file), data)
        loaded = JSONProcessor.read_jsonl(str(jsonl_file))

        assert len(loaded) == 2
        assert loaded[0]['id'] == 1

    def test_json_validate_structure(self):
        """Test JSON structure validation."""
        data = {'name': 'Alice', 'age': 30}
        valid, errors = JSONProcessor.validate_structure(
            data,
            required_keys=['name', 'age']
        )

        assert valid
        assert len(errors) == 0

    def test_json_merge(self):
        """Test JSON merging."""
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'b': 3, 'c': 4}

        merged = JSONProcessor.merge(dict1, dict2)

        assert merged['a'] == 1
        assert merged['b'] == 3  # Later value wins
        assert merged['c'] == 4

    def test_json_flatten(self):
        """Test JSON flattening."""
        data = {'a': {'b': {'c': 1}}}
        flat = JSONProcessor.flatten(data)

        assert 'a.b.c' in flat
        assert flat['a.b.c'] == 1

    def test_yaml_read_write(self, tmp_path):
        """Test YAML reading and writing."""
        yaml_file = tmp_path / "test.yaml"
        data = {'name': 'Alice', 'age': 30}

        YAMLProcessor.write(str(yaml_file), data)
        loaded = YAMLProcessor.read(str(yaml_file))

        assert loaded == data

    def test_yaml_multi_document(self, tmp_path):
        """Test multi-document YAML."""
        yaml_file = tmp_path / "multi.yaml"
        docs = [{'doc': 1}, {'doc': 2}]

        YAMLProcessor.write_all(str(yaml_file), docs)
        loaded = YAMLProcessor.read_all(str(yaml_file))

        assert len(loaded) == 2

    def test_json_to_yaml_conversion(self, tmp_path):
        """Test JSON to YAML conversion."""
        json_file = tmp_path / "test.json"
        yaml_file = tmp_path / "test.yaml"

        data = {'name': 'Alice', 'age': 30}
        JSONProcessor.write(str(json_file), data)

        DataConverter.json_to_yaml(str(json_file), str(yaml_file))

        loaded = YAMLProcessor.read(str(yaml_file))
        assert loaded == data

    def test_yaml_to_json_conversion(self, tmp_path):
        """Test YAML to JSON conversion."""
        yaml_file = tmp_path / "test.yaml"
        json_file = tmp_path / "test.json"

        data = {'name': 'Alice', 'age': 30}
        YAMLProcessor.write(str(yaml_file), data)

        DataConverter.yaml_to_json(str(yaml_file), str(json_file))

        loaded = JSONProcessor.read(str(json_file))
        assert loaded == data


# ============= Integration Tests =============

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_data_pipeline(self, tmp_path):
        """Test complete data processing pipeline."""
        # Create test data
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Alice', None],
            'age': [25, 30, 35, 25, 28],
            'salary': [50000, 60000, 70000, 50000, None]
        })

        csv_file = tmp_path / "input.csv"
        df.to_csv(csv_file, index=False)

        # Load and process
        processor = CSVProcessor(str(csv_file))
        processor.load()

        # Clean
        processor.clean(drop_duplicates=True)

        # Transform
        processor.transform(
            column_transformations={'salary': lambda x: x * 1.1}
        )

        # Save
        output_file = tmp_path / "output.csv"
        processor.save(str(output_file))

        # Verify
        result_df = pd.read_csv(output_file)
        assert len(result_df) < len(df)  # Duplicates removed

    def test_statistical_analysis_pipeline(self):
        """Test statistical analysis pipeline."""
        # Create data
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(100, 15, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })

        # Analyze
        analyzer = StatisticalAnalyzer(df)

        # Get statistics
        summary = analyzer.get_summary_stats('value')
        assert summary.count == 100

        # Detect outliers
        outliers, stats = analyzer.detect_outliers('value', method='iqr')
        assert 'n_outliers' in stats

        # Group comparison
        df['group'] = np.random.choice(['X', 'Y'], 100)
        comparison = analyzer.compare_groups('value', 'group')
        assert 'p_value' in comparison


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
