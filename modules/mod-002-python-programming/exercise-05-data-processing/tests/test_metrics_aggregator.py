#!/usr/bin/env python3
"""
Test suite for ML Metrics Aggregator.

Tests pandas and numpy operations, data processing, and aggregations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add solutions directory to path
solutions_dir = Path(__file__).parent.parent / "solutions"
sys.path.insert(0, str(solutions_dir))

from metrics_aggregator import (
    MetricsAggregator,
    ModelPerformance,
    ComparisonResult,
    MetricsError,
    DataLoadError,
    AggregationError
)


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def sample_metrics_data():
    """Create sample metrics data."""
    np.random.seed(42)

    data = []
    for i in range(50):
        data.append({
            'experiment_id': f'exp_{i:03d}',
            'model_name': np.random.choice(['model_a', 'model_b', 'model_c']),
            'dataset_name': np.random.choice(['dataset_1', 'dataset_2']),
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
            'accuracy': np.random.uniform(0.7, 0.95),
            'f1_score': np.random.uniform(0.65, 0.90),
            'precision': np.random.uniform(0.70, 0.92),
            'recall': np.random.uniform(0.68, 0.88)
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_metrics_data):
    """Create sample CSV file."""
    filepath = tmp_path / "metrics.csv"
    sample_metrics_data.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def aggregator():
    """Create MetricsAggregator instance."""
    return MetricsAggregator()


@pytest.fixture
def loaded_aggregator(aggregator, sample_csv_file):
    """Create aggregator with loaded data."""
    aggregator.load_metrics(str(sample_csv_file))
    return aggregator


# ===========================
# Initialization Tests
# ===========================

class TestInitialization:
    """Test MetricsAggregator initialization."""

    def test_initialization(self, aggregator):
        """Test aggregator initialization."""
        assert aggregator.metrics_df is None
        assert len(aggregator.required_columns) > 0

    def test_required_columns(self, aggregator):
        """Test required columns are defined."""
        assert 'model_name' in aggregator.required_columns
        assert 'experiment_id' in aggregator.required_columns
        assert 'timestamp' in aggregator.required_columns


# ===========================
# Data Loading Tests
# ===========================

class TestDataLoading:
    """Test data loading functionality."""

    def test_load_csv(self, aggregator, sample_csv_file):
        """Test loading CSV file."""
        df = aggregator.load_metrics(str(sample_csv_file))

        assert df is not None
        assert len(df) == 50
        assert aggregator.metrics_df is not None

    def test_load_missing_file(self, aggregator):
        """Test loading nonexistent file raises error."""
        with pytest.raises(DataLoadError):
            aggregator.load_metrics('nonexistent.csv')

    def test_load_missing_columns(self, aggregator, tmp_path):
        """Test loading file with missing required columns."""
        # Create file missing required column
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        filepath = tmp_path / "invalid.csv"
        df.to_csv(filepath, index=False)

        with pytest.raises(DataLoadError, match="Missing required columns"):
            aggregator.load_metrics(str(filepath))

    def test_timestamp_parsing(self, aggregator, sample_csv_file):
        """Test timestamp column is parsed correctly."""
        aggregator.load_metrics(str(sample_csv_file))

        assert pd.api.types.is_datetime64_any_dtype(
            aggregator.metrics_df['timestamp']
        )

    def test_load_multiple_files(self, aggregator, tmp_path, sample_metrics_data):
        """Test loading multiple files."""
        # Create two files
        file1 = tmp_path / "metrics1.csv"
        file2 = tmp_path / "metrics2.csv"

        sample_metrics_data[:25].to_csv(file1, index=False)
        sample_metrics_data[25:].to_csv(file2, index=False)

        # Load both
        aggregator.load_metrics_batch([str(file1), str(file2)])

        assert len(aggregator.metrics_df) == 50


# ===========================
# Aggregation Tests
# ===========================

class TestAggregation:
    """Test aggregation functionality."""

    def test_aggregate_by_model(self, loaded_aggregator):
        """Test aggregating by model."""
        result = loaded_aggregator.aggregate_by_model(['accuracy'])

        assert 'model_name' in result.columns
        assert 'accuracy_mean' in result.columns
        assert 'accuracy_std' in result.columns
        assert len(result) > 0

    def test_aggregate_by_model_all_metrics(self, loaded_aggregator):
        """Test aggregating all metrics."""
        result = loaded_aggregator.aggregate_by_model()

        # Should have aggregations for all metrics
        assert 'accuracy_mean' in result.columns
        assert 'f1_score_mean' in result.columns

    def test_aggregate_empty_data(self, aggregator):
        """Test aggregation with no data raises error."""
        with pytest.raises(AggregationError, match="No metrics loaded"):
            aggregator.aggregate_by_model()

    def test_aggregate_by_dataset(self, loaded_aggregator):
        """Test aggregating by dataset."""
        result = loaded_aggregator.aggregate_by_dataset(['accuracy'])

        assert 'model_name' in result.columns
        assert 'dataset_name' in result.columns
        assert 'accuracy_mean' in result.columns

    def test_aggregate_by_time_period(self, loaded_aggregator):
        """Test aggregating by time period."""
        result = loaded_aggregator.aggregate_by_time_period(
            period='W',
            metrics=['accuracy']
        )

        assert 'timestamp' in result.columns
        assert 'model_name' in result.columns
        assert 'accuracy_mean' in result.columns


# ===========================
# Model Comparison Tests
# ===========================

class TestModelComparison:
    """Test model comparison functionality."""

    def test_compare_two_models(self, loaded_aggregator):
        """Test comparing two models."""
        comparisons = loaded_aggregator.compare_models(
            ['model_a', 'model_b'],
            'accuracy'
        )

        assert len(comparisons) == 1
        assert isinstance(comparisons[0], ComparisonResult)
        assert comparisons[0].model_a == 'model_a'
        assert comparisons[0].model_b == 'model_b'

    def test_compare_three_models(self, loaded_aggregator):
        """Test comparing three models."""
        comparisons = loaded_aggregator.compare_models(
            ['model_a', 'model_b', 'model_c'],
            'accuracy'
        )

        # Should have 3 comparisons (A-B, A-C, B-C)
        assert len(comparisons) == 3

    def test_comparison_calculations(self, loaded_aggregator):
        """Test comparison calculations are correct."""
        comparisons = loaded_aggregator.compare_models(
            ['model_a', 'model_b'],
            'accuracy'
        )

        comp = comparisons[0]
        expected_diff = comp.a_mean - comp.b_mean

        assert comp.difference == pytest.approx(expected_diff, rel=1e-6)

    def test_compare_nonexistent_metric(self, loaded_aggregator):
        """Test comparing nonexistent metric raises error."""
        with pytest.raises(AggregationError, match="not found"):
            loaded_aggregator.compare_models(
                ['model_a', 'model_b'],
                'nonexistent_metric'
            )


# ===========================
# Best Model Tests
# ===========================

class TestBestModel:
    """Test best model identification."""

    def test_get_best_model(self, loaded_aggregator):
        """Test getting best model."""
        best = loaded_aggregator.get_best_model('accuracy')

        assert best in ['model_a', 'model_b', 'model_c']

    def test_get_best_model_minimize(self, loaded_aggregator):
        """Test getting best model with minimize=True."""
        # Add a loss metric
        loaded_aggregator.metrics_df['loss'] = 1 - loaded_aggregator.metrics_df['accuracy']

        best = loaded_aggregator.get_best_model('loss', minimize=True)

        assert best in ['model_a', 'model_b', 'model_c']

    def test_get_best_model_nonexistent_metric(self, loaded_aggregator):
        """Test getting best model for nonexistent metric."""
        with pytest.raises(AggregationError):
            loaded_aggregator.get_best_model('nonexistent')


# ===========================
# Model Performance Tests
# ===========================

class TestModelPerformance:
    """Test model performance analysis."""

    def test_get_model_performance(self, loaded_aggregator):
        """Test getting model performance."""
        perf = loaded_aggregator.get_model_performance('model_a', 'accuracy')

        assert isinstance(perf, ModelPerformance)
        assert perf.model_name == 'model_a'
        assert 0 <= perf.metric_mean <= 1
        assert perf.metric_std >= 0
        assert perf.n_experiments > 0

    def test_get_performance_nonexistent_model(self, loaded_aggregator):
        """Test getting performance for nonexistent model."""
        with pytest.raises(AggregationError, match="not found"):
            loaded_aggregator.get_model_performance('nonexistent_model', 'accuracy')

    def test_performance_statistics(self, loaded_aggregator):
        """Test performance statistics are correct."""
        perf = loaded_aggregator.get_model_performance('model_a', 'accuracy')

        # Min should be <= mean <= max
        assert perf.metric_min <= perf.metric_mean <= perf.metric_max


# ===========================
# Filtering Tests
# ===========================

class TestFiltering:
    """Test data filtering functionality."""

    def test_filter_by_date_range(self, loaded_aggregator):
        """Test filtering by date range."""
        filtered = loaded_aggregator.filter_by_date_range(
            '2024-01-10',
            '2024-01-20'
        )

        assert filtered.metrics_df is not None
        assert len(filtered.metrics_df) < len(loaded_aggregator.metrics_df)

    def test_filter_preserves_data(self, loaded_aggregator):
        """Test filtering creates new aggregator."""
        original_len = len(loaded_aggregator.metrics_df)

        filtered = loaded_aggregator.filter_by_date_range(
            '2024-01-10',
            '2024-01-20'
        )

        # Original should be unchanged
        assert len(loaded_aggregator.metrics_df) == original_len


# ===========================
# Trend Analysis Tests
# ===========================

class TestTrendAnalysis:
    """Test trend analysis functionality."""

    def test_calculate_trend(self, loaded_aggregator):
        """Test calculating trend."""
        trend = loaded_aggregator.calculate_trend(
            'model_a',
            'accuracy',
            window=3
        )

        assert isinstance(trend, pd.Series)
        assert len(trend) > 0

    def test_trend_nonexistent_model(self, loaded_aggregator):
        """Test trend for nonexistent model."""
        with pytest.raises(AggregationError, match="not found"):
            loaded_aggregator.calculate_trend(
                'nonexistent_model',
                'accuracy'
            )


# ===========================
# Export Tests
# ===========================

class TestExport:
    """Test export functionality."""

    def test_export_csv(self, loaded_aggregator, tmp_path):
        """Test exporting to CSV."""
        output_path = tmp_path / "report.csv"
        loaded_aggregator.export_report(str(output_path), format='csv')

        assert output_path.exists()

        # Verify can be read back
        df = pd.read_csv(output_path)
        assert len(df) > 0

    def test_export_json(self, loaded_aggregator, tmp_path):
        """Test exporting to JSON."""
        output_path = tmp_path / "report.json"
        loaded_aggregator.export_report(str(output_path), format='json')

        assert output_path.exists()

    def test_export_without_data(self, aggregator, tmp_path):
        """Test export without loaded data."""
        with pytest.raises(AggregationError):
            aggregator.export_report(str(tmp_path / "report.csv"))


# ===========================
# Summary Statistics Tests
# ===========================

class TestSummaryStatistics:
    """Test summary statistics functionality."""

    def test_get_summary_statistics(self, loaded_aggregator):
        """Test getting summary statistics."""
        summaries = loaded_aggregator.get_summary_statistics()

        assert 'overall' in summaries
        assert 'by_model' in summaries
        assert 'correlation' in summaries

        # Check overall is a describe() output
        assert isinstance(summaries['overall'], pd.DataFrame)

    def test_get_metrics_info(self, loaded_aggregator):
        """Test getting metrics info."""
        info = loaded_aggregator.get_metrics_info()

        assert 'total_experiments' in info
        assert 'unique_models' in info
        assert 'metric_columns' in info
        assert info['total_experiments'] > 0


# ===========================
# Data Optimization Tests
# ===========================

class TestDataOptimization:
    """Test data optimization functionality."""

    def test_dtype_optimization(self, aggregator, sample_csv_file):
        """Test dtype optimization."""
        # Load data
        aggregator.load_metrics(str(sample_csv_file))

        # Check that model_name is categorical
        assert aggregator.metrics_df['model_name'].dtype.name == 'category'

    def test_memory_efficiency(self, aggregator, sample_csv_file):
        """Test memory usage is tracked."""
        aggregator.load_metrics(str(sample_csv_file))

        info = aggregator.get_metrics_info()
        assert 'memory_usage_mb' in info
        assert info['memory_usage_mb'] > 0


# ===========================
# NumPy Integration Tests
# ===========================

class TestNumPyIntegration:
    """Test NumPy integration."""

    def test_numpy_calculations(self, loaded_aggregator):
        """Test NumPy calculations in aggregations."""
        result = loaded_aggregator.aggregate_by_model(['accuracy'])

        # Mean should be same as pandas mean
        model_a_data = loaded_aggregator.metrics_df[
            loaded_aggregator.metrics_df['model_name'] == 'model_a'
        ]['accuracy']

        expected_mean = np.mean(model_a_data)
        actual_mean = result[
            result['model_name'] == 'model_a'
        ]['accuracy_mean'].values[0]

        assert actual_mean == pytest.approx(expected_mean, rel=1e-6)


# ===========================
# Edge Cases Tests
# ===========================

class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dataframe_operations(self, aggregator):
        """Test operations on empty DataFrame."""
        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=[
            'experiment_id', 'model_name', 'timestamp', 'accuracy'
        ])
        aggregator.metrics_df = empty_df

        with pytest.raises(AggregationError):
            aggregator.aggregate_by_model()

    def test_single_model(self, aggregator, tmp_path):
        """Test with single model."""
        # Create data with single model
        data = pd.DataFrame({
            'experiment_id': ['exp_1', 'exp_2'],
            'model_name': ['model_a', 'model_a'],
            'timestamp': pd.date_range('2024-01-01', periods=2),
            'accuracy': [0.9, 0.91]
        })

        filepath = tmp_path / "single_model.csv"
        data.to_csv(filepath, index=False)

        aggregator.load_metrics(str(filepath))
        result = aggregator.aggregate_by_model(['accuracy'])

        assert len(result) == 1
        assert result['model_name'].iloc[0] == 'model_a'

    def test_perfect_scores(self, aggregator, tmp_path):
        """Test with perfect scores."""
        data = pd.DataFrame({
            'experiment_id': ['exp_1', 'exp_2'],
            'model_name': ['model_a', 'model_a'],
            'timestamp': pd.date_range('2024-01-01', periods=2),
            'accuracy': [1.0, 1.0]
        })

        filepath = tmp_path / "perfect.csv"
        data.to_csv(filepath, index=False)

        aggregator.load_metrics(str(filepath))
        perf = aggregator.get_model_performance('model_a', 'accuracy')

        assert perf.metric_mean == 1.0
        assert perf.metric_std == 0.0


# ===========================
# Integration Tests
# ===========================

@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, tmp_path):
        """Test complete workflow from load to export."""
        # Create data
        data = pd.DataFrame({
            'experiment_id': [f'exp_{i}' for i in range(20)],
            'model_name': ['model_a'] * 10 + ['model_b'] * 10,
            'dataset_name': ['dataset_1'] * 20,
            'timestamp': pd.date_range('2024-01-01', periods=20),
            'accuracy': np.random.uniform(0.8, 0.95, 20),
            'f1_score': np.random.uniform(0.75, 0.90, 20)
        })

        filepath = tmp_path / "workflow_data.csv"
        data.to_csv(filepath, index=False)

        # Initialize and load
        aggregator = MetricsAggregator()
        aggregator.load_metrics(str(filepath))

        # Aggregate
        model_stats = aggregator.aggregate_by_model()
        assert len(model_stats) == 2

        # Compare
        comparisons = aggregator.compare_models(['model_a', 'model_b'], 'accuracy')
        assert len(comparisons) == 1

        # Get best
        best = aggregator.get_best_model('accuracy')
        assert best in ['model_a', 'model_b']

        # Export
        output_path = tmp_path / "workflow_report.csv"
        aggregator.export_report(str(output_path))
        assert output_path.exists()


# ===========================
# Run Tests
# ===========================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=metrics_aggregator", "--cov-report=term-missing"])
