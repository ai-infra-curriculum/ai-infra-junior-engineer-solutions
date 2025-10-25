#!/usr/bin/env python3
"""
ML Metrics Aggregation Tool

A comprehensive system for aggregating, analyzing, and reporting ML experiment metrics.

Features:
- Load metrics from multiple sources
- Aggregate by model, dataset, time period
- Statistical analysis and comparisons
- Best model identification
- Export and reporting
- Performance optimized for large datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================
# Data Classes
# ===========================

@dataclass
class ModelPerformance:
    """Model performance summary."""
    model_name: str
    metric_mean: float
    metric_std: float
    metric_min: float
    metric_max: float
    n_experiments: int


@dataclass
class ComparisonResult:
    """Model comparison result."""
    model_a: str
    model_b: str
    metric: str
    a_mean: float
    b_mean: float
    difference: float
    percent_improvement: float
    winner: str


# ===========================
# Custom Exceptions
# ===========================

class MetricsError(Exception):
    """Base exception for metrics errors."""
    pass


class DataLoadError(MetricsError):
    """Error loading metrics data."""
    pass


class AggregationError(MetricsError):
    """Error during aggregation."""
    pass


# ===========================
# Metrics Aggregator
# ===========================

class MetricsAggregator:
    """Aggregate and analyze ML experiment metrics."""

    def __init__(self):
        """Initialize metrics aggregator."""
        self.metrics_df: Optional[pd.DataFrame] = None
        self.required_columns = ['model_name', 'experiment_id', 'timestamp']
        logger.info("MetricsAggregator initialized")

    def load_metrics(
        self,
        filepath: str,
        format: str = 'csv'
    ) -> pd.DataFrame:
        """
        Load metrics from file.

        Args:
            filepath: Path to metrics file
            format: File format ('csv', 'json', 'parquet')

        Returns:
            Loaded DataFrame

        Raises:
            DataLoadError: If loading fails
        """
        try:
            if format == 'csv':
                df = pd.read_csv(filepath)
            elif format == 'json':
                df = pd.read_json(filepath)
            elif format == 'parquet':
                df = pd.read_parquet(filepath)
            else:
                raise DataLoadError(f"Unsupported format: {format}")

            # Validate required columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                raise DataLoadError(f"Missing required columns: {missing_cols}")

            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Optimize dtypes
            df = self._optimize_dtypes(df)

            # Store or append
            if self.metrics_df is None:
                self.metrics_df = df
            else:
                self.metrics_df = pd.concat([self.metrics_df, df], ignore_index=True)

            logger.info(f"Loaded {len(df)} metrics from {filepath}")
            return df

        except Exception as e:
            raise DataLoadError(f"Failed to load metrics: {e}")

    def load_metrics_batch(
        self,
        filepaths: List[str],
        format: str = 'csv'
    ) -> pd.DataFrame:
        """
        Load metrics from multiple files.

        Args:
            filepaths: List of file paths
            format: File format

        Returns:
            Combined DataFrame
        """
        dfs = []
        for filepath in filepaths:
            try:
                df = self.load_metrics(filepath, format)
                dfs.append(df)
            except DataLoadError as e:
                logger.warning(f"Skipped {filepath}: {e}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined)} total metrics from {len(dfs)} files")
            return combined
        else:
            raise DataLoadError("No files loaded successfully")

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes for memory efficiency.

        Args:
            df: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        # Convert object columns to category if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # If less than 50% unique
                df[col] = df[col].astype('category')

        # Downcast numeric types
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        return df

    def aggregate_by_model(
        self,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate metrics by model.

        Args:
            metrics: List of metric columns to aggregate (all if None)

        Returns:
            Aggregated DataFrame

        Raises:
            AggregationError: If aggregation fails
        """
        if self.metrics_df is None or self.metrics_df.empty:
            raise AggregationError("No metrics loaded")

        try:
            # Determine metric columns
            if metrics is None:
                metrics = self._get_metric_columns()

            # Aggregate
            agg_dict = {
                metric: ['mean', 'std', 'min', 'max', 'count']
                for metric in metrics
            }

            aggregated = self.metrics_df.groupby('model_name').agg(agg_dict)

            # Flatten column names
            aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
            aggregated = aggregated.reset_index()

            logger.info(f"Aggregated metrics for {len(aggregated)} models")
            return aggregated

        except Exception as e:
            raise AggregationError(f"Aggregation failed: {e}")

    def aggregate_by_dataset(
        self,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate metrics by dataset.

        Args:
            metrics: List of metric columns

        Returns:
            Aggregated DataFrame
        """
        if 'dataset_name' not in self.metrics_df.columns:
            raise AggregationError("No dataset_name column found")

        if metrics is None:
            metrics = self._get_metric_columns()

        agg_dict = {metric: ['mean', 'std', 'min', 'max'] for metric in metrics}
        aggregated = self.metrics_df.groupby(['model_name', 'dataset_name']).agg(agg_dict)
        aggregated.columns = ['_'.join(col) for col in aggregated.columns]

        return aggregated.reset_index()

    def aggregate_by_time_period(
        self,
        period: str = 'D',
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate metrics by time period.

        Args:
            period: Time period ('D' for day, 'W' for week, 'M' for month)
            metrics: List of metric columns

        Returns:
            Time-aggregated DataFrame
        """
        if self.metrics_df is None:
            raise AggregationError("No metrics loaded")

        if metrics is None:
            metrics = self._get_metric_columns()

        # Set timestamp as index
        df = self.metrics_df.set_index('timestamp')

        # Resample and aggregate
        agg_dict = {metric: ['mean', 'std', 'count'] for metric in metrics}
        aggregated = df.groupby('model_name').resample(period).agg(agg_dict)

        aggregated.columns = ['_'.join(col) for col in aggregated.columns]
        return aggregated.reset_index()

    def compare_models(
        self,
        model_names: List[str],
        metric: str = 'accuracy'
    ) -> List[ComparisonResult]:
        """
        Compare multiple models.

        Args:
            model_names: List of model names to compare
            metric: Metric to compare

        Returns:
            List of comparison results
        """
        if self.metrics_df is None:
            raise AggregationError("No metrics loaded")

        if metric not in self.metrics_df.columns:
            raise AggregationError(f"Metric {metric} not found")

        results = []

        # Compare each pair
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                # Get metrics for each model
                a_metrics = self.metrics_df[
                    self.metrics_df['model_name'] == model_a
                ][metric]
                b_metrics = self.metrics_df[
                    self.metrics_df['model_name'] == model_b
                ][metric]

                if a_metrics.empty or b_metrics.empty:
                    continue

                a_mean = a_metrics.mean()
                b_mean = b_metrics.mean()
                diff = a_mean - b_mean
                percent = (diff / b_mean) * 100 if b_mean != 0 else 0

                winner = model_a if a_mean > b_mean else model_b

                result = ComparisonResult(
                    model_a=model_a,
                    model_b=model_b,
                    metric=metric,
                    a_mean=float(a_mean),
                    b_mean=float(b_mean),
                    difference=float(diff),
                    percent_improvement=float(percent),
                    winner=winner
                )
                results.append(result)

        logger.info(f"Compared {len(model_names)} models")
        return results

    def get_best_model(
        self,
        metric: str = 'accuracy',
        minimize: bool = False
    ) -> str:
        """
        Find best performing model.

        Args:
            metric: Metric to use for ranking
            minimize: If True, lower is better

        Returns:
            Name of best model
        """
        if self.metrics_df is None or metric not in self.metrics_df.columns:
            raise AggregationError(f"Cannot find best model for {metric}")

        # Calculate mean for each model
        model_means = self.metrics_df.groupby('model_name')[metric].mean()

        # Find best
        if minimize:
            best_model = model_means.idxmin()
        else:
            best_model = model_means.idxmax()

        logger.info(f"Best model: {best_model} ({metric}={model_means[best_model]:.4f})")
        return str(best_model)

    def get_model_performance(
        self,
        model_name: str,
        metric: str = 'accuracy'
    ) -> ModelPerformance:
        """
        Get performance summary for specific model.

        Args:
            model_name: Model name
            metric: Metric to summarize

        Returns:
            ModelPerformance object
        """
        if self.metrics_df is None:
            raise AggregationError("No metrics loaded")

        model_data = self.metrics_df[self.metrics_df['model_name'] == model_name]

        if model_data.empty:
            raise AggregationError(f"Model {model_name} not found")

        if metric not in model_data.columns:
            raise AggregationError(f"Metric {metric} not found")

        metric_values = model_data[metric]

        return ModelPerformance(
            model_name=model_name,
            metric_mean=float(metric_values.mean()),
            metric_std=float(metric_values.std()),
            metric_min=float(metric_values.min()),
            metric_max=float(metric_values.max()),
            n_experiments=len(model_data)
        )

    def filter_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> 'MetricsAggregator':
        """
        Filter metrics by date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            New MetricsAggregator with filtered data
        """
        if self.metrics_df is None:
            raise AggregationError("No metrics loaded")

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        filtered = self.metrics_df[
            (self.metrics_df['timestamp'] >= start) &
            (self.metrics_df['timestamp'] <= end)
        ].copy()

        # Create new aggregator
        new_aggregator = MetricsAggregator()
        new_aggregator.metrics_df = filtered

        logger.info(f"Filtered to {len(filtered)} metrics between {start_date} and {end_date}")
        return new_aggregator

    def calculate_trend(
        self,
        model_name: str,
        metric: str = 'accuracy',
        window: int = 7
    ) -> pd.Series:
        """
        Calculate trend (moving average) for model metric.

        Args:
            model_name: Model name
            metric: Metric to analyze
            window: Window size for moving average

        Returns:
            Series with moving average
        """
        if self.metrics_df is None:
            raise AggregationError("No metrics loaded")

        model_data = self.metrics_df[
            self.metrics_df['model_name'] == model_name
        ].sort_values('timestamp')

        if model_data.empty:
            raise AggregationError(f"Model {model_name} not found")

        trend = model_data.set_index('timestamp')[metric].rolling(window=window).mean()

        return trend

    def export_report(
        self,
        filepath: str,
        format: str = 'csv',
        include_summary: bool = True
    ) -> None:
        """
        Export metrics report.

        Args:
            filepath: Output file path
            format: Export format ('csv', 'json', 'excel')
            include_summary: Include summary statistics
        """
        if self.metrics_df is None:
            raise AggregationError("No metrics loaded")

        try:
            if include_summary:
                # Create summary
                summary = self.aggregate_by_model()
            else:
                summary = self.metrics_df

            # Export
            if format == 'csv':
                summary.to_csv(filepath, index=False)
            elif format == 'json':
                summary.to_json(filepath, orient='records', indent=2)
            elif format == 'excel':
                summary.to_excel(filepath, index=False, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Exported report to {filepath}")

        except Exception as e:
            raise MetricsError(f"Export failed: {e}")

    def get_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive summary statistics.

        Returns:
            Dictionary with various summary DataFrames
        """
        if self.metrics_df is None:
            raise AggregationError("No metrics loaded")

        summaries = {}

        # Overall statistics
        metric_cols = self._get_metric_columns()
        summaries['overall'] = self.metrics_df[metric_cols].describe()

        # Per-model statistics
        summaries['by_model'] = self.aggregate_by_model(metric_cols)

        # Per-dataset statistics (if available)
        if 'dataset_name' in self.metrics_df.columns:
            summaries['by_dataset'] = self.aggregate_by_dataset(metric_cols)

        # Correlation matrix
        summaries['correlation'] = self.metrics_df[metric_cols].corr()

        return summaries

    def _get_metric_columns(self) -> List[str]:
        """
        Identify metric columns (numeric, excluding IDs and timestamps).

        Returns:
            List of metric column names
        """
        if self.metrics_df is None:
            return []

        # Get numeric columns
        numeric_cols = self.metrics_df.select_dtypes(include=[np.number]).columns

        # Exclude ID and count columns
        exclude_patterns = ['id', 'count', 'index']
        metric_cols = [
            col for col in numeric_cols
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]

        return metric_cols

    def get_metrics_info(self) -> Dict[str, any]:
        """
        Get information about loaded metrics.

        Returns:
            Dictionary with metrics information
        """
        if self.metrics_df is None:
            return {'status': 'No metrics loaded'}

        return {
            'total_experiments': len(self.metrics_df),
            'unique_models': self.metrics_df['model_name'].nunique(),
            'unique_datasets': self.metrics_df['dataset_name'].nunique() if 'dataset_name' in self.metrics_df.columns else 0,
            'metric_columns': self._get_metric_columns(),
            'date_range': (
                self.metrics_df['timestamp'].min(),
                self.metrics_df['timestamp'].max()
            ) if 'timestamp' in self.metrics_df.columns else None,
            'memory_usage_mb': self.metrics_df.memory_usage(deep=True).sum() / 1024**2
        }


# ===========================
# Demo Function
# ===========================

def demo():
    """Demonstrate the Metrics Aggregator."""
    print("=" * 70)
    print("ML Metrics Aggregation Tool Demo")
    print("=" * 70)

    # Create sample metrics data
    np.random.seed(42)

    data = []
    models = ['model_a', 'model_b', 'model_c']
    datasets = ['dataset_1', 'dataset_2']
    start_date = pd.Timestamp('2024-01-01')

    for i in range(100):
        model = np.random.choice(models)
        dataset = np.random.choice(datasets)
        timestamp = start_date + pd.Timedelta(days=i % 30)

        # Generate metrics with some model-specific patterns
        if model == 'model_a':
            accuracy = np.random.normal(0.90, 0.02)
            f1_score = np.random.normal(0.88, 0.02)
        elif model == 'model_b':
            accuracy = np.random.normal(0.85, 0.03)
            f1_score = np.random.normal(0.83, 0.03)
        else:
            accuracy = np.random.normal(0.92, 0.015)
            f1_score = np.random.normal(0.90, 0.015)

        data.append({
            'experiment_id': f'exp_{i:04d}',
            'model_name': model,
            'dataset_name': dataset,
            'timestamp': timestamp,
            'accuracy': np.clip(accuracy, 0, 1),
            'f1_score': np.clip(f1_score, 0, 1),
            'precision': np.clip(np.random.normal(0.88, 0.03), 0, 1),
            'recall': np.clip(np.random.normal(0.87, 0.03), 0, 1)
        })

    # Save sample data
    sample_df = pd.DataFrame(data)
    sample_df.to_csv('sample_metrics.csv', index=False)

    # Initialize aggregator
    print("\n1. Loading metrics...")
    aggregator = MetricsAggregator()
    aggregator.load_metrics('sample_metrics.csv')

    # Get info
    print("\n2. Metrics info:")
    info = aggregator.get_metrics_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Aggregate by model
    print("\n3. Aggregating by model...")
    model_stats = aggregator.aggregate_by_model(['accuracy', 'f1_score'])
    print(model_stats[['model_name', 'accuracy_mean', 'f1_score_mean']])

    # Compare models
    print("\n4. Comparing models...")
    comparisons = aggregator.compare_models(['model_a', 'model_b', 'model_c'], 'accuracy')
    for comp in comparisons:
        print(f"   {comp.model_a} vs {comp.model_b}: "
              f"{comp.percent_improvement:+.2f}% (winner: {comp.winner})")

    # Find best model
    print("\n5. Finding best model...")
    best = aggregator.get_best_model('accuracy')
    print(f"   Best model: {best}")

    # Get model performance
    print("\n6. Model performance details...")
    perf = aggregator.get_model_performance('model_c', 'accuracy')
    print(f"   {perf.model_name}:")
    print(f"   - Mean: {perf.metric_mean:.4f}")
    print(f"   - Std: {perf.metric_std:.4f}")
    print(f"   - Range: [{perf.metric_min:.4f}, {perf.metric_max:.4f}]")
    print(f"   - Experiments: {perf.n_experiments}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
