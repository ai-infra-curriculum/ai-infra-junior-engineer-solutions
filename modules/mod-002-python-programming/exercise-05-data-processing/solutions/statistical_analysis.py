#!/usr/bin/env python3
"""
Statistical Analysis and Data Insights

Tools for statistical analysis, correlation, hypothesis testing, and insights generation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StatisticalSummary:
    """Statistical summary of a dataset."""
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float


@dataclass
class CorrelationAnalysis:
    """Correlation analysis results."""
    method: str
    correlation_matrix: pd.DataFrame
    strong_correlations: List[Tuple[str, str, float]]


class StatisticalAnalyzer:
    """Perform statistical analysis on data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize statistical analyzer.

        Args:
            df: DataFrame to analyze
        """
        self.df = df

    def get_summary_stats(
        self,
        column: str
    ) -> StatisticalSummary:
        """
        Get comprehensive statistical summary for a column.

        Args:
            column: Column to analyze

        Returns:
            Statistical summary

        Example:
            >>> analyzer = StatisticalAnalyzer(df)
            >>> summary = analyzer.get_summary_stats('price')
        """
        data = self.df[column].dropna()

        return StatisticalSummary(
            count=len(data),
            mean=float(data.mean()),
            median=float(data.median()),
            std=float(data.std()),
            min=float(data.min()),
            max=float(data.max()),
            q25=float(data.quantile(0.25)),
            q75=float(data.quantile(0.75)),
            skewness=float(data.skew()),
            kurtosis=float(data.kurtosis())
        )

    def describe_all(self) -> pd.DataFrame:
        """
        Get descriptive statistics for all numeric columns.

        Returns:
            DataFrame with descriptive statistics

        Example:
            >>> analyzer = StatisticalAnalyzer(df)
            >>> stats = analyzer.describe_all()
        """
        return self.df.describe()

    def get_correlation_matrix(
        self,
        method: str = 'pearson',
        threshold: float = 0.7
    ) -> CorrelationAnalysis:
        """
        Calculate correlation matrix and identify strong correlations.

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Threshold for strong correlation

        Returns:
            Correlation analysis results

        Example:
            >>> analyzer = StatisticalAnalyzer(df)
            >>> corr = analyzer.get_correlation_matrix(method='pearson')
        """
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)

        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) >= threshold:
                    strong_correlations.append((col1, col2, float(corr_value)))

        logger.info(f"Found {len(strong_correlations)} strong correlations (threshold={threshold})")

        return CorrelationAnalysis(
            method=method,
            correlation_matrix=corr_matrix,
            strong_correlations=strong_correlations
        )

    def detect_outliers(
        self,
        column: str,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Detect outliers in a column.

        Args:
            column: Column to analyze
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection

        Returns:
            Tuple of (outlier boolean mask, outlier statistics)

        Example:
            >>> analyzer = StatisticalAnalyzer(df)
            >>> outliers, stats = analyzer.detect_outliers('price', method='iqr')
        """
        data = self.df[column].dropna()

        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = (data < lower_bound) | (data > upper_bound)
            stats_dict = {
                'method': 'IQR',
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'n_outliers': int(outliers.sum()),
                'outlier_percentage': float(outliers.sum() / len(data) * 100)
            }

        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > threshold

            stats_dict = {
                'method': 'Z-Score',
                'threshold': threshold,
                'n_outliers': int(outliers.sum()),
                'outlier_percentage': float(outliers.sum() / len(data) * 100)
            }

        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold

            stats_dict = {
                'method': 'Modified Z-Score',
                'threshold': threshold,
                'n_outliers': int(outliers.sum()),
                'outlier_percentage': float(outliers.sum() / len(data) * 100)
            }

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Detected {stats_dict['n_outliers']} outliers in {column}")
        return outliers, stats_dict

    def test_normality(
        self,
        column: str,
        method: str = 'shapiro'
    ) -> Tuple[float, float, bool]:
        """
        Test if data follows normal distribution.

        Args:
            column: Column to test
            method: Test method ('shapiro', 'kstest')

        Returns:
            Tuple of (statistic, p_value, is_normal)

        Example:
            >>> analyzer = StatisticalAnalyzer(df)
            >>> stat, p_value, is_normal = analyzer.test_normality('values')
        """
        data = self.df[column].dropna()

        if method == 'shapiro':
            statistic, p_value = stats.shapiro(data)
        elif method == 'kstest':
            statistic, p_value = stats.kstest(data, 'norm')
        else:
            raise ValueError(f"Unknown method: {method}")

        is_normal = p_value > 0.05  # Common significance level

        logger.info(f"Normality test ({method}): p-value={p_value:.4f}, is_normal={is_normal}")
        return float(statistic), float(p_value), is_normal

    def compare_groups(
        self,
        column: str,
        group_column: str,
        test: str = 'ttest'
    ) -> Dict[str, Any]:
        """
        Compare two groups statistically.

        Args:
            column: Column with values to compare
            group_column: Column defining groups
            test: Statistical test ('ttest', 'mannwhitneyu', 'anova')

        Returns:
            Dictionary with test results

        Example:
            >>> analyzer = StatisticalAnalyzer(df)
            >>> result = analyzer.compare_groups('sales', 'region', test='ttest')
        """
        groups = self.df.groupby(group_column)[column].apply(list)

        if test == 'ttest' and len(groups) == 2:
            group_values = list(groups.values)
            statistic, p_value = stats.ttest_ind(group_values[0], group_values[1])
            test_name = "T-Test"

        elif test == 'mannwhitneyu' and len(groups) == 2:
            group_values = list(groups.values)
            statistic, p_value = stats.mannwhitneyu(group_values[0], group_values[1])
            test_name = "Mann-Whitney U Test"

        elif test == 'anova':
            group_values = list(groups.values)
            statistic, p_value = stats.f_oneway(*group_values)
            test_name = "ANOVA"

        else:
            raise ValueError(f"Invalid test '{test}' for {len(groups)} groups")

        significant = p_value < 0.05

        result = {
            'test': test_name,
            'n_groups': len(groups),
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'group_means': {name: np.mean(values) for name, values in groups.items()}
        }

        logger.info(f"{test_name}: p-value={p_value:.4f}, significant={significant}")
        return result

    def calculate_percentiles(
        self,
        column: str,
        percentiles: List[float] = [25, 50, 75, 90, 95, 99]
    ) -> Dict[float, float]:
        """
        Calculate percentiles for a column.

        Args:
            column: Column to analyze
            percentiles: List of percentiles to calculate

        Returns:
            Dictionary mapping percentiles to values

        Example:
            >>> analyzer = StatisticalAnalyzer(df)
            >>> pcts = analyzer.calculate_percentiles('price', [50, 75, 90])
        """
        data = self.df[column].dropna()
        result = {pct: float(np.percentile(data, pct)) for pct in percentiles}

        logger.info(f"Calculated {len(percentiles)} percentiles for {column}")
        return result

    def get_value_counts(
        self,
        column: str,
        normalize: bool = False,
        top_n: Optional[int] = None
    ) -> pd.Series:
        """
        Get value counts for a column.

        Args:
            column: Column to analyze
            normalize: Return proportions instead of counts
            top_n: Return only top N values

        Returns:
            Series with value counts

        Example:
            >>> analyzer = StatisticalAnalyzer(df)
            >>> counts = analyzer.get_value_counts('category', top_n=10)
        """
        counts = self.df[column].value_counts(normalize=normalize)

        if top_n:
            counts = counts.head(top_n)

        return counts

    def calculate_moving_statistics(
        self,
        column: str,
        window: int,
        statistics: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        Calculate moving statistics.

        Args:
            column: Column to analyze
            window: Window size
            statistics: Statistics to calculate ('mean', 'std', 'min', 'max')

        Returns:
            DataFrame with moving statistics

        Example:
            >>> analyzer = StatisticalAnalyzer(df)
            >>> moving = analyzer.calculate_moving_statistics('price', window=7)
        """
        result = pd.DataFrame()

        for stat in statistics:
            if stat == 'mean':
                result[f'{column}_ma_{window}'] = self.df[column].rolling(window).mean()
            elif stat == 'std':
                result[f'{column}_std_{window}'] = self.df[column].rolling(window).std()
            elif stat == 'min':
                result[f'{column}_min_{window}'] = self.df[column].rolling(window).min()
            elif stat == 'max':
                result[f'{column}_max_{window}'] = self.df[column].rolling(window).max()

        logger.info(f"Calculated moving statistics for {column}, window={window}")
        return result


class DataInsights:
    """Generate insights from data."""

    @staticmethod
    def identify_trends(
        df: pd.DataFrame,
        time_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """
        Identify trends in time series data.

        Args:
            df: DataFrame
            time_column: Time column
            value_column: Value column

        Returns:
            Dictionary with trend analysis

        Example:
            >>> insights = DataInsights.identify_trends(df, 'date', 'sales')
        """
        df = df.sort_values(time_column)

        # Calculate linear regression slope
        x = np.arange(len(df))
        y = df[value_column].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Determine trend direction
        if slope > 0:
            trend = 'increasing'
        elif slope < 0:
            trend = 'decreasing'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'start_value': float(y[0]),
            'end_value': float(y[-1]),
            'percent_change': float((y[-1] - y[0]) / y[0] * 100) if y[0] != 0 else None
        }

    @staticmethod
    def find_top_contributors(
        df: pd.DataFrame,
        category_column: str,
        value_column: str,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Find top contributors to total value.

        Args:
            df: DataFrame
            category_column: Category column
            value_column: Value column
            n: Number of top contributors

        Returns:
            DataFrame with top contributors and their contribution percentages

        Example:
            >>> top = DataInsights.find_top_contributors(df, 'product', 'revenue', n=5)
        """
        total = df[value_column].sum()

        contributions = (
            df.groupby(category_column)[value_column]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )

        result = pd.DataFrame({
            'value': contributions,
            'percentage': contributions / total * 100
        })

        return result


if __name__ == '__main__':
    print("=== Statistical Analysis Examples ===\n")

    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'value': np.random.normal(100, 15, 1000),
        'price': np.random.lognormal(4, 0.5, 1000),
        'quantity': np.random.randint(1, 100, 1000),
    })

    analyzer = StatisticalAnalyzer(df)

    print("1. Summary Statistics:")
    summary = analyzer.get_summary_stats('value')
    print(f"   Mean: {summary.mean:.2f}")
    print(f"   Median: {summary.median:.2f}")
    print(f"   Std: {summary.std:.2f}")

    print("\n2. Correlation Analysis:")
    corr = analyzer.get_correlation_matrix(threshold=0.3)
    print(f"   Found {len(corr.strong_correlations)} strong correlations")

    print("\n3. Outlier Detection:")
    outliers, outlier_stats = analyzer.detect_outliers('price', method='iqr')
    print(f"   Outliers: {outlier_stats['n_outliers']} ({outlier_stats['outlier_percentage']:.2f}%)")

    print("\n4. Normality Test:")
    stat, p_value, is_normal = analyzer.test_normality('value')
    print(f"   P-value: {p_value:.4f}, Is Normal: {is_normal}")

    print("\n5. Group Comparison:")
    df['group'] = np.random.choice(['X', 'Y'], 1000)
    comparison = analyzer.compare_groups('value', 'group', test='ttest')
    print(f"   P-value: {comparison['p_value']:.4f}, Significant: {comparison['significant']}")

    print("\n✓ Statistical analysis examples completed")
