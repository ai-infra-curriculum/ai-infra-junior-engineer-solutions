#!/usr/bin/env python3
"""
Data Transformation and Feature Engineering

Tools for transforming, reshaping, and engineering features from data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransformer:
    """Transform and reshape data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize data transformer.

        Args:
            df: DataFrame to transform
        """
        self.df = df.copy()

    def pivot_data(
        self,
        index: Union[str, List[str]],
        columns: str,
        values: str,
        aggfunc: str = 'mean'
    ) -> pd.DataFrame:
        """
        Pivot data from long to wide format.

        Args:
            index: Column(s) to use as index
            columns: Column to pivot into columns
            values: Column with values to aggregate
            aggfunc: Aggregation function

        Returns:
            Pivoted DataFrame

        Example:
            >>> transformer = DataTransformer(df)
            >>> pivoted = transformer.pivot_data(
            ...     index='date',
            ...     columns='metric',
            ...     values='value'
            ... )
        """
        result = self.df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc
        )

        logger.info(f"Pivoted data: {result.shape}")
        return result

    def melt_data(
        self,
        id_vars: List[str],
        value_vars: Optional[List[str]] = None,
        var_name: str = 'variable',
        value_name: str = 'value'
    ) -> pd.DataFrame:
        """
        Melt data from wide to long format.

        Args:
            id_vars: Columns to keep as identifiers
            value_vars: Columns to melt (None for all except id_vars)
            var_name: Name for variable column
            value_name: Name for value column

        Returns:
            Melted DataFrame

        Example:
            >>> transformer = DataTransformer(df)
            >>> melted = transformer.melt_data(
            ...     id_vars=['date'],
            ...     value_vars=['metric1', 'metric2']
            ... )
        """
        result = pd.melt(
            self.df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )

        logger.info(f"Melted data: {result.shape}")
        return result

    def normalize_numeric(
        self,
        columns: List[str],
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Normalize numeric columns.

        Args:
            columns: Columns to normalize
            method: Normalization method ('standard', 'minmax')

        Returns:
            DataFrame with normalized columns

        Example:
            >>> transformer = DataTransformer(df)
            >>> df_norm = transformer.normalize_numeric(['price', 'quantity'])
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.df[columns] = scaler.fit_transform(self.df[columns])
        logger.info(f"Normalized {len(columns)} columns using {method}")

        return self.df

    def bin_numeric(
        self,
        column: str,
        bins: Union[int, List[float]],
        labels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Bin numeric column into categories.

        Args:
            column: Column to bin
            bins: Number of bins or bin edges
            labels: Labels for bins

        Returns:
            DataFrame with binned column

        Example:
            >>> transformer = DataTransformer(df)
            >>> df = transformer.bin_numeric(
            ...     'age',
            ...     bins=[0, 18, 30, 50, 100],
            ...     labels=['child', 'young', 'adult', 'senior']
            ... )
        """
        binned_col = f"{column}_binned"
        self.df[binned_col] = pd.cut(self.df[column], bins=bins, labels=labels)
        logger.info(f"Binned {column} into {binned_col}")

        return self.df

    def encode_categorical(
        self,
        columns: List[str],
        method: str = 'label'
    ) -> pd.DataFrame:
        """
        Encode categorical columns.

        Args:
            columns: Columns to encode
            method: Encoding method ('label', 'onehot')

        Returns:
            DataFrame with encoded columns

        Example:
            >>> transformer = DataTransformer(df)
            >>> df = transformer.encode_categorical(['category'], method='onehot')
        """
        if method == 'label':
            for col in columns:
                le = LabelEncoder()
                self.df[f"{col}_encoded"] = le.fit_transform(self.df[col])

        elif method == 'onehot':
            self.df = pd.get_dummies(
                self.df,
                columns=columns,
                prefix=columns
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Encoded {len(columns)} categorical columns using {method}")
        return self.df

    def create_lag_features(
        self,
        column: str,
        lags: List[int],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series.

        Args:
            column: Column to create lags for
            lags: List of lag periods
            group_by: Column to group by (for panel data)

        Returns:
            DataFrame with lag features

        Example:
            >>> transformer = DataTransformer(df)
            >>> df = transformer.create_lag_features('sales', lags=[1, 7, 30])
        """
        for lag in lags:
            lag_col = f"{column}_lag_{lag}"

            if group_by:
                self.df[lag_col] = self.df.groupby(group_by)[column].shift(lag)
            else:
                self.df[lag_col] = self.df[column].shift(lag)

        logger.info(f"Created {len(lags)} lag features for {column}")
        return self.df

    def create_rolling_features(
        self,
        column: str,
        windows: List[int],
        functions: List[str] = ['mean'],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features.

        Args:
            column: Column to create rolling features for
            windows: List of window sizes
            functions: List of functions ('mean', 'std', 'min', 'max')
            group_by: Column to group by

        Returns:
            DataFrame with rolling features

        Example:
            >>> transformer = DataTransformer(df)
            >>> df = transformer.create_rolling_features(
            ...     'price',
            ...     windows=[7, 30],
            ...     functions=['mean', 'std']
            ... )
        """
        for window in windows:
            for func in functions:
                feature_name = f"{column}_rolling_{func}_{window}"

                if group_by:
                    rolling = self.df.groupby(group_by)[column].rolling(window)
                else:
                    rolling = self.df[column].rolling(window)

                if func == 'mean':
                    self.df[feature_name] = rolling.mean().values
                elif func == 'std':
                    self.df[feature_name] = rolling.std().values
                elif func == 'min':
                    self.df[feature_name] = rolling.min().values
                elif func == 'max':
                    self.df[feature_name] = rolling.max().values

        logger.info(f"Created rolling features for {column}")
        return self.df

    def create_date_features(self, date_column: str) -> pd.DataFrame:
        """
        Extract features from date column.

        Args:
            date_column: Date column to extract features from

        Returns:
            DataFrame with date features

        Example:
            >>> transformer = DataTransformer(df)
            >>> df = transformer.create_date_features('timestamp')
        """
        # Ensure datetime type
        self.df[date_column] = pd.to_datetime(self.df[date_column])

        # Extract features
        self.df[f"{date_column}_year"] = self.df[date_column].dt.year
        self.df[f"{date_column}_month"] = self.df[date_column].dt.month
        self.df[f"{date_column}_day"] = self.df[date_column].dt.day
        self.df[f"{date_column}_dayofweek"] = self.df[date_column].dt.dayofweek
        self.df[f"{date_column}_quarter"] = self.df[date_column].dt.quarter
        self.df[f"{date_column}_hour"] = self.df[date_column].dt.hour
        self.df[f"{date_column}_is_weekend"] = self.df[date_column].dt.dayofweek.isin([5, 6])

        logger.info(f"Created date features from {date_column}")
        return self.df

    def apply_custom_transformation(
        self,
        transformations: Dict[str, Callable]
    ) -> pd.DataFrame:
        """
        Apply custom transformations to columns.

        Args:
            transformations: Dictionary mapping column names to transformation functions

        Returns:
            Transformed DataFrame

        Example:
            >>> transformer = DataTransformer(df)
            >>> df = transformer.apply_custom_transformation({
            ...     'price': lambda x: np.log1p(x),
            ...     'quantity': lambda x: x ** 2
            ... })
        """
        for col, transform_func in transformations.items():
            if col in self.df.columns:
                self.df[col] = transform_func(self.df[col])
                logger.info(f"Applied transformation to {col}")

        return self.df

    def get_result(self) -> pd.DataFrame:
        """Get transformed DataFrame."""
        return self.df


class FeatureEngineer:
    """Create derived features from data."""

    @staticmethod
    def create_interaction_features(
        df: pd.DataFrame,
        column_pairs: List[Tuple[str, str]],
        operations: List[str] = ['multiply']
    ) -> pd.DataFrame:
        """
        Create interaction features between column pairs.

        Args:
            df: DataFrame
            column_pairs: List of column pairs to interact
            operations: Operations to perform ('multiply', 'add', 'subtract', 'divide')

        Returns:
            DataFrame with interaction features

        Example:
            >>> df = FeatureEngineer.create_interaction_features(
            ...     df,
            ...     column_pairs=[('price', 'quantity')],
            ...     operations=['multiply']
            ... )
        """
        result = df.copy()

        for col1, col2 in column_pairs:
            for op in operations:
                if op == 'multiply':
                    result[f"{col1}_x_{col2}"] = result[col1] * result[col2]
                elif op == 'add':
                    result[f"{col1}_plus_{col2}"] = result[col1] + result[col2]
                elif op == 'subtract':
                    result[f"{col1}_minus_{col2}"] = result[col1] - result[col2]
                elif op == 'divide':
                    result[f"{col1}_div_{col2}"] = result[col1] / result[col2].replace(0, np.nan)

        logger.info(f"Created interaction features for {len(column_pairs)} pairs")
        return result

    @staticmethod
    def create_polynomial_features(
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features.

        Args:
            df: DataFrame
            columns: Columns to create polynomial features for
            degree: Polynomial degree

        Returns:
            DataFrame with polynomial features

        Example:
            >>> df = FeatureEngineer.create_polynomial_features(
            ...     df,
            ...     columns=['age', 'income'],
            ...     degree=2
            ... )
        """
        result = df.copy()

        for col in columns:
            for d in range(2, degree + 1):
                result[f"{col}_pow_{d}"] = result[col] ** d

        logger.info(f"Created polynomial features up to degree {degree}")
        return result

    @staticmethod
    def create_ratio_features(
        df: pd.DataFrame,
        numerator_cols: List[str],
        denominator_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create ratio features.

        Args:
            df: DataFrame
            numerator_cols: Numerator columns
            denominator_cols: Denominator columns

        Returns:
            DataFrame with ratio features

        Example:
            >>> df = FeatureEngineer.create_ratio_features(
            ...     df,
            ...     numerator_cols=['profit'],
            ...     denominator_cols=['revenue']
            ... )
        """
        result = df.copy()

        for num_col in numerator_cols:
            for den_col in denominator_cols:
                feature_name = f"{num_col}_per_{den_col}"
                result[feature_name] = result[num_col] / result[den_col].replace(0, np.nan)

        logger.info(f"Created {len(numerator_cols) * len(denominator_cols)} ratio features")
        return result


if __name__ == '__main__':
    print("=== Data Transformation Examples ===\n")

    # Create sample data
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.randn(100) * 10 + 50,
        'quantity': np.random.randint(1, 100, 100),
    })

    print("1. Original Data:")
    print(df.head())

    # Transform data
    transformer = DataTransformer(df)

    print("\n2. Create Date Features:")
    df_transformed = transformer.create_date_features('date')
    print(f"   Added date features: {[c for c in df_transformed.columns if 'date' in c and c != 'date']}")

    print("\n3. Create Lag Features:")
    df_transformed = transformer.create_lag_features('value', lags=[1, 7])
    print(f"   Columns: {list(df_transformed.columns)}")

    print("\n4. Encode Categorical:")
    df_transformed = transformer.encode_categorical(['category'], method='onehot')
    print(f"   Total columns after encoding: {len(df_transformed.columns)}")

    print("\n5. Create Interaction Features:")
    df_with_interactions = FeatureEngineer.create_interaction_features(
        df,
        column_pairs=[('value', 'quantity')],
        operations=['multiply', 'divide']
    )
    print(f"   Created features: {[c for c in df_with_interactions.columns if '_x_' in c or '_div_' in c]}")

    print("\n✓ Data transformation examples completed")
