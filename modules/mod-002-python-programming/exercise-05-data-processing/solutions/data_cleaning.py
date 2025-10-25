#!/usr/bin/env python3
"""
Data Cleaning and Validation

Comprehensive data cleaning, validation, and normalization tools.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Statistics from data cleaning operation."""
    rows_before: int
    rows_after: int
    rows_removed: int
    columns_before: int
    columns_after: int
    missing_values_filled: int
    outliers_removed: int


class DataCleaner:
    """Clean and validate data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize data cleaner.

        Args:
            df: DataFrame to clean
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.stats = None

    def handle_missing_values(
        self,
        strategy: str = 'drop',
        fill_values: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None
    ) -> 'DataCleaner':
        """
        Handle missing values.

        Args:
            strategy: Strategy ('drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'constant')
            fill_values: Dictionary of column:value for constant strategy
            columns: Specific columns to handle (None for all)

        Returns:
            Self for chaining

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaner.handle_missing_values(strategy='mean')
        """
        if columns is None:
            columns = self.df.columns.tolist()

        if strategy == 'drop':
            self.df = self.df.dropna(subset=columns)
            logger.info("Dropped rows with missing values")

        elif strategy == 'mean':
            for col in columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].mean())

        elif strategy == 'median':
            for col in columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())

        elif strategy == 'mode':
            for col in columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        elif strategy == 'forward_fill':
            self.df[columns] = self.df[columns].fillna(method='ffill')

        elif strategy == 'backward_fill':
            self.df[columns] = self.df[columns].fillna(method='bfill')

        elif strategy == 'constant' and fill_values:
            self.df = self.df.fillna(fill_values)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"Handled missing values using strategy: {strategy}")
        return self

    def remove_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> 'DataCleaner':
        """
        Remove duplicate rows.

        Args:
            subset: Columns to consider for duplicates (None for all)
            keep: Which duplicate to keep ('first', 'last', False)

        Returns:
            Self for chaining

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaner.remove_duplicates(subset=['email'])
        """
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = before - len(self.df)

        logger.info(f"Removed {removed} duplicate rows")
        return self

    def remove_outliers(
        self,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> 'DataCleaner':
        """
        Remove outliers from numeric columns.

        Args:
            columns: Columns to check for outliers
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection (IQR multiplier or Z-score)

        Returns:
            Self for chaining

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaner.remove_outliers(['price', 'quantity'], method='iqr')
        """
        before = len(self.df)

        for col in columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                logger.warning(f"Skipping non-numeric column: {col}")
                continue

            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                self.df = self.df[
                    (self.df[col] >= lower) & (self.df[col] <= upper)
                ]

            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]

            else:
                raise ValueError(f"Unknown method: {method}")

        removed = before - len(self.df)
        logger.info(f"Removed {removed} outliers from {len(columns)} columns")
        return self

    def normalize_text(
        self,
        columns: List[str],
        lowercase: bool = True,
        remove_whitespace: bool = True,
        remove_special_chars: bool = False
    ) -> 'DataCleaner':
        """
        Normalize text columns.

        Args:
            columns: Columns to normalize
            lowercase: Convert to lowercase
            remove_whitespace: Remove extra whitespace
            remove_special_chars: Remove special characters

        Returns:
            Self for chaining

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaner.normalize_text(['name', 'description'], lowercase=True)
        """
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column not found: {col}")
                continue

            if lowercase:
                self.df[col] = self.df[col].str.lower()

            if remove_whitespace:
                self.df[col] = self.df[col].str.strip()
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)

            if remove_special_chars:
                self.df[col] = self.df[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

        logger.info(f"Normalized {len(columns)} text columns")
        return self

    def validate_format(
        self,
        column: str,
        pattern: str,
        remove_invalid: bool = True
    ) -> 'DataCleaner':
        """
        Validate column values against a regex pattern.

        Args:
            column: Column to validate
            pattern: Regex pattern
            remove_invalid: Remove invalid rows

        Returns:
            Self for chaining

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaner.validate_format('email', r'^[\w\.-]+@[\w\.-]+\.\w+$')
        """
        valid_mask = self.df[column].str.match(pattern, na=False)
        invalid_count = (~valid_mask).sum()

        if remove_invalid:
            self.df = self.df[valid_mask]
            logger.info(f"Removed {invalid_count} rows with invalid {column} format")
        else:
            logger.info(f"Found {invalid_count} rows with invalid {column} format")

        return self

    def standardize_dates(
        self,
        columns: List[str],
        format: Optional[str] = None
    ) -> 'DataCleaner':
        """
        Standardize date columns.

        Args:
            columns: Date columns to standardize
            format: Target date format (None for datetime object)

        Returns:
            Self for chaining

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaner.standardize_dates(['created_at', 'updated_at'])
        """
        for col in columns:
            if col not in self.df.columns:
                continue

            # Convert to datetime
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

            # Format if specified
            if format:
                self.df[col] = self.df[col].dt.strftime(format)

        logger.info(f"Standardized {len(columns)} date columns")
        return self

    def cap_values(
        self,
        column: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None
    ) -> 'DataCleaner':
        """
        Cap values at specified limits.

        Args:
            column: Column to cap
            lower: Lower limit (None for no lower cap)
            upper: Upper limit (None for no upper cap)

        Returns:
            Self for chaining

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaner.cap_values('age', lower=0, upper=100)
        """
        if lower is not None:
            self.df[column] = self.df[column].clip(lower=lower)

        if upper is not None:
            self.df[column] = self.df[column].clip(upper=upper)

        logger.info(f"Capped values in {column}")
        return self

    def convert_types(
        self,
        type_map: Dict[str, str]
    ) -> 'DataCleaner':
        """
        Convert column data types.

        Args:
            type_map: Dictionary mapping columns to target types

        Returns:
            Self for chaining

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaner.convert_types({'age': 'int32', 'price': 'float64'})
        """
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(dtype)
                except Exception as e:
                    logger.error(f"Failed to convert {col} to {dtype}: {e}")

        logger.info(f"Converted types for {len(type_map)} columns")
        return self

    def get_result(self) -> pd.DataFrame:
        """
        Get cleaned DataFrame.

        Returns:
            Cleaned DataFrame

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaned_df = cleaner.handle_missing_values().remove_duplicates().get_result()
        """
        return self.df

    def get_cleaning_stats(self) -> CleaningStats:
        """
        Get cleaning statistics.

        Returns:
            Cleaning statistics

        Example:
            >>> cleaner = DataCleaner(df)
            >>> cleaner.handle_missing_values()
            >>> stats = cleaner.get_cleaning_stats()
        """
        return CleaningStats(
            rows_before=len(self.original_df),
            rows_after=len(self.df),
            rows_removed=len(self.original_df) - len(self.df),
            columns_before=len(self.original_df.columns),
            columns_after=len(self.df.columns),
            missing_values_filled=0,  # Would need to track this
            outliers_removed=0  # Would need to track this
        )


class DataValidator:
    """Validate data quality."""

    @staticmethod
    def validate_not_null(df: pd.DataFrame, columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that columns have no null values.

        Args:
            df: DataFrame to validate
            columns: Columns to check

        Returns:
            Tuple of (is_valid, list of columns with nulls)

        Example:
            >>> valid, errors = DataValidator.validate_not_null(df, ['id', 'name'])
        """
        errors = []
        for col in columns:
            if df[col].isnull().any():
                errors.append(col)

        return len(errors) == 0, errors

    @staticmethod
    def validate_unique(df: pd.DataFrame, columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that columns have unique values.

        Args:
            df: DataFrame to validate
            columns: Columns to check

        Returns:
            Tuple of (is_valid, list of columns with duplicates)

        Example:
            >>> valid, errors = DataValidator.validate_unique(df, ['email'])
        """
        errors = []
        for col in columns:
            if df[col].duplicated().any():
                errors.append(col)

        return len(errors) == 0, errors

    @staticmethod
    def validate_range(
        df: pd.DataFrame,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> Tuple[bool, int]:
        """
        Validate that values are within range.

        Args:
            df: DataFrame to validate
            column: Column to check
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Tuple of (is_valid, count of invalid values)

        Example:
            >>> valid, count = DataValidator.validate_range(df, 'age', min_value=0, max_value=120)
        """
        mask = pd.Series([True] * len(df))

        if min_value is not None:
            mask &= df[column] >= min_value

        if max_value is not None:
            mask &= df[column] <= max_value

        invalid_count = (~mask).sum()
        return invalid_count == 0, int(invalid_count)

    @staticmethod
    def validate_data_types(
        df: pd.DataFrame,
        expected_types: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate column data types.

        Args:
            df: DataFrame to validate
            expected_types: Dictionary mapping columns to expected types

        Returns:
            Tuple of (is_valid, list of columns with wrong types)

        Example:
            >>> valid, errors = DataValidator.validate_data_types(df, {'age': 'int64'})
        """
        errors = []
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    errors.append(f"{col}: expected {expected_type}, got {actual_type}")

        return len(errors) == 0, errors


if __name__ == '__main__':
    print("=== Data Cleaning Examples ===\n")

    # Create sample data with issues
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', '  David  ', None],
        'age': [25, 30, 150, 25, 35, 28],  # Outlier: 150
        'email': ['alice@test.com', 'invalid-email', 'charlie@test.com', 'alice@test.com', 'david@test.com', None],
        'salary': [50000, 60000, None, 50000, 70000, 55000],
    })

    print("Original Data:")
    print(df)
    print()

    # Clean data
    cleaner = DataCleaner(df)
    cleaned = (
        cleaner
        .handle_missing_values(strategy='median', columns=['salary'])
        .remove_duplicates(subset=['name', 'email'])
        .remove_outliers(['age'], method='iqr', threshold=1.5)
        .normalize_text(['name'], lowercase=True, remove_whitespace=True)
        .get_result()
    )

    print("Cleaned Data:")
    print(cleaned)
    print()

    # Get stats
    stats = cleaner.get_cleaning_stats()
    print("Cleaning Stats:")
    print(f"  Rows: {stats.rows_before} -> {stats.rows_after} (removed {stats.rows_removed})")

    print("\n✓ Data cleaning examples completed")
