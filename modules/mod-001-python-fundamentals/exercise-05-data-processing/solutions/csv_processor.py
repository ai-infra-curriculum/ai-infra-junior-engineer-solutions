#!/usr/bin/env python3
"""
CSV Data Processing and Analysis

Comprehensive CSV processing tools for data cleaning, transformation, and analysis.
"""

import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    duplicate_rows: int
    data_types: Dict[str, str]
    unique_values: Dict[str, int]


class CSVProcessor:
    """Process and analyze CSV data."""

    def __init__(self, filepath: str):
        """
        Initialize CSV processor.

        Args:
            filepath: Path to CSV file
        """
        self.filepath = Path(filepath)
        self.df: Optional[pd.DataFrame] = None

    def load(
        self,
        encoding: str = 'utf-8',
        delimiter: str = ',',
        dtype: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.

        Args:
            encoding: File encoding
            delimiter: Column delimiter
            dtype: Dictionary of column data types

        Returns:
            Loaded DataFrame

        Example:
            >>> processor = CSVProcessor("data.csv")
            >>> df = processor.load()
        """
        logger.info(f"Loading CSV from {self.filepath}")

        try:
            self.df = pd.read_csv(
                self.filepath,
                encoding=encoding,
                delimiter=delimiter,
                dtype=dtype
            )
            logger.info(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
            return self.df

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def load_in_chunks(
        self,
        chunk_size: int = 10000,
        processor: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Load large CSV file in chunks.

        Args:
            chunk_size: Number of rows per chunk
            processor: Function to process each chunk

        Returns:
            Processed DataFrame

        Example:
            >>> processor = CSVProcessor("large_data.csv")
            >>> df = processor.load_in_chunks(chunk_size=5000)
        """
        logger.info(f"Loading CSV in chunks of {chunk_size}")

        chunks = []
        for chunk in pd.read_csv(self.filepath, chunksize=chunk_size):
            if processor:
                chunk = processor(chunk)
            chunks.append(chunk)

        self.df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(self.df)} total rows")
        return self.df

    def get_quality_report(self) -> DataQualityReport:
        """
        Generate data quality report.

        Returns:
            Data quality report

        Example:
            >>> processor = CSVProcessor("data.csv")
            >>> processor.load()
            >>> report = processor.get_quality_report()
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")

        # Missing values
        missing = self.df.isnull().sum().to_dict()
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).to_dict()

        # Duplicates
        duplicates = self.df.duplicated().sum()

        # Data types
        dtypes = self.df.dtypes.astype(str).to_dict()

        # Unique values
        unique = {col: self.df[col].nunique() for col in self.df.columns}

        return DataQualityReport(
            total_rows=len(self.df),
            total_columns=len(self.df.columns),
            missing_values=missing,
            missing_percentage=missing_pct,
            duplicate_rows=int(duplicates),
            data_types=dtypes,
            unique_values=unique
        )

    def clean(
        self,
        drop_duplicates: bool = True,
        fill_missing: Optional[Dict[str, Any]] = None,
        drop_missing: bool = False,
        remove_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Clean DataFrame.

        Args:
            drop_duplicates: Remove duplicate rows
            fill_missing: Dictionary mapping columns to fill values
            drop_missing: Drop rows with missing values
            remove_columns: List of columns to remove

        Returns:
            Cleaned DataFrame

        Example:
            >>> processor = CSVProcessor("data.csv")
            >>> processor.load()
            >>> df_clean = processor.clean(drop_duplicates=True)
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")

        logger.info("Cleaning data...")
        original_rows = len(self.df)

        # Remove duplicates
        if drop_duplicates:
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            removed = before - len(self.df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")

        # Fill missing values
        if fill_missing:
            self.df = self.df.fillna(fill_missing)
            logger.info(f"Filled missing values for {len(fill_missing)} columns")

        # Drop missing values
        if drop_missing:
            before = len(self.df)
            self.df = self.df.dropna()
            removed = before - len(self.df)
            if removed > 0:
                logger.info(f"Dropped {removed} rows with missing values")

        # Remove columns
        if remove_columns:
            self.df = self.df.drop(columns=remove_columns)
            logger.info(f"Removed {len(remove_columns)} columns")

        logger.info(f"Cleaned data: {original_rows} -> {len(self.df)} rows")
        return self.df

    def filter(
        self,
        conditions: Optional[Dict[str, Any]] = None,
        query_string: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter DataFrame based on conditions.

        Args:
            conditions: Dictionary of column:value conditions
            query_string: Pandas query string

        Returns:
            Filtered DataFrame

        Example:
            >>> processor = CSVProcessor("data.csv")
            >>> processor.load()
            >>> df = processor.filter(conditions={'age': lambda x: x > 25})
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")

        if conditions:
            for col, condition in conditions.items():
                if callable(condition):
                    self.df = self.df[self.df[col].apply(condition)]
                else:
                    self.df = self.df[self.df[col] == condition]

        if query_string:
            self.df = self.df.query(query_string)

        logger.info(f"Filtered to {len(self.df)} rows")
        return self.df

    def transform(
        self,
        column_transformations: Dict[str, Callable] = None,
        new_columns: Dict[str, Callable] = None
    ) -> pd.DataFrame:
        """
        Transform columns in DataFrame.

        Args:
            column_transformations: Dict mapping columns to transformation functions
            new_columns: Dict mapping new column names to creation functions

        Returns:
            Transformed DataFrame

        Example:
            >>> processor = CSVProcessor("data.csv")
            >>> processor.load()
            >>> df = processor.transform(
            ...     column_transformations={'price': lambda x: x * 1.1},
            ...     new_columns={'total': lambda df: df['price'] * df['quantity']}
            ... )
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")

        # Transform existing columns
        if column_transformations:
            for col, transform_func in column_transformations.items():
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(transform_func)
                    logger.info(f"Transformed column: {col}")

        # Create new columns
        if new_columns:
            for col, create_func in new_columns.items():
                self.df[col] = create_func(self.df)
                logger.info(f"Created column: {col}")

        return self.df

    def aggregate(
        self,
        group_by: List[str],
        aggregations: Dict[str, Union[str, List[str]]]
    ) -> pd.DataFrame:
        """
        Aggregate data by groups.

        Args:
            group_by: Columns to group by
            aggregations: Dictionary mapping columns to aggregation functions

        Returns:
            Aggregated DataFrame

        Example:
            >>> processor = CSVProcessor("sales.csv")
            >>> processor.load()
            >>> agg = processor.aggregate(
            ...     group_by=['region'],
            ...     aggregations={'sales': ['sum', 'mean'], 'quantity': 'sum'}
            ... )
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")

        result = self.df.groupby(group_by).agg(aggregations)
        logger.info(f"Aggregated into {len(result)} groups")
        return result

    def sort(
        self,
        by: Union[str, List[str]],
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Sort DataFrame.

        Args:
            by: Column(s) to sort by
            ascending: Sort in ascending order

        Returns:
            Sorted DataFrame

        Example:
            >>> processor = CSVProcessor("data.csv")
            >>> processor.load()
            >>> df = processor.sort(by='age', ascending=False)
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")

        self.df = self.df.sort_values(by=by, ascending=ascending)
        logger.info(f"Sorted by {by}")
        return self.df

    def save(
        self,
        output_path: str,
        format: str = 'csv',
        **kwargs
    ) -> None:
        """
        Save DataFrame to file.

        Args:
            output_path: Output file path
            format: Output format ('csv', 'json', 'parquet', 'excel')
            **kwargs: Additional arguments for to_csv, to_json, etc.

        Example:
            >>> processor = CSVProcessor("data.csv")
            >>> processor.load()
            >>> processor.save("output.csv")
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")

        output_path = Path(output_path)

        if format == 'csv':
            self.df.to_csv(output_path, index=False, **kwargs)
        elif format == 'json':
            self.df.to_json(output_path, **kwargs)
        elif format == 'parquet':
            self.df.to_parquet(output_path, **kwargs)
        elif format == 'excel':
            self.df.to_excel(output_path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved to {output_path} ({format})")


class CSVMerger:
    """Merge multiple CSV files."""

    @staticmethod
    def merge_files(
        filepaths: List[str],
        how: str = 'outer',
        on: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Merge multiple CSV files.

        Args:
            filepaths: List of CSV file paths
            how: Merge type ('inner', 'outer', 'left', 'right')
            on: Columns to merge on

        Returns:
            Merged DataFrame

        Example:
            >>> files = ["data1.csv", "data2.csv", "data3.csv"]
            >>> merged = CSVMerger.merge_files(files, how='outer')
        """
        if not filepaths:
            raise ValueError("No files provided")

        dfs = [pd.read_csv(fp) for fp in filepaths]
        logger.info(f"Loaded {len(dfs)} files")

        if on:
            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df, how=how, on=on)
        else:
            result = pd.concat(dfs, ignore_index=True)

        logger.info(f"Merged into {len(result)} rows")
        return result

    @staticmethod
    def concatenate_files(filepaths: List[str]) -> pd.DataFrame:
        """
        Concatenate multiple CSV files vertically.

        Args:
            filepaths: List of CSV file paths

        Returns:
            Concatenated DataFrame

        Example:
            >>> files = ["jan.csv", "feb.csv", "mar.csv"]
            >>> combined = CSVMerger.concatenate_files(files)
        """
        dfs = [pd.read_csv(fp) for fp in filepaths]
        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Concatenated {len(dfs)} files into {len(result)} rows")
        return result


if __name__ == '__main__':
    print("=== CSV Processor Examples ===\n")

    # Create sample data
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        temp_file = f.name
        f.write("name,age,city,salary\n")
        f.write("Alice,30,NYC,100000\n")
        f.write("Bob,25,LA,80000\n")
        f.write("Charlie,35,Chicago,120000\n")
        f.write("Alice,30,NYC,100000\n")  # Duplicate

    try:
        # Load and process
        processor = CSVProcessor(temp_file)
        processor.load()

        print("1. Data Quality Report:")
        report = processor.get_quality_report()
        print(f"   Total rows: {report.total_rows}")
        print(f"   Duplicate rows: {report.duplicate_rows}")

        print("\n2. Clean Data:")
        processor.clean(drop_duplicates=True)
        print(f"   Rows after cleaning: {len(processor.df)}")

        print("\n3. Transform Data:")
        processor.transform(
            column_transformations={'salary': lambda x: x * 1.1},
            new_columns={'bonus': lambda df: df['salary'] * 0.1}
        )
        print(f"   Columns: {list(processor.df.columns)}")

        print("\n4. Aggregate Data:")
        agg = processor.aggregate(
            group_by=['city'],
            aggregations={'salary': ['mean', 'count']}
        )
        print(f"   Aggregation results:\n{agg}")

    finally:
        import os
        os.unlink(temp_file)

    print("\n✓ CSV processing examples completed")
