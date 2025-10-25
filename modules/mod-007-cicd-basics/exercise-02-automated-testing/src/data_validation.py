"""Data Validation Module"""

import logging
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_schema(
    df: pd.DataFrame,
    expected_schema: Dict[str, str],
) -> Tuple[bool, List[str]]:
    """
    Validate dataframe schema.

    Args:
        df: Input dataframe
        expected_schema: Dict of {column_name: expected_dtype}

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check for missing columns
    expected_cols = set(expected_schema.keys())
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Check for extra columns
    extra = actual_cols - expected_cols
    if extra:
        errors.append(f"Extra columns: {extra}")

    # Check data types
    for col, expected_dtype in expected_schema.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if expected_dtype not in actual_dtype:
                errors.append(f"Column '{col}': expected {expected_dtype}, got {actual_dtype}")

    is_valid = len(errors) == 0
    return is_valid, errors


def check_missing_values(df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, float]:
    """Check for missing values."""
    missing = df.isnull().sum() / len(df)
    problematic = missing[missing > threshold].to_dict()

    if problematic:
        logger.warning(f"Columns with >{threshold:.1%} missing: {problematic}")

    return problematic


def check_value_ranges(
    df: pd.DataFrame,
    ranges: Dict[str, Tuple[float, float]],
) -> List[str]:
    """Check if values are within expected ranges."""
    violations = []

    for col, (min_val, max_val) in ranges.items():
        if col not in df.columns:
            continue

        out_of_range = (df[col] < min_val) | (df[col] > max_val)
        count = out_of_range.sum()

        if count > 0:
            violations.append(f"{col}: {count} values out of range [{min_val}, {max_val}]")

    return violations


def check_duplicates(df: pd.DataFrame) -> int:
    """Check for duplicate rows."""
    dup_count = df.duplicated().sum()

    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate rows")

    return dup_count


def check_data_drift(
    df_new: pd.DataFrame,
    df_reference: pd.DataFrame,
    columns: List[str] = None,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """Check for data drift using KS test."""
    from scipy.stats import ks_2samp

    if columns is None:
        columns = df_new.select_dtypes(include=[np.number]).columns

    drift_detected = {}

    for col in columns:
        if col not in df_new.columns or col not in df_reference.columns:
            continue

        statistic, p_value = ks_2samp(df_new[col].dropna(), df_reference[col].dropna())

        if p_value < threshold:
            drift_detected[col] = p_value
            logger.warning(f"Data drift detected in '{col}': p-value={p_value:.4f}")

    return drift_detected
