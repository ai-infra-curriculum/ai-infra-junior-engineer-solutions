"""
Feature Engineering Module

Functions for creating and transforming features for ML models.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2,
    include_bias: bool = False,
) -> pd.DataFrame:
    """
    Create polynomial features.

    Args:
        df: Input dataframe
        columns: Columns to create polynomial features from
        degree: Polynomial degree
        include_bias: Whether to include bias column

    Returns:
        Dataframe with polynomial features added
    """
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_features = poly.fit_transform(df[columns])

    # Create feature names
    feature_names = poly.get_feature_names_out(columns)

    # Create new dataframe with polynomial features
    df_poly = pd.DataFrame(
        poly_features,
        columns=feature_names,
        index=df.index
    )

    # Combine with original dataframe (excluding original columns)
    df_result = pd.concat([
        df.drop(columns=columns),
        df_poly
    ], axis=1)

    logger.info(f"Created {len(feature_names)} polynomial features (degree={degree})")

    return df_result


def create_interaction_features(
    df: pd.DataFrame,
    column_pairs: List[tuple[str, str]],
    operations: List[str] = ["multiply"],
) -> pd.DataFrame:
    """
    Create interaction features between column pairs.

    Args:
        df: Input dataframe
        column_pairs: List of (col1, col2) tuples
        operations: List of operations ('multiply', 'add', 'subtract', 'divide')

    Returns:
        Dataframe with interaction features added
    """
    df_result = df.copy()

    for col1, col2 in column_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            logger.warning(f"Columns '{col1}' or '{col2}' not found, skipping")
            continue

        for op in operations:
            if op == "multiply":
                df_result[f"{col1}_x_{col2}"] = df[col1] * df[col2]
            elif op == "add":
                df_result[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
            elif op == "subtract":
                df_result[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
            elif op == "divide":
                # Avoid division by zero
                df_result[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)
            else:
                logger.warning(f"Unknown operation: {op}")

    added = len(df_result.columns) - len(df.columns)
    logger.info(f"Created {added} interaction features")

    return df_result


def create_binned_features(
    df: pd.DataFrame,
    column: str,
    bins: int = 5,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create binned (discretized) features.

    Args:
        df: Input dataframe
        column: Column to bin
        bins: Number of bins or list of bin edges
        labels: Labels for bins

    Returns:
        Dataframe with binned feature added
    """
    df_result = df.copy()

    try:
        df_result[f"{column}_binned"] = pd.cut(
            df[column],
            bins=bins,
            labels=labels,
            duplicates='drop'
        )

        logger.info(f"Created binned feature for '{column}' with {bins} bins")
    except Exception as e:
        logger.error(f"Failed to bin column '{column}': {e}")
        raise

    return df_result


def create_date_features(
    df: pd.DataFrame,
    date_column: str,
    features: List[str] = ["year", "month", "day", "dayofweek"],
) -> pd.DataFrame:
    """
    Extract features from datetime column.

    Args:
        df: Input dataframe
        date_column: Name of datetime column
        features: List of features to extract

    Returns:
        Dataframe with date features added
    """
    df_result = df.copy()

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df_result[date_column]):
        df_result[date_column] = pd.to_datetime(df_result[date_column])

    dt = df_result[date_column].dt

    feature_map = {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "dayofweek": dt.dayofweek,
        "dayofyear": dt.dayofyear,
        "quarter": dt.quarter,
        "weekofyear": dt.isocalendar().week,
        "hour": dt.hour,
        "minute": dt.minute,
        "is_weekend": (dt.dayofweek >= 5).astype(int),
        "is_month_start": dt.is_month_start.astype(int),
        "is_month_end": dt.is_month_end.astype(int),
    }

    for feature in features:
        if feature in feature_map:
            df_result[f"{date_column}_{feature}"] = feature_map[feature]
            logger.info(f"Created date feature: {date_column}_{feature}")
        else:
            logger.warning(f"Unknown date feature: {feature}")

    return df_result


def create_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: List[int],
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create lagged features for time series.

    Args:
        df: Input dataframe
        column: Column to create lags from
        lags: List of lag periods
        group_by: Column to group by (for panel data)

    Returns:
        Dataframe with lag features added
    """
    df_result = df.copy()

    for lag in lags:
        if group_by:
            df_result[f"{column}_lag_{lag}"] = df_result.groupby(group_by)[column].shift(lag)
        else:
            df_result[f"{column}_lag_{lag}"] = df_result[column].shift(lag)

        logger.info(f"Created lag feature: {column}_lag_{lag}")

    return df_result


def create_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: List[int],
    functions: List[str] = ["mean"],
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create rolling window features.

    Args:
        df: Input dataframe
        column: Column to compute rolling features from
        windows: List of window sizes
        functions: List of aggregation functions
        group_by: Column to group by

    Returns:
        Dataframe with rolling features added
    """
    df_result = df.copy()

    func_map = {
        "mean": "mean",
        "std": "std",
        "min": "min",
        "max": "max",
        "sum": "sum",
    }

    for window in windows:
        for func in functions:
            if func not in func_map:
                logger.warning(f"Unknown function: {func}")
                continue

            feature_name = f"{column}_rolling_{func}_{window}"

            if group_by:
                df_result[feature_name] = (
                    df_result.groupby(group_by)[column]
                    .rolling(window=window, min_periods=1)
                    .agg(func_map[func])
                    .reset_index(level=0, drop=True)
                )
            else:
                df_result[feature_name] = (
                    df_result[column]
                    .rolling(window=window, min_periods=1)
                    .agg(func_map[func])
                )

            logger.info(f"Created rolling feature: {feature_name}")

    return df_result


def create_text_features(
    df: pd.DataFrame,
    text_column: str,
    features: List[str] = ["length", "word_count"],
) -> pd.DataFrame:
    """
    Extract features from text column.

    Args:
        df: Input dataframe
        text_column: Name of text column
        features: List of features to extract

    Returns:
        Dataframe with text features added
    """
    df_result = df.copy()

    text_series = df_result[text_column].fillna("").astype(str)

    if "length" in features:
        df_result[f"{text_column}_length"] = text_series.str.len()
        logger.info(f"Created text feature: {text_column}_length")

    if "word_count" in features:
        df_result[f"{text_column}_word_count"] = text_series.str.split().str.len()
        logger.info(f"Created text feature: {text_column}_word_count")

    if "char_count" in features:
        df_result[f"{text_column}_char_count"] = text_series.str.replace(" ", "").str.len()

    if "unique_words" in features:
        df_result[f"{text_column}_unique_words"] = text_series.str.split().apply(
            lambda x: len(set(x)) if isinstance(x, list) else 0
        )

    if "avg_word_length" in features:
        df_result[f"{text_column}_avg_word_length"] = text_series.apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )

    return df_result


def select_features_by_correlation(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = 0.1,
    method: str = "pearson",
) -> List[str]:
    """
    Select features based on correlation with target.

    Args:
        df: Input dataframe
        target_column: Target column name
        threshold: Minimum absolute correlation to keep feature
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        List of selected feature names
    """
    # Calculate correlations
    correlations = df.corr(method=method)[target_column].abs()

    # Select features above threshold (excluding target itself)
    selected = correlations[
        (correlations >= threshold) & (correlations.index != target_column)
    ].sort_values(ascending=False)

    logger.info(f"Selected {len(selected)} features with |correlation| >= {threshold}")

    return selected.index.tolist()


def select_features_by_variance(
    df: pd.DataFrame,
    threshold: float = 0.01,
) -> List[str]:
    """
    Select features based on variance threshold.

    Args:
        df: Input dataframe
        threshold: Minimum variance to keep feature

    Returns:
        List of selected feature names
    """
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=threshold)

    # Only consider numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    selector.fit(df[numeric_cols])

    # Get selected features
    selected_mask = selector.get_support()
    selected_features = numeric_cols[selected_mask].tolist()

    logger.info(f"Selected {len(selected_features)} features with variance >= {threshold}")

    return selected_features
