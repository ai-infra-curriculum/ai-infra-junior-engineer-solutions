"""
Data Preprocessing Module

Functions for cleaning and preparing data for ML models.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def clean_data(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    drop_na_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Clean data by removing duplicates and handling missing values.

    Args:
        df: Input dataframe
        remove_duplicates: Whether to remove duplicate rows
        drop_na_threshold: Drop columns with more than this fraction of NaNs

    Returns:
        Cleaned dataframe
    """
    logger.info(f"Cleaning data with shape {df.shape}")

    df_clean = df.copy()

    # Remove duplicates
    if remove_duplicates:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = before - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")

    # Drop columns with too many missing values
    threshold = int(len(df_clean) * drop_na_threshold)
    df_clean = df_clean.dropna(thresh=threshold, axis=1)

    logger.info(f"Cleaned data shape: {df_clean.shape}")
    return df_clean


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Handle missing values in dataframe.

    Args:
        df: Input dataframe
        strategy: Imputation strategy ('mean', 'median', 'mode', 'forward_fill', 'drop')
        columns: Columns to impute (None = all numeric columns)

    Returns:
        Dataframe with imputed values
    """
    if df.isnull().sum().sum() == 0:
        logger.info("No missing values found")
        return df

    df_filled = df.copy()

    if columns is None:
        columns = df_filled.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"Handling missing values with strategy: {strategy}")

    for col in columns:
        if col not in df_filled.columns:
            continue

        missing_count = df_filled[col].isnull().sum()
        if missing_count == 0:
            continue

        logger.info(f"Imputing {missing_count} missing values in '{col}'")

        if strategy == "mean":
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
        elif strategy == "median":
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
        elif strategy == "mode":
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
        elif strategy == "forward_fill":
            df_filled[col].fillna(method="ffill", inplace=True)
        elif strategy == "drop":
            df_filled = df_filled.dropna(subset=[col])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return df_filled


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers from dataframe.

    Args:
        df: Input dataframe
        columns: Columns to check for outliers (None = all numeric)
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: IQR multiplier or z-score threshold

    Returns:
        Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    df_clean = df.copy()
    mask = pd.Series([True] * len(df_clean))

    for col in columns:
        if col not in df_clean.columns:
            continue

        if method == "iqr":
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            col_mask = (df_clean[col] >= lower) & (df_clean[col] <= upper)
        elif method == "zscore":
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            col_mask = z_scores < threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        outliers_removed = (~col_mask).sum()
        if outliers_removed > 0:
            logger.info(f"Removing {outliers_removed} outliers from '{col}'")

        mask &= col_mask

    df_clean = df_clean[mask]
    logger.info(f"Removed {(~mask).sum()} total outlier rows")

    return df_clean


def normalize_data(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "standard",
) -> tuple[pd.DataFrame, Union[StandardScaler, Dict]]:
    """
    Normalize numerical columns.

    Args:
        df: Input dataframe
        columns: Columns to normalize (None = all numeric)
        method: Normalization method ('standard', 'minmax', 'robust')

    Returns:
        Tuple of (normalized dataframe, scaler object)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    df_norm = df.copy()

    if method == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_norm[columns] = scaler.fit_transform(df_norm[columns])
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df_norm[columns] = scaler.fit_transform(df_norm[columns])
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        df_norm[columns] = scaler.fit_transform(df_norm[columns])
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Normalized {len(columns)} columns using {method} scaling")

    return df_norm, scaler


def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "onehot",
) -> tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables.

    Args:
        df: Input dataframe
        columns: Columns to encode (None = all object columns)
        method: Encoding method ('onehot', 'label', 'target')

    Returns:
        Tuple of (encoded dataframe, encoder dict)
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    df_encoded = df.copy()
    encoders = {}

    for col in columns:
        if col not in df_encoded.columns:
            continue

        logger.info(f"Encoding '{col}' with {method} encoding")

        if method == "onehot":
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            encoders[col] = list(dummies.columns)

        elif method == "label":
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
            encoders[col] = encoder

        else:
            raise ValueError(f"Unknown method: {method}")

    return df_encoded, encoders


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    Args:
        df: Input dataframe
        target_column: Name of target column

    Returns:
        Tuple of (features dataframe, target series)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info(f"Split data: Features shape {X.shape}, Target shape {y.shape}")

    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        X: Features
        y: Target
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split by target

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    logger.info(
        f"Split data: Train={len(X_train)}, Test={len(X_test)}, "
        f"Test ratio={test_size:.2%}"
    )

    return X_train, X_test, y_train, y_test
