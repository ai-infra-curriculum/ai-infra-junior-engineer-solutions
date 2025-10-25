"""Data preprocessing module."""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and clean data for ML pipelines."""

    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        logger.info("DataPreprocessor initialized")

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "mean",
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """Handle missing values in DataFrame.

        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                     ("mean", "median", "mode", "drop", "fill")
            fill_value: Value to use when strategy is "fill"

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        logger.info(f"Handling missing values using strategy: {strategy}")

        if strategy == "drop":
            return df.dropna()
        elif strategy == "fill" and fill_value is not None:
            return df.fillna(fill_value)
        elif strategy in ["mean", "median"]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        elif strategy == "mode":
            for col in df.columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def remove_duplicates(self, df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
        """Remove duplicate rows.

        Args:
            df: Input DataFrame
            keep: Which duplicates to keep ("first", "last", False)

        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Removing duplicate rows")
        n_before = len(df)
        df = df.drop_duplicates(keep=keep)
        n_after = len(df)
        logger.info(f"Removed {n_before - n_after} duplicate rows")
        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Encode categorical variables.

        Args:
            df: Input DataFrame
            columns: List of columns to encode (None = auto-detect)

        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        logger.info(f"Encoding {len(columns)} categorical columns")

        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))

        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """Scale numeric features.

        Args:
            df: Input DataFrame
            columns: List of columns to scale (None = all numeric)
            fit: Whether to fit the scaler or just transform

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Scaling {len(columns)} numeric columns")

        if fit:
            df[columns] = self.scaler.fit_transform(df[columns])
        else:
            df[columns] = self.scaler.transform(df[columns])

        return df

    def process(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_missing: bool = True,
        encode_categorical: bool = True,
        scale_features: bool = True
    ) -> pd.DataFrame:
        """Run complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicates
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical variables
            scale_features: Whether to scale numeric features

        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Starting preprocessing: {len(df)} rows, {len(df.columns)} columns")

        if remove_duplicates:
            df = self.remove_duplicates(df)

        if handle_missing:
            df = self.handle_missing_values(df, strategy="mean")

        if encode_categorical:
            df = self.encode_categorical(df)

        if scale_features:
            df = self.scale_features(df)

        logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        return df
