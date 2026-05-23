"""Feature engineering module for creating ML features."""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for machine learning."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineer.

        Args:
            config: Configuration for feature engineering
        """
        self.config = config or {}
        self.pca = None
        self.feature_selector = None
        logger.info("FeatureEngineer initialized")

    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """Create polynomial features.

        Args:
            df: Input DataFrame
            columns: Columns to create polynomials for
            degree: Polynomial degree

        Returns:
            DataFrame with polynomial features added
        """
        df = df.copy()
        logger.info(f"Creating polynomial features (degree={degree}) for {len(columns)} columns")

        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                for d in range(2, degree + 1):
                    df[f"{col}_pow{d}"] = df[col] ** d

        return df

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[tuple]
    ) -> pd.DataFrame:
        """Create interaction features between column pairs.

        Args:
            df: Input DataFrame
            column_pairs: List of (col1, col2) tuples

        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()
        logger.info(f"Creating {len(column_pairs)} interaction features")

        for col1, col2 in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                    df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-6)

        return df

    def create_aggregation_features(
        self,
        df: pd.DataFrame,
        group_by: str,
        agg_columns: List[str],
        agg_funcs: List[str] = None
    ) -> pd.DataFrame:
        """Create aggregation features.

        Args:
            df: Input DataFrame
            group_by: Column to group by
            agg_columns: Columns to aggregate
            agg_funcs: Aggregation functions (default: mean, std, min, max)

        Returns:
            DataFrame with aggregation features added
        """
        if agg_funcs is None:
            agg_funcs = ['mean', 'std', 'min', 'max']

        df = df.copy()
        logger.info(f"Creating aggregation features grouped by '{group_by}'")

        for col in agg_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                for func in agg_funcs:
                    agg_result = df.groupby(group_by)[col].transform(func)
                    df[f"{col}_{func}_by_{group_by}"] = agg_result

        return df

    def create_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Extract time-based features from date column.

        Args:
            df: Input DataFrame
            date_column: Name of datetime column

        Returns:
            DataFrame with time features added
        """
        df = df.copy()
        logger.info(f"Creating time features from '{date_column}'")

        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df[f"{date_column}_year"] = df[date_column].dt.year
            df[f"{date_column}_month"] = df[date_column].dt.month
            df[f"{date_column}_day"] = df[date_column].dt.day
            df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek
            df[f"{date_column}_quarter"] = df[date_column].dt.quarter
            df[f"{date_column}_dayofyear"] = df[date_column].dt.dayofyear
            df[f"{date_column}_is_weekend"] = (df[date_column].dt.dayofweek >= 5).astype(int)

        return df

    def apply_pca(
        self,
        df: pd.DataFrame,
        n_components: Optional[int] = None,
        variance_ratio: float = 0.95,
        fit: bool = True
    ) -> pd.DataFrame:
        """Apply PCA dimensionality reduction.

        Args:
            df: Input DataFrame (numeric only)
            n_components: Number of components (None = auto based on variance)
            variance_ratio: Minimum variance to preserve
            fit: Whether to fit PCA or just transform

        Returns:
            DataFrame with PCA components
        """
        logger.info("Applying PCA dimensionality reduction")

        if fit:
            if n_components is None:
                self.pca = PCA(n_components=variance_ratio)
            else:
                self.pca = PCA(n_components=n_components)

            components = self.pca.fit_transform(df)
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted yet")
            components = self.pca.transform(df)

        # Create DataFrame with PCA components
        pca_df = pd.DataFrame(
            components,
            columns=[f"pca_{i}" for i in range(components.shape[1])],
            index=df.index
        )

        logger.info(f"PCA reduced {df.shape[1]} features to {pca_df.shape[1]} components")

        return pca_df

    def select_k_best_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        fit: bool = True
    ) -> pd.DataFrame:
        """Select k best features using statistical tests.

        Args:
            X: Feature DataFrame
            y: Target variable
            k: Number of features to select
            fit: Whether to fit selector or just transform

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting {k} best features")

        if fit:
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_columns = X.columns[self.feature_selector.get_support()].tolist()
        else:
            if self.feature_selector is None:
                raise ValueError("Feature selector not fitted yet")
            X_selected = self.feature_selector.transform(X)
            selected_columns = X.columns[self.feature_selector.get_support()].tolist()

        result_df = pd.DataFrame(X_selected, columns=selected_columns, index=X.index)
        logger.info(f"Selected features: {selected_columns}")

        return result_df

    def transform(
        self,
        df: pd.DataFrame,
        create_polynomials: bool = False,
        polynomial_degree: int = 2,
        create_interactions: bool = False,
        apply_pca_transform: bool = False,
        pca_components: Optional[int] = None
    ) -> pd.DataFrame:
        """Apply complete feature engineering pipeline.

        Args:
            df: Input DataFrame
            create_polynomials: Whether to create polynomial features
            polynomial_degree: Degree for polynomial features
            create_interactions: Whether to create interaction features
            apply_pca_transform: Whether to apply PCA
            pca_components: Number of PCA components

        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Starting feature engineering: {len(df.columns)} features")

        result_df = df.copy()

        # Create polynomial features
        if create_polynomials:
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            result_df = self.create_polynomial_features(
                result_df,
                numeric_cols[:5],  # Limit to first 5 to avoid explosion
                degree=polynomial_degree
            )

        # Create interaction features
        if create_interactions:
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                pairs = [(numeric_cols[i], numeric_cols[i + 1])
                         for i in range(min(3, len(numeric_cols) - 1))]
                result_df = self.create_interaction_features(result_df, pairs)

        # Apply PCA
        if apply_pca_transform:
            numeric_df = result_df.select_dtypes(include=[np.number])
            pca_df = self.apply_pca(numeric_df, n_components=pca_components, fit=True)
            result_df = pd.concat([result_df.select_dtypes(exclude=[np.number]), pca_df], axis=1)

        logger.info(f"Feature engineering complete: {len(result_df.columns)} features")

        return result_df
