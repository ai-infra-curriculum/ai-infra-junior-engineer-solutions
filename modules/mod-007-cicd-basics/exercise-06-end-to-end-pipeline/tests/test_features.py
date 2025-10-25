"""Tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test feature engineering."""

    def test_create_polynomial_features(self, sample_features):
        """Test polynomial feature creation."""
        engineer = FeatureEngineer()
        columns = sample_features.columns[:3].tolist()

        result = engineer.create_polynomial_features(
            sample_features,
            columns,
            degree=2
        )

        # Check that polynomial features were created
        assert len(result.columns) > len(sample_features.columns)
        assert any("_pow2" in col for col in result.columns)

    def test_create_interaction_features(self, sample_features):
        """Test interaction feature creation."""
        engineer = FeatureEngineer()
        pairs = [
            (sample_features.columns[0], sample_features.columns[1]),
            (sample_features.columns[2], sample_features.columns[3])
        ]

        result = engineer.create_interaction_features(sample_features, pairs)

        # Check that interaction features were created
        assert len(result.columns) > len(sample_features.columns)
        assert any("_x_" in col for col in result.columns)
        assert any("_div_" in col for col in result.columns)

    def test_create_aggregation_features(self):
        """Test aggregation feature creation."""
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B", "C", "C"],
            "value": [1, 2, 3, 4, 5, 6]
        })

        engineer = FeatureEngineer()
        result = engineer.create_aggregation_features(
            df,
            group_by="group",
            agg_columns=["value"],
            agg_funcs=["mean", "std"]
        )

        # Check that aggregation features were created
        assert "value_mean_by_group" in result.columns
        assert "value_std_by_group" in result.columns

    def test_create_time_features(self):
        """Test time feature creation."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
            "value": range(100)
        })

        engineer = FeatureEngineer()
        result = engineer.create_time_features(df, "date")

        # Check that time features were created
        assert "date_year" in result.columns
        assert "date_month" in result.columns
        assert "date_day" in result.columns
        assert "date_dayofweek" in result.columns
        assert "date_is_weekend" in result.columns

    def test_apply_pca(self, sample_features):
        """Test PCA dimensionality reduction."""
        engineer = FeatureEngineer()
        result = engineer.apply_pca(sample_features, n_components=5, fit=True)

        # Check that dimensions were reduced
        assert result.shape[1] == 5
        assert all("pca_" in col for col in result.columns)

    def test_select_k_best_features(self, sample_features, sample_target):
        """Test feature selection."""
        engineer = FeatureEngineer()
        result = engineer.select_k_best_features(
            sample_features,
            sample_target,
            k=10,
            fit=True
        )

        # Check that features were selected
        assert result.shape[1] == 10
        assert len(result) == len(sample_features)

    def test_transform_complete_pipeline(self, sample_features):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        result = engineer.transform(
            sample_features,
            create_polynomials=True,
            polynomial_degree=2,
            create_interactions=True,
            apply_pca_transform=False
        )

        # Check that features were created
        assert len(result.columns) >= len(sample_features.columns)

    def test_transform_with_pca(self, sample_features):
        """Test feature engineering with PCA."""
        engineer = FeatureEngineer()
        result = engineer.transform(
            sample_features,
            create_polynomials=False,
            create_interactions=False,
            apply_pca_transform=True,
            pca_components=10
        )

        # Check PCA was applied
        assert result.shape[1] == 10
