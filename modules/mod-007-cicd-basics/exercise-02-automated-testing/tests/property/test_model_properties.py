"""
Property-Based Tests for ML Models

Tests invariants and properties using hypothesis.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings

from src.data_preprocessing import normalize_data
from src.model_inference import predict


@pytest.mark.property
class TestModelProperties:
    """Property-based tests for model behavior."""

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000), min_size=5, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_normalization_bounds(self, values):
        """Property: Normalized values should be bounded."""
        df = pd.DataFrame({'feature': values})
        normalized, _ = normalize_data(df, method="minmax")

        # MinMax normalized values should be between 0 and 1
        assert (normalized['feature'] >= 0).all()
        assert (normalized['feature'] <= 1).all()

    def test_prediction_shape_invariance(self, trained_model):
        """Property: Prediction output shape should match input."""
        for n_samples in [1, 10, 100]:
            X = pd.DataFrame(np.random.rand(n_samples, 20), columns=[f'feature_{i}' for i in range(20)])
            predictions = predict(trained_model, X)

            assert len(predictions) == n_samples

    def test_model_output_type(self, trained_model):
        """Property: Model should always output valid class labels."""
        X = pd.DataFrame(np.random.rand(50, 20), columns=[f'feature_{i}' for i in range(20)])
        predictions = predict(trained_model, X)

        # Predictions should be integers
        assert predictions.dtype in [np.int32, np.int64]

        # All predictions should be valid classes (0 or 1)
        assert set(predictions).issubset({0, 1})
