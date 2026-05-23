"""
Tests for ML Model
Comprehensive test suite for the MLModel class
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from ml_model import (
    MLModel,
    create_train_test_split,
    load_sample_data,
)


class TestMLModel:
    """Test suite for MLModel class."""

    @pytest.fixture
    def sample_data(self):
        """Fixture to provide sample data for tests."""
        X, y, _ = load_sample_data()
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)
        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def trained_model(self, sample_data):
        """Fixture to provide a trained model."""
        X_train, _, y_train, _ = sample_data
        model = MLModel(n_estimators=10, max_depth=3, random_state=42)
        model.train(X_train, y_train)
        return model

    def test_model_initialization(self):
        """Test that model initializes with correct parameters."""
        model = MLModel(n_estimators=50, max_depth=10, random_state=123)

        assert model.n_estimators == 50
        assert model.max_depth == 10
        assert model.random_state == 123
        assert not model.is_trained()
        assert model.model is None

    def test_model_initialization_defaults(self):
        """Test that model uses default parameters correctly."""
        model = MLModel()

        assert model.n_estimators == 100
        assert model.max_depth is None
        assert model.random_state == 42

    def test_train_success(self, sample_data):
        """Test that model trains successfully."""
        X_train, _, y_train, _ = sample_data
        model = MLModel()

        result = model.train(X_train, y_train)

        assert model.is_trained()
        assert model.model is not None
        assert result is model  # Check method chaining

    def test_train_empty_data(self):
        """Test that training with empty data raises ValueError."""
        model = MLModel()
        X_empty = np.array([]).reshape(0, 4)
        y_empty = np.array([])

        with pytest.raises(ValueError, match="Training data cannot be empty"):
            model.train(X_empty, y_empty)

    def test_train_mismatched_shapes(self):
        """Test that training with mismatched X and y raises ValueError."""
        model = MLModel()
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 50)  # Different size

        with pytest.raises(ValueError, match="must have the same number of samples"):
            model.train(X, y)

    def test_predict_success(self, trained_model, sample_data):
        """Test that model makes predictions successfully."""
        _, X_test, _, _ = sample_data

        predictions = trained_model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1, 2] for pred in predictions)  # Valid classes

    def test_predict_before_training(self):
        """Test that predicting before training raises RuntimeError."""
        model = MLModel()
        X = np.random.rand(10, 4)

        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.predict(X)

    def test_predict_empty_data(self, trained_model):
        """Test that predicting with empty data raises ValueError."""
        X_empty = np.array([]).reshape(0, 4)

        with pytest.raises(ValueError, match="Input data cannot be empty"):
            trained_model.predict(X_empty)

    def test_predict_proba_success(self, trained_model, sample_data):
        """Test that model predicts probabilities successfully."""
        _, X_test, _, _ = sample_data

        probabilities = trained_model.predict_proba(X_test)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_test), 3)  # 3 classes
        # Check probabilities sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        # Check all probabilities are between 0 and 1
        assert np.all((probabilities >= 0) & (probabilities <= 1))

    def test_predict_proba_before_training(self):
        """Test that predicting probabilities before training raises error."""
        model = MLModel()
        X = np.random.rand(10, 4)

        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.predict_proba(X)

    def test_evaluate_success(self, trained_model, sample_data):
        """Test that model evaluates successfully."""
        _, X_test, _, y_test = sample_data

        accuracy, report = trained_model.evaluate(X_test, y_test)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report

    def test_evaluate_before_training(self):
        """Test that evaluating before training raises RuntimeError."""
        model = MLModel()
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 3, 10)

        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.evaluate(X, y)

    def test_get_feature_importance(self, trained_model):
        """Test that feature importance is retrieved correctly."""
        importances = trained_model.get_feature_importance()

        assert isinstance(importances, np.ndarray)
        assert len(importances) == 4  # Iris has 4 features
        assert all(imp >= 0 for imp in importances)
        # Importances should sum to approximately 1
        assert np.isclose(importances.sum(), 1.0, atol=0.01)

    def test_get_feature_importance_before_training(self):
        """Test that getting feature importance before training raises error."""
        model = MLModel()

        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.get_feature_importance()

    def test_is_trained_false(self):
        """Test that is_trained returns False for untrained model."""
        model = MLModel()
        assert not model.is_trained()

    def test_is_trained_true(self, trained_model):
        """Test that is_trained returns True for trained model."""
        assert trained_model.is_trained()


class TestDataFunctions:
    """Test suite for data utility functions."""

    def test_load_sample_data(self):
        """Test that sample data loads correctly."""
        X, y, feature_names = load_sample_data()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
        assert X.shape == (150, 4)  # Iris dataset
        assert y.shape == (150,)
        assert len(feature_names) == 4

    def test_create_train_test_split(self):
        """Test that data splitting works correctly."""
        X, y, _ = load_sample_data()

        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Check shapes
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        # Check split ratio
        test_ratio = len(X_test) / len(X)
        assert abs(test_ratio - 0.2) < 0.05  # Allow 5% tolerance

    def test_create_train_test_split_different_sizes(self):
        """Test data splitting with different test sizes."""
        X, y, _ = load_sample_data()

        for test_size in [0.1, 0.3, 0.5]:
            X_train, X_test, y_train, y_test = create_train_test_split(
                X, y, test_size=test_size
            )

            test_ratio = len(X_test) / len(X)
            assert abs(test_ratio - test_size) < 0.05


class TestModelIntegration:
    """Integration tests for the complete ML workflow."""

    def test_end_to_end_workflow(self):
        """Test the complete training and prediction workflow."""
        # Load data
        X, y, feature_names = load_sample_data()

        # Split data
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)

        # Create and train model
        model = MLModel(n_estimators=50, max_depth=5, random_state=42)
        model.train(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Evaluate
        accuracy, report = model.evaluate(X_test, y_test)

        # Get feature importance
        importances = model.get_feature_importance()

        # Assertions
        assert model.is_trained()
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 3)
        assert accuracy > 0.7  # Model should have reasonable accuracy
        assert "precision" in report
        assert len(importances) == len(feature_names)

    def test_reproducibility(self):
        """Test that model training is reproducible with same random state."""
        X, y, _ = load_sample_data()
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, random_state=42
        )

        # Train two models with same random state
        model1 = MLModel(n_estimators=50, random_state=42)
        model1.train(X_train, y_train)
        pred1 = model1.predict(X_test)

        model2 = MLModel(n_estimators=50, random_state=42)
        model2.train(X_train, y_train)
        pred2 = model2.predict(X_test)

        # Predictions should be identical
        assert np.array_equal(pred1, pred2)

    def test_model_with_different_hyperparameters(self):
        """Test models with different hyperparameters."""
        X, y, _ = load_sample_data()
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)

        # Test different hyperparameters
        configs = [
            {"n_estimators": 10, "max_depth": 3},
            {"n_estimators": 100, "max_depth": 10},
            {"n_estimators": 50, "max_depth": None},
        ]

        for config in configs:
            model = MLModel(**config, random_state=42)
            model.train(X_train, y_train)
            accuracy, _ = model.evaluate(X_test, y_test)

            assert model.is_trained()
            assert 0.0 <= accuracy <= 1.0


# Performance benchmark tests
@pytest.mark.benchmark
class TestModelPerformance:
    """Performance benchmark tests."""

    def test_training_performance(self, benchmark):
        """Benchmark model training performance."""
        X, y, _ = load_sample_data()
        X_train, _, y_train, _ = create_train_test_split(X, y)

        model = MLModel(n_estimators=100, random_state=42)

        def train():
            model.train(X_train, y_train)

        benchmark(train)

    def test_prediction_performance(self, benchmark):
        """Benchmark model prediction performance."""
        X, y, _ = load_sample_data()
        X_train, X_test, y_train, _ = create_train_test_split(X, y)

        model = MLModel(n_estimators=100, random_state=42)
        model.train(X_train, y_train)

        benchmark(model.predict, X_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=ml_model", "--cov-report=term"])
