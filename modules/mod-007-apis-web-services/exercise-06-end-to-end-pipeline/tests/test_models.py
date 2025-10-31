"""Tests for model training and evaluation modules."""

import pytest
import pandas as pd
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator


class TestModelTrainer:
    """Test model training."""

    def test_train_random_forest(self, sample_features, sample_target):
        """Test training random forest model."""
        trainer = ModelTrainer(
            experiment_name="test-rf",
            model_type="random_forest"
        )

        model, metrics = trainer.train(
            sample_features,
            sample_target,
            test_size=0.2,
            tune_hyperparameters=False
        )

        # Check model was trained
        assert model is not None
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5  # Should be better than random

    def test_train_gradient_boosting(self, sample_features, sample_target):
        """Test training gradient boosting model."""
        trainer = ModelTrainer(
            experiment_name="test-gb",
            model_type="gradient_boosting"
        )

        model, metrics = trainer.train(
            sample_features,
            sample_target,
            test_size=0.2,
            tune_hyperparameters=False
        )

        assert model is not None
        assert "accuracy" in metrics

    def test_train_logistic_regression(self, sample_features, sample_target):
        """Test training logistic regression model."""
        trainer = ModelTrainer(
            experiment_name="test-lr",
            model_type="logistic"
        )

        model, metrics = trainer.train(
            sample_features,
            sample_target,
            test_size=0.2,
            tune_hyperparameters=False
        )

        assert model is not None
        assert "accuracy" in metrics

    def test_train_with_hyperparameter_tuning(self, sample_features, sample_target):
        """Test training with hyperparameter tuning."""
        trainer = ModelTrainer(
            experiment_name="test-tuning",
            model_type="random_forest"
        )

        # Use smaller dataset for faster tuning
        X_small = sample_features.head(200)
        y_small = sample_target.head(200)

        model, metrics = trainer.train(
            X_small,
            y_small,
            test_size=0.2,
            tune_hyperparameters=True,
            cv_folds=3
        )

        assert model is not None
        assert "accuracy" in metrics

    def test_invalid_model_type(self, sample_features, sample_target):
        """Test invalid model type."""
        trainer = ModelTrainer(
            experiment_name="test-invalid",
            model_type="invalid_model"
        )

        with pytest.raises(ValueError):
            trainer.train(sample_features, sample_target)


class TestModelEvaluator:
    """Test model evaluation."""

    @pytest.fixture
    def trained_model(self, sample_features, sample_target):
        """Create a trained model for testing."""
        trainer = ModelTrainer(
            experiment_name="test-eval",
            model_type="random_forest"
        )
        model, _ = trainer.train(
            sample_features,
            sample_target,
            test_size=0.2,
            tune_hyperparameters=False
        )
        return model

    def test_evaluate(self, trained_model, sample_features, sample_target):
        """Test model evaluation."""
        evaluator = ModelEvaluator()

        # Use last 200 samples for testing
        X_test = sample_features.tail(200)
        y_test = sample_target.tail(200)

        metrics = evaluator.evaluate(
            trained_model,
            X_test,
            y_test,
            log_to_mlflow=False
        )

        # Check all metrics are present
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # Check metrics are in valid range
        for metric_name, metric_value in metrics.items():
            assert 0 <= metric_value <= 1

    def test_get_confusion_matrix(self, trained_model, sample_features, sample_target):
        """Test confusion matrix generation."""
        evaluator = ModelEvaluator()

        X_test = sample_features.tail(200)
        y_test = sample_target.tail(200)

        cm = evaluator.get_confusion_matrix(trained_model, X_test, y_test)

        # Check confusion matrix shape
        assert cm.shape == (2, 2)  # Binary classification

    def test_get_classification_report(self, trained_model, sample_features, sample_target):
        """Test classification report generation."""
        evaluator = ModelEvaluator()

        X_test = sample_features.tail(200)
        y_test = sample_target.tail(200)

        report = evaluator.get_classification_report(trained_model, X_test, y_test)

        # Check report is a string
        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report

    def test_check_production_readiness_pass(self):
        """Test production readiness check (passing)."""
        evaluator = ModelEvaluator()

        metrics = {
            "accuracy": 0.90,
            "precision": 0.85,
            "recall": 0.85
        }

        is_ready, message = evaluator.check_production_readiness(metrics)

        assert is_ready is True
        assert "ready for production" in message.lower()

    def test_check_production_readiness_fail(self):
        """Test production readiness check (failing)."""
        evaluator = ModelEvaluator()

        metrics = {
            "accuracy": 0.70,
            "precision": 0.65,
            "recall": 0.60
        }

        is_ready, message = evaluator.check_production_readiness(metrics)

        assert is_ready is False
        assert "NOT ready" in message
