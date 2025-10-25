"""
Integration Tests for Training Pipeline

Tests end-to-end training workflows.
"""

import pytest
import pandas as pd
from src.data_preprocessing import clean_data, handle_missing_values, normalize_data, split_features_target, create_train_test_split
from src.model_training import train_model, save_model, load_model
from src.model_inference import predict
from src.model_evaluation import evaluate_classification


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""

    def test_end_to_end_training(self, large_sample_data, tmp_path):
        """Test complete training pipeline from raw data to predictions."""
        # Step 1: Clean data
        cleaned = clean_data(large_sample_data)
        assert len(cleaned) > 0

        # Step 2: Handle missing values
        filled = handle_missing_values(cleaned, strategy="mean")
        assert filled.isnull().sum().sum() == 0

        # Step 3: Split features and target
        X, y = split_features_target(filled, 'target')

        # Step 4: Normalize
        X_norm, scaler = normalize_data(X, method="standard")

        # Step 5: Train/test split
        X_train, X_test, y_train, y_test = create_train_test_split(
            X_norm, y, test_size=0.2, random_state=42
        )

        # Step 6: Train model
        model = train_model(X_train, y_train, model_type="random_forest", random_state=42)
        assert model is not None

        # Step 7: Save model
        model_path = tmp_path / "model.pkl"
        save_model(model, str(model_path))
        assert model_path.exists()

        # Step 8: Load model
        loaded_model = load_model(str(model_path))
        assert loaded_model is not None

        # Step 9: Make predictions
        predictions = predict(loaded_model, X_test)
        assert len(predictions) == len(X_test)

        # Step 10: Evaluate
        metrics = evaluate_classification(y_test, predictions)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_model_reproducibility(self, train_test_data):
        """Test that training produces reproducible results."""
        X_train, X_test, y_train, _ = train_test_data

        # Train two models with same seed
        model1 = train_model(X_train, y_train, random_state=42)
        model2 = train_model(X_train, y_train, random_state=42)

        # Predictions should be identical
        pred1 = predict(model1, X_test)
        pred2 = predict(model2, X_test)

        assert (pred1 == pred2).all()

    def test_different_model_types(self, train_test_data):
        """Test pipeline with different model types."""
        X_train, X_test, y_train, y_test = train_test_data

        model_types = ["random_forest", "logistic_regression", "decision_tree"]

        for model_type in model_types:
            model = train_model(X_train, y_train, model_type=model_type, random_state=42)
            predictions = predict(model, X_test)
            metrics = evaluate_classification(y_test, predictions)

            assert metrics['accuracy'] > 0.5  # Better than random
