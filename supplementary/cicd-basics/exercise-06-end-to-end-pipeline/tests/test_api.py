"""Tests for API serving module."""

import pytest
from fastapi.testclient import TestClient
from src.serve.api import create_app
from src.models.train import ModelTrainer
import pandas as pd


@pytest.fixture
def test_model(sample_features, sample_target, tmp_path):
    """Create and save a test model."""
    trainer = ModelTrainer(
        experiment_name="test-api",
        model_type="random_forest"
    )

    # Train on small subset
    X_small = sample_features.head(100)
    y_small = sample_target.head(100)

    model, _ = trainer.train(X_small, y_small, test_size=0.2)

    # Save model
    model_path = str(tmp_path / "test_model")
    trainer.save_model(model_path)

    return model_path


@pytest.fixture
def test_client(test_model):
    """Create test client."""
    app = create_app(model_uri=test_model)
    return TestClient(app)


class TestAPI:
    """Test FastAPI endpoints."""

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_predict_endpoint(self, test_client):
        """Test prediction endpoint."""
        # Create sample features
        features = [
            {f"feature_{i}": 0.5 for i in range(20)}
        ]

        response = test_client.post(
            "/predict",
            json={"features": features}
        )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 1

    def test_predict_multiple_samples(self, test_client):
        """Test prediction with multiple samples."""
        features = [
            {f"feature_{i}": i * 0.1 for i in range(20)}
            for _ in range(5)
        ]

        response = test_client.post(
            "/predict",
            json={"features": features}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 5

    def test_model_info_endpoint(self, test_client):
        """Test model info endpoint."""
        # First make a prediction to load the model
        features = [{f"feature_{i}": 0.5 for i in range(20)}]
        test_client.post("/predict", json={"features": features})

        # Then get model info
        response = test_client.get("/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "model_uri" in data
        assert "model_type" in data

    def test_predict_invalid_features(self, test_client):
        """Test prediction with invalid features."""
        response = test_client.post(
            "/predict",
            json={"features": [{"invalid": "data"}]}
        )

        # Should return error (might be 500 or 422)
        assert response.status_code in [422, 500]
