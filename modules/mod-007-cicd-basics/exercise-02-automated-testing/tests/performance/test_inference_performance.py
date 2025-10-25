"""
Performance Tests

Benchmark tests for training and inference performance.
"""

import pytest
import time
import pandas as pd
import numpy as np

from src.model_training import train_model
from src.model_inference import predict, batch_predict


@pytest.mark.performance
@pytest.mark.slow
class TestInferencePerformance:
    """Performance benchmarks for model inference."""

    def test_inference_latency(self, trained_model, benchmark):
        """Benchmark single prediction latency."""
        X = pd.DataFrame(np.random.rand(1, 20), columns=[f'feature_{i}' for i in range(20)])

        def run_prediction():
            return predict(trained_model, X)

        result = benchmark(run_prediction)
        assert result is not None

    def test_batch_inference_throughput(self, trained_model):
        """Test batch prediction throughput."""
        X = pd.DataFrame(np.random.rand(1000, 20), columns=[f'feature_{i}' for i in range(20)])

        start_time = time.time()
        predictions = batch_predict(trained_model, X, batch_size=100)
        elapsed_time = time.time() - start_time

        throughput = len(X) / elapsed_time

        assert len(predictions) == len(X)
        assert throughput > 100  # At least 100 predictions/second
        print(f"Throughput: {throughput:.0f} predictions/second")

    def test_training_time(self, train_test_data):
        """Test that training completes within reasonable time."""
        X_train, _, y_train, _ = train_test_data

        start_time = time.time()
        model = train_model(X_train, y_train, model_type="random_forest",
                           hyperparameters={"n_estimators": 10})
        elapsed_time = time.time() - start_time

        assert elapsed_time < 10  # Training should take less than 10 seconds
        print(f"Training time: {elapsed_time:.2f} seconds")
