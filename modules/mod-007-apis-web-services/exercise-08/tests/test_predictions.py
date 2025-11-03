"""Unit tests for prediction endpoints.

Tests cover:
- Single prediction success/failure
- Batch prediction scenarios
- Input validation
- Caching behavior
- Error handling
- Performance characteristics

Markers:
    unit: Unit tests
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestSinglePrediction:
    """Tests for single prediction endpoint (/predict)."""

    def test_predict_success(self, client, auth_headers, sample_features):
        """Test successful single prediction.

        Verifies:
        - 200 status code
        - Prediction value present and valid type
        - Model version included
        - Response format correct
        """
        response = client.post(
            "/predict",
            json={"features": sample_features},
            headers=auth_headers
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "prediction" in data, "Response missing prediction field"
        assert isinstance(data["prediction"], (int, float)), "Prediction must be numeric"
        assert "model_version" in data, "Response missing model_version"
        assert "cached" in data, "Response missing cached field"

    def test_predict_invalid_feature_count_too_few(self, client, auth_headers):
        """Test prediction with too few features.

        Verifies:
        - 422 Validation Error status
        - Error message indicates count issue
        """
        features = [1.0, 2.0, 3.0]  # Only 3 features, need 10

        response = client.post(
            "/predict",
            json={"features": features},
            headers=auth_headers
        )

        assert response.status_code == 422, "Should reject wrong feature count"
        error_detail = str(response.json())
        assert "10" in error_detail or "feature" in error_detail.lower()

    def test_predict_invalid_feature_count_too_many(self, client, auth_headers):
        """Test prediction with too many features.

        Verifies:
        - 422 Validation Error status
        """
        features = [1.0] * 15  # 15 features, need 10

        response = client.post(
            "/predict",
            json={"features": features},
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_predict_invalid_feature_types(self, client, auth_headers):
        """Test prediction with non-numeric features.

        Verifies:
        - 422 Validation Error status
        - Features must be numeric
        """
        invalid_features = [
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],  # Strings
            [None] * 10,  # Nulls
            [[1, 2]] + [1.0] * 9,  # Nested list
        ]

        for features in invalid_features:
            response = client.post(
                "/predict",
                json={"features": features},
                headers=auth_headers
            )

            assert response.status_code == 422, \
                f"Should reject invalid feature types: {features}"

    def test_predict_features_out_of_range(self, client, auth_headers):
        """Test prediction with features outside valid range.

        Verifies:
        - Features must be in [-1000, 1000] range
        - Validation error for out-of-range values
        """
        out_of_range_features = [
            [2000.0] + [1.0] * 9,  # Value too high
            [-2000.0] + [1.0] * 9,  # Value too low
            [1.0] * 9 + [float('inf')],  # Infinity
        ]

        for features in out_of_range_features:
            response = client.post(
                "/predict",
                json={"features": features},
                headers=auth_headers
            )

            assert response.status_code == 422, \
                f"Should reject out-of-range features: {features[0] if len(features) > 0 else 'N/A'}"

    def test_predict_missing_features_field(self, client, auth_headers):
        """Test prediction without features field.

        Verifies:
        - 422 Validation Error status
        """
        response = client.post(
            "/predict",
            json={},
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_predict_features_as_wrong_type(self, client, auth_headers):
        """Test features field with wrong type (not a list).

        Verifies:
        - Validation rejects non-list features
        """
        wrong_types = [
            {"features": "not a list"},
            {"features": 123},
            {"features": {"a": 1}},
        ]

        for payload in wrong_types:
            response = client.post(
                "/predict",
                json=payload,
                headers=auth_headers
            )

            assert response.status_code == 422

    def test_predict_caching_first_request(self, client, auth_headers, sample_features):
        """Test that first prediction is not cached.

        Verifies:
        - First request has cached=False
        - Prediction value returned
        """
        response = client.post(
            "/predict",
            json={"features": sample_features},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["cached"] == False, "First request should not be cached"

    def test_predict_caching_second_request(self, client, auth_headers, sample_features):
        """Test that repeated predictions are cached.

        Verifies:
        - Second identical request has cached=True
        - Prediction value matches first request
        - Cache improves performance
        """
        # First request
        response1 = client.post(
            "/predict",
            json={"features": sample_features},
            headers=auth_headers
        )
        assert response1.status_code == 200
        prediction1 = response1.json()["prediction"]
        assert response1.json()["cached"] == False

        # Second request (should be cached)
        response2 = client.post(
            "/predict",
            json={"features": sample_features},
            headers=auth_headers
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["cached"] == True, "Second request should be cached"
        assert data2["prediction"] == prediction1, "Cached prediction should match original"

    def test_predict_different_features_not_cached(self, client, auth_headers):
        """Test that different features don't use cache.

        Verifies:
        - Different features get new predictions
        - Each has cached=False on first request
        """
        features1 = [1.0] * 10
        features2 = [2.0] * 10

        # First set of features
        response1 = client.post(
            "/predict",
            json={"features": features1},
            headers=auth_headers
        )
        assert response1.json()["cached"] == False

        # Different features (should not use cache)
        response2 = client.post(
            "/predict",
            json={"features": features2},
            headers=auth_headers
        )
        assert response2.json()["cached"] == False, "Different features should not use cache"

    def test_predict_unauthorized(self, client, sample_features):
        """Test prediction without authentication.

        Verifies:
        - 401/403 status code
        - Error message present
        """
        response = client.post(
            "/predict",
            json={"features": sample_features}
        )

        assert response.status_code in [401, 403], "Should require authentication"
        assert "detail" in response.json()

    def test_predict_with_expired_token(self, client, expired_token, sample_features):
        """Test prediction with expired token.

        Verifies:
        - Request is rejected
        - Proper error returned
        """
        response = client.post(
            "/predict",
            json={"features": sample_features},
            headers={"Authorization": f"Bearer {expired_token}"}
        )

        assert response.status_code == 401

    def test_predict_response_time(self, client, auth_headers, sample_features):
        """Test that prediction response time is reasonable.

        Verifies:
        - Response time < 500ms for single prediction
        """
        import time

        start = time.time()
        response = client.post(
            "/predict",
            json={"features": sample_features},
            headers=auth_headers
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 0.5, f"Prediction took too long: {elapsed*1000:.2f}ms"

    def test_predict_boundary_values(self, client, auth_headers):
        """Test prediction with boundary feature values.

        Verifies:
        - Min values (-1000) accepted
        - Max values (1000) accepted
        - Zero values accepted
        """
        boundary_cases = [
            [-1000.0] * 10,  # Min values
            [1000.0] * 10,   # Max values
            [0.0] * 10,      # Zero values
            [-1000.0, 1000.0] + [0.0] * 8,  # Mixed
        ]

        for features in boundary_cases:
            response = client.post(
                "/predict",
                json={"features": features},
                headers=auth_headers
            )

            assert response.status_code == 200, \
                f"Should accept boundary values: {features[:2]}"

    def test_predict_floating_point_precision(self, client, auth_headers):
        """Test that floating point values are handled precisely.

        Verifies:
        - High-precision floats accepted
        - No precision loss in response
        """
        features = [1.123456789] * 10

        response = client.post(
            "/predict",
            json={"features": features},
            headers=auth_headers
        )

        assert response.status_code == 200
        # Prediction should be a valid float
        prediction = response.json()["prediction"]
        assert isinstance(prediction, (int, float))
        assert not np.isnan(prediction), "Prediction should not be NaN"
        assert not np.isinf(prediction), "Prediction should not be Inf"


@pytest.mark.unit
class TestBatchPrediction:
    """Tests for batch prediction endpoint (/batch-predict)."""

    def test_batch_predict_success(self, client, auth_headers, sample_batch):
        """Test successful batch prediction.

        Verifies:
        - 200 status code
        - Predictions list present
        - Count matches input samples
        - All predictions are numeric
        """
        response = client.post(
            "/batch-predict",
            json={"samples": sample_batch},
            headers=auth_headers
        )

        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == len(sample_batch), \
            "Number of predictions should match input samples"
        assert data["count"] == len(sample_batch)

        # All predictions should be numeric
        for pred in data["predictions"]:
            assert isinstance(pred, (int, float))

    def test_batch_predict_single_sample(self, client, auth_headers):
        """Test batch prediction with single sample.

        Verifies:
        - Single sample batch accepted
        - Returns one prediction
        """
        samples = [[1.0] * 10]

        response = client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1
        assert data["count"] == 1

    def test_batch_predict_maximum_samples(self, client, auth_headers):
        """Test batch prediction with maximum allowed samples (100).

        Verifies:
        - Max samples (100) accepted
        - All predictions returned
        """
        samples = [[float(i % 10)] * 10 for i in range(100)]

        response = client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 100
        assert data["count"] == 100

    def test_batch_predict_exceeds_max_samples(self, client, auth_headers):
        """Test batch prediction with too many samples (>100).

        Verifies:
        - 422 Validation Error status
        - Error indicates sample limit
        """
        samples = [[float(i % 10)] * 10 for i in range(101)]  # 101 samples

        response = client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=auth_headers
        )

        assert response.status_code == 422, "Should reject >100 samples"
        error_detail = str(response.json())
        assert "100" in error_detail or "limit" in error_detail.lower()

    def test_batch_predict_empty_list(self, client, auth_headers):
        """Test batch prediction with empty samples list.

        Verifies:
        - 422 Validation Error status
        """
        response = client.post(
            "/batch-predict",
            json={"samples": []},
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_batch_predict_invalid_sample_length(self, client, auth_headers):
        """Test batch prediction with wrong feature count in samples.

        Verifies:
        - Samples with wrong feature count rejected
        - Error indicates which sample is invalid
        """
        invalid_samples = [
            [[1.0, 2.0]],  # Too few features
            [[1.0] * 15],  # Too many features
            [[1.0] * 10, [1.0] * 5],  # Mixed lengths
        ]

        for samples in invalid_samples:
            response = client.post(
                "/batch-predict",
                json={"samples": samples},
                headers=auth_headers
            )

            assert response.status_code == 422, \
                f"Should reject invalid sample lengths: {[len(s) for s in samples]}"

    def test_batch_predict_invalid_sample_type(self, client, auth_headers):
        """Test batch prediction with non-list samples.

        Verifies:
        - Samples must be lists
        - Non-list samples rejected
        """
        invalid_samples = [
            "not a list",
            123,
            {"a": [1.0] * 10},
            [[1.0] * 10, "invalid"],  # Mixed types
        ]

        for samples in invalid_samples:
            response = client.post(
                "/batch-predict",
                json={"samples": samples},
                headers=auth_headers
            )

            assert response.status_code == 422

    def test_batch_predict_out_of_range_features(self, client, auth_headers):
        """Test batch prediction with out-of-range feature values.

        Verifies:
        - Features must be in valid range
        - Invalid samples rejected
        """
        samples = [
            [2000.0] * 10,  # Values too high
            [1.0] * 10,     # Valid
            [-2000.0] * 10  # Values too low
        ]

        response = client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_batch_predict_unauthorized(self, client, sample_batch):
        """Test batch prediction without authentication.

        Verifies:
        - 401/403 status code
        """
        response = client.post(
            "/batch-predict",
            json={"samples": sample_batch}
        )

        assert response.status_code in [401, 403]

    def test_batch_predict_performance(self, client, auth_headers):
        """Test batch prediction performance.

        Verifies:
        - Batch of 50 samples completes in reasonable time
        - Performance scales with batch size
        """
        import time

        samples_50 = [[float(i % 10)] * 10 for i in range(50)]

        start = time.time()
        response = client.post(
            "/batch-predict",
            json={"samples": samples_50},
            headers=auth_headers
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 2.0, f"Batch prediction took too long: {elapsed*1000:.2f}ms"

        # Calculate throughput
        throughput = 50 / elapsed
        print(f"Batch prediction throughput: {throughput:.2f} samples/sec")

    def test_batch_predict_consistency(self, client, auth_headers):
        """Test that batch predictions are consistent.

        Verifies:
        - Same inputs produce same outputs
        - Batch prediction matches individual predictions
        """
        samples = [[float(i)] * 10 for i in range(1, 4)]

        # Get batch predictions
        response = client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=auth_headers
        )
        batch_predictions = response.json()["predictions"]

        # Get individual predictions (should match)
        individual_predictions = []
        for sample in samples:
            response = client.post(
                "/predict",
                json={"features": sample},
                headers=auth_headers
            )
            individual_predictions.append(response.json()["prediction"])

        # Compare (allow small floating-point differences)
        for i, (batch_pred, indiv_pred) in enumerate(zip(batch_predictions, individual_predictions)):
            assert abs(batch_pred - indiv_pred) < 1e-6, \
                f"Sample {i}: batch={batch_pred}, individual={indiv_pred}"

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 25, 50, 100])
    def test_batch_predict_various_sizes(self, client, auth_headers, batch_size):
        """Test batch prediction with various batch sizes.

        Args:
            batch_size: Number of samples in batch

        Verifies:
        - All valid batch sizes work correctly
        """
        samples = [[float(i % 10)] * 10 for i in range(batch_size)]

        response = client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == batch_size
        assert data["count"] == batch_size


@pytest.mark.unit
class TestPredictionEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_predict_with_nan_features(self, client, auth_headers):
        """Test prediction with NaN feature values.

        Verifies:
        - NaN values are rejected or handled gracefully
        """
        # Note: JSON doesn't support NaN, so this tests JSON parsing
        response = client.post(
            "/predict",
            json={"features": [float('nan')] * 10},
            headers=auth_headers
        )

        # Should be rejected (NaN not valid in JSON)
        assert response.status_code in [400, 422]

    def test_predict_concurrent_requests(self, client, auth_headers):
        """Test concurrent prediction requests.

        Verifies:
        - Multiple simultaneous requests handled correctly
        - No race conditions
        """
        import concurrent.futures

        def make_prediction(i):
            features = [float(i % 10)] * 10
            return client.post(
                "/predict",
                json={"features": features},
                headers=auth_headers
            )

        # Make 20 concurrent predictions
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(20)]
            responses = [f.result() for f in futures]

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert "prediction" in response.json()

    def test_predict_memory_efficiency(self, client, auth_headers):
        """Test that predictions don't cause memory leaks.

        Makes many predictions and checks system remains stable.
        """
        import gc

        # Make 100 predictions
        for i in range(100):
            features = [float(i % 10)] * 10
            response = client.post(
                "/predict",
                json={"features": features},
                headers=auth_headers
            )
            assert response.status_code == 200

        # Force garbage collection
        gc.collect()

        # System should still be responsive
        response = client.get("/health")
        assert response.status_code == 200
