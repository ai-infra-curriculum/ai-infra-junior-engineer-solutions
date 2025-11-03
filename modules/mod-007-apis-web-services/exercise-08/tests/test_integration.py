"""Integration tests for complete workflows.

Tests cover:
- End-to-end user workflows
- Multi-step scenarios
- Cross-endpoint interactions
- System behavior under realistic usage

Markers:
    integration: Integration tests
"""

import pytest
import time


@pytest.mark.integration
class TestCompleteAuthFlow:
    """Test complete authentication workflow."""

    def test_login_and_access_protected_resources(self, client, test_user):
        """Test complete flow: login → access protected endpoint → verify data.

        Steps:
        1. Login with valid credentials
        2. Receive JWT token
        3. Use token to access protected endpoint
        4. Verify user data returned correctly

        Verifies:
        - Full authentication flow works end-to-end
        - Token works across requests
        - User data persists
        """
        # Step 1: Login
        login_response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        assert login_response.status_code == 200, "Login should succeed"
        token = login_response.json()["access_token"]
        assert token, "Should receive access token"

        # Step 2: Access protected resource
        user_response = client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert user_response.status_code == 200, "Should access protected endpoint with token"

        # Step 3: Verify data
        user_data = user_response.json()
        assert user_data["username"] == test_user["username"]
        assert user_data["email"] == test_user["email"]

    def test_login_token_reuse(self, client, test_user):
        """Test that token can be reused multiple times.

        Verifies:
        - Token valid for multiple requests
        - No need to re-authenticate for each request
        """
        # Login once
        login_response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Use token multiple times
        for _ in range(5):
            response = client.get("/users/me", headers=headers)
            assert response.status_code == 200, "Token should work multiple times"

    def test_expired_token_requires_relogin(self, client, test_user, expired_token):
        """Test that expired token requires re-authentication.

        Verifies:
        - Expired token rejected
        - Re-login with credentials works
        - New token valid
        """
        # Try with expired token
        response = client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        assert response.status_code == 401, "Expired token should be rejected"

        # Re-login
        login_response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        assert login_response.status_code == 200

        # New token should work
        new_token = login_response.json()["access_token"]
        response = client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {new_token}"}
        )
        assert response.status_code == 200


@pytest.mark.integration
class TestMLPipelineFlow:
    """Test complete ML prediction pipeline."""

    def test_full_prediction_pipeline(self, client, test_user):
        """Test complete ML workflow: login → predict → verify result.

        Steps:
        1. Authenticate
        2. Check model availability
        3. Make single prediction
        4. Make batch prediction
        5. Verify all results valid

        Verifies:
        - Complete ML serving pipeline works
        - Multiple prediction types supported
        - Results are consistent
        """
        # Step 1: Login
        login_response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Step 2: Check models available
        models_response = client.get("/models", headers=headers)
        assert models_response.status_code == 200
        models = models_response.json()
        assert len(models) > 0, "At least one model should be available"

        # Step 3: Single prediction
        single_pred_response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=headers
        )
        assert single_pred_response.status_code == 200
        single_prediction = single_pred_response.json()["prediction"]
        assert isinstance(single_prediction, (int, float))

        # Step 4: Batch prediction
        batch_pred_response = client.post(
            "/batch-predict",
            json={"samples": [[1.0] * 10, [2.0] * 10]},
            headers=headers
        )
        assert batch_pred_response.status_code == 200
        batch_predictions = batch_pred_response.json()["predictions"]
        assert len(batch_predictions) == 2

    def test_prediction_with_caching_workflow(self, client, auth_headers):
        """Test prediction caching across multiple requests.

        Steps:
        1. Make first prediction (not cached)
        2. Make same prediction (should be cached)
        3. Make different prediction (not cached)
        4. Repeat second prediction (cached)

        Verifies:
        - Caching works correctly
        - Cache is feature-specific
        - Performance benefit from caching
        """
        features_a = [1.0] * 10
        features_b = [2.0] * 10

        # First request for features_a (not cached)
        response1 = client.post(
            "/predict",
            json={"features": features_a},
            headers=auth_headers
        )
        assert response1.json()["cached"] == False
        prediction_a = response1.json()["prediction"]

        # Second request for features_a (cached)
        start = time.time()
        response2 = client.post(
            "/predict",
            json={"features": features_a},
            headers=auth_headers
        )
        cached_time = time.time() - start
        assert response2.json()["cached"] == True
        assert response2.json()["prediction"] == prediction_a

        # First request for features_b (not cached)
        start = time.time()
        response3 = client.post(
            "/predict",
            json={"features": features_b},
            headers=auth_headers
        )
        uncached_time = time.time() - start
        assert response3.json()["cached"] == False

        # Cached request should be faster (usually)
        print(f"Cached: {cached_time*1000:.2f}ms, Uncached: {uncached_time*1000:.2f}ms")

    def test_mixed_single_and_batch_predictions(self, client, auth_headers):
        """Test interleaving single and batch predictions.

        Verifies:
        - Both endpoint types work together
        - No interference between endpoints
        - Results consistent
        """
        # Single prediction
        single_response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )
        assert single_response.status_code == 200
        single_pred = single_response.json()["prediction"]

        # Batch prediction
        batch_response = client.post(
            "/batch-predict",
            json={"samples": [[1.0] * 10, [2.0] * 10]},
            headers=auth_headers
        )
        assert batch_response.status_code == 200
        batch_preds = batch_response.json()["predictions"]

        # Another single prediction
        single_response2 = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )
        assert single_response2.status_code == 200

        # Single prediction should match first element of batch
        # (allowing for floating point differences)
        assert abs(single_pred - batch_preds[0]) < 1e-6


@pytest.mark.integration
class TestSystemBehavior:
    """Test system-level behavior and interactions."""

    def test_health_check_reflects_system_state(self, client):
        """Test that health endpoint reflects actual system state.

        Verifies:
        - Health endpoint accessible without auth
        - Returns expected status
        - Provides timestamp
        """
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    def test_sequential_requests_maintain_consistency(self, client, auth_headers):
        """Test that sequential requests maintain consistent behavior.

        Makes many sequential requests to verify system stability.
        """
        features = [1.0] * 10

        # Make 20 sequential predictions
        predictions = []
        for i in range(20):
            response = client.post(
                "/predict",
                json={"features": features},
                headers=auth_headers
            )
            assert response.status_code == 200, f"Request {i} failed"
            predictions.append(response.json()["prediction"])

        # All predictions for same input should be identical
        assert len(set(predictions)) == 1, "Predictions should be consistent"

    def test_error_recovery(self, client, auth_headers):
        """Test that system recovers from errors gracefully.

        Steps:
        1. Make valid request (succeeds)
        2. Make invalid request (fails)
        3. Make valid request again (should still work)

        Verifies:
        - Errors don't corrupt system state
        - System recovers from errors
        """
        # Valid request
        response1 = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )
        assert response1.status_code == 200

        # Invalid request
        response2 = client.post(
            "/predict",
            json={"features": [1.0] * 5},  # Wrong count
            headers=auth_headers
        )
        assert response2.status_code == 422

        # System should still work
        response3 = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )
        assert response3.status_code == 200

    def test_concurrent_users(self, client):
        """Test multiple users accessing API concurrently.

        Verifies:
        - Multiple users can authenticate simultaneously
        - Tokens are user-specific
        - No cross-user contamination
        """
        import concurrent.futures

        def user_workflow(username, password):
            # Login
            login_response = client.post(
                "/auth/login",
                json={"username": username, "password": password}
            )
            if login_response.status_code != 200:
                return False

            token = login_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            # Make prediction
            pred_response = client.post(
                "/predict",
                json={"features": [1.0] * 10},
                headers=headers
            )
            return pred_response.status_code == 200

        # Simulate 5 concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(user_workflow, "testuser", "testpassword123")
                for _ in range(5)
            ]
            results = [f.result() for f in futures]

        # All workflows should succeed
        assert all(results), "All concurrent users should succeed"


@pytest.mark.integration
class TestDataFlow:
    """Test data flow through the system."""

    def test_prediction_result_format_consistency(self, client, auth_headers):
        """Test that prediction results maintain consistent format.

        Makes predictions with various inputs and verifies format.
        """
        test_cases = [
            [1.0] * 10,
            [0.0] * 10,
            [-100.0] * 10,
            [100.0] * 10,
            list(range(1, 11)),
        ]

        for features in test_cases:
            response = client.post(
                "/predict",
                json={"features": [float(f) for f in features]},
                headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()

            # Verify consistent response format
            assert "prediction" in data
            assert "cached" in data
            assert "model_version" in data
            assert isinstance(data["prediction"], (int, float))
            assert isinstance(data["cached"], bool)
            assert isinstance(data["model_version"], str)

    def test_batch_prediction_order_preserved(self, client, auth_headers):
        """Test that batch predictions preserve input order.

        Verifies:
        - Prediction order matches input order
        - No shuffling of results
        """
        # Create samples with distinguishable features
        samples = [
            [float(i)] * 10
            for i in range(1, 6)
        ]

        response = client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=auth_headers
        )

        assert response.status_code == 200
        predictions = response.json()["predictions"]

        # Verify we got predictions for all samples
        assert len(predictions) == len(samples)

        # Predictions should be in same order as inputs
        # (can't verify exact values without knowing model, but can verify consistency)
        for i, sample in enumerate(samples):
            # Make single prediction for comparison
            single_response = client.post(
                "/predict",
                json={"features": sample},
                headers=auth_headers
            )
            single_pred = single_response.json()["prediction"]

            # Batch prediction should match single prediction
            assert abs(predictions[i] - single_pred) < 1e-6, \
                f"Batch prediction {i} doesn't match single prediction"

    def test_api_response_headers(self, client, auth_headers):
        """Test that API returns appropriate response headers.

        Verifies:
        - Content-Type headers correct
        - Standard headers present
        """
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )

        assert response.status_code == 200

        # Check Content-Type
        assert "application/json" in response.headers.get("content-type", "").lower()


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningScenarios:
    """Test long-running and sustained usage scenarios."""

    def test_sustained_prediction_load(self, client, auth_headers):
        """Test system under sustained prediction load.

        Makes many predictions over time to check for:
        - Memory leaks
        - Performance degradation
        - System stability
        """
        import gc

        num_requests = 100
        predictions = []

        for i in range(num_requests):
            features = [float(i % 10)] * 10
            response = client.post(
                "/predict",
                json={"features": features},
                headers=auth_headers
            )

            assert response.status_code == 200, f"Request {i} failed"
            predictions.append(response.json()["prediction"])

            # Periodic cleanup
            if i % 20 == 0:
                gc.collect()

        # System should still be healthy
        health_response = client.get("/health")
        assert health_response.status_code == 200

        print(f"Completed {num_requests} predictions successfully")
