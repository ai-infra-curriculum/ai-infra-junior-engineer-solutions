"""Locust load testing scenarios for ML API.

This file defines realistic user behavior patterns for load testing:
- Authentication workflows
- Prediction requests with varying patterns
- Mixed single and batch predictions
- Realistic think times

Usage:
    locust -f locustfile.py --host=http://localhost:8000

    # Run headless with specific users
    locust -f locustfile.py --host=http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 5m --headless

    # Generate HTML report
    locust -f locustfile.py --host=http://localhost:8000 \
           --users 100 --spawn-rate 10 --run-time 10m \
           --html report.html --headless

Scenarios:
    - MLAPIUser: Realistic ML prediction workload (70% weight)
    - HeavyBatchUser: Heavy batch prediction workload (20% weight)
    - LightPollingUser: Light monitoring/health check workload (10% weight)
"""

from locust import HttpUser, task, between, tag
import random
import json


class MLAPIUser(HttpUser):
    """Simulates typical ML API user behavior.

    Workflow:
    1. Login once at start
    2. Mix of single and batch predictions
    3. Occasional health checks
    4. Realistic think time between requests

    This represents the most common usage pattern.
    """

    # Wait 1-3 seconds between tasks (realistic user behavior)
    wait_time = between(1, 3)

    # Weight: 70% of total traffic
    weight = 70

    def on_start(self):
        """Called when user starts. Handles authentication."""
        # Login and store token
        response = self.client.post(
            "/auth/login",
            json={
                "username": "testuser",
                "password": "testpassword123"
            }
        )

        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            # Mark as failure for metrics
            self.environment.runner.stats.log_error(
                "POST", "/auth/login", "Login failed"
            )
            self.headers = {}

    @task(50)
    @tag("prediction", "single")
    def predict_single(self):
        """Make single prediction request.

        Most common task - 50% of all requests.
        Uses random features to simulate different inputs.
        """
        # Generate random features
        features = [random.uniform(-10, 10) for _ in range(10)]

        with self.client.post(
            "/predict",
            json={"features": features},
            headers=self.headers,
            catch_response=True,
            name="/predict [single]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response format
                if "prediction" in data and "model_version" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(20)
    @tag("prediction", "batch")
    def predict_batch(self):
        """Make batch prediction request.

        20% of all requests.
        Batch size varies between 2-10 samples.
        """
        # Generate random batch
        batch_size = random.randint(2, 10)
        samples = [
            [random.uniform(-10, 10) for _ in range(10)]
            for _ in range(batch_size)
        ]

        with self.client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=self.headers,
            catch_response=True,
            name=f"/batch-predict [size={batch_size}]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data.get("predictions", [])) == batch_size:
                    response.success()
                else:
                    response.failure("Prediction count mismatch")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(15)
    @tag("prediction", "cached")
    def predict_cached(self):
        """Make prediction with commonly used features.

        15% of all requests.
        Uses fixed features to test caching behavior.
        """
        # Use fixed features that should be cached
        features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        with self.client.post(
            "/predict",
            json={"features": features},
            headers=self.headers,
            catch_response=True,
            name="/predict [cached]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Check if cached (should be after first request)
                if "cached" in data:
                    response.success()
                else:
                    response.failure("Missing 'cached' field")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(10)
    @tag("metadata")
    def check_user_info(self):
        """Check current user info.

        10% of all requests.
        Simulates users checking their account status.
        """
        with self.client.get(
            "/users/me",
            headers=self.headers,
            catch_response=True,
            name="/users/me"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "username" in data and "email" in data:
                    response.success()
                else:
                    response.failure("Invalid user response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    @tag("metadata")
    def list_models(self):
        """List available models.

        5% of all requests.
        Simulates users checking available models.
        """
        self.client.get(
            "/models",
            headers=self.headers,
            name="/models"
        )


class HeavyBatchUser(HttpUser):
    """Simulates heavy batch prediction workload.

    Workflow:
    1. Login once at start
    2. Primarily large batch predictions
    3. Less frequent requests (longer think time)

    This represents power users running large batches.
    """

    # Longer wait time (3-8 seconds) as batch processing takes longer
    wait_time = between(3, 8)

    # Weight: 20% of total traffic
    weight = 20

    def on_start(self):
        """Authenticate at start."""
        response = self.client.post(
            "/auth/login",
            json={
                "username": "testuser",
                "password": "testpassword123"
            }
        )

        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}

    @task(80)
    @tag("prediction", "batch", "heavy")
    def large_batch_prediction(self):
        """Make large batch predictions.

        Primary task - 80% of requests.
        Uses large batches (20-50 samples).
        """
        # Generate large batch
        batch_size = random.randint(20, 50)
        samples = [
            [random.uniform(-100, 100) for _ in range(10)]
            for _ in range(batch_size)
        ]

        with self.client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=self.headers,
            catch_response=True,
            name=f"/batch-predict [large, size={batch_size}]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("count") == batch_size:
                    response.success()
                else:
                    response.failure("Count mismatch")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(20)
    @tag("prediction", "single")
    def single_prediction(self):
        """Occasional single prediction.

        20% of requests.
        Even heavy users make some single predictions.
        """
        features = [random.uniform(-100, 100) for _ in range(10)]

        self.client.post(
            "/predict",
            json={"features": features},
            headers=self.headers,
            name="/predict [from heavy user]"
        )


class LightPollingUser(HttpUser):
    """Simulates monitoring/health check traffic.

    Workflow:
    1. No authentication needed
    2. Frequent health checks
    3. Occasional model list checks
    4. Short think time (rapid polling)

    This represents monitoring systems and health checkers.
    """

    # Short wait time (0.5-2 seconds) for polling behavior
    wait_time = between(0.5, 2)

    # Weight: 10% of total traffic
    weight = 10

    @task(90)
    @tag("monitoring", "health")
    def health_check(self):
        """Frequent health checks.

        90% of requests.
        No authentication required.
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health [polling]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data.get('status')}")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(10)
    @tag("monitoring")
    def check_models(self):
        """Occasional model list check.

        10% of requests.
        Monitoring systems checking available models.
        """
        self.client.get(
            "/models",
            name="/models [monitoring]"
        )


class StressTestUser(HttpUser):
    """Aggressive user for stress testing.

    This user type is designed for stress tests and should NOT be
    used during normal load testing. Activate with --tags stress.

    Workflow:
    - No wait time between requests
    - Maximum rate of requests
    - Large payloads
    """

    # No wait time - maximum stress
    wait_time = between(0, 0.1)

    # Disabled by default - use --tags stress to enable
    weight = 0

    def on_start(self):
        """Authenticate at start."""
        response = self.client.post(
            "/auth/login",
            json={
                "username": "testuser",
                "password": "testpassword123"
            }
        )

        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}

    @task
    @tag("stress")
    def stress_predict(self):
        """Maximum rate prediction requests."""
        features = [random.random() * 100 for _ in range(10)]
        self.client.post(
            "/predict",
            json={"features": features},
            headers=self.headers,
            name="/predict [stress]"
        )

    @task
    @tag("stress")
    def stress_batch(self):
        """Maximum rate batch predictions."""
        # Use maximum allowed batch size
        samples = [[random.random() * 100 for _ in range(10)] for _ in range(100)]
        self.client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=self.headers,
            name="/batch-predict [stress]"
        )
