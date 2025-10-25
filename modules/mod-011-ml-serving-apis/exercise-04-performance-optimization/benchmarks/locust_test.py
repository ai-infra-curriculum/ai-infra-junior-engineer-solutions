"""Locust load testing for ML API.

Run with:
    locust -f locust_test.py --host=http://localhost:8000

Web UI: http://localhost:8089
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner
import random
import json
import time


class MLAPIUser(HttpUser):
    """
    Simulated user for ML API load testing.

    Simulates realistic user behavior with:
    - Different task types
    - Variable wait times
    - Weighted request distribution
    """

    # Wait 1-3 seconds between requests
    wait_time = between(1, 3)

    def on_start(self):
        """Called when user starts (login, etc.)."""
        # Authenticate if needed
        # self.client.post("/api/v1/auth/login", json={...})
        pass

    @task(10)
    def predict_single(self):
        """
        Single prediction (most common operation).

        Weight: 10 (happens 10x more than training)
        """
        features = [random.random() for _ in range(5)]

        with self.client.post(
            "/api/v1/predict",
            json={"features": features},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                result = response.json()
                # Validate response
                if "prediction" in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(5)
    def predict_async(self):
        """
        Async prediction with polling.

        Weight: 5 (50% as common as single prediction)
        """
        features = [random.random() for _ in range(5)]

        # Submit async task
        response = self.client.post(
            "/api/v1/predict/async",
            json={"features": features}
        )

        if response.status_code == 200:
            task_id = response.json()["task_id"]

            # Poll for result (max 5 attempts)
            for _ in range(5):
                time.sleep(0.5)
                status_response = self.client.get(
                    f"/api/v1/tasks/{task_id}",
                    name="/api/v1/tasks/[id]"  # Group in stats
                )

                if status_response.status_code == 200:
                    status = status_response.json()["status"]
                    if status == "SUCCESS":
                        break

    @task(3)
    def predict_batch(self):
        """
        Batch prediction.

        Weight: 3 (30% as common as single)
        """
        batch_size = random.randint(5, 20)
        instances = [[random.random() for _ in range(5)] for _ in range(batch_size)]

        self.client.post(
            "/api/v1/predict/batch",
            json={"instances": instances},
            name="/api/v1/predict/batch"
        )

    @task(2)
    def health_check(self):
        """
        Health check.

        Weight: 2
        """
        self.client.get("/health")

    @task(1)
    def get_metrics(self):
        """
        Get Prometheus metrics.

        Weight: 1 (least common)
        """
        self.client.get("/metrics")


class PowerUser(HttpUser):
    """Power user with higher request rates."""

    wait_time = between(0.5, 1.5)

    @task(5)
    def predict_batch_large(self):
        """Large batch predictions."""
        batch_size = random.randint(50, 100)
        instances = [[random.random() for _ in range(5)] for _ in range(batch_size)]

        self.client.post(
            "/api/v1/predict/batch",
            json={"instances": instances, "batch_size": 20},
            name="/api/v1/predict/batch (large)"
        )

    @task(2)
    def predict_single(self):
        """Regular predictions."""
        features = [random.random() for _ in range(5)]
        self.client.post("/api/v1/predict", json={"features": features})


class StressTestUser(HttpUser):
    """
    Stress test user - no wait time.

    Use sparingly to test system limits.
    """

    wait_time = between(0, 0.1)

    @task
    def rapid_fire_predictions(self):
        """Rapid predictions to stress test."""
        features = [random.random() for _ in range(5)]
        self.client.post("/api/v1/predict", json={"features": features})


# ====================
# Event Handlers
# ====================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("\n" + "=" * 60)
    print("LOCUST LOAD TEST STARTING")
    print("=" * 60)
    print(f"Host: {environment.host}")
    print(f"Users: {environment.parsed_options.num_users if hasattr(environment, 'parsed_options') else 'N/A'}")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("\n" + "=" * 60)
    print("LOCUST LOAD TEST COMPLETE")
    print("=" * 60)

    # Print summary stats
    stats = environment.stats
    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Failure Rate: {stats.total.fail_ratio:.2%}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f} ms")
    print(f"Min Response Time: {stats.total.min_response_time:.2f} ms")
    print(f"Max Response Time: {stats.total.max_response_time:.2f} ms")
    print(f"RPS: {stats.total.total_rps:.2f}")

    print("\n" + "=" * 60 + "\n")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Called for each request."""
    # Custom logic for each request
    # Could send to external monitoring, log slow requests, etc.

    if exception:
        print(f"Request failed: {name} - {exception}")

    # Log slow requests
    if response_time > 1000:  # > 1 second
        print(f"SLOW REQUEST: {name} took {response_time:.0f}ms")


# ====================
# Custom Scenarios
# ====================

class RampUpScenario(MLAPIUser):
    """
    Gradually increase load.

    Simulates realistic traffic ramp-up.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def wait_time_func(self):
        """Dynamic wait time based on elapsed time."""
        elapsed = time.time() - self.start_time

        if elapsed < 60:
            # First minute: slow
            return random.uniform(3, 5)
        elif elapsed < 120:
            # Second minute: medium
            return random.uniform(1, 3)
        else:
            # After 2 minutes: fast
            return random.uniform(0.5, 1.5)

    wait_time = wait_time_func


class SpikeTestUser(HttpUser):
    """
    Simulate traffic spikes.

    Alternates between normal and high load.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spike_mode = False
        self.last_toggle = time.time()

    def wait_time_func(self):
        """Toggle between spike and normal."""
        # Toggle mode every 30 seconds
        if time.time() - self.last_toggle > 30:
            self.spike_mode = not self.spike_mode
            self.last_toggle = time.time()

        if self.spike_mode:
            return random.uniform(0.1, 0.3)  # Spike
        else:
            return random.uniform(2, 4)  # Normal

    wait_time = wait_time_func

    @task
    def predict(self):
        features = [random.random() for _ in range(5)]
        self.client.post("/api/v1/predict", json={"features": features})


# ====================
# Helper Functions
# ====================

def print_summary(stats):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    # Overall stats
    print(f"\nTotal Requests: {stats.total.num_requests:,}")
    print(f"Total Failures: {stats.total.num_failures:,}")
    print(f"Failure Rate: {stats.total.fail_ratio:.2%}")
    print(f"RPS: {stats.total.total_rps:.2f}")

    # Response times
    print(f"\nResponse Times:")
    print(f"  Average: {stats.total.avg_response_time:.2f} ms")
    print(f"  Min: {stats.total.min_response_time:.2f} ms")
    print(f"  Max: {stats.total.max_response_time:.2f} ms")
    print(f"  P50: {stats.total.get_response_time_percentile(0.5):.2f} ms")
    print(f"  P95: {stats.total.get_response_time_percentile(0.95):.2f} ms")
    print(f"  P99: {stats.total.get_response_time_percentile(0.99):.2f} ms")

    # Per-endpoint stats
    print(f"\nPer-Endpoint Stats:")
    for name, stat in stats.entries.items():
        if stat.num_requests > 0:
            print(f"\n  {name}:")
            print(f"    Requests: {stat.num_requests:,}")
            print(f"    Failures: {stat.num_failures:,}")
            print(f"    Avg: {stat.avg_response_time:.2f} ms")
            print(f"    P95: {stat.get_response_time_percentile(0.95):.2f} ms")

    print("\n" + "=" * 60)


# ====================
# Usage Examples
# ====================

if __name__ == "__main__":
    print("""
    LOCUST LOAD TESTING FOR ML API
    ==============================

    1. Start API server:
       uvicorn api:app --reload

    2. Run Locust (Web UI):
       locust -f locust_test.py --host=http://localhost:8000

       Open http://localhost:8089
       Configure users and spawn rate

    3. Run Locust (Headless):
       locust -f locust_test.py --host=http://localhost:8000 \\
              --users=100 --spawn-rate=10 --run-time=60s --headless

    4. Distributed Load Test:
       # Start master
       locust -f locust_test.py --master --host=http://localhost:8000

       # Start workers (on same or different machines)
       locust -f locust_test.py --worker --master-host=localhost

    5. Different User Types:
       locust -f locust_test.py --host=http://localhost:8000 \\
              MLAPIUser:70,PowerUser:20,StressTestUser:10

    SCENARIOS:
    - MLAPIUser: Normal user behavior
    - PowerUser: Higher request rate
    - StressTestUser: Maximum stress
    - RampUpScenario: Gradual load increase
    - SpikeTestUser: Traffic spikes

    METRICS TO WATCH:
    - RPS (Requests Per Second)
    - P95/P99 latency
    - Failure rate
    - Resource utilization (CPU, memory)
    """)
