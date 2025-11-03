"""Locust load testing configuration.

Run with:
    locust -f locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 to configure and start the test.
"""

from locust import HttpUser, task, between, events
import random


class MLAPIUser(HttpUser):
    """Simulated user making requests to ML API."""

    # Wait 1-3 seconds between requests
    wait_time = between(1, 3)

    def on_start(self):
        """Get authentication token when user starts."""
        response = self.client.post(
            "/login",
            json={
                "username": "admin",
                "password": "password"
            }
        )

        if response.status_code == 200:
            self.token = response.json()["token"]
        else:
            print(f"Login failed: {response.status_code} - {response.text}")
            self.token = None

    @task(10)
    def predict_single(self):
        """Make a single prediction (high frequency)."""
        if not self.token:
            return

        # Generate random features
        features = [random.uniform(-100, 100) for _ in range(10)]

        self.client.post(
            "/predict",
            json={"features": features},
            headers={"Authorization": f"Bearer {self.token}"},
            name="/predict (single)"
        )

    @task(3)
    def predict_batch(self):
        """Make a batch prediction (lower frequency)."""
        if not self.token:
            return

        # Generate 5-20 random samples
        num_samples = random.randint(5, 20)
        samples = [
            [random.uniform(-100, 100) for _ in range(10)]
            for _ in range(num_samples)
        ]

        self.client.post(
            "/batch-predict",
            json={"samples": samples},
            headers={"Authorization": f"Bearer {self.token}"},
            name=f"/batch-predict ({num_samples} samples)"
        )

    @task(2)
    def model_info(self):
        """Get model information (low frequency)."""
        self.client.get("/model-info", name="/model-info")

    @task(1)
    def health_check(self):
        """Check health (lowest frequency)."""
        self.client.get("/health", name="/health")

    @task(1)
    def cache_stats(self):
        """Get cache statistics (requires auth)."""
        if not self.token:
            return

        self.client.get(
            "/cache/stats",
            headers={"Authorization": f"Bearer {self.token}"},
            name="/cache/stats"
        )


@events.init_command_line_parser.add_listener
def _(parser):
    """Add custom command line options."""
    parser.add_argument(
        "--framework",
        type=str,
        default="fastapi",
        choices=["flask", "fastapi"],
        help="Framework to test (flask or fastapi)"
    )


@events.test_start.add_listener
def _(environment, **kwargs):
    """Log when test starts."""
    print(f"\n{'='*60}")
    print(f"Starting load test")
    print(f"Target: {environment.host}")
    print(f"{'='*60}\n")


@events.test_stop.add_listener
def _(environment, **kwargs):
    """Log when test stops."""
    print(f"\n{'='*60}")
    print(f"Load test completed")
    print(f"{'='*60}\n")
