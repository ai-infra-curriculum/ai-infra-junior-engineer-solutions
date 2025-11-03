"""Compare Flask and FastAPI performance.

This script benchmarks both implementations to measure:
- Sequential request throughput
- Concurrent request throughput
- Latency percentiles (P50, P95, P99)
- Response time consistency

Run both Flask and FastAPI servers before executing this script:
- Flask: python src/flask_app.py
- FastAPI: python src/fastapi_app.py
"""

import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import json

# Configuration
FLASK_URL = "http://localhost:5000"
FASTAPI_URL = "http://localhost:8000"
NUM_REQUESTS_SEQUENTIAL = 100  # Reduced for faster testing
NUM_REQUESTS_CONCURRENT = 1000
CONCURRENCY = 10

# Test features for predictions
TEST_FEATURES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


class BenchmarkResult:
    """Store benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.total_time = 0.0
        self.latencies: List[float] = []
        self.errors = 0

    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        if self.total_time == 0:
            return 0.0
        return len(self.latencies) / self.total_time

    @property
    def avg_latency(self) -> float:
        """Calculate average latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies) * 1000

    @property
    def median_latency(self) -> float:
        """Calculate median latency (P50) in milliseconds."""
        if not self.latencies:
            return 0.0
        return statistics.median(self.latencies) * 1000

    @property
    def p95_latency(self) -> float:
        """Calculate P95 latency in milliseconds."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(0.95 * len(sorted_latencies))
        return sorted_latencies[index] * 1000

    @property
    def p99_latency(self) -> float:
        """Calculate P99 latency in milliseconds."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(0.99 * len(sorted_latencies))
        return sorted_latencies[index] * 1000

    @property
    def min_latency(self) -> float:
        """Calculate minimum latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return min(self.latencies) * 1000

    @property
    def max_latency(self) -> float:
        """Calculate maximum latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return max(self.latencies) * 1000

    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'=' * 70}")
        print(f"Results for {self.name}")
        print(f"{'=' * 70}")
        print(f"  Total requests: {len(self.latencies)}")
        print(f"  Errors: {self.errors}")
        print(f"  Total time: {self.total_time:.2f}s")
        print(f"  Throughput: {self.requests_per_second:.2f} req/s")
        print()
        print(f"  Latency statistics:")
        print(f"    Min:    {self.min_latency:7.2f} ms")
        print(f"    Avg:    {self.avg_latency:7.2f} ms")
        print(f"    P50:    {self.median_latency:7.2f} ms")
        print(f"    P95:    {self.p95_latency:7.2f} ms")
        print(f"    P99:    {self.p99_latency:7.2f} ms")
        print(f"    Max:    {self.max_latency:7.2f} ms")


def get_token(base_url: str) -> str:
    """Get authentication token.

    Args:
        base_url: Base URL of the API

    Returns:
        JWT token

    Raises:
        Exception: If authentication fails
    """
    try:
        response = requests.post(
            f"{base_url}/login",
            json={"username": "admin", "password": "password"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()['token']
    except Exception as e:
        raise Exception(f"Failed to get token from {base_url}: {e}")


def make_prediction(base_url: str, token: str, features: list) -> Tuple[float, bool]:
    """Make a single prediction request.

    Args:
        base_url: Base URL of the API
        token: Authentication token
        features: Feature vector

    Returns:
        Tuple of (elapsed_time, success)
    """
    start = time.time()

    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": features},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )

        elapsed = time.time() - start

        if response.status_code == 200:
            return (elapsed, True)
        else:
            print(f"  ⚠️  Request failed with status {response.status_code}: {response.text[:100]}")
            return (elapsed, False)

    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"  ⚠️  Request timed out after {elapsed:.2f}s")
        return (elapsed, False)
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ⚠️  Request error: {e}")
        return (elapsed, False)


def check_api_health(base_url: str) -> bool:
    """Check if API is healthy and responding.

    Args:
        base_url: Base URL of the API

    Returns:
        True if API is healthy, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('model_loaded', False)
        return False
    except Exception:
        return False


def benchmark_sequential(base_url: str, name: str, num_requests: int) -> BenchmarkResult:
    """Benchmark API with sequential requests.

    Args:
        base_url: Base URL of the API
        name: Name for this benchmark
        num_requests: Number of requests to make

    Returns:
        BenchmarkResult with statistics
    """
    print(f"\n{'=' * 70}")
    print(f"Benchmarking {name} - Sequential Requests")
    print(f"{'=' * 70}")
    print(f"  Making {num_requests} sequential requests...")

    result = BenchmarkResult(f"{name} Sequential")

    try:
        # Get token
        print(f"  Getting authentication token...")
        token = get_token(base_url)

        # Make sequential requests
        start_time = time.time()

        for i in range(num_requests):
            latency, success = make_prediction(base_url, token, TEST_FEATURES)

            if success:
                result.latencies.append(latency)
            else:
                result.errors += 1

            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{num_requests} requests completed")

        result.total_time = time.time() - start_time

    except Exception as e:
        print(f"  ❌ Benchmark failed: {e}")

    return result


def benchmark_concurrent(base_url: str, name: str, num_requests: int, concurrency: int) -> BenchmarkResult:
    """Benchmark API with concurrent requests.

    Args:
        base_url: Base URL of the API
        name: Name for this benchmark
        num_requests: Total number of requests to make
        concurrency: Number of concurrent workers

    Returns:
        BenchmarkResult with statistics
    """
    print(f"\n{'=' * 70}")
    print(f"Benchmarking {name} - Concurrent Requests")
    print(f"{'=' * 70}")
    print(f"  Making {num_requests} requests with {concurrency} concurrent workers...")

    result = BenchmarkResult(f"{name} Concurrent")

    try:
        # Get token
        print(f"  Getting authentication token...")
        token = get_token(base_url)

        # Make concurrent requests
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(make_prediction, base_url, token, TEST_FEATURES)
                for _ in range(num_requests)
            ]

            completed = 0
            for future in as_completed(futures):
                latency, success = future.result()

                if success:
                    result.latencies.append(latency)
                else:
                    result.errors += 1

                completed += 1

                # Progress indicator
                if completed % 100 == 0:
                    print(f"    Progress: {completed}/{num_requests} requests completed")

        result.total_time = time.time() - start_time

    except Exception as e:
        print(f"  ❌ Benchmark failed: {e}")

    return result


def compare_results(flask_result: BenchmarkResult, fastapi_result: BenchmarkResult):
    """Compare and print results from both APIs.

    Args:
        flask_result: Flask benchmark results
        fastapi_result: FastAPI benchmark results
    """
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}\n")

    # Throughput comparison
    print("Throughput (requests/second):")
    print(f"  Flask:   {flask_result.requests_per_second:7.2f} req/s")
    print(f"  FastAPI: {fastapi_result.requests_per_second:7.2f} req/s")

    if flask_result.requests_per_second > 0:
        improvement = (
            (fastapi_result.requests_per_second - flask_result.requests_per_second)
            / flask_result.requests_per_second * 100
        )
        print(f"  Improvement: {improvement:+.1f}%")
        if improvement > 0:
            print(f"  🏆 Winner: FastAPI")
        else:
            print(f"  🏆 Winner: Flask")

    # Latency comparison
    print("\nLatency P95 (milliseconds):")
    print(f"  Flask:   {flask_result.p95_latency:7.2f} ms")
    print(f"  FastAPI: {fastapi_result.p95_latency:7.2f} ms")

    if flask_result.p95_latency > 0:
        improvement = (
            (flask_result.p95_latency - fastapi_result.p95_latency)
            / flask_result.p95_latency * 100
        )
        print(f"  Improvement: {improvement:+.1f}%")
        if improvement > 0:
            print(f"  🏆 Winner: FastAPI")
        else:
            print(f"  🏆 Winner: Flask")

    # Error comparison
    print("\nErrors:")
    print(f"  Flask:   {flask_result.errors}")
    print(f"  FastAPI: {fastapi_result.errors}")


def main():
    """Main benchmark execution."""
    print("\n" + "=" * 70)
    print("FLASK vs FASTAPI PERFORMANCE COMPARISON")
    print("=" * 70)

    # Check if both APIs are running
    print("\nChecking API availability...")

    flask_healthy = check_api_health(FLASK_URL)
    fastapi_healthy = check_api_health(FASTAPI_URL)

    print(f"  Flask ({FLASK_URL}):   {'✅ Healthy' if flask_healthy else '❌ Unavailable'}")
    print(f"  FastAPI ({FASTAPI_URL}): {'✅ Healthy' if fastapi_healthy else '❌ Unavailable'}")

    if not flask_healthy or not fastapi_healthy:
        print("\n❌ Error: Both APIs must be running and healthy")
        print("\nStart the servers with:")
        print(f"  Flask:   python src/flask_app.py")
        print(f"  FastAPI: python src/fastapi_app.py")
        return

    # Run sequential benchmarks
    print("\n" + "=" * 70)
    print("PHASE 1: SEQUENTIAL REQUESTS")
    print("=" * 70)

    flask_seq = benchmark_sequential(FLASK_URL, "Flask", NUM_REQUESTS_SEQUENTIAL)
    flask_seq.print_summary()

    fastapi_seq = benchmark_sequential(FASTAPI_URL, "FastAPI", NUM_REQUESTS_SEQUENTIAL)
    fastapi_seq.print_summary()

    compare_results(flask_seq, fastapi_seq)

    # Run concurrent benchmarks
    print("\n" + "=" * 70)
    print("PHASE 2: CONCURRENT REQUESTS")
    print("=" * 70)

    flask_conc = benchmark_concurrent(FLASK_URL, "Flask", NUM_REQUESTS_CONCURRENT, CONCURRENCY)
    flask_conc.print_summary()

    fastapi_conc = benchmark_concurrent(FASTAPI_URL, "FastAPI", NUM_REQUESTS_CONCURRENT, CONCURRENCY)
    fastapi_conc.print_summary()

    compare_results(flask_conc, fastapi_conc)

    # Final summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  1. Sequential throughput: FastAPI is {((fastapi_seq.requests_per_second / flask_seq.requests_per_second - 1) * 100):+.1f}% vs Flask")
    print(f"  2. Concurrent throughput: FastAPI is {((fastapi_conc.requests_per_second / flask_conc.requests_per_second - 1) * 100):+.1f}% vs Flask")
    print(f"  3. P95 latency (sequential): FastAPI is {((1 - fastapi_seq.p95_latency / flask_seq.p95_latency) * 100):+.1f}% faster")
    print(f"  4. P95 latency (concurrent): FastAPI is {((1 - fastapi_conc.p95_latency / flask_conc.p95_latency) * 100):+.1f}% faster")


if __name__ == "__main__":
    main()
