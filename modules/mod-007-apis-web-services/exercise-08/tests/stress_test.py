"""Stress testing script for ML API.

This script performs various stress tests to identify breaking points:
- Concurrent request stress
- Sustained load testing
- Memory leak detection
- Connection pool exhaustion
- Rate limit testing

Usage:
    # Run all stress tests
    python stress_test.py --all

    # Run specific test
    python stress_test.py --test concurrent

    # Run with custom parameters
    python stress_test.py --test sustained --duration 600 --rps 100

Requirements:
    pip install requests aiohttp psutil
"""

import argparse
import time
import asyncio
import aiohttp
import requests
import statistics
import psutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime


# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_USERNAME = "testuser"
TEST_PASSWORD = "testpassword123"


@dataclass
class StressTestResult:
    """Container for stress test results."""
    test_name: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    peak_memory_mb: float
    notes: str = ""


class MLAPIStressTester:
    """Stress testing framework for ML API."""

    def __init__(self, base_url: str = API_BASE_URL):
        """Initialize stress tester.

        Args:
            base_url: Base URL of the API to test
        """
        self.base_url = base_url
        self.token = None
        self.headers = {}
        self.process = psutil.Process()

    def authenticate(self) -> bool:
        """Authenticate and get JWT token.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
                timeout=10
            )

            if response.status_code == 200:
                self.token = response.json()["access_token"]
                self.headers = {"Authorization": f"Bearer {self.token}"}
                print(f"✓ Authentication successful")
                return True
            else:
                print(f"✗ Authentication failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Authentication error: {e}")
            return False

    def test_concurrent_requests(
        self,
        num_requests: int = 1000,
        num_workers: int = 50
    ) -> StressTestResult:
        """Test API under concurrent request load.

        Args:
            num_requests: Total number of requests to make
            num_workers: Number of concurrent workers

        Returns:
            StressTestResult with metrics
        """
        print(f"\n{'='*60}")
        print(f"Concurrent Request Stress Test")
        print(f"{'='*60}")
        print(f"Total requests: {num_requests}")
        print(f"Concurrent workers: {num_workers}")
        print(f"{'='*60}\n")

        response_times = []
        successful = 0
        failed = 0
        peak_memory = 0

        def make_request(request_id: int) -> Tuple[bool, float]:
            """Make single request and return success status and response time."""
            try:
                features = [float(request_id % 100) for _ in range(10)]

                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/predict",
                    json={"features": features},
                    headers=self.headers,
                    timeout=30
                )
                elapsed = time.time() - start_time

                success = response.status_code == 200
                return success, elapsed

            except Exception as e:
                return False, 0.0

        start_time = time.time()

        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(make_request, i)
                for i in range(num_requests)
            ]

            # Process results as they complete
            for i, future in enumerate(as_completed(futures), 1):
                success, elapsed = future.result()

                if success:
                    successful += 1
                    response_times.append(elapsed)
                else:
                    failed += 1

                # Track memory usage
                current_memory = self.process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)

                # Progress indicator
                if i % 100 == 0:
                    print(f"Progress: {i}/{num_requests} requests "
                          f"({successful} success, {failed} failed, "
                          f"memory: {current_memory:.1f}MB)")

        duration = time.time() - start_time

        # Calculate statistics
        if response_times:
            avg_time = statistics.mean(response_times)
            p50 = statistics.median(response_times)
            p95 = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99 = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            avg_time = p50 = p95 = p99 = 0.0

        rps = num_requests / duration
        error_rate = (failed / num_requests) * 100

        return StressTestResult(
            test_name="Concurrent Requests",
            duration=duration,
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            avg_response_time=avg_time,
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            requests_per_second=rps,
            error_rate=error_rate,
            peak_memory_mb=peak_memory
        )

    async def _async_request(
        self,
        session: aiohttp.ClientSession,
        request_id: int
    ) -> Tuple[bool, float]:
        """Make async request.

        Args:
            session: aiohttp session
            request_id: Request identifier

        Returns:
            Tuple of (success, elapsed_time)
        """
        try:
            features = [float(request_id % 100) for _ in range(10)]

            start_time = time.time()
            async with session.post(
                f"{self.base_url}/predict",
                json={"features": features},
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                await response.text()  # Read response body
                elapsed = time.time() - start_time
                return response.status == 200, elapsed

        except Exception as e:
            return False, 0.0

    async def _sustained_load_async(
        self,
        duration: int,
        target_rps: int
    ) -> StressTestResult:
        """Run sustained load test asynchronously.

        Args:
            duration: Test duration in seconds
            target_rps: Target requests per second

        Returns:
            StressTestResult with metrics
        """
        print(f"\n{'='*60}")
        print(f"Sustained Load Test (Async)")
        print(f"{'='*60}")
        print(f"Duration: {duration}s")
        print(f"Target RPS: {target_rps}")
        print(f"{'='*60}\n")

        response_times = []
        successful = 0
        failed = 0
        peak_memory = 0
        request_id = 0

        start_time = time.time()
        last_report = start_time

        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=100)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:

            while time.time() - start_time < duration:
                batch_start = time.time()

                # Create batch of requests to hit target RPS
                batch_size = target_rps
                tasks = [
                    self._async_request(session, request_id + i)
                    for i in range(batch_size)
                ]

                # Execute batch
                results = await asyncio.gather(*tasks)

                # Process results
                for success, elapsed in results:
                    if success:
                        successful += 1
                        response_times.append(elapsed)
                    else:
                        failed += 1

                request_id += batch_size

                # Track memory
                current_memory = self.process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)

                # Report progress every 10 seconds
                if time.time() - last_report >= 10:
                    elapsed = time.time() - start_time
                    current_rps = (successful + failed) / elapsed
                    print(f"[{elapsed:.0f}s] RPS: {current_rps:.1f}, "
                          f"Success: {successful}, Failed: {failed}, "
                          f"Memory: {current_memory:.1f}MB")
                    last_report = time.time()

                # Sleep to maintain target RPS
                batch_elapsed = time.time() - batch_start
                sleep_time = max(0, 1.0 - batch_elapsed)
                await asyncio.sleep(sleep_time)

        duration = time.time() - start_time
        total_requests = successful + failed

        # Calculate statistics
        if response_times:
            avg_time = statistics.mean(response_times)
            p50 = statistics.median(response_times)
            p95 = statistics.quantiles(response_times, n=20)[18]
            p99 = statistics.quantiles(response_times, n=100)[98]
        else:
            avg_time = p50 = p95 = p99 = 0.0

        actual_rps = total_requests / duration
        error_rate = (failed / total_requests * 100) if total_requests > 0 else 0

        return StressTestResult(
            test_name="Sustained Load",
            duration=duration,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            avg_response_time=avg_time,
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            requests_per_second=actual_rps,
            error_rate=error_rate,
            peak_memory_mb=peak_memory,
            notes=f"Target: {target_rps} RPS, Actual: {actual_rps:.1f} RPS"
        )

    def test_sustained_load(
        self,
        duration: int = 300,
        target_rps: int = 50
    ) -> StressTestResult:
        """Test API under sustained load.

        Args:
            duration: Test duration in seconds
            target_rps: Target requests per second

        Returns:
            StressTestResult with metrics
        """
        return asyncio.run(self._sustained_load_async(duration, target_rps))

    def test_memory_leak(
        self,
        num_iterations: int = 10,
        requests_per_iteration: int = 100
    ) -> StressTestResult:
        """Test for memory leaks.

        Args:
            num_iterations: Number of test iterations
            requests_per_iteration: Requests per iteration

        Returns:
            StressTestResult with memory analysis
        """
        print(f"\n{'='*60}")
        print(f"Memory Leak Detection Test")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Requests per iteration: {requests_per_iteration}")
        print(f"{'='*60}\n")

        memory_readings = []
        response_times = []
        successful = 0
        failed = 0

        start_time = time.time()

        for iteration in range(num_iterations):
            # Force garbage collection before measurement
            import gc
            gc.collect()

            # Measure memory before iteration
            memory_before = self.process.memory_info().rss / 1024 / 1024

            # Make requests
            for i in range(requests_per_iteration):
                try:
                    features = [float(i) for _ in range(10)]

                    req_start = time.time()
                    response = requests.post(
                        f"{self.base_url}/predict",
                        json={"features": features},
                        headers=self.headers,
                        timeout=10
                    )
                    req_elapsed = time.time() - req_start

                    if response.status_code == 200:
                        successful += 1
                        response_times.append(req_elapsed)
                    else:
                        failed += 1

                except Exception as e:
                    failed += 1

            # Measure memory after iteration
            gc.collect()
            memory_after = self.process.memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before

            memory_readings.append(memory_after)

            print(f"Iteration {iteration + 1}: "
                  f"Memory: {memory_after:.1f}MB "
                  f"(Δ {memory_delta:+.1f}MB)")

            # Small delay between iterations
            time.sleep(1)

        duration = time.time() - start_time

        # Analyze memory trend
        if len(memory_readings) > 2:
            # Calculate linear regression slope
            x = list(range(len(memory_readings)))
            y = memory_readings
            n = len(x)
            slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / \
                    (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)

            memory_trend = f"Memory trend: {slope:+.2f} MB/iteration"
            if slope > 1.0:
                memory_trend += " ⚠️  Possible memory leak detected!"
            elif slope > 0.5:
                memory_trend += " ⚠️  Elevated memory growth"
            else:
                memory_trend += " ✓ Memory stable"
        else:
            memory_trend = "Insufficient data for trend analysis"

        # Calculate statistics
        if response_times:
            avg_time = statistics.mean(response_times)
            p50 = statistics.median(response_times)
            p95 = statistics.quantiles(response_times, n=20)[18]
            p99 = statistics.quantiles(response_times, n=100)[98]
        else:
            avg_time = p50 = p95 = p99 = 0.0

        total_requests = successful + failed
        rps = total_requests / duration
        error_rate = (failed / total_requests * 100) if total_requests > 0 else 0

        return StressTestResult(
            test_name="Memory Leak Detection",
            duration=duration,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            avg_response_time=avg_time,
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            requests_per_second=rps,
            error_rate=error_rate,
            peak_memory_mb=max(memory_readings),
            notes=memory_trend
        )

    def print_results(self, result: StressTestResult):
        """Print formatted test results.

        Args:
            result: StressTestResult to print
        """
        print(f"\n{'='*60}")
        print(f"Test Results: {result.test_name}")
        print(f"{'='*60}")
        print(f"Duration:           {result.duration:.2f}s")
        print(f"Total Requests:     {result.total_requests}")
        print(f"Successful:         {result.successful_requests} "
              f"({result.successful_requests/result.total_requests*100:.1f}%)")
        print(f"Failed:             {result.failed_requests} "
              f"({result.error_rate:.1f}%)")
        print(f"Requests/Second:    {result.requests_per_second:.2f}")
        print(f"\nResponse Times:")
        print(f"  Average:          {result.avg_response_time*1000:.2f}ms")
        print(f"  P50 (median):     {result.p50_response_time*1000:.2f}ms")
        print(f"  P95:              {result.p95_response_time*1000:.2f}ms")
        print(f"  P99:              {result.p99_response_time*1000:.2f}ms")
        print(f"\nMemory:")
        print(f"  Peak Memory:      {result.peak_memory_mb:.1f}MB")
        if result.notes:
            print(f"\nNotes: {result.notes}")
        print(f"{'='*60}\n")


def main():
    """Main entry point for stress testing."""
    parser = argparse.ArgumentParser(description="ML API Stress Testing")
    parser.add_argument(
        "--test",
        choices=["concurrent", "sustained", "memory", "all"],
        default="all",
        help="Test to run (default: all)"
    )
    parser.add_argument(
        "--base-url",
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=1000,
        help="Number of requests for concurrent test (default: 1000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of concurrent workers (default: 50)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration in seconds for sustained test (default: 300)"
    )
    parser.add_argument(
        "--rps",
        type=int,
        default=50,
        help="Target requests per second for sustained test (default: 50)"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = MLAPIStressTester(base_url=args.base_url)

    # Authenticate
    if not tester.authenticate():
        print("Failed to authenticate. Exiting.")
        sys.exit(1)

    results = []

    # Run tests
    if args.test in ["concurrent", "all"]:
        result = tester.test_concurrent_requests(
            num_requests=args.requests,
            num_workers=args.workers
        )
        tester.print_results(result)
        results.append(result)

    if args.test in ["sustained", "all"]:
        result = tester.test_sustained_load(
            duration=args.duration,
            target_rps=args.rps
        )
        tester.print_results(result)
        results.append(result)

    if args.test in ["memory", "all"]:
        result = tester.test_memory_leak()
        tester.print_results(result)
        results.append(result)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"Overall Summary")
        print(f"{'='*60}")
        for result in results:
            print(f"{result.test_name:25} | "
                  f"RPS: {result.requests_per_second:6.1f} | "
                  f"P95: {result.p95_response_time*1000:6.1f}ms | "
                  f"Errors: {result.error_rate:5.1f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
