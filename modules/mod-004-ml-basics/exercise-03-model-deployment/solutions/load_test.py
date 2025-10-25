#!/usr/bin/env python3
"""
load_test.py - Load testing tool for deployed ML models

Description:
    Performance testing tool for ML model APIs with support for
    concurrent requests, throughput measurement, and latency analysis.

Usage:
    python load_test.py --url http://localhost:8000/predict --requests 1000 --concurrency 10

Options:
    --url URL              API endpoint URL
    --requests N           Total number of requests (default: 1000)
    --concurrency N        Number of concurrent workers (default: 10)
    --duration SECONDS     Test duration in seconds (overrides --requests)
    --data FILE            Input data file (JSON)
    --report FILE          Save report to file
    --verbose              Verbose output
    --help                 Display this help message
"""

import argparse
import json
import time
import asyncio
import aiohttp
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import sys

# Colors for output
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


@dataclass
class RequestResult:
    """Single request result"""
    success: bool
    latency_ms: float
    status_code: int
    error: str = ""
    timestamp: float = 0.0


@dataclass
class LoadTestResults:
    """Load test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    requests_per_second: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    error_rate: float


class LoadTester:
    """Load testing orchestrator"""

    def __init__(self, url: str, total_requests: int = 1000,
                 concurrency: int = 10, duration: float = None,
                 data: Dict = None, verbose: bool = False):
        """
        Initialize load tester

        Args:
            url: API endpoint URL
            total_requests: Total number of requests
            concurrency: Number of concurrent workers
            duration: Test duration in seconds (optional)
            data: Request data
            verbose: Verbose output
        """
        self.url = url
        self.total_requests = total_requests
        self.concurrency = concurrency
        self.duration = duration
        self.data = data or {"data": [[1.0, 2.0, 3.0]]}
        self.verbose = verbose

        self.results: List[RequestResult] = []
        self.start_time = None
        self.end_time = None

    async def make_request(self, session: aiohttp.ClientSession) -> RequestResult:
        """Make single API request"""
        start_time = time.time()

        try:
            async with session.post(self.url, json=self.data) as response:
                latency = (time.time() - start_time) * 1000  # ms
                await response.text()  # Read response

                return RequestResult(
                    success=response.status == 200,
                    latency_ms=latency,
                    status_code=response.status,
                    timestamp=start_time
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return RequestResult(
                success=False,
                latency_ms=latency,
                status_code=0,
                error=str(e),
                timestamp=start_time
            )

    async def worker(self, worker_id: int, session: aiohttp.ClientSession,
                    queue: asyncio.Queue):
        """Worker coroutine"""
        while True:
            try:
                await queue.get()

                # Make request
                result = await self.make_request(session)
                self.results.append(result)

                if self.verbose:
                    status = "✓" if result.success else "✗"
                    print(f"Worker {worker_id}: {status} Latency: {result.latency_ms:.2f}ms")

                queue.task_done()

            except asyncio.CancelledError:
                break

    async def run_load_test(self):
        """Run load test"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Running Load Test{Colors.RESET}")
        print("=" * 80)
        print(f"URL: {self.url}")
        print(f"Total Requests: {self.total_requests}")
        print(f"Concurrency: {self.concurrency}")
        if self.duration:
            print(f"Duration: {self.duration}s")
        print("=" * 80 + "\n")

        # Create queue
        queue = asyncio.Queue()

        # Fill queue
        for _ in range(self.total_requests):
            queue.put_nowait(None)

        # Create session
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Create workers
            workers = [
                asyncio.create_task(self.worker(i, session, queue))
                for i in range(self.concurrency)
            ]

            # Start timer
            self.start_time = time.time()

            # Wait for completion or timeout
            if self.duration:
                try:
                    await asyncio.wait_for(queue.join(), timeout=self.duration)
                except asyncio.TimeoutError:
                    pass
            else:
                await queue.join()

            # End timer
            self.end_time = time.time()

            # Cancel workers
            for worker in workers:
                worker.cancel()

            await asyncio.gather(*workers, return_exceptions=True)

        print(f"\n{Colors.GREEN}Test completed!{Colors.RESET}\n")

    def analyze_results(self) -> LoadTestResults:
        """Analyze test results"""
        if not self.results:
            return None

        # Filter successful requests
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        # Calculate latencies
        latencies = [r.latency_ms for r in successful]

        if not latencies:
            latencies = [0]

        # Sort for percentiles
        latencies_sorted = sorted(latencies)

        # Calculate percentiles
        p95_idx = int(len(latencies_sorted) * 0.95)
        p99_idx = int(len(latencies_sorted) * 0.99)

        # Total time
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0

        # Calculate RPS
        rps = len(self.results) / total_time if total_time > 0 else 0

        return LoadTestResults(
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_time_seconds=total_time,
            requests_per_second=rps,
            latency_mean_ms=statistics.mean(latencies),
            latency_median_ms=statistics.median(latencies),
            latency_p95_ms=latencies_sorted[p95_idx] if latencies_sorted else 0,
            latency_p99_ms=latencies_sorted[p99_idx] if latencies_sorted else 0,
            latency_min_ms=min(latencies) if latencies else 0,
            latency_max_ms=max(latencies) if latencies else 0,
            error_rate=len(failed) / len(self.results) if self.results else 0
        )

    def print_results(self, results: LoadTestResults):
        """Print results to console"""
        print(f"{Colors.BOLD}{Colors.CYAN}Load Test Results{Colors.RESET}")
        print("=" * 80)

        # Request statistics
        print(f"\n{Colors.BOLD}Requests:{Colors.RESET}")
        print(f"  Total: {results.total_requests}")
        print(f"  Successful: {Colors.GREEN}{results.successful_requests}{Colors.RESET}")
        print(f"  Failed: {Colors.RED if results.failed_requests > 0 else Colors.GREEN}{results.failed_requests}{Colors.RESET}")
        print(f"  Error Rate: {results.error_rate * 100:.2f}%")

        # Throughput
        print(f"\n{Colors.BOLD}Throughput:{Colors.RESET}")
        print(f"  Total Time: {results.total_time_seconds:.2f}s")
        print(f"  Requests/sec: {Colors.GREEN}{results.requests_per_second:.2f}{Colors.RESET}")

        # Latency statistics
        print(f"\n{Colors.BOLD}Latency (ms):{Colors.RESET}")
        print(f"  Mean: {results.latency_mean_ms:.2f}")
        print(f"  Median: {results.latency_median_ms:.2f}")
        print(f"  P95: {results.latency_p95_ms:.2f}")
        print(f"  P99: {results.latency_p99_ms:.2f}")
        print(f"  Min: {results.latency_min_ms:.2f}")
        print(f"  Max: {results.latency_max_ms:.2f}")

        print("\n" + "=" * 80)

    def save_report(self, results: LoadTestResults, filename: str):
        """Save results to JSON file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'url': self.url,
                'total_requests': self.total_requests,
                'concurrency': self.concurrency,
                'duration': self.duration
            },
            'results': asdict(results)
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{Colors.GREEN}Report saved to: {filename}{Colors.RESET}")


async def main():
    parser = argparse.ArgumentParser(
        description='Load testing tool for ML model APIs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--url', type=str, required=True,
                       help='API endpoint URL')
    parser.add_argument('--requests', type=int, default=1000,
                       help='Total number of requests (default: 1000)')
    parser.add_argument('--concurrency', type=int, default=10,
                       help='Number of concurrent workers (default: 10)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Test duration in seconds (overrides --requests)')
    parser.add_argument('--data', type=str, default=None,
                       help='Input data file (JSON)')
    parser.add_argument('--report', type=str, default=None,
                       help='Save report to file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Load data if provided
    data = None
    if args.data:
        with open(args.data) as f:
            data = json.load(f)

    # Create load tester
    tester = LoadTester(
        url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        duration=args.duration,
        data=data,
        verbose=args.verbose
    )

    # Run test
    await tester.run_load_test()

    # Analyze results
    results = tester.analyze_results()

    if results:
        # Print results
        tester.print_results(results)

        # Save report if requested
        if args.report:
            tester.save_report(results, args.report)
    else:
        print(f"{Colors.RED}No results to analyze{Colors.RESET}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
