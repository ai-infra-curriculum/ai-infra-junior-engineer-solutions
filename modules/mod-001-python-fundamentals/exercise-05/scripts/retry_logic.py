#!/usr/bin/env python3
"""
Retry Logic and Resilience for ML Workflows

Demonstrates retry mechanisms with exponential backoff for handling transient failures.
"""

import time
import random
from typing import Callable, Any, Optional, List
from functools import wraps
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3,
                      initial_delay: float = 1.0,
                      backoff_factor: float = 2.0,
                      exceptions: tuple = (Exception,)):
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.2f}s... Error: {e}"
                    )

                    time.sleep(delay)
                    delay *= backoff_factor

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


@retry_with_backoff(max_retries=3, initial_delay=0.5)
def download_model(model_url: str, failure_rate: float = 0.7) -> dict:
    """
    Simulate model download with potential failures.

    Args:
        model_url: URL to download from
        failure_rate: Probability of failure (for simulation)

    Returns:
        Download result dictionary
    """
    if random.random() < failure_rate:
        raise ConnectionError(f"Failed to download from {model_url}")

    logger.info(f"✓ Successfully downloaded model from {model_url}")
    return {"url": model_url, "status": "downloaded", "size_mb": 150}


@retry_with_backoff(max_retries=5, initial_delay=1.0, backoff_factor=1.5)
def load_from_storage(file_path: str, failure_rate: float = 0.5) -> dict:
    """
    Load file from storage with retry logic.

    Args:
        file_path: Path to file in storage
        failure_rate: Probability of failure (for simulation)

    Returns:
        Loaded data dictionary
    """
    if random.random() < failure_rate:
        raise IOError(f"Storage temporarily unavailable: {file_path}")

    logger.info(f"✓ Successfully loaded from {file_path}")
    return {"path": file_path, "data": "model_weights", "size_mb": 200}


@retry_with_backoff(
    max_retries=3,
    initial_delay=0.5,
    exceptions=(ConnectionError, TimeoutError)
)
def fetch_dataset(api_endpoint: str) -> dict:
    """
    Fetch dataset from API with specific exception handling.

    Args:
        api_endpoint: API endpoint URL

    Returns:
        Dataset dictionary
    """
    if random.random() < 0.6:
        if random.random() < 0.5:
            raise ConnectionError("Connection to API failed")
        else:
            raise TimeoutError("API request timed out")

    return {"endpoint": api_endpoint, "samples": 10000, "features": 50}


class ResilientDataLoader:
    """Data loader with automatic retry and fallback to backup sources"""

    def __init__(self, primary_source: str, backup_sources: List[str]):
        """
        Initialize resilient data loader.

        Args:
            primary_source: Primary data source
            backup_sources: List of backup data sources
        """
        self.primary_source = primary_source
        self.backup_sources = backup_sources

    def load_data(self) -> Optional[dict]:
        """
        Load data with fallback to backup sources.

        Returns:
            Loaded data or None if all sources fail
        """
        sources = [self.primary_source] + self.backup_sources

        for i, source in enumerate(sources):
            try:
                logger.info(f"Attempting to load from source {i + 1}/{len(sources)}: {source}")
                data = self._load_from_source(source)
                logger.info(f"✓ Successfully loaded from {source}")
                return data
            except Exception as e:
                logger.warning(f"✗ Failed to load from {source}: {e}")

                if i == len(sources) - 1:
                    logger.error("All data sources failed")
                    return None
                else:
                    logger.info(f"Trying next source...")

        return None

    @retry_with_backoff(max_retries=2, initial_delay=0.5)
    def _load_from_source(self, source: str) -> dict:
        """
        Load from specific source with retry.

        Args:
            source: Data source identifier

        Returns:
            Loaded data dictionary
        """
        # Simulate loading with high failure rate
        if random.random() < 0.6:
            raise IOError(f"Source unavailable: {source}")

        return {"source": source, "data": list(range(1000)), "size_mb": 50}


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures"""

    def __init__(self, failure_threshold: int = 3, timeout: float = 60.0):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting reset
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            # Check if timeout has elapsed
            if time.time() - self.last_failure_time >= self.timeout:
                logger.info("Circuit breaker: Attempting reset (half-open)")
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN - not attempting call")

        try:
            result = func(*args, **kwargs)

            # Success - reset if half-open
            if self.state == "half-open":
                logger.info("Circuit breaker: Reset successful (closed)")
                self.state = "closed"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker: OPENED after {self.failure_count} failures")

            raise


def main():
    """Demonstrate retry logic and resilience patterns"""
    print("=" * 70)
    print("Retry Logic and Resilience")
    print("=" * 70)
    print()

    # Example 1: Simple retry with backoff
    print("Example 1: Download Model with Retry")
    print("-" * 70)
    try:
        result = download_model("https://example.com/model.h5")
        print(f"✓ Download result: {result}")
    except ConnectionError as e:
        print(f"✗ Download failed after all retries: {e}")
    print()

    # Example 2: Storage loading with different backoff
    print("Example 2: Load from Storage with Custom Backoff")
    print("-" * 70)
    try:
        data = load_from_storage("/storage/models/resnet50.pkl")
        print(f"✓ Loaded: {data}")
    except IOError as e:
        print(f"✗ Load failed after all retries: {e}")
    print()

    # Example 3: Specific exception handling
    print("Example 3: API Fetch with Specific Exception Types")
    print("-" * 70)
    try:
        dataset = fetch_dataset("https://api.example.com/dataset")
        print(f"✓ Dataset: {dataset}")
    except (ConnectionError, TimeoutError) as e:
        print(f"✗ Fetch failed: {e}")
    print()

    # Example 4: Resilient loader with fallback
    print("Example 4: Resilient Data Loader with Multiple Sources")
    print("-" * 70)
    loader = ResilientDataLoader(
        primary_source="s3://primary-bucket/data.csv",
        backup_sources=[
            "s3://backup1-bucket/data.csv",
            "s3://backup2-bucket/data.csv",
            "local://fallback/data.csv"
        ]
    )

    data = loader.load_data()
    if data:
        print(f"✓ Data loaded successfully")
        print(f"  Source: {data['source']}")
        print(f"  Size: {data['size_mb']}MB")
    else:
        print("✗ Failed to load from all sources")
    print()

    # Example 5: Circuit breaker
    print("Example 5: Circuit Breaker Pattern")
    print("-" * 70)

    def flaky_service():
        """Simulated flaky service"""
        if random.random() < 0.8:  # 80% failure rate
            raise ConnectionError("Service unavailable")
        return "success"

    breaker = CircuitBreaker(failure_threshold=3, timeout=5.0)

    for i in range(6):
        try:
            result = breaker.call(flaky_service)
            print(f"Call {i + 1}: ✓ {result} (state: {breaker.state})")
        except Exception as e:
            print(f"Call {i + 1}: ✗ {type(e).__name__}: {e} (state: {breaker.state})")
        time.sleep(0.2)
    print()

    # Example 6: Retry statistics
    print("Example 6: Retry Statistics")
    print("-" * 70)

    success_count = 0
    total_attempts = 10

    for i in range(total_attempts):
        try:
            download_model(f"https://example.com/model_{i}.h5", failure_rate=0.6)
            success_count += 1
        except ConnectionError:
            pass

    print(f"Success rate: {success_count}/{total_attempts} "
          f"({100 * success_count / total_attempts:.1f}%)")
    print()

    print("=" * 70)
    print("✓ Retry logic demonstration complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
