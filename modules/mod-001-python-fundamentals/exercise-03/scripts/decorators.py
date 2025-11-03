#!/usr/bin/env python3
"""
Decorators for Common ML Patterns

Demonstrates practical decorators for ML workflows including timing, logging,
retry logic, caching, and input validation.
"""

import time
import functools
from typing import Any, Callable
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Useful for profiling ML operations and identifying bottlenecks.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that logs execution time

    Examples:
        >>> @timing_decorator
        ... def train_model():
        ...     time.sleep(1)
        ...     return "done"
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:.4f} seconds")

        return result

    return wrapper


def log_calls(func: Callable) -> Callable:
    """
    Decorator to log function calls with arguments and return values.

    Useful for debugging and tracking function invocations in ML pipelines.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that logs calls

    Examples:
        >>> @log_calls
        ... def preprocess_data(data, normalize=True):
        ...     return data
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Format arguments for logging
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        logger.info(f"Calling {func.__name__}({signature})")

        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} returned {result!r}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure.

    Useful for operations that might fail transiently (network calls,
    file I/O, external API calls).

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay: Delay between retries in seconds (default: 1.0)

    Returns:
        Decorator function

    Examples:
        >>> @retry(max_attempts=3, delay=0.5)
        ... def load_model(path):
        ...     # Might fail due to network issues
        ...     return model
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempts}/{max_attempts}). "
                        f"Retrying in {delay}s... Error: {e}"
                    )
                    time.sleep(delay)

        return wrapper
    return decorator


def cache_results(func: Callable) -> Callable:
    """
    Decorator to cache function results (memoization).

    Useful for expensive computations that are called repeatedly with
    the same arguments (feature extraction, metric computation, etc.).

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with caching

    Examples:
        >>> @cache_results
        ... def compute_features(data_id):
        ...     # Expensive feature extraction
        ...     return features
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from args and kwargs
        # Note: This simple implementation won't work with unhashable types
        cache_key = str(args) + str(sorted(kwargs.items()))

        if cache_key in cache:
            logger.info(f"Cache hit for {func.__name__}")
            return cache[cache_key]

        logger.info(f"Cache miss for {func.__name__}, computing...")
        result = func(*args, **kwargs)
        cache[cache_key] = result

        return result

    # Attach cache for inspection/clearing
    wrapper.cache = cache
    wrapper.cache_clear = lambda: cache.clear()

    return wrapper


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.

    Uses validator functions to check argument values before execution.
    Useful for ensuring data quality and catching errors early.

    Args:
        **validators: Dict mapping parameter names to validator functions

    Returns:
        Decorator function

    Examples:
        >>> @validate_inputs(
        ...     learning_rate=lambda x: 0 < x < 1,
        ...     batch_size=lambda x: isinstance(x, int) and x > 0
        ... )
        ... def configure_training(learning_rate, batch_size):
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature for argument binding
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each argument
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}' "
                            f"with value {value}"
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def count_calls(func: Callable) -> Callable:
    """
    Decorator to count function calls.

    Useful for monitoring how often certain operations are performed.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that counts calls
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        logger.info(f"{func.__name__} called {wrapper.call_count} times")
        return func(*args, **kwargs)

    wrapper.call_count = 0
    return wrapper


def rate_limit(calls_per_second: float):
    """
    Decorator to rate-limit function calls.

    Useful for API calls, database queries, or other operations
    that need throttling.

    Args:
        calls_per_second: Maximum calls per second

    Returns:
        Decorator function
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # Use list to make it mutable

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.info(
                    f"Rate limiting {func.__name__}, sleeping {sleep_time:.3f}s"
                )
                time.sleep(sleep_time)

            last_called[0] = time.time()
            return func(*args, **kwargs)

        return wrapper
    return decorator


# Example functions with decorators

@timing_decorator
@log_calls
def train_model(epochs: int, batch_size: int) -> float:
    """Simulate model training."""
    logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    time.sleep(0.5)  # Simulate training time
    return 0.92


@retry(max_attempts=3, delay=0.5)
def load_model_from_storage(model_path: str) -> str:
    """Simulate loading model from storage (might fail)."""
    import random

    if random.random() < 0.5:
        raise IOError(f"Failed to load model from {model_path}")

    return f"Model loaded from {model_path}"


@cache_results
@timing_decorator
def compute_expensive_metric(data_size: int) -> float:
    """Simulate expensive computation."""
    logger.info(f"Computing metric for data size {data_size}")
    time.sleep(1.0)
    return data_size * 0.001


@validate_inputs(
    learning_rate=lambda x: 0 < x < 1,
    batch_size=lambda x: isinstance(x, int) and x > 0
)
def configure_training(learning_rate: float, batch_size: int) -> dict:
    """Configure training with validation."""
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }


@count_calls
def preprocess_batch(batch_data: list) -> list:
    """Preprocess a batch of data."""
    return [x * 2 for x in batch_data]


@rate_limit(calls_per_second=2.0)
def call_external_api(endpoint: str) -> dict:
    """Simulate API call with rate limiting."""
    logger.info(f"Calling API endpoint: {endpoint}")
    return {"status": "success", "data": [1, 2, 3]}


def main():
    """Demonstrate decorator patterns."""
    print("=" * 70)
    print("Decorators for Common ML Patterns")
    print("=" * 70)
    print()

    # Example 1: Timing and logging
    print("Example 1: Timing and Logging Decorators")
    print("-" * 70)
    accuracy = train_model(epochs=10, batch_size=32)
    print(f"Training accuracy: {accuracy}")
    print()

    # Example 2: Retry decorator
    print("\nExample 2: Retry Decorator")
    print("-" * 70)
    print("Attempting to load model (may require retries)...")
    try:
        model = load_model_from_storage("/models/resnet50.h5")
        print(f"Success: {model}")
    except IOError as e:
        print(f"Failed after all retries: {e}")
    print()

    # Example 3: Caching decorator
    print("\nExample 3: Caching Decorator")
    print("-" * 70)
    print("First call (cache miss):")
    result1 = compute_expensive_metric(1000)
    print(f"Result: {result1}")

    print("\nSecond call with same argument (cache hit):")
    result2 = compute_expensive_metric(1000)
    print(f"Result: {result2}")

    print("\nThird call with different argument (cache miss):")
    result3 = compute_expensive_metric(2000)
    print(f"Result: {result3}")

    print(f"\nCache size: {len(compute_expensive_metric.cache)}")
    print()

    # Example 4: Input validation
    print("\nExample 4: Input Validation Decorator")
    print("-" * 70)
    print("Valid configuration:")
    try:
        config1 = configure_training(learning_rate=0.001, batch_size=32)
        print(f"  {config1}")
    except ValueError as e:
        print(f"  Error: {e}")

    print("\nInvalid configuration (learning_rate out of range):")
    try:
        config2 = configure_training(learning_rate=1.5, batch_size=32)
        print(f"  {config2}")
    except ValueError as e:
        print(f"  Error: {e}")

    print("\nInvalid configuration (batch_size not int):")
    try:
        config3 = configure_training(learning_rate=0.001, batch_size=32.5)
        print(f"  {config3}")
    except ValueError as e:
        print(f"  Error: {e}")
    print()

    # Example 5: Call counting
    print("\nExample 5: Call Counting Decorator")
    print("-" * 70)
    for i in range(5):
        batch = [1, 2, 3, 4, 5]
        result = preprocess_batch(batch)
    print(f"Final call count: {preprocess_batch.call_count}")
    print()

    # Example 6: Rate limiting
    print("\nExample 6: Rate Limiting Decorator")
    print("-" * 70)
    print("Making 5 API calls (limited to 2 calls/second):")
    start = time.time()
    for i in range(5):
        response = call_external_api(f"/api/data/{i}")
        print(f"  Call {i+1}: {response['status']}")
    elapsed = time.time() - start
    print(f"Total time: {elapsed:.2f}s (expected: ~2.0s for 5 calls)")
    print()

    # Example 7: Stacking multiple decorators
    print("\nExample 7: Stacking Multiple Decorators")
    print("-" * 70)

    @timing_decorator
    @cache_results
    @log_calls
    def complex_computation(x: int, y: int) -> int:
        """Demonstration of decorator stacking."""
        time.sleep(0.2)
        return x ** 2 + y ** 2

    print("First call (all decorators applied):")
    result = complex_computation(3, 4)
    print(f"Result: {result}")

    print("\nSecond call (cache hit, timing still applied):")
    result = complex_computation(3, 4)
    print(f"Result: {result}")
    print()

    print("=" * 70)
    print("✓ All decorator demonstrations completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
