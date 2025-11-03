"""Timing decorator for performance monitoring."""

import time
import functools
from typing import Callable, Any


def timer(func: Callable) -> Callable:
    """
    Decorator to measure and print function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that prints execution time

    Examples:
        >>> @timer
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()  # doctest: +SKIP
        slow_function took 1.0001 seconds
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result

    return wrapper


def timer_with_units(unit: str = "seconds") -> Callable:
    """
    Decorator factory to measure and print function execution time with custom units.

    Args:
        unit: Time unit ('seconds', 'milliseconds', 'microseconds')

    Returns:
        Decorator function

    Examples:
        >>> @timer_with_units("milliseconds")
        ... def fast_function():
        ...     pass
        >>> fast_function()  # doctest: +SKIP
        fast_function took 0.0100 milliseconds
    """
    unit_multipliers = {
        "seconds": 1,
        "milliseconds": 1000,
        "microseconds": 1_000_000,
    }

    if unit not in unit_multipliers:
        raise ValueError(f"Invalid unit: {unit}. Must be one of {list(unit_multipliers.keys())}")

    multiplier = unit_multipliers[unit]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = (end_time - start_time) * multiplier
            print(f"{func.__name__} took {elapsed_time:.4f} {unit}")
            return result

        return wrapper

    return decorator
