"""Retry decorator with exponential backoff."""

import time
import functools
from typing import Callable, Any, Tuple, Type


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator to retry function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Multiplier for delay after each attempt (default: 2.0)
        exceptions: Tuple of exception types to catch and retry (default: (Exception,))

    Returns:
        Decorated function that retries on failure

    Examples:
        >>> @retry(max_attempts=3, delay=0.1)
        ... def unreliable_function():
        ...     import random
        ...     if random.random() < 0.5:
        ...         raise ValueError("Random failure")
        ...     return "Success"
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        print(
                            f"{func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {e}"
                        )
                        raise

                    print(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            # This should never be reached, but for type safety
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def retry_on_condition(
    max_attempts: int = 3,
    delay: float = 1.0,
    condition: Callable[[Any], bool] = lambda x: x is None,
) -> Callable:
    """
    Decorator to retry function until result meets condition.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
        condition: Function to test result; retry if returns True

    Returns:
        Decorated function that retries based on result

    Examples:
        >>> @retry_on_condition(max_attempts=3, condition=lambda x: x is None)
        ... def get_data():
        ...     return {"data": "value"}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                result = func(*args, **kwargs)

                if not condition(result):
                    return result

                if attempt < max_attempts:
                    print(
                        f"{func.__name__} attempt {attempt}/{max_attempts} - "
                        f"result didn't meet condition. Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)

            # Return last result even if condition not met
            print(
                f"{func.__name__} exhausted {max_attempts} attempts. "
                f"Returning last result."
            )
            return result

        return wrapper

    return decorator
