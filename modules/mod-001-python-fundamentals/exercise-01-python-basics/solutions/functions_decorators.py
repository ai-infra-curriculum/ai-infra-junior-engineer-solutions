#!/usr/bin/env python3
"""
Functions and Decorators

Comprehensive examples of Python functions, decorators, closures, and functional programming.
"""

import functools
import time
from typing import Callable, Any, List, Dict, Optional
from datetime import datetime


# ============= Basic Functions =============

def calculate_stats(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.

    Args:
        numbers: List of numeric values

    Returns:
        Dictionary containing min, max, mean, median

    Raises:
        ValueError: If list is empty

    Example:
        >>> calculate_stats([1, 2, 3, 4, 5])
        {'min': 1, 'max': 5, 'mean': 3.0, 'median': 3.0}
    """
    if not numbers:
        raise ValueError("Cannot calculate stats for empty list")

    sorted_numbers = sorted(numbers)
    n = len(numbers)

    median = (
        sorted_numbers[n // 2]
        if n % 2 == 1
        else (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
    )

    return {
        'min': min(numbers),
        'max': max(numbers),
        'mean': sum(numbers) / len(numbers),
        'median': median,
        'count': n,
    }


def fibonacci(n: int) -> List[int]:
    """
    Generate Fibonacci sequence up to n terms.

    Args:
        n: Number of terms to generate

    Returns:
        List of Fibonacci numbers

    Example:
        >>> fibonacci(7)
        [0, 1, 1, 2, 3, 5, 8]
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])

    return fib


def fibonacci_generator(n: int):
    """
    Generate Fibonacci sequence lazily using a generator.

    Args:
        n: Number of terms to generate

    Yields:
        Next Fibonacci number

    Example:
        >>> list(fibonacci_generator(5))
        [0, 1, 1, 2, 3]
    """
    a, b = 0, 1
    count = 0

    while count < n:
        yield a
        a, b = b, a + b
        count += 1


# ============= Higher-Order Functions =============

def apply_operation(numbers: List[float], operation: Callable[[float], float]) -> List[float]:
    """
    Apply a function to each number in the list.

    Args:
        numbers: List of numbers
        operation: Function to apply to each number

    Returns:
        List of transformed numbers

    Example:
        >>> apply_operation([1, 2, 3], lambda x: x ** 2)
        [1, 4, 9]
    """
    return [operation(num) for num in numbers]


def filter_data(items: List[Any], predicate: Callable[[Any], bool]) -> List[Any]:
    """
    Filter items based on a predicate function.

    Args:
        items: List of items to filter
        predicate: Function that returns True for items to keep

    Returns:
        Filtered list

    Example:
        >>> filter_data([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        [2, 4]
    """
    return [item for item in items if predicate(item)]


def reduce_with_func(items: List[Any], func: Callable[[Any, Any], Any], initial: Any = None) -> Any:
    """
    Reduce a list to a single value using a function.

    Args:
        items: List of items
        func: Binary function to apply cumulatively
        initial: Initial value

    Returns:
        Reduced value

    Example:
        >>> reduce_with_func([1, 2, 3, 4], lambda x, y: x + y, 0)
        10
    """
    if initial is None:
        if not items:
            raise ValueError("Cannot reduce empty sequence without initial value")
        result = items[0]
        items = items[1:]
    else:
        result = initial

    for item in items:
        result = func(result, item)

    return result


# ============= Decorators =============

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to decorate

    Returns:
        Wrapped function that prints execution time

    Example:
        >>> @timing_decorator
        ... def slow_function():
        ...     time.sleep(1)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper


def retry_decorator(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds

    Returns:
        Decorator function

    Example:
        >>> @retry_decorator(max_attempts=3)
        ... def flaky_function():
        ...     # Might fail sometimes
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def cache_decorator(func: Callable) -> Callable:
    """
    Simple caching decorator (memoization).

    Args:
        func: Function to cache results for

    Returns:
        Wrapped function with caching

    Example:
        >>> @cache_decorator
        ... def expensive_computation(n):
        ...     return n ** 2
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    wrapper.cache = cache  # Expose cache for testing
    return wrapper


def validate_types(**type_hints):
    """
    Decorator to validate function argument types.

    Args:
        **type_hints: Keyword arguments mapping parameter names to expected types

    Returns:
        Decorator function

    Example:
        >>> @validate_types(x=int, y=int)
        ... def add(x, y):
        ...     return x + y
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function parameter names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate types
            for param_name, expected_type in type_hints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_calls(func: Callable) -> Callable:
    """
    Decorator to log function calls with arguments and results.

    Args:
        func: Function to log

    Returns:
        Wrapped function with logging

    Example:
        >>> @log_calls
        ... def add(a, b):
        ...     return a + b
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        print(f"[{timestamp}] Calling {func.__name__}({signature})")

        try:
            result = func(*args, **kwargs)
            print(f"[{timestamp}] {func.__name__} returned {result!r}")
            return result
        except Exception as e:
            print(f"[{timestamp}] {func.__name__} raised {type(e).__name__}: {e}")
            raise

    return wrapper


# ============= Closures =============

def make_multiplier(factor: float) -> Callable[[float], float]:
    """
    Create a multiplier function with a specific factor.

    Args:
        factor: Multiplication factor

    Returns:
        Function that multiplies its input by factor

    Example:
        >>> times_three = make_multiplier(3)
        >>> times_three(4)
        12
    """
    def multiplier(x: float) -> float:
        return x * factor
    return multiplier


def make_counter(start: int = 0) -> Callable[[], int]:
    """
    Create a counter function that increments on each call.

    Args:
        start: Starting value

    Returns:
        Function that returns next count on each call

    Example:
        >>> counter = make_counter(10)
        >>> counter()
        11
        >>> counter()
        12
    """
    count = [start]  # Use list to allow modification in closure

    def counter() -> int:
        count[0] += 1
        return count[0]

    return counter


def make_accumulator(initial: float = 0) -> Callable[[float], float]:
    """
    Create an accumulator function.

    Args:
        initial: Initial accumulated value

    Returns:
        Function that adds to accumulator and returns total

    Example:
        >>> acc = make_accumulator()
        >>> acc(5)
        5
        >>> acc(10)
        15
    """
    total = [initial]

    def accumulator(value: float) -> float:
        total[0] += value
        return total[0]

    return accumulator


# ============= Partial Functions =============

def create_partial_function():
    """
    Demonstrate partial function application.

    Returns:
        Dictionary of partial function examples
    """
    def power(base: float, exponent: float) -> float:
        return base ** exponent

    # Create partial functions
    square = functools.partial(power, exponent=2)
    cube = functools.partial(power, exponent=3)

    return {
        'square_of_5': square(5),      # 25
        'cube_of_3': cube(3),          # 27
    }


# ============= Lambda Functions =============

# Common lambda function examples
LAMBDA_EXAMPLES = {
    'square': lambda x: x ** 2,
    'add': lambda x, y: x + y,
    'is_even': lambda x: x % 2 == 0,
    'get_name': lambda person: person.get('name', 'Unknown'),
    'sort_by_second': lambda x: x[1],
}


# ============= Practical Examples =============

@timing_decorator
@cache_decorator
def expensive_fibonacci(n: int) -> int:
    """
    Calculate nth Fibonacci number (with caching and timing).

    Args:
        n: Position in Fibonacci sequence

    Returns:
        Fibonacci number at position n

    Example:
        >>> expensive_fibonacci(10)
        55
    """
    if n <= 1:
        return n
    return expensive_fibonacci(n - 1) + expensive_fibonacci(n - 2)


@validate_types(text=str, n=int)
def repeat_string(text: str, n: int) -> str:
    """
    Repeat string n times (with type validation).

    Args:
        text: String to repeat
        n: Number of repetitions

    Returns:
        Repeated string

    Example:
        >>> repeat_string("Hi", 3)
        'HiHiHi'
    """
    return text * n


def compose(*functions):
    """
    Compose multiple functions into a single function.

    Args:
        *functions: Functions to compose (applied right to left)

    Returns:
        Composed function

    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> f = compose(add_one, double)
        >>> f(3)  # (3 * 2) + 1
        7
    """
    def composed(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return composed


if __name__ == '__main__':
    print("=== Functions and Decorators Examples ===\n")

    # Basic functions
    print("Statistics:", calculate_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print("Fibonacci:", fibonacci(10))

    # Decorators
    print("\nTimed Fibonacci:")
    result = expensive_fibonacci(10)
    print(f"Result: {result}")

    # Closures
    print("\nClosures:")
    times_three = make_multiplier(3)
    print(f"3 × 5 = {times_three(5)}")

    counter = make_counter(0)
    print(f"Counter: {counter()}, {counter()}, {counter()}")

    # Lambda and higher-order functions
    print("\nHigher-order functions:")
    numbers = [1, 2, 3, 4, 5]
    print(f"Squares: {apply_operation(numbers, lambda x: x ** 2)}")
    print(f"Evens: {filter_data(numbers, lambda x: x % 2 == 0)}")

    # Function composition
    print("\nFunction composition:")
    add_one = lambda x: x + 1
    double = lambda x: x * 2
    f = compose(add_one, double)
    print(f"compose(+1, ×2)(5) = {f(5)}")
