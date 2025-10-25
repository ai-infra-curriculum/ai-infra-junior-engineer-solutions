#!/usr/bin/env python3
"""
Error Handling and Exception Management

Comprehensive examples of exception handling, custom exceptions, and error recovery.
"""

import logging
from typing import Any, Optional, Callable, Dict, List
from contextlib import contextmanager
from functools import wraps


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= Custom Exceptions =============

class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class DataProcessingError(Exception):
    """Raised when data processing fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class RetryableError(Exception):
    """Raised when operation should be retried."""
    pass


# ============= Basic Error Handling =============

def safe_divide(a: float, b: float) -> Optional[float]:
    """
    Safely divide two numbers.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Result of division or None if error

    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        None
    """
    try:
        return a / b
    except ZeroDivisionError:
        logger.error(f"Cannot divide {a} by zero")
        return None
    except TypeError:
        logger.error(f"Invalid types for division: {type(a)}, {type(b)}")
        return None


def safe_int_conversion(value: str) -> Optional[int]:
    """
    Safely convert string to integer.

    Args:
        value: String to convert

    Returns:
        Integer value or None if conversion fails

    Example:
        >>> safe_int_conversion("123")
        123
        >>> safe_int_conversion("abc")
        None
    """
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Cannot convert '{value}' to integer")
        return None


def safe_list_access(lst: List[Any], index: int, default: Any = None) -> Any:
    """
    Safely access list element.

    Args:
        lst: List to access
        index: Index to access
        default: Default value if index out of range

    Returns:
        Element at index or default value

    Example:
        >>> safe_list_access([1, 2, 3], 1)
        2
        >>> safe_list_access([1, 2, 3], 10, default=-1)
        -1
    """
    try:
        return lst[index]
    except IndexError:
        logger.warning(f"Index {index} out of range for list of length {len(lst)}")
        return default


# ============= Validation with Exceptions =============

def validate_age(age: int) -> int:
    """
    Validate age value.

    Args:
        age: Age to validate

    Returns:
        Validated age

    Raises:
        ValidationError: If age is invalid

    Example:
        >>> validate_age(25)
        25
        >>> validate_age(-5)
        Traceback (most recent call last):
            ...
        ValidationError: Age must be non-negative
    """
    if not isinstance(age, int):
        raise ValidationError(f"Age must be an integer, got {type(age).__name__}")

    if age < 0:
        raise ValidationError("Age must be non-negative")

    if age > 150:
        raise ValidationError("Age must be less than 150")

    return age


def validate_email(email: str) -> str:
    """
    Validate email address.

    Args:
        email: Email to validate

    Returns:
        Validated email

    Raises:
        ValidationError: If email is invalid

    Example:
        >>> validate_email("test@example.com")
        'test@example.com'
    """
    if not isinstance(email, str):
        raise ValidationError(f"Email must be a string, got {type(email).__name__}")

    if '@' not in email:
        raise ValidationError("Email must contain @")

    if '.' not in email.split('@')[1]:
        raise ValidationError("Email domain must contain a dot")

    return email.lower()


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        Validated configuration

    Raises:
        ConfigurationError: If configuration is invalid

    Example:
        >>> validate_config({"api_key": "abc123", "timeout": 30})
        {'api_key': 'abc123', 'timeout': 30}
    """
    required_fields = ['api_key', 'timeout']

    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required field: {field}")

    if not isinstance(config['timeout'], (int, float)):
        raise ConfigurationError("timeout must be a number")

    if config['timeout'] <= 0:
        raise ConfigurationError("timeout must be positive")

    return config


# ============= Context Managers for Error Handling =============

@contextmanager
def error_handler(error_message: str, default_return: Any = None, reraise: bool = False):
    """
    Context manager for error handling.

    Args:
        error_message: Message to log on error
        default_return: Default value to return on error
        reraise: Whether to re-raise the exception

    Yields:
        None

    Example:
        >>> with error_handler("Failed to process data"):
        ...     # risky operation
        ...     pass
    """
    try:
        yield
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        if reraise:
            raise
        return default_return


@contextmanager
def suppress_exception(*exceptions):
    """
    Context manager to suppress specific exceptions.

    Args:
        *exceptions: Exception types to suppress

    Yields:
        None

    Example:
        >>> with suppress_exception(FileNotFoundError):
        ...     open("nonexistent.txt")
    """
    try:
        yield
    except exceptions as e:
        logger.debug(f"Suppressed exception: {type(e).__name__}: {e}")
        pass


# ============= Retry Decorator =============

def retry(max_attempts: int = 3, delay: float = 0, exceptions: tuple = (Exception,)):
    """
    Decorator to retry function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorator function

    Example:
        >>> @retry(max_attempts=3)
        ... def unreliable_function():
        ...     # might fail
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time

            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}"
                    )

                    if attempt < max_attempts:
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


# ============= Exception Chaining =============

def process_data_with_context(data: Any) -> Any:
    """
    Process data with exception context.

    Args:
        data: Data to process

    Returns:
        Processed data

    Raises:
        DataProcessingError: If processing fails

    Example:
        >>> process_data_with_context({"value": 10})
        {'value': 10, 'processed': True}
    """
    try:
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        if 'value' not in data:
            raise KeyError("Data must contain 'value' key")

        result = data.copy()
        result['processed'] = True
        return result

    except (ValueError, KeyError) as e:
        # Chain exceptions to preserve context
        raise DataProcessingError(f"Failed to process data: {data}") from e


# ============= Finally and Cleanup =============

class ResourceManager:
    """Example of proper resource management."""

    def __init__(self, resource_name: str):
        """
        Initialize resource manager.

        Args:
            resource_name: Name of resource
        """
        self.resource_name = resource_name
        self.resource = None
        logger.info(f"ResourceManager created for {resource_name}")

    def acquire(self):
        """Acquire resource."""
        logger.info(f"Acquiring resource: {self.resource_name}")
        self.resource = f"Resource_{self.resource_name}"
        return self.resource

    def release(self):
        """Release resource."""
        if self.resource:
            logger.info(f"Releasing resource: {self.resource_name}")
            self.resource = None

    def use_resource(self):
        """
        Use resource with proper cleanup.

        Example:
            >>> manager = ResourceManager("test")
            >>> manager.use_resource()
            'Used Resource_test'
        """
        try:
            self.acquire()
            if not self.resource:
                raise RuntimeError("Failed to acquire resource")

            # Use resource
            logger.info(f"Using resource: {self.resource}")
            return f"Used {self.resource}"

        finally:
            # Always cleanup, even if error occurs
            self.release()


# ============= Practical Examples =============

def safe_file_read(filepath: str) -> Optional[str]:
    """
    Safely read file with comprehensive error handling.

    Args:
        filepath: Path to file

    Returns:
        File contents or None if error

    Example:
        >>> # content = safe_file_read("test.txt")
        >>> pass
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None

    except PermissionError:
        logger.error(f"Permission denied: {filepath}")
        return None

    except UnicodeDecodeError:
        logger.error(f"Unable to decode file as UTF-8: {filepath}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error reading {filepath}: {e}")
        return None


@retry(max_attempts=3, delay=1.0)
def fetch_data(url: str) -> Dict[str, Any]:
    """
    Fetch data from URL with retries.

    Args:
        url: URL to fetch from

    Returns:
        Fetched data

    Raises:
        RetryableError: If fetch fails after retries

    Example:
        >>> # data = fetch_data("https://api.example.com/data")
        >>> pass
    """
    # Simulated fetch that might fail
    import random

    if random.random() < 0.7:  # 70% chance of failure
        raise RetryableError(f"Failed to fetch from {url}")

    return {"url": url, "data": "sample data"}


if __name__ == '__main__':
    print("=== Error Handling Examples ===\n")

    # Basic error handling
    print("Safe Division:")
    print(f"  10 / 2 = {safe_divide(10, 2)}")
    print(f"  10 / 0 = {safe_divide(10, 0)}")

    # Validation
    print("\nValidation:")
    try:
        validate_age(25)
        print("  Age 25: Valid")
    except ValidationError as e:
        print(f"  Age 25: Invalid - {e}")

    try:
        validate_age(-5)
        print("  Age -5: Valid")
    except ValidationError as e:
        print(f"  Age -5: Invalid - {e}")

    # Resource management
    print("\nResource Management:")
    manager = ResourceManager("test_resource")
    result = manager.use_resource()
    print(f"  {result}")

    # Context manager
    print("\nContext Manager Error Handling:")
    with error_handler("Test operation failed"):
        print("  Performing safe operation")

    # Safe file operations
    print("\nSafe File Read:")
    content = safe_file_read("nonexistent.txt")
    print(f"  Result: {content}")

    print("\n✓ Error handling examples completed")
