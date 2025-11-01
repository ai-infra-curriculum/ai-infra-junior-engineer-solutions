"""
Structured Logging Configuration

Provides structured logging setup for ML applications with JSON formatting,
log rotation, and context propagation.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import traceback


# Global logger configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging

    Outputs log records as JSON objects for easier parsing by
    log aggregation systems (ELK, Splunk, etc.)
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """
    Add contextual information to log records

    Useful for adding request IDs, user IDs, or other context
    that should appear in all logs.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize context filter

        Args:
            context: Dictionary of context to add to all log records
        """
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context to record

        Args:
            record: Log record to modify

        Returns:
            Always True (don't filter any records)
        """
        if not hasattr(record, "extra_fields"):
            record.extra_fields = {}

        record.extra_fields.update(self.context)
        return True

    def set_context(self, key: str, value: Any):
        """
        Update context

        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value

    def clear_context(self):
        """Clear all context"""
        self.context.clear()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Setup application logging

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Use JSON formatting
        context: Initial context to add to all logs

    Returns:
        Configured root logger
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add context filter
    if context:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for module

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter for adding structured context

    Example:
        logger = LoggerAdapter(get_logger(__name__), {"request_id": "123"})
        logger.info("Processing request")
    """

    def process(
        self,
        msg: str,
        kwargs: Dict[str, Any]
    ) -> tuple:
        """
        Add extra fields to log record

        Args:
            msg: Log message
            kwargs: Keyword arguments

        Returns:
            Processed message and kwargs
        """
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        # Add adapter context
        kwargs["extra"].update(self.extra)

        return msg, kwargs


class PerformanceLogger:
    """
    Context manager for logging operation performance

    Example:
        with PerformanceLogger("model_inference"):
            result = model.predict(image)
    """

    def __init__(
        self,
        operation: str,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO
    ):
        """
        Initialize performance logger

        Args:
            operation: Name of operation being timed
            logger: Logger to use (default: root logger)
            level: Log level for performance logs
        """
        self.operation = operation
        self.logger = logger or logging.getLogger()
        self.level = level
        self.start_time = None

    def __enter__(self):
        """Start timing"""
        self.start_time = datetime.utcnow()
        self.logger.log(
            self.level,
            f"Starting operation: {self.operation}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        End timing and log duration

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        duration = (datetime.utcnow() - self.start_time).total_seconds()

        if exc_type:
            self.logger.log(
                logging.ERROR,
                f"Operation failed: {self.operation} "
                f"(duration: {duration:.3f}s)",
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            self.logger.log(
                self.level,
                f"Operation completed: {self.operation} "
                f"(duration: {duration:.3f}s)"
            )


def log_function_call(func):
    """
    Decorator to log function calls with arguments and results

    Example:
        @log_function_call
        def process_data(data):
            return processed

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """
    logger = get_logger(func.__module__)

    def wrapper(*args, **kwargs):
        logger.debug(
            f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
        )
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(
                f"{func.__name__} raised {type(e).__name__}: {e}",
                exc_info=True
            )
            raise

    return wrapper


def configure_ml_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    json_format: bool = False
):
    """
    Configure logging for ML applications

    Sets up:
    - Console logging (stdout)
    - File logging with rotation
    - Structured logging (optional JSON)

    Args:
        log_dir: Directory for log files
        level: Log level
        json_format: Use JSON formatting
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Application log
    app_log = log_path / "app.log"

    # Setup logging
    setup_logging(
        level=level,
        log_file=str(app_log),
        json_format=json_format,
        context={
            "service": "ml-inference-api",
            "environment": "production"
        }
    )

    # Suppress noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = get_logger(__name__)
    logger.info(f"Logging configured: level={level}, dir={log_dir}")


# Example usage
if __name__ == "__main__":
    # Basic setup
    configure_ml_logging(level="DEBUG")

    # Get logger
    logger = get_logger(__name__)

    # Test logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Performance logging
    with PerformanceLogger("test_operation", logger):
        import time
        time.sleep(0.1)

    # Structured logging with context
    adapter = LoggerAdapter(
        logger,
        {"request_id": "req-123", "user_id": "user-456"}
    )
    adapter.info("Processing user request")

    # Function decorator
    @log_function_call
    def example_function(x, y):
        return x + y

    result = example_function(5, 3)
