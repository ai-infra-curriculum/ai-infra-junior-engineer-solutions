"""
Structured logging configuration using structlog.

All logs are JSON-formatted with:
- Timestamp
- Log level
- Message
- Request ID (correlation)
- Trace ID and Span ID (from OpenTelemetry)
- Contextual metadata
"""

import logging
import sys
import structlog
from typing import Any, Dict
from pythonjsonlogger import jsonlogger
from opentelemetry import trace


def setup_logging(log_level: str = "INFO", log_format: str = "json"):
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('json' or 'console')
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

    # Processors for structlog
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_trace_context,  # Add OpenTelemetry trace context
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        # JSON format for production
        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Configure JSON formatter for root logger
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.handlers = [handler]
        root_logger.setLevel(getattr(logging, log_level.upper()))

    else:
        # Console format for development
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer(colors=True)
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def add_trace_context(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """
    Add OpenTelemetry trace context to log records.

    This enables correlation between logs and traces.

    Args:
        logger: Logger instance
        method_name: Method name
        event_dict: Event dictionary

    Returns:
        Modified event dictionary with trace context
    """
    # Get current span
    span = trace.get_current_span()
    if span != trace.INVALID_SPAN:
        ctx = span.get_span_context()
        if ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
            event_dict["trace_flags"] = format(ctx.trace_flags, "02x")

    return event_dict


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict):
        """
        Add custom fields to JSON log records.

        Args:
            log_record: Log record dictionary
            record: LogRecord instance
            message_dict: Message dictionary
        """
        super().add_fields(log_record, record, message_dict)

        # Rename fields for consistency
        if "levelname" in log_record:
            log_record["level"] = log_record.pop("levelname")

        # Add logger name
        log_record["logger"] = record.name

        # Ensure timestamp is present
        if "timestamp" not in log_record:
            log_record["timestamp"] = self.formatTime(record, self.datefmt)


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """
    Get a configured structlog logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Context manager for request-scoped logging
class LogContext:
    """
    Context manager for adding request-scoped context to logs.

    Usage:
        with LogContext(request_id="123", user_id="456"):
            logger.info("Processing request")
            # All logs within this block will include request_id and user_id
    """

    def __init__(self, **kwargs):
        self.context = kwargs

    def __enter__(self):
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        structlog.contextvars.clear_contextvars()


# Helper functions for common logging patterns
def log_request_start(
    logger: structlog.stdlib.BoundLogger,
    method: str,
    path: str,
    request_id: str,
    **kwargs
):
    """
    Log the start of an HTTP request.

    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        request_id: Request ID
        **kwargs: Additional context
    """
    logger.info(
        "Request started",
        method=method,
        path=path,
        request_id=request_id,
        **kwargs
    )


def log_request_complete(
    logger: structlog.stdlib.BoundLogger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: str,
    **kwargs
):
    """
    Log the completion of an HTTP request.

    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        request_id: Request ID
        **kwargs: Additional context
    """
    logger.info(
        "Request completed",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=round(duration_ms, 2),
        request_id=request_id,
        **kwargs
    )


def log_inference_start(
    logger: structlog.stdlib.BoundLogger,
    model_name: str,
    request_id: str,
    **kwargs
):
    """
    Log the start of model inference.

    Args:
        logger: Logger instance
        model_name: Model name
        request_id: Request ID
        **kwargs: Additional context
    """
    logger.info(
        "Inference started",
        model_name=model_name,
        request_id=request_id,
        **kwargs
    )


def log_inference_complete(
    logger: structlog.stdlib.BoundLogger,
    model_name: str,
    duration_ms: float,
    prediction_class: str,
    confidence: float,
    request_id: str,
    **kwargs
):
    """
    Log the completion of model inference.

    Args:
        logger: Logger instance
        model_name: Model name
        duration_ms: Inference duration in milliseconds
        prediction_class: Predicted class
        confidence: Prediction confidence
        request_id: Request ID
        **kwargs: Additional context
    """
    logger.info(
        "Inference completed",
        model_name=model_name,
        duration_ms=round(duration_ms, 2),
        prediction_class=prediction_class,
        confidence=round(confidence, 4),
        request_id=request_id,
        **kwargs
    )


def log_error(
    logger: structlog.stdlib.BoundLogger,
    error: Exception,
    context: str,
    **kwargs
):
    """
    Log an error with full context.

    Args:
        logger: Logger instance
        error: Exception instance
        context: Error context description
        **kwargs: Additional context
    """
    logger.error(
        f"{context}: {str(error)}",
        error_type=type(error).__name__,
        error_message=str(error),
        **kwargs,
        exc_info=True
    )
