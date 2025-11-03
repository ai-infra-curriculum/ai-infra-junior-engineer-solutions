"""Structured logging utilities for ML pipelines."""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredLogger:
    """
    Logger that outputs structured JSON logs for better parsing and analysis.

    Attributes:
        logger_name: Name of the logger
        log_level: Logging level (default: INFO)
    """

    def __init__(self, logger_name: str, log_level: int = logging.INFO):
        """
        Initialize structured logger.

        Args:
            logger_name: Name of the logger
            log_level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        # Create console handler with JSON formatter
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(JsonFormatter())

        # Remove existing handlers and add ours
        self.logger.handlers.clear()
        self.logger.addHandler(handler)

    def log(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log structured message with context.

        Args:
            level: Log level ('debug', 'info', 'warning', 'error', 'critical')
            message: Log message
            context: Additional context dictionary
            **kwargs: Additional key-value pairs to include
        """
        extra = {"context": context or {}, **kwargs}

        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info level message."""
        self.log("info", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error level message."""
        self.log("error", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning level message."""
        self.log("warning", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug level message."""
        self.log("debug", message, **kwargs)


class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as JSON.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context if present
        if hasattr(record, "context"):
            log_data["context"] = record.context

        # Add any additional fields from extra
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "context",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


def log_ml_event(
    logger: StructuredLogger,
    event_type: str,
    model_name: str,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> None:
    """
    Log ML-specific event with standard structure.

    Args:
        logger: StructuredLogger instance
        event_type: Type of event ('training_start', 'training_end', 'prediction', etc.)
        model_name: Name of the ML model
        metrics: Dictionary of metric values
        **kwargs: Additional context
    """
    context = {
        "event_type": event_type,
        "model_name": model_name,
        "metrics": metrics or {},
        **kwargs,
    }

    logger.info(f"ML Event: {event_type}", context=context)
