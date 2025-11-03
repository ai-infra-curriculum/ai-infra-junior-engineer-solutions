"""Tests for structured logging utilities."""

import json
import logging
from ml_infra_utils.logging import StructuredLogger, JsonFormatter, log_ml_event


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_logger_initialization(self):
        """Test logger initializes correctly."""
        logger = StructuredLogger("test_logger", logging.INFO)
        assert logger.logger.name == "test_logger"
        assert logger.logger.level == logging.INFO

    def test_logger_info(self, capsys):
        """Test info level logging."""
        logger = StructuredLogger("test_logger", logging.INFO)
        logger.info("Test message", extra_field="value")

        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip())

        assert log_data["message"] == "Test message"
        assert log_data["level"] == "INFO"
        assert log_data["extra_field"] == "value"

    def test_logger_error(self, capsys):
        """Test error level logging."""
        logger = StructuredLogger("test_logger", logging.ERROR)
        logger.error("Error occurred", error_code=500)

        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip())

        assert log_data["message"] == "Error occurred"
        assert log_data["level"] == "ERROR"
        assert log_data["error_code"] == 500

    def test_logger_with_context(self, capsys):
        """Test logging with context dictionary."""
        logger = StructuredLogger("test_logger", logging.INFO)
        context = {"user_id": "123", "request_id": "abc"}
        logger.info("Request processed", context=context)

        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip())

        assert log_data["context"]["user_id"] == "123"
        assert log_data["context"]["request_id"] == "abc"

    def test_logger_debug_below_threshold(self, capsys):
        """Test debug messages not logged when level is INFO."""
        logger = StructuredLogger("test_logger", logging.INFO)
        logger.debug("Debug message")

        captured = capsys.readouterr()
        # Debug message should not be logged
        assert captured.out == ""


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_json_formatter_format(self):
        """Test JSON formatter produces valid JSON."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["message"] == "Test message"
        assert log_data["level"] == "INFO"
        assert "timestamp" in log_data
        assert log_data["line"] == 10

    def test_json_formatter_with_context(self):
        """Test JSON formatter includes context."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.context = {"key": "value"}

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["context"]["key"] == "value"


class TestMLLogging:
    """Tests for ML-specific logging."""

    def test_log_ml_event(self, capsys):
        """Test ML event logging."""
        logger = StructuredLogger("ml_logger", logging.INFO)
        metrics = {"accuracy": 0.95, "loss": 0.05}

        log_ml_event(
            logger,
            event_type="training_end",
            model_name="model_v1",
            metrics=metrics,
            epoch=10,
        )

        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip())

        assert log_data["message"] == "ML Event: training_end"
        assert log_data["context"]["event_type"] == "training_end"
        assert log_data["context"]["model_name"] == "model_v1"
        assert log_data["context"]["metrics"]["accuracy"] == 0.95
        assert log_data["context"]["epoch"] == 10

    def test_log_ml_event_without_metrics(self, capsys):
        """Test ML event logging without metrics."""
        logger = StructuredLogger("ml_logger", logging.INFO)

        log_ml_event(
            logger,
            event_type="training_start",
            model_name="model_v2",
        )

        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip())

        assert log_data["context"]["event_type"] == "training_start"
        assert log_data["context"]["metrics"] == {}
