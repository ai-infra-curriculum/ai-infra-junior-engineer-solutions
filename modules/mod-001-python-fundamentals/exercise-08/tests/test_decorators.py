"""Tests for decorator utilities."""

import pytest
import time
from ml_infra_utils.decorators import timer, timer_with_units, retry, retry_on_condition


class TestTimer:
    """Tests for timer decorator."""

    def test_timer_execution(self, capsys):
        """Test timer decorator prints execution time."""

        @timer
        def slow_function():
            time.sleep(0.1)
            return "done"

        result = slow_function()
        captured = capsys.readouterr()

        assert result == "done"
        assert "slow_function took" in captured.out
        assert "seconds" in captured.out

    def test_timer_with_units_seconds(self, capsys):
        """Test timer with seconds unit."""

        @timer_with_units("seconds")
        def test_func():
            return "done"

        test_func()
        captured = capsys.readouterr()
        assert "seconds" in captured.out

    def test_timer_with_units_milliseconds(self, capsys):
        """Test timer with milliseconds unit."""

        @timer_with_units("milliseconds")
        def test_func():
            return "done"

        test_func()
        captured = capsys.readouterr()
        assert "milliseconds" in captured.out

    def test_timer_with_invalid_unit_raises(self):
        """Test timer with invalid unit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid unit"):
            @timer_with_units("hours")
            def test_func():
                pass


class TestRetry:
    """Tests for retry decorator."""

    def test_retry_success_first_try(self):
        """Test retry with successful first attempt."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failure(self, capsys):
        """Test retry with success after initial failures."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = fails_twice()
        captured = capsys.readouterr()

        assert result == "success"
        assert call_count == 3
        assert "Retrying" in captured.out

    def test_retry_exhausts_attempts(self, capsys):
        """Test retry exhausts all attempts and raises."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        captured = capsys.readouterr()
        assert call_count == 3
        assert "failed after 3 attempts" in captured.out

    def test_retry_with_specific_exception(self):
        """Test retry only catches specified exceptions."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        def fails_with_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Different exception")

        with pytest.raises(TypeError):
            fails_with_type_error()

        assert call_count == 1  # Should not retry for TypeError

    def test_retry_with_exponential_backoff(self):
        """Test retry with exponential backoff."""
        timestamps = []

        @retry(max_attempts=3, delay=0.1, backoff=2.0)
        def track_timing():
            timestamps.append(time.time())
            if len(timestamps) < 3:
                raise ValueError("Not yet")
            return "success"

        track_timing()
        assert len(timestamps) == 3

        # Check delays are approximately exponential
        # First delay ~0.1s, second delay ~0.2s
        delay1 = timestamps[1] - timestamps[0]
        delay2 = timestamps[2] - timestamps[1]
        assert 0.08 < delay1 < 0.15  # Allow some tolerance
        assert 0.18 < delay2 < 0.25


class TestRetryOnCondition:
    """Tests for retry_on_condition decorator."""

    def test_retry_on_condition_success(self):
        """Test retry_on_condition with successful result."""
        call_count = 0

        @retry_on_condition(max_attempts=3, delay=0.01, condition=lambda x: x is None)
        def returns_value():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return None
            return "success"

        result = returns_value()
        assert result == "success"
        assert call_count == 2

    def test_retry_on_condition_exhausts(self, capsys):
        """Test retry_on_condition exhausts attempts."""

        @retry_on_condition(max_attempts=3, delay=0.01, condition=lambda x: x is None)
        def always_returns_none():
            return None

        result = always_returns_none()
        captured = capsys.readouterr()

        assert result is None
        assert "exhausted 3 attempts" in captured.out

    def test_retry_on_condition_custom_condition(self):
        """Test retry_on_condition with custom condition."""
        call_count = 0

        @retry_on_condition(
            max_attempts=3, delay=0.01, condition=lambda x: x < 10
        )
        def increment_value():
            nonlocal call_count
            call_count += 1
            return call_count * 5

        result = increment_value()
        # First call returns 5 (< 10, retry)
        # Second call returns 10 (>= 10, stop)
        assert result == 10
        assert call_count == 2
