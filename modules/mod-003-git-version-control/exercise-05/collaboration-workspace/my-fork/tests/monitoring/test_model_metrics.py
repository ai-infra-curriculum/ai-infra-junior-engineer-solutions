"""Tests for model metrics."""

import pytest
from src.monitoring.model_metrics import ModelMetrics


def test_model_metrics_initialization():
    """Test metrics initialization."""
    metrics = ModelMetrics()
    assert metrics.predictions == []
    assert metrics.ground_truth == []
    assert metrics.latencies == []


def test_add_prediction():
    """Test adding predictions."""
    metrics = ModelMetrics()
    metrics.add_prediction(0.95, truth=1.0, latency=0.05)
    metrics.add_prediction(0.87, truth=1.0, latency=0.03)

    assert len(metrics.predictions) == 2
    assert len(metrics.ground_truth) == 2
    assert len(metrics.latencies) == 2


def test_calculate_accuracy():
    """Test accuracy calculation."""
    metrics = ModelMetrics()
    metrics.add_prediction(0.9, truth=1.0)
    metrics.add_prediction(0.8, truth=1.0)
    metrics.add_prediction(0.2, truth=0.0)

    accuracy = metrics.calculate_accuracy()
    assert accuracy == pytest.approx(100.0)


def test_get_average_latency():
    """Test average latency calculation."""
    metrics = ModelMetrics()
    metrics.add_prediction(0.9, latency=0.05)
    metrics.add_prediction(0.8, latency=0.03)

    avg_latency = metrics.get_average_latency()
    assert avg_latency == pytest.approx(0.04)


def test_get_summary():
    """Test metrics summary."""
    metrics = ModelMetrics()
    metrics.add_prediction(0.9, truth=1.0, latency=0.05)
    metrics.add_prediction(0.8, truth=1.0, latency=0.03)

    summary = metrics.get_summary()
    assert summary["total_predictions"] == 2
    assert "accuracy" in summary
    assert "average_latency" in summary
    assert "uptime_seconds" in summary
