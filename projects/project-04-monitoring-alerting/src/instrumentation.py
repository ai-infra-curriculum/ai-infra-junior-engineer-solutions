"""Prometheus metrics instrumentation for the ML inference API.

Provides:
- Application info, request, and ML-specific metrics.
- A Flask middleware that records request lifecycle automatically.
- A background collector that publishes process-level resource stats.
- A ``/metrics`` endpoint helper compatible with Prometheus scrapers.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Callable, Dict, Optional

from flask import Flask, Response, g, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

try:
    import psutil  # optional dependency for SystemMetricsCollector
except ImportError:  # pragma: no cover - psutil is recommended but optional
    psutil = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom registry — exposes a controlled set of metrics and avoids polluting
# the global default registry when this module is imported from tests.
# ---------------------------------------------------------------------------

registry = CollectorRegistry()


# ---------------------------------------------------------------------------
# Application info
# ---------------------------------------------------------------------------

app_info = Info("app_info", "Application information", registry=registry)
app_info.info(
    {
        "version": "1.0.0",
        "environment": "production",
        "service": "ml-api",
        "model_version": "resnet50-v1",
    }
)


# ---------------------------------------------------------------------------
# HTTP request metrics
# ---------------------------------------------------------------------------

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry,
)

http_request_size_bytes = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000],
    registry=registry,
)

http_response_size_bytes = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000],
    registry=registry,
)

http_requests_in_flight = Gauge(
    "http_requests_in_flight",
    "Current number of HTTP requests being processed",
    registry=registry,
)


# ---------------------------------------------------------------------------
# ML model metrics
# ---------------------------------------------------------------------------

model_predictions_total = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["model_name", "prediction_class"],
    registry=registry,
)

model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

model_prediction_confidence = Histogram(
    "model_prediction_confidence",
    "Distribution of model prediction confidence scores",
    ["model_name"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    registry=registry,
)

model_accuracy = Gauge(
    "model_accuracy",
    "Current model accuracy (0-1)",
    ["model_name"],
    registry=registry,
)

model_prediction_errors_total = Counter(
    "model_prediction_errors_total",
    "Total number of prediction errors",
    ["model_name", "error_type"],
    registry=registry,
)


# ---------------------------------------------------------------------------
# Data quality metrics
# ---------------------------------------------------------------------------

data_drift_score = Gauge(
    "data_drift_score",
    "Data drift score (0-1, higher = more drift)",
    ["feature_name"],
    registry=registry,
)

missing_features_total = Counter(
    "missing_features_total",
    "Total requests with missing features",
    ["feature_name"],
    registry=registry,
)

invalid_requests_total = Counter(
    "invalid_requests_total",
    "Total invalid requests",
    ["reason"],
    registry=registry,
)


# ---------------------------------------------------------------------------
# Infrastructure metrics
# ---------------------------------------------------------------------------

memory_usage_bytes = Gauge(
    "process_memory_usage_bytes",
    "Current memory usage in bytes",
    registry=registry,
)

cpu_usage_percent = Gauge(
    "process_cpu_usage_percent",
    "Current CPU usage percentage",
    registry=registry,
)

active_db_connections = Gauge(
    "active_db_connections",
    "Number of active database connections",
    registry=registry,
)


# ---------------------------------------------------------------------------
# Business metrics
# ---------------------------------------------------------------------------

business_predictions_total = Counter(
    "business_predictions_total",
    "Total predictions broken down by customer tier",
    ["customer_tier"],
    registry=registry,
)


# ---------------------------------------------------------------------------
# Flask middleware
# ---------------------------------------------------------------------------


class MetricsMiddleware:
    """Flask middleware that records request-lifecycle metrics."""

    def __init__(self, app: Flask) -> None:
        self.app = app
        self.setup_middleware()

    def setup_middleware(self) -> None:
        app = self.app

        @app.before_request
        def _before_request() -> None:
            g.start_time = time.time()
            http_requests_in_flight.inc()
            try:
                size = len(request.get_data(cache=True))
            except Exception:
                size = 0
            http_request_size_bytes.labels(
                method=request.method,
                endpoint=request.endpoint or "unknown",
            ).observe(size)

        @app.after_request
        def _after_request(response: Response) -> Response:
            start_time = getattr(g, "start_time", None)
            endpoint = request.endpoint or "unknown"
            if start_time is not None:
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=request.method, endpoint=endpoint
                ).observe(duration)
            http_requests_total.labels(
                method=request.method,
                endpoint=endpoint,
                status=str(response.status_code),
            ).inc()
            response_size = response.calculate_content_length() or 0
            http_response_size_bytes.labels(
                method=request.method, endpoint=endpoint
            ).observe(response_size)
            http_requests_in_flight.dec()
            return response

        @app.teardown_request
        def _teardown_request(exception: Optional[BaseException] = None) -> None:
            # Decrement only when after_request did not run (i.e., an
            # unhandled exception bypassed it).
            if exception is not None:
                try:
                    http_requests_in_flight.dec()
                except ValueError:
                    # Gauge would clamp at 0 if already decremented.
                    pass

    @staticmethod
    def track_prediction(
        model_name: str,
        prediction_class: str,
        confidence: float,
        inference_time: float,
    ) -> None:
        model_predictions_total.labels(
            model_name=model_name, prediction_class=prediction_class
        ).inc()
        model_inference_duration_seconds.labels(model_name=model_name).observe(
            inference_time
        )
        model_prediction_confidence.labels(model_name=model_name).observe(confidence)

    @staticmethod
    def track_data_quality(
        missing_features: Dict[str, int],
        drift_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        for feature_name, count in missing_features.items():
            missing_features_total.labels(feature_name=feature_name).inc(count)
        if drift_scores:
            for feature_name, score in drift_scores.items():
                data_drift_score.labels(feature_name=feature_name).set(score)

    @staticmethod
    def update_model_accuracy(model_name: str, accuracy: float) -> None:
        model_accuracy.labels(model_name=model_name).set(accuracy)


# ---------------------------------------------------------------------------
# System metrics collector
# ---------------------------------------------------------------------------


class SystemMetricsCollector:
    """Background poller that publishes process-level resource stats."""

    def __init__(self, interval: int = 15) -> None:
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def collect_once(self) -> None:
        if psutil is None:
            logger.debug("psutil not installed; skipping system metrics collection.")
            return
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_bytes.set(memory_info.rss)
        # interval=None returns the value since the last call; this is non-
        # blocking when called inside a polling loop.
        cpu_usage_percent.set(process.cpu_percent(interval=None))

    def start_background_collection(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        def _loop() -> None:
            # Prime cpu_percent so subsequent calls return meaningful values.
            if psutil is not None:
                psutil.Process().cpu_percent(interval=None)
            while not self._stop_event.is_set():
                try:
                    self.collect_once()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Failed to collect system metrics")
                self._stop_event.wait(self.interval)

        self._thread = threading.Thread(
            target=_loop, name="system-metrics", daemon=True
        )
        self._thread.start()
        logger.info(
            "Started system metrics collection (interval=%ss)", self.interval
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)


# ---------------------------------------------------------------------------
# Metrics endpoint + utility decorator
# ---------------------------------------------------------------------------


def metrics_endpoint() -> Response:
    """Return all registered metrics in Prometheus text format."""
    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)


def timed(
    metric: Optional[Histogram] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Callable[[Callable], Callable]:
    """Decorate a function to record its duration into ``metric``.

    Args:
        metric: a Prometheus Histogram. If ``None`` the duration is only
            logged at DEBUG level.
        labels: label values applied via ``.labels(**labels)`` before
            observation.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                if metric is not None:
                    target = metric.labels(**labels) if labels else metric
                    target.observe(duration)
                logger.debug("%s took %.4fs", func.__name__, duration)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------


def _demo_app() -> Flask:  # pragma: no cover - manual entrypoint
    import random

    from flask import jsonify

    app = Flask(__name__)
    MetricsMiddleware(app)

    @app.route("/metrics")
    def _metrics() -> Response:
        return metrics_endpoint()

    @app.route("/predict", methods=["POST"])
    def _predict():
        time.sleep(random.uniform(0.01, 0.1))
        prediction_class = random.choice(["cat", "dog", "bird"])
        confidence = random.uniform(0.7, 0.99)
        inference_time = random.uniform(0.01, 0.1)
        MetricsMiddleware.track_prediction(
            model_name="resnet50",
            prediction_class=prediction_class,
            confidence=confidence,
            inference_time=inference_time,
        )
        return jsonify({"prediction": prediction_class, "confidence": confidence})

    @app.route("/health")
    def _health():
        return jsonify({"status": "healthy"})

    SystemMetricsCollector(interval=15).start_background_collection()
    return app


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    logging.basicConfig(level=logging.INFO)
    _demo_app().run(host="0.0.0.0", port=5000)
