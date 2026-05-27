"""Kubernetes-ready model serving API.

Builds on project-01's Flask serving API with:
- Prometheus metrics exposed on ``/metrics``.
- Separate liveness (``/health/live``) and readiness (``/health/ready``)
  probes — plus a combined ``/health`` endpoint for convenience.
- Configuration from environment variables (ConfigMap-injected).
- Structured logging.
- Graceful SIGTERM handling for zero-downtime deployments.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from .model import ModelLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME: str = os.getenv("MODEL_NAME", "resnet50")
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "32"))
PORT: int = int(os.getenv("PORT", "5000"))
SHUTDOWN_GRACE_SECONDS: float = float(os.getenv("SHUTDOWN_GRACE_SECONDS", "2.0"))
MODEL_LOAD_SECONDS: float = float(os.getenv("MODEL_LOAD_SECONDS", "1.0"))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    """Configure root logger with a stream handler + standard formatter."""
    root = logging.getLogger("model-api")
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    # Avoid duplicating handlers on repeated imports (e.g., gunicorn workers).
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(handler)
    root.propagate = False
    return root


logger = setup_logging()


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

request_count = Counter(
    "model_api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

request_duration = Histogram(
    "model_api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

prediction_count = Counter(
    "model_api_predictions_total",
    "Total predictions made",
    ["model_name", "status"],
)

inference_duration = Histogram(
    "model_api_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_name"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
)

model_loaded_gauge = Gauge(
    "model_api_model_loaded",
    "Whether the model is loaded and ready (1) or not (0)",
    ["model_name", "version"],
)

active_connections = Gauge(
    "model_api_active_connections",
    "Number of active requests being processed",
)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class ApplicationState:
    """Track liveness, readiness, and graceful-shutdown state."""

    def __init__(self) -> None:
        self.is_ready: bool = False
        self.is_alive: bool = True
        self.model_loaded: bool = False
        self.shutdown_event: threading.Event = threading.Event()
        self.model: Optional[ModelLoader] = None
        self.start_time: float = time.time()

    def mark_ready(self) -> None:
        self.is_ready = True
        logger.info("Application marked READY")

    def mark_not_ready(self) -> None:
        self.is_ready = False
        logger.info("Application marked NOT READY")

    def mark_shutdown(self) -> None:
        self.is_alive = False
        self.is_ready = False
        self.shutdown_event.set()
        logger.info("Application marked SHUTDOWN")

    def uptime_seconds(self) -> float:
        return time.time() - self.start_time


app_state = ApplicationState()


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.before_request
def before_request() -> None:
    """Start request timing and bump the in-flight gauge."""
    request.start_time = time.time()  # type: ignore[attr-defined]
    active_connections.inc()
    logger.info(
        "request_received method=%s path=%s remote_addr=%s",
        request.method,
        request.path,
        request.remote_addr,
    )


@app.after_request
def after_request(response: Response) -> Response:
    """Record metrics and emit a structured log line per response."""
    start_time = getattr(request, "start_time", None)
    duration = time.time() - start_time if start_time is not None else 0.0
    endpoint = request.endpoint or "unknown"
    request_duration.labels(method=request.method, endpoint=endpoint).observe(duration)
    request_count.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=str(response.status_code),
    ).inc()
    active_connections.dec()
    logger.info(
        "request_complete method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.path,
        response.status_code,
        duration * 1000,
    )
    return response


@app.teardown_request
def teardown_request(exception: Optional[BaseException] = None) -> None:
    if exception is not None:
        # after_request did not run; keep the gauge balanced.
        try:
            active_connections.dec()
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Health-check endpoints
# ---------------------------------------------------------------------------


def _health_payload(status: str) -> Dict[str, Any]:
    return {
        "status": status,
        "model_loaded": app_state.model_loaded,
        "model_name": MODEL_NAME,
        "uptime_seconds": round(app_state.uptime_seconds(), 2),
    }


@app.route("/health", methods=["GET"])
def health():
    """Combined liveness+readiness check for simple deployments."""
    if app_state.is_alive and app_state.model_loaded:
        return jsonify(_health_payload("healthy")), 200
    return jsonify(_health_payload("unhealthy")), 503


@app.route("/health/live", methods=["GET"])
def liveness():
    """Liveness probe — fail only when the process should be restarted."""
    if app_state.is_alive:
        return jsonify({"status": "alive"}), 200
    return jsonify({"status": "shutting_down"}), 503


@app.route("/health/ready", methods=["GET"])
def readiness():
    """Readiness probe — fail when the pod should be removed from rotation."""
    if app_state.is_ready and app_state.model_loaded:
        return jsonify({"status": "ready"}), 200
    return (
        jsonify(
            {
                "status": "not_ready",
                "model_loaded": app_state.model_loaded,
            }
        ),
        503,
    )


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus scrape target — returns text-format metrics."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------


@app.route("/predict", methods=["POST"])
def predict():
    """Run inference on a batch of instances."""
    if not app_state.model_loaded or app_state.model is None:
        return jsonify({"error": "Model is not loaded yet"}), 503

    if not request.is_json:
        return (
            jsonify({"error": "Content-Type must be application/json"}),
            400,
        )

    data = request.get_json(silent=True) or {}
    if "instances" not in data:
        return jsonify({"error": "Missing 'instances' in request"}), 400

    instances = data["instances"]
    if not isinstance(instances, list):
        return jsonify({"error": "'instances' must be a list"}), 400
    if len(instances) == 0:
        return jsonify({"error": "'instances' must not be empty"}), 400
    if len(instances) > MAX_BATCH_SIZE:
        return (
            jsonify(
                {
                    "error": "Batch too large",
                    "max_batch_size": MAX_BATCH_SIZE,
                    "received": len(instances),
                }
            ),
            400,
        )

    start_time = time.time()
    try:
        predictions = app_state.model.predict(instances)
        inference_seconds = time.time() - start_time
        prediction_count.labels(model_name=MODEL_NAME, status="success").inc(
            len(predictions)
        )
        inference_duration.labels(model_name=MODEL_NAME).observe(inference_seconds)
        return (
            jsonify(
                {
                    "predictions": predictions,
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "inference_time_ms": round(inference_seconds * 1000, 2),
                }
            ),
            200,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        prediction_count.labels(model_name=MODEL_NAME, status="error").inc()
        return jsonify({"error": "Prediction failed", "details": str(exc)}), 500


@app.route("/", methods=["GET"])
def index():
    """Root endpoint — returns service metadata."""
    return (
        jsonify(
            {
                "service": "Model Serving API",
                "version": "2.0",
                "model": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "endpoints": {
                    "predict": "/predict",
                    "health": "/health",
                    "liveness": "/health/live",
                    "readiness": "/health/ready",
                    "metrics": "/metrics",
                },
            }
        ),
        200,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model() -> None:
    """Load the model and flip readiness state."""
    logger.info("Loading model %s (version=%s)", MODEL_NAME, MODEL_VERSION)
    try:
        loader = ModelLoader(
            model_name=MODEL_NAME,
            version=MODEL_VERSION,
            load_seconds=MODEL_LOAD_SECONDS,
        ).load()
    except Exception:
        logger.exception("Failed to load model — aborting startup")
        sys.exit(1)
    app_state.model = loader
    app_state.model_loaded = True
    model_loaded_gauge.labels(model_name=MODEL_NAME, version=MODEL_VERSION).set(1)
    app_state.mark_ready()
    logger.info("Model %s is ready to serve traffic", MODEL_NAME)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


def handle_shutdown(signum: int, _frame: Any) -> None:
    """Handle SIGTERM/SIGINT for zero-downtime rollouts."""
    logger.info("Received signal %s — beginning graceful shutdown", signum)
    app_state.mark_not_ready()
    model_loaded_gauge.labels(model_name=MODEL_NAME, version=MODEL_VERSION).set(0)
    # Give the load balancer a moment to drain traffic.
    time.sleep(SHUTDOWN_GRACE_SECONDS)
    app_state.mark_shutdown()
    logger.info("Shutdown complete — exiting")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------


def initialize_application() -> None:
    """Run all startup work — currently just model loading."""
    logger.info(
        "Initializing application: model=%s version=%s max_batch=%d port=%d",
        MODEL_NAME,
        MODEL_VERSION,
        MAX_BATCH_SIZE,
        PORT,
    )
    load_model()


# Allow gunicorn / WSGI servers to import the app without immediately
# blocking on model load — they should call ``initialize_application``
# explicitly during their startup hooks (or rely on the test fixture
# below).
if os.getenv("MODEL_AUTOLOAD", "1") == "1":
    initialize_application()


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug)
