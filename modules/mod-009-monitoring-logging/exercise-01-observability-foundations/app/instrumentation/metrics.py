"""
Prometheus metrics for the inference gateway.

Implements the Four Golden Signals:
- Latency: How long requests take
- Traffic: Rate of requests
- Errors: Rate of failed requests
- Saturation: How full the service is
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Optional


# ====================================================================================
# Application Info
# ====================================================================================

app_info = Info(
    "app",
    "Application information"
)


# ====================================================================================
# Golden Signal 1: LATENCY
# ====================================================================================

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "endpoint", "status_code"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Model inference latency in seconds",
    labelnames=["model_name"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0)
)


# ====================================================================================
# Golden Signal 2: TRAFFIC
# ====================================================================================

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "endpoint", "status_code"]
)

model_predictions_total = Counter(
    "model_predictions_total",
    "Total model predictions",
    labelnames=["model_name", "prediction_class"]
)


# ====================================================================================
# Golden Signal 3: ERRORS
# ====================================================================================

http_request_exceptions_total = Counter(
    "http_request_exceptions_total",
    "Total HTTP request exceptions",
    labelnames=["method", "endpoint", "exception_type"]
)

model_inference_errors_total = Counter(
    "model_inference_errors_total",
    "Total model inference errors",
    labelnames=["model_name", "error_type"]
)


# ====================================================================================
# Golden Signal 4: SATURATION
# ====================================================================================

inference_queue_size = Gauge(
    "inference_queue_size",
    "Current size of the inference queue"
)

model_memory_usage_bytes = Gauge(
    "model_memory_usage_bytes",
    "Model memory usage in bytes",
    labelnames=["model_name"]
)

active_requests = Gauge(
    "active_requests",
    "Number of requests currently being processed",
    labelnames=["endpoint"]
)


# ====================================================================================
# Business Metrics
# ====================================================================================

prediction_confidence = Histogram(
    "prediction_confidence",
    "Distribution of prediction confidence scores",
    labelnames=["model_name", "prediction_class"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
)

image_size_bytes = Histogram(
    "image_size_bytes",
    "Distribution of input image sizes in bytes",
    buckets=(1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000)
)


# ====================================================================================
# Model Loading Metrics
# ====================================================================================

model_load_duration_seconds = Gauge(
    "model_load_duration_seconds",
    "Time taken to load the model in seconds",
    labelnames=["model_name"]
)

model_loaded = Gauge(
    "model_loaded",
    "Whether the model is currently loaded (1) or not (0)",
    labelnames=["model_name"]
)


# ====================================================================================
# Helper Functions
# ====================================================================================

def record_request(method: str, endpoint: str, duration: float, status_code: int):
    """
    Record HTTP request metrics.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        duration: Request duration in seconds
        status_code: HTTP status code
    """
    # Normalize endpoint to avoid high cardinality
    normalized_endpoint = _normalize_endpoint(endpoint)

    # Record latency
    http_request_duration_seconds.labels(
        method=method,
        endpoint=normalized_endpoint,
        status_code=status_code
    ).observe(duration)

    # Record traffic
    http_requests_total.labels(
        method=method,
        endpoint=normalized_endpoint,
        status_code=status_code
    ).inc()


def record_inference(
    model_name: str,
    duration: float,
    prediction_class: str,
    confidence: float
):
    """
    Record model inference metrics.

    Args:
        model_name: Name of the model
        duration: Inference duration in seconds
        prediction_class: Predicted class
        confidence: Prediction confidence score (0-1)
    """
    # Record latency
    model_inference_duration_seconds.labels(
        model_name=model_name
    ).observe(duration)

    # Record traffic
    model_predictions_total.labels(
        model_name=model_name,
        prediction_class=prediction_class
    ).inc()

    # Record confidence distribution
    prediction_confidence.labels(
        model_name=model_name,
        prediction_class=prediction_class
    ).observe(confidence)


def record_error(
    method: str,
    endpoint: str,
    exception_type: str
):
    """
    Record HTTP request error.

    Args:
        method: HTTP method
        endpoint: API endpoint
        exception_type: Type of exception
    """
    normalized_endpoint = _normalize_endpoint(endpoint)

    http_request_exceptions_total.labels(
        method=method,
        endpoint=normalized_endpoint,
        exception_type=exception_type
    ).inc()


def record_model_error(model_name: str, error_type: str):
    """
    Record model inference error.

    Args:
        model_name: Name of the model
        error_type: Type of error
    """
    model_inference_errors_total.labels(
        model_name=model_name,
        error_type=error_type
    ).inc()


def set_queue_size(size: int):
    """
    Set current inference queue size.

    Args:
        size: Current queue size
    """
    inference_queue_size.set(size)


def set_model_memory(model_name: str, memory_bytes: int):
    """
    Set model memory usage.

    Args:
        model_name: Name of the model
        memory_bytes: Memory usage in bytes
    """
    model_memory_usage_bytes.labels(model_name=model_name).set(memory_bytes)


def set_model_loaded(model_name: str, loaded: bool):
    """
    Set model loaded status.

    Args:
        model_name: Name of the model
        loaded: Whether model is loaded
    """
    model_loaded.labels(model_name=model_name).set(1 if loaded else 0)


def increment_active_requests(endpoint: str):
    """Increment active request count for endpoint."""
    active_requests.labels(endpoint=_normalize_endpoint(endpoint)).inc()


def decrement_active_requests(endpoint: str):
    """Decrement active request count for endpoint."""
    active_requests.labels(endpoint=_normalize_endpoint(endpoint)).dec()


def _normalize_endpoint(endpoint: str) -> str:
    """
    Normalize endpoint path to avoid high cardinality.

    Examples:
        /predict/123 -> /predict/{id}
        /users/abc-def -> /users/{id}

    Args:
        endpoint: Raw endpoint path

    Returns:
        Normalized endpoint path
    """
    # List of known endpoints
    known_endpoints = {
        "/predict",
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/openapi.json",
    }

    # If exact match, return as-is
    if endpoint in known_endpoints:
        return endpoint

    # If starts with known endpoint, normalize parameters
    for known in known_endpoints:
        if endpoint.startswith(known + "/"):
            return f"{known}/{{id}}"

    # Default: return first path segment only
    parts = endpoint.split("/")
    if len(parts) > 1:
        return f"/{parts[1]}"

    return endpoint


def initialize_metrics(app_name: str, app_version: str, environment: str):
    """
    Initialize application info metrics.

    Args:
        app_name: Application name
        app_version: Application version
        environment: Environment (dev, staging, prod)
    """
    app_info.info({
        "app_name": app_name,
        "version": app_version,
        "environment": environment,
    })
