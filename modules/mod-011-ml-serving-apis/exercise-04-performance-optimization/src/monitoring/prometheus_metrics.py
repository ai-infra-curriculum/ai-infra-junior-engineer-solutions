"""Prometheus metrics for ML API monitoring.

Tracks:
- Request rates
- Response times
- Model inference latency
- Error rates
- Queue depth
- Cache hit rates
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    generate_latest,
    REGISTRY,
)
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
from typing import Callable


# ====================
# Define Metrics
# ====================

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

REQUEST_IN_PROGRESS = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint']
)

# ML prediction metrics
PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total ML predictions made',
    ['model_version', 'prediction_type', 'status']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_duration_seconds',
    'ML prediction latency in seconds',
    ['model_version', 'prediction_type'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Model loading metrics
MODEL_LOAD_TIME = Histogram(
    'ml_model_load_duration_seconds',
    'Time to load ML model in seconds',
    ['model_version'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

MODEL_LOADED = Gauge(
    'ml_model_loaded',
    'Whether model is loaded (1) or not (0)',
    ['model_version']
)

# Preprocessing metrics
PREPROCESSING_LATENCY = Histogram(
    'ml_preprocessing_duration_seconds',
    'Feature preprocessing latency',
    ['preprocessing_type'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1)
)

# Cache metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

CACHE_SIZE = Gauge(
    'cache_size_bytes',
    'Current cache size in bytes',
    ['cache_type']
)

# Queue metrics (Celery)
QUEUE_DEPTH = Gauge(
    'task_queue_depth',
    'Number of tasks in queue',
    ['queue_name']
)

TASK_EXECUTION_TIME = Histogram(
    'task_execution_duration_seconds',
    'Task execution time',
    ['task_name', 'status'],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0)
)

# Error metrics
ERROR_COUNT = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'endpoint']
)

# Resource metrics
MEMORY_USAGE = Gauge(
    'process_memory_bytes',
    'Process memory usage in bytes'
)

CPU_USAGE = Gauge(
    'process_cpu_percent',
    'Process CPU usage percentage'
)

# Model metadata
MODEL_INFO = Info(
    'ml_model_info',
    'ML model metadata'
)

# Business metrics
ACTIVE_USERS = Gauge(
    'active_users',
    'Number of active users'
)

REQUESTS_PER_USER = Summary(
    'requests_per_user',
    'Requests per user distribution'
)


# ====================
# Prometheus Middleware
# ====================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for Prometheus metrics.

    Automatically tracks:
    - Request count
    - Request latency
    - In-progress requests
    - Status codes
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track metrics."""

        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        endpoint = request.url.path

        # Track in-progress requests
        REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()

        # Track request latency
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Record metrics
            status = response.status_code
            latency = time.time() - start_time

            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(latency)

            # Add latency header
            response.headers["X-Process-Time"] = f"{latency:.4f}"

            return response

        except Exception as e:
            # Track errors
            ERROR_COUNT.labels(
                error_type=type(e).__name__,
                endpoint=endpoint
            ).inc()

            raise

        finally:
            # Decrement in-progress
            REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()


# ====================
# Helper Functions
# ====================

def track_prediction(
    model_version: str,
    prediction_type: str,
    latency: float,
    status: str = "success"
):
    """
    Track prediction metrics.

    Args:
        model_version: Version of model used
        prediction_type: Type of prediction (single, batch, etc.)
        latency: Prediction latency in seconds
        status: Success or failure
    """
    PREDICTION_COUNT.labels(
        model_version=model_version,
        prediction_type=prediction_type,
        status=status
    ).inc()

    PREDICTION_LATENCY.labels(
        model_version=model_version,
        prediction_type=prediction_type
    ).observe(latency)


def track_cache(cache_type: str, hit: bool):
    """
    Track cache hit/miss.

    Args:
        cache_type: Type of cache (redis, memory, etc.)
        hit: Whether it was a hit (True) or miss (False)
    """
    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()


def get_cache_hit_rate(cache_type: str) -> float:
    """
    Calculate cache hit rate.

    Args:
        cache_type: Type of cache

    Returns:
        Hit rate as percentage
    """
    hits = CACHE_HITS.labels(cache_type=cache_type)._value.get()
    misses = CACHE_MISSES.labels(cache_type=cache_type)._value.get()

    total = hits + misses
    if total == 0:
        return 0.0

    return (hits / total) * 100


def track_model_load(model_version: str, load_time: float):
    """
    Track model loading.

    Args:
        model_version: Version of model
        load_time: Time to load in seconds
    """
    MODEL_LOAD_TIME.labels(model_version=model_version).observe(load_time)
    MODEL_LOADED.labels(model_version=model_version).set(1)


def update_queue_depth(queue_name: str, depth: int):
    """
    Update queue depth metric.

    Args:
        queue_name: Name of queue
        depth: Number of tasks in queue
    """
    QUEUE_DEPTH.labels(queue_name=queue_name).set(depth)


def track_task_execution(task_name: str, duration: float, status: str = "success"):
    """
    Track async task execution.

    Args:
        task_name: Name of Celery task
        duration: Execution time in seconds
        status: Success or failure
    """
    TASK_EXECUTION_TIME.labels(
        task_name=task_name,
        status=status
    ).observe(duration)


def update_resource_metrics():
    """Update CPU and memory metrics."""
    try:
        import psutil
        process = psutil.Process()

        # Memory
        memory_info = process.memory_info()
        MEMORY_USAGE.set(memory_info.rss)

        # CPU
        cpu_percent = process.cpu_percent(interval=0.1)
        CPU_USAGE.set(cpu_percent)

    except ImportError:
        pass  # psutil not installed


def set_model_info(version: str, framework: str, input_shape: str):
    """
    Set model metadata.

    Args:
        version: Model version
        framework: ML framework (pytorch, tensorflow, etc.)
        input_shape: Expected input shape
    """
    MODEL_INFO.info({
        'version': version,
        'framework': framework,
        'input_shape': input_shape,
    })


# ====================
# FastAPI Integration
# ====================

def setup_metrics(app: FastAPI):
    """
    Setup Prometheus metrics for FastAPI app.

    Args:
        app: FastAPI application
    """
    # Add middleware
    app.add_middleware(PrometheusMiddleware)

    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        # Update resource metrics
        update_resource_metrics()

        # Generate metrics
        return Response(
            content=generate_latest(REGISTRY),
            media_type="text/plain"
        )

    print("✓ Prometheus metrics configured")
    print("  Endpoint: /metrics")


# ====================
# Example Usage
# ====================

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    import random

    app = FastAPI(title="Metrics Demo")

    # Setup metrics
    setup_metrics(app)

    # Set model info
    set_model_info(
        version="v1.0.0",
        framework="pytorch",
        input_shape="(batch, 10)"
    )

    @app.get("/")
    async def root():
        return {"message": "Metrics demo"}

    @app.post("/predict")
    async def predict():
        # Simulate prediction
        start = time.time()
        time.sleep(random.uniform(0.01, 0.1))
        latency = time.time() - start

        # Track metrics
        track_prediction(
            model_version="v1.0.0",
            prediction_type="single",
            latency=latency,
            status="success"
        )

        return {"prediction": 1, "latency": latency}

    @app.get("/cached")
    async def cached_endpoint():
        # Simulate cache hit/miss
        is_hit = random.random() > 0.3
        track_cache("redis", is_hit)

        return {"cache_hit": is_hit}

    print("\n" + "=" * 60)
    print("Prometheus Metrics Demo")
    print("=" * 60)
    print("API: http://localhost:8000")
    print("Metrics: http://localhost:8000/metrics")
    print("\nMake requests to /predict and /cached to generate metrics")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
