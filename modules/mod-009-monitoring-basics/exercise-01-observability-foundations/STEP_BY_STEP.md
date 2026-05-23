# Step-by-Step Implementation Guide: Observability Foundations Lab

## Overview

This guide walks you through building a production-ready FastAPI inference service with comprehensive observability using **Prometheus metrics**, **structured logging**, and **OpenTelemetry tracing**. By the end, you'll have a fully instrumented ML inference service with SLO tracking.

**Estimated Time**: 3-4 hours
**Difficulty**: Beginner → Intermediate

## Table of Contents

1. [Phase 1: Project Setup](#phase-1-project-setup)
2. [Phase 2: Core FastAPI Application](#phase-2-core-fastapi-application)
3. [Phase 3: Prometheus Metrics Instrumentation](#phase-3-prometheus-metrics-instrumentation)
4. [Phase 4: Structured Logging](#phase-4-structured-logging)
5. [Phase 5: OpenTelemetry Tracing](#phase-5-opentelemetry-tracing)
6. [Phase 6: ML Model Integration](#phase-6-ml-model-integration)
7. [Phase 7: Health Checks and SLO Endpoints](#phase-7-health-checks-and-slo-endpoints)
8. [Phase 8: Docker and Observability Stack](#phase-8-docker-and-observability-stack)
9. [Phase 9: Testing and Validation](#phase-9-testing-and-validation)
10. [Phase 10: Production Deployment](#phase-10-production-deployment)

---

## Phase 1: Project Setup

### 1.1 Create Project Structure

```bash
# Create project directory
mkdir -p exercise-01-observability-foundations
cd exercise-01-observability-foundations

# Create Python application structure
mkdir -p app/{api,core,instrumentation,models}
mkdir -p app/{api,core,instrumentation,models}
mkdir -p tests/{unit,integration}
mkdir -p config docs scripts

# Create __init__.py files
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
touch app/instrumentation/__init__.py
touch app/models/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
```

### 1.2 Create Requirements Files

**`requirements.txt`** (Production dependencies):

```txt
# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# ML Libraries
torch==2.1.2
torchvision==0.16.2
pillow==10.2.0

# Observability - Metrics
prometheus-client==0.19.0

# Observability - Logging
structlog==24.1.0
python-json-logger==2.0.7

# Observability - Tracing
opentelemetry-api==1.22.0
opentelemetry-sdk==1.22.0
opentelemetry-instrumentation-fastapi==0.43b0
opentelemetry-exporter-otlp==1.22.0
opentelemetry-exporter-jaeger==1.22.0

# Configuration
pydantic==2.5.3
pydantic-settings==2.1.0
python-dotenv==1.0.0

# Utilities
httpx==0.26.0
```

**`requirements-dev.txt`** (Development dependencies):

```txt
-r requirements.txt

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
httpx==0.26.0

# Code Quality
black==23.12.1
isort==5.13.2
flake8==7.0.0
mypy==1.8.0

# Load Testing
locust==2.20.0
```

### 1.3 Create Environment Configuration

**`.env.example`**:

```bash
# Application Configuration
APP_NAME=inference-gateway
APP_VERSION=1.0.0
APP_DESCRIPTION="ML Inference Gateway with Observability"
LOG_LEVEL=INFO
WORKERS=4

# Model Configuration
MODEL_NAME=resnet50
MODEL_WARMUP=true
MODEL_DEVICE=cpu

# Observability - Metrics
ENABLE_METRICS=true
METRICS_PORT=8000

# Observability - Logging
LOG_FORMAT=json
LOG_OUTPUT=stdout

# Observability - Tracing
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
OTEL_SERVICE_NAME=inference-gateway
OTEL_SERVICE_VERSION=1.0.0

# Performance
MAX_QUEUE_SIZE=100
REQUEST_TIMEOUT=30
GRACEFUL_SHUTDOWN_TIMEOUT=10

# Development
RELOAD=false
DEBUG=false
```

Copy to `.env`:
```bash
cp .env.example .env
```

---

## Phase 2: Core FastAPI Application

### 2.1 Configuration Management

**`app/core/config.py`**:

```python
"""Application configuration using Pydantic Settings."""
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = Field(default="inference-gateway", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_description: str = Field(default="ML Inference Gateway", description="App description")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    workers: int = Field(default=4, ge=1, le=16)

    # Model
    model_name: str = "resnet50"
    model_warmup: bool = True
    model_device: Literal["cpu", "cuda"] = "cpu"

    # Observability - Metrics
    enable_metrics: bool = True
    metrics_port: int = 8000

    # Observability - Logging
    log_format: Literal["json", "text"] = "json"
    log_output: Literal["stdout", "file"] = "stdout"

    # Observability - Tracing
    enable_tracing: bool = True
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "inference-gateway"
    otel_service_version: str = "1.0.0"

    # Performance
    max_queue_size: int = 100
    request_timeout: int = 30
    graceful_shutdown_timeout: int = 10

    # Development
    reload: bool = False
    debug: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
```

### 2.2 Custom Exceptions

**`app/core/exceptions.py`**:

```python
"""Custom exception classes."""


class InferenceError(Exception):
    """Base exception for inference errors."""
    pass


class ModelNotLoadedError(InferenceError):
    """Raised when model is not loaded."""
    pass


class InvalidInputError(InferenceError):
    """Raised when input validation fails."""
    pass


class PredictionError(InferenceError):
    """Raised when prediction fails."""
    pass


class QueueFullError(InferenceError):
    """Raised when request queue is full."""
    pass
```

---

## Phase 3: Prometheus Metrics Instrumentation

### 3.1 Metrics Module

**`app/instrumentation/metrics.py`**:

```python
"""Prometheus metrics for inference gateway."""
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Optional

# ====================
# Application Info
# ====================
app_info = Info(
    'inference_gateway',
    'Inference Gateway application information'
)

# ====================
# Four Golden Signals
# ====================

# 1. LATENCY: How long requests take
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint', 'status'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
)

# 2. TRAFFIC: Rate of requests
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# 3. ERRORS: Rate of failed requests
http_errors_total = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'error_type']
)

# 4. SATURATION: How full the service is
inference_queue_size = Gauge(
    'inference_queue_size',
    'Current number of requests in inference queue'
)

# ====================
# ML-Specific Metrics
# ====================

model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference latency in seconds',
    ['model_name'],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
)

model_predictions_total = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_name', 'prediction_class']
)

model_prediction_confidence = Histogram(
    'model_prediction_confidence',
    'Model prediction confidence scores',
    ['model_name'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# ====================
# Resource Utilization
# ====================

model_memory_usage_bytes = Gauge(
    'model_memory_usage_bytes',
    'Memory used by loaded model in bytes',
    ['model_name']
)

# ====================
# Helper Functions
# ====================

def record_http_request(
    method: str,
    endpoint: str,
    status: int,
    duration: float
):
    """Record an HTTP request with all relevant metrics."""
    status_str = f"{status // 100}xx"

    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status=status_str
    ).inc()

    http_request_duration_seconds.labels(
        method=method,
        endpoint=endpoint,
        status=status_str
    ).observe(duration)


def record_http_error(
    method: str,
    endpoint: str,
    error_type: str
):
    """Record an HTTP error."""
    http_errors_total.labels(
        method=method,
        endpoint=endpoint,
        error_type=error_type
    ).inc()


def record_model_prediction(
    model_name: str,
    prediction_class: str,
    confidence: float,
    duration: float
):
    """Record a model prediction with all metrics."""
    model_predictions_total.labels(
        model_name=model_name,
        prediction_class=prediction_class
    ).inc()

    model_inference_duration_seconds.labels(
        model_name=model_name
    ).observe(duration)

    model_prediction_confidence.labels(
        model_name=model_name
    ).observe(confidence)
```

### 3.2 Observability Middleware

**`app/instrumentation/middleware.py`**:

```python
"""FastAPI middleware for observability."""
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.instrumentation.metrics import record_http_request, inference_queue_size
from app.instrumentation.logging import get_logger

logger = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware to add observability to all requests."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._active_requests = 0

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Track queue size (saturation metric)
        self._active_requests += 1
        inference_queue_size.set(self._active_requests)

        # Start timing
        start_time = time.time()

        # Log request received
        logger.info(
            "request_received",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Record metrics
            record_http_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration
            )

            # Log successful response
            logger.info(
                "request_completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2)
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            duration = time.time() - start_time

            # Log error
            logger.error(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(exc),
                duration_ms=round(duration * 1000, 2),
                exc_info=True
            )

            raise

        finally:
            # Decrement active requests
            self._active_requests -= 1
            inference_queue_size.set(self._active_requests)
```

---

## Phase 4: Structured Logging

### 4.1 Logging Configuration

**`app/instrumentation/logging.py`**:

```python
"""Structured logging configuration using structlog."""
import logging
import sys
from typing import Optional

import structlog
from opentelemetry import trace

from app.core.config import settings


def setup_logging():
    """Configure structured logging with structlog."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )

    # Processors for structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Add trace context processor if tracing enabled
    if settings.enable_tracing:
        processors.append(add_trace_context)

    # Format as JSON or console
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def add_trace_context(logger, method_name, event_dict):
    """Add OpenTelemetry trace context to logs."""
    span = trace.get_current_span()
    if span:
        span_context = span.get_span_context()
        if span_context.is_valid:
            event_dict["trace_id"] = format(span_context.trace_id, "032x")
            event_dict["span_id"] = format(span_context.span_id, "016x")
    return event_dict


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Initialize logging on module import
setup_logging()
```

---

## Phase 5: OpenTelemetry Tracing

### 5.1 Tracing Configuration

**`app/instrumentation/tracing.py`**:

```python
"""OpenTelemetry tracing configuration."""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource

from app.core.config import settings


def setup_tracing():
    """Configure OpenTelemetry tracing."""

    if not settings.enable_tracing:
        return None

    # Create resource with service information
    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": settings.otel_service_version,
    })

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Configure OTLP exporter (for Jaeger/Tempo/etc)
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint,
        insecure=True  # Use insecure for local development
    )

    # Add span processor
    tracer_provider.add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )

    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider


def instrument_fastapi(app):
    """Instrument FastAPI application with automatic tracing."""
    if settings.enable_tracing:
        FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str):
    """Get a tracer instance."""
    return trace.get_tracer(name)
```

---

## Phase 6: ML Model Integration

### 6.1 Inference Model

**`app/models/inference.py`**:

```python
"""ML model inference with observability."""
import time
from typing import Tuple
from io import BytesIO

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

from app.core.config import settings
from app.core.exceptions import ModelNotLoadedError, PredictionError
from app.instrumentation.metrics import record_model_prediction, model_memory_usage_bytes
from app.instrumentation.logging import get_logger
from app.instrumentation.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class ImageClassifier:
    """Image classification model with observability."""

    def __init__(self):
        self.model = None
        self.device = torch.device(settings.model_device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.classes = None

    def load_model(self):
        """Load the ResNet-50 model."""
        with tracer.start_as_current_span("load_model") as span:
            span.set_attribute("model.name", settings.model_name)

            logger.info("loading_model", model_name=settings.model_name)

            start_time = time.time()

            # Load model with pretrained weights
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.to(self.device)
            self.model.eval()

            # Load class names
            self.classes = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

            # Record memory usage
            if settings.model_device == "cuda":
                memory_bytes = torch.cuda.memory_allocated()
            else:
                # Estimate CPU memory (rough approximation)
                memory_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

            model_memory_usage_bytes.labels(
                model_name=settings.model_name
            ).set(memory_bytes)

            load_time = time.time() - start_time

            logger.info(
                "model_loaded",
                model_name=settings.model_name,
                load_time_seconds=round(load_time, 2),
                memory_bytes=memory_bytes,
                device=str(self.device)
            )

            span.set_attribute("model.load_time_seconds", load_time)
            span.set_attribute("model.memory_bytes", memory_bytes)

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess image for model input."""
        with tracer.start_as_current_span("preprocess_image"):
            try:
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
                tensor = self.transform(image).unsqueeze(0)
                return tensor.to(self.device)
            except Exception as e:
                logger.error("image_preprocess_failed", error=str(e))
                raise PredictionError(f"Failed to preprocess image: {e}")

    async def predict(self, image_bytes: bytes) -> Tuple[str, float]:
        """Run inference and return prediction with confidence."""
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded. Call load_model() first.")

        with tracer.start_as_current_span("model_inference") as span:
            start_time = time.time()

            try:
                # Preprocess
                input_tensor = self.preprocess_image(image_bytes)

                # Inference
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)

                # Get prediction
                prediction_class = self.classes[predicted_idx.item()]
                confidence_score = confidence.item()

                # Record metrics
                inference_duration = time.time() - start_time
                record_model_prediction(
                    model_name=settings.model_name,
                    prediction_class=prediction_class,
                    confidence=confidence_score,
                    duration=inference_duration
                )

                # Add span attributes
                span.set_attribute("model.name", settings.model_name)
                span.set_attribute("prediction.class", prediction_class)
                span.set_attribute("prediction.confidence", confidence_score)
                span.set_attribute("inference.duration_seconds", inference_duration)

                # Log prediction
                logger.info(
                    "prediction_completed",
                    model_name=settings.model_name,
                    prediction=prediction_class,
                    confidence=round(confidence_score, 4),
                    duration_ms=round(inference_duration * 1000, 2)
                )

                return prediction_class, confidence_score

            except Exception as e:
                logger.error(
                    "prediction_failed",
                    model_name=settings.model_name,
                    error=str(e),
                    exc_info=True
                )
                raise PredictionError(f"Prediction failed: {e}")


# Global model instance
classifier = ImageClassifier()
```

---

## Phase 7: Health Checks and SLO Endpoints

### 7.1 API Routes

**`app/api/routes.py`**:

```python
"""API endpoints for inference gateway."""
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime

from app.models.inference import classifier
from app.core.exceptions import ModelNotLoadedError, PredictionError
from app.instrumentation.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    status: str
    model_loaded: bool
    timestamp: datetime


class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: str
    confidence: float
    request_id: str


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Liveness probe - is the service running?

    Returns 200 if the service is alive.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow()
    )


@router.get("/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check():
    """
    Readiness probe - is the service ready to accept traffic?

    Returns 200 if model is loaded and service is ready.
    """
    model_loaded = classifier.model is not None

    return ReadinessResponse(
        status="ready" if model_loaded else "not_ready",
        model_loaded=model_loaded,
        timestamp=datetime.utcnow()
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file to classify")
):
    """
    Predict the class of an uploaded image.

    - **file**: Image file (JPEG, PNG, etc.)
    - Returns prediction class and confidence score
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Run inference
        prediction, confidence = await classifier.predict(image_bytes)

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            request_id=request_id
        )

    except ModelNotLoadedError as e:
        logger.error("model_not_loaded", request_id=request_id, error=str(e))
        raise HTTPException(status_code=503, detail="Model not loaded")

    except PredictionError as e:
        logger.error("prediction_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    except Exception as e:
        logger.error("unexpected_error", request_id=request_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 7.2 Main Application

**`app/main.py`**:

```python
"""FastAPI application entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import settings
from app.api.routes import router
from app.models.inference import classifier
from app.instrumentation.middleware import ObservabilityMiddleware
from app.instrumentation.tracing import setup_tracing, instrument_fastapi
from app.instrumentation.logging import get_logger
from app.instrumentation.metrics import app_info

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("application_starting", app_name=settings.app_name, version=settings.app_version)

    # Set application info metric
    app_info.info({
        'version': settings.app_version,
        'name': settings.app_name,
        'description': settings.app_description
    })

    # Setup tracing
    setup_tracing()

    # Load ML model
    if settings.model_warmup:
        logger.info("warming_up_model")
        classifier.load_model()
        logger.info("model_warmup_complete")

    logger.info("application_ready")

    yield

    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan
)

# Add observability middleware
app.add_middleware(ObservabilityMiddleware)

# Instrument with OpenTelemetry
instrument_fastapi(app)

# Include API routes
app.include_router(router)


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }
```

---

## Phase 8: Docker and Observability Stack

### 8.1 Dockerfile

**`Dockerfile`**:

```dockerfile
# Multi-stage build for smaller image

FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ===================
# Production Stage
# ===================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY app/ ./app/
COPY .env.example .env

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 Docker Compose with Observability Stack

**`docker-compose.yml`**:

```yaml
version: '3.8'

services:
  # ==================
  # Inference Gateway
  # ==================
  inference-gateway:
    build: .
    container_name: inference-gateway
    ports:
      - "8000:8000"
    environment:
      - APP_NAME=inference-gateway
      - LOG_LEVEL=INFO
      - ENABLE_METRICS=true
      - ENABLE_TRACING=true
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
      - MODEL_WARMUP=true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

  # ==================
  # Prometheus
  # ==================
  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - monitoring

  # ==================
  # Jaeger (Tracing)
  # ==================
  jaeger:
    image: jaegertracing/all-in-one:1.53
    container_name: jaeger
    ports:
      - "16686:16686"  # Jaeger UI
      - "4318:4318"    # OTLP gRPC receiver
      - "4317:4317"    # OTLP HTTP receiver
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - monitoring

volumes:
  prometheus_data:

networks:
  monitoring:
    driver: bridge
```

### 8.3 Prometheus Configuration

**`config/prometheus.yml`**:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'local'
    environment: 'dev'

scrape_configs:
  # Inference Gateway
  - job_name: 'inference-gateway'
    static_configs:
      - targets: ['inference-gateway:8000']
        labels:
          service: 'inference-gateway'
          team: 'ml-platform'

  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

---

## Phase 9: Testing and Validation

### 9.1 Create Test Image

```bash
# Download a test image
curl -o test_image.jpg https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg
```

### 9.2 Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f inference-gateway

# Wait for model to load (30-60 seconds)
```

### 9.3 Test Endpoints

```bash
# 1. Health Check
curl http://localhost:8000/health
# Expected: {"status":"healthy","timestamp":"2025-10-24T..."}

# 2. Readiness Check
curl http://localhost:8000/ready
# Expected: {"status":"ready","model_loaded":true,...}

# 3. Make Prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
# Expected: {"prediction":"tabby","confidence":0.85,...}

# 4. Check Metrics
curl http://localhost:8000/metrics | grep http_requests_total
# Expected: http_requests_total{endpoint="/predict",method="POST",status="2xx"} 1.0
```

### 9.4 Access Monitoring UIs

```bash
# Open Prometheus
open http://localhost:9090

# Try queries:
# - http_requests_total
# - rate(http_requests_total[5m])
# - histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Open Jaeger
open http://localhost:16686

# Search for traces from "inference-gateway"
```

### 9.5 Load Testing

Create **`scripts/load_test.sh`**:

```bash
#!/bin/bash

echo "Running load test..."
echo "Sending 100 requests to /predict endpoint"

for i in {1..100}; do
    curl -X POST http://localhost:8000/predict \
      -F "file=@test_image.jpg" \
      -s -o /dev/null -w "%{http_code}\n" &

    # Limit concurrent requests
    if (( i % 10 == 0 )); then
        wait
        echo "Completed $i requests"
    fi
done

wait
echo "Load test complete!"
echo ""
echo "Check Prometheus for metrics:"
echo "  http://localhost:9090"
echo ""
echo "Check Jaeger for traces:"
echo "  http://localhost:16686"
```

Run load test:
```bash
chmod +x scripts/load_test.sh
./scripts/load_test.sh
```

---

## Phase 10: Production Deployment

### 10.1 Production Configuration Checklist

- [ ] Set `LOG_LEVEL=INFO` (not DEBUG)
- [ ] Set `MODEL_WARMUP=true` for faster startup
- [ ] Configure resource limits in Docker/Kubernetes
- [ ] Set up persistent storage for Prometheus data
- [ ] Configure long-term storage (Thanos, Cortex, or Mimir)
- [ ] Set up Alertmanager for alert notifications
- [ ] Configure TLS/HTTPS for production endpoints
- [ ] Set up proper CORS policies
- [ ] Enable rate limiting
- [ ] Configure log aggregation (Loki, ELK)

### 10.2 Kubernetes Deployment (Optional)

Create **`k8s/deployment.yaml`**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-gateway
  labels:
    app: inference-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference-gateway
  template:
    metadata:
      labels:
        app: inference-gateway
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: inference-gateway
        image: inference-gateway:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENABLE_METRICS
          value: "true"
        - name: ENABLE_TRACING
          value: "true"
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger-collector:4318"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## Troubleshooting

### Model Loading Issues

```bash
# Check if model is loaded
curl http://localhost:8000/ready

# If not loaded, check logs
docker-compose logs inference-gateway | grep "loading_model"
```

### Metrics Not Appearing

```bash
# Verify metrics endpoint works
curl http://localhost:8000/metrics

# Check Prometheus targets
open http://localhost:9090/targets

# Verify Prometheus can scrape
docker-compose logs prometheus | grep inference-gateway
```

### Traces Not Showing in Jaeger

```bash
# Check OTLP endpoint is reachable
docker-compose logs jaeger

# Verify tracing is enabled
curl http://localhost:8000/ | jq

# Check application logs for tracing errors
docker-compose logs inference-gateway | grep -i trace
```

---

## Next Steps

After completing this exercise, you'll build upon this foundation in:

1. **Exercise 02**: Deploy production Prometheus stack with recording rules and SLO tracking
2. **Exercise 03**: Create comprehensive Grafana dashboards for visualization
3. **Exercise 04**: Implement centralized logging with Loki or ELK stack
4. **Exercise 05**: Configure alerting and incident response workflows

---

## Summary

**What You Built**:
- ✅ FastAPI inference service with PyTorch ResNet-50
- ✅ Prometheus metrics (Four Golden Signals + ML-specific metrics)
- ✅ Structured JSON logging with correlation IDs
- ✅ OpenTelemetry distributed tracing
- ✅ Health and readiness probes
- ✅ Complete observability stack (Prometheus + Jaeger)
- ✅ Production-ready Docker deployment

**Key Metrics Exposed**:
- HTTP request duration, count, errors
- Model inference latency and predictions
- Queue size (saturation)
- Model memory usage
- Prediction confidence distribution

**Production Readiness**:
- Multi-stage Docker builds
- Non-root container user
- Health checks for Kubernetes
- Graceful shutdown handling
- Comprehensive error handling
- Request ID tracking across all logs and traces

You now have a solid foundation in observability for ML systems! 🎉
