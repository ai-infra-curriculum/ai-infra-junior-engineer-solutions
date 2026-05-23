"""
Inference Gateway - FastAPI Application with Full Observability.

This application demonstrates comprehensive observability with:
- Prometheus metrics (Four Golden Signals)
- Structured JSON logging with trace correlation
- OpenTelemetry distributed tracing
- Health checks and SLO tracking
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.core.config import settings
from app.api import routes
from app.models.inference import get_model_service
from app.instrumentation.logging import setup_logging, get_logger
from app.instrumentation.metrics import initialize_metrics
from app.instrumentation.tracing import setup_tracing, instrument_fastapi
from app.instrumentation.middleware import ObservabilityMiddleware

# Setup logging first
setup_logging(log_level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting Inference Gateway",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        model_name=settings.model_name
    )

    # Initialize observability
    initialize_metrics(
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=settings.environment
    )
    logger.info("Metrics initialized")

    if settings.enable_tracing and settings.otel_exporter_otlp_endpoint:
        setup_tracing(
            service_name=settings.otel_service_name,
            service_version=settings.app_version,
            otlp_endpoint=settings.otel_exporter_otlp_endpoint,
            enable_console_export=settings.debug
        )
        logger.info(
            "Tracing initialized",
            otlp_endpoint=settings.otel_exporter_otlp_endpoint
        )

    # Load model
    try:
        model_service = get_model_service()
        model_service.load_model()

        if settings.model_warmup:
            model_service.warmup()

        logger.info(
            "Model loaded and ready",
            model_name=settings.model_name,
            device=settings.model_device
        )
    except Exception as e:
        logger.error(
            "Failed to load model",
            error=str(e),
            exc_info=True
        )
        # Continue startup even if model fails to load
        # Readiness check will fail until model is loaded

    logger.info("Inference Gateway started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Inference Gateway")


# Create FastAPI application
app = FastAPI(
    title="Inference Gateway",
    description="ML inference service with comprehensive observability",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Add observability middleware
app.add_middleware(ObservabilityMiddleware)

# Instrument with OpenTelemetry
if settings.enable_tracing:
    instrument_fastapi(app)

# Include API routes
app.include_router(routes.router)

# Mount Prometheus metrics endpoint
if settings.enable_metrics:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with service information.
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health",
        "ready": "/ready"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
