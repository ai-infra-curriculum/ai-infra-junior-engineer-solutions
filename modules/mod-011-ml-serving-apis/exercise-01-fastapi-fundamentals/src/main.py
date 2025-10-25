"""
FastAPI ML Prediction API

A production-ready ML serving API built with FastAPI demonstrating:
- Async endpoints
- Pydantic validation
- Error handling
- Background tasks
- Middleware
- API documentation
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.models.ml_model import MLModel
from src.routers import health, predictions
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

# Global model instance
model_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.

    This replaces the deprecated @app.on_event decorators.
    """
    # Startup
    global model_instance
    logger.info("Starting up FastAPI application...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Model path: {settings.MODEL_PATH}")

    try:
        model_instance = MLModel(model_path=settings.MODEL_PATH)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application...")
    model_instance = None


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description="Production ML Prediction API built with FastAPI",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    response.headers["X-Server-Time"] = datetime.utcnow().isoformat()

    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Time: {process_time:.4f}s"
    )

    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(
        f"Incoming request: {request.method} {request.url.path} "
        f"from {request.client.host}"
    )
    response = await call_next(request)
    return response


# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(
    predictions.router,
    prefix="/api/v1",
    tags=["predictions"]
)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# Get model instance (for dependency injection)
def get_model() -> MLModel:
    """Dependency to get the global model instance."""
    if model_instance is None:
        raise RuntimeError("Model not loaded")
    return model_instance


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
