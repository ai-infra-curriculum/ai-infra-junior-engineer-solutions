"""
API routes for the inference gateway.
"""

import time
from datetime import datetime
from typing import Dict
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import (
    ModelNotLoadedException,
    InvalidInputException,
    ModelInferenceException,
)
from app.models.inference import get_model_service
from app.instrumentation.logging import get_logger
from app.instrumentation.metrics import record_inference, record_model_error, image_size_bytes
from app.instrumentation.tracing import TracedOperation, set_span_attribute

logger = get_logger(__name__)
router = APIRouter()


# ==================== Response Models ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    status: str
    model_loaded: bool
    model_name: str


class PredictionClass(BaseModel):
    """Single prediction class."""
    class_id: int
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    """Prediction response."""
    request_id: str
    model_name: str
    prediction: PredictionClass
    top5: list[PredictionClass]
    inference_time_ms: float


# ==================== Health Endpoints ====================

@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Liveness probe - is the service running?

    Returns basic health status without checking dependencies.
    Used by Kubernetes liveness probes.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version=settings.app_version
    )


@router.get("/ready", response_model=ReadinessResponse, tags=["health"])
async def readiness_check():
    """
    Readiness probe - is the service ready to accept traffic?

    Checks if the model is loaded and ready for inference.
    Used by Kubernetes readiness probes.
    """
    model_service = get_model_service()
    is_ready = model_service.is_ready()

    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail="Service not ready - model not loaded"
        )

    return ReadinessResponse(
        status="ready",
        model_loaded=is_ready,
        model_name=settings.model_name
    )


# ==================== Inference Endpoints ====================

@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file for classification")
):
    """
    Predict image class using ResNet-50 model.

    Args:
        request: FastAPI request object
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        Prediction with top 5 classes and confidence scores

    Raises:
        HTTPException: If model is not loaded or inference fails
    """
    request_id = getattr(request.state, "request_id", "unknown")

    with TracedOperation("predict_endpoint", {"endpoint": "/predict"}):
        try:
            # Validate content type
            if not file.content_type or not file.content_type.startswith("image/"):
                raise InvalidInputException(
                    f"Invalid content type: {file.content_type}. Expected image/*"
                )

            # Read image bytes
            image_bytes = await file.read()

            # Record image size metric
            image_size_bytes.observe(len(image_bytes))

            # Add span attributes
            set_span_attribute("image.size_bytes", len(image_bytes))
            set_span_attribute("image.content_type", file.content_type)

            logger.info(
                "Processing prediction request",
                request_id=request_id,
                filename=file.filename,
                content_type=file.content_type,
                size_bytes=len(image_bytes)
            )

            # Get model service
            model_service = get_model_service()

            # Run inference
            start_time = time.time()
            result = model_service.predict(image_bytes)
            inference_duration = (time.time() - start_time) * 1000

            # Record metrics
            record_inference(
                model_name=result["model_name"],
                duration=inference_duration / 1000,  # Convert to seconds
                prediction_class=result["prediction"]["class_name"],
                confidence=result["prediction"]["confidence"]
            )

            # Add span attributes
            set_span_attribute("prediction.class", result["prediction"]["class_name"])
            set_span_attribute("prediction.confidence", result["prediction"]["confidence"])
            set_span_attribute("inference.duration_ms", inference_duration)

            logger.info(
                "Prediction completed successfully",
                request_id=request_id,
                prediction_class=result["prediction"]["class_name"],
                confidence=result["prediction"]["confidence"],
                inference_time_ms=round(inference_duration, 2)
            )

            return PredictionResponse(
                request_id=request_id,
                **result
            )

        except ModelNotLoadedException as e:
            logger.error(
                "Model not loaded",
                request_id=request_id,
                error=str(e)
            )
            record_model_error(settings.model_name, "model_not_loaded")
            raise HTTPException(status_code=503, detail=str(e))

        except InvalidInputException as e:
            logger.error(
                "Invalid input",
                request_id=request_id,
                error=str(e)
            )
            record_model_error(settings.model_name, "invalid_input")
            raise HTTPException(status_code=400, detail=str(e))

        except ModelInferenceException as e:
            logger.error(
                "Inference failed",
                request_id=request_id,
                error=str(e)
            )
            record_model_error(settings.model_name, "inference_failed")
            raise HTTPException(status_code=500, detail=str(e))

        except Exception as e:
            logger.error(
                "Unexpected error",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            record_model_error(settings.model_name, "unexpected_error")
            raise HTTPException(status_code=500, detail="Internal server error")


# ==================== Info Endpoints ====================

@router.get("/info", tags=["info"])
async def info():
    """
    Get service information.

    Returns:
        Service metadata including version, model, configuration
    """
    model_service = get_model_service()

    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "model": {
            "name": settings.model_name,
            "device": settings.model_device,
            "loaded": model_service.is_ready(),
        },
        "slo": {
            "availability_target": settings.slo_availability_target,
            "latency_p99_target_ms": settings.slo_latency_p99_target_ms,
            "latency_p95_target_ms": settings.slo_latency_p95_target_ms,
        },
        "observability": {
            "metrics_enabled": settings.enable_metrics,
            "tracing_enabled": settings.enable_tracing,
            "log_level": settings.log_level,
        }
    }
