"""Health check endpoints."""

from datetime import datetime
from fastapi import APIRouter, Depends

from src.config import settings
from src.models.ml_model import MLModel
from src.models.schemas import HealthResponse
from src.main import get_model

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(model: MLModel = Depends(get_model)) -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status response
    """
    model_loaded = model.model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        version=settings.API_VERSION,
        model_loaded=model_loaded,
        timestamp=datetime.utcnow(),
    )


@router.get("/ready")
async def readiness_check(model: MLModel = Depends(get_model)):
    """
    Readiness check for Kubernetes.

    Returns 200 if ready, 503 if not ready.
    """
    if model.model is None:
        return {"status": "not ready"}, 503

    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes.

    Always returns 200 if the process is running.
    """
    return {"status": "alive"}
