"""Prediction endpoints."""

from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from src.config import settings
from src.models.ml_model import MLModel
from src.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    APIInfo,
)
from src.main import get_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


def log_prediction_background(request: PredictionRequest, response: PredictionResponse):
    """Log prediction in background."""
    logger.info(
        f"Prediction logged: features={request.features}, "
        f"prediction={response.prediction}"
    )


@router.get("/info", response_model=APIInfo)
async def get_api_info() -> APIInfo:
    """Get API information."""
    return APIInfo(
        name=settings.API_TITLE,
        version=settings.API_VERSION,
        model_type=settings.MODEL_TYPE,
        model_version=settings.MODEL_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model: MLModel = Depends(get_model),
) -> PredictionResponse:
    """
    Make a single prediction.

    Args:
        request: Prediction request with features
        background_tasks: FastAPI background tasks
        model: ML model instance

    Returns:
        Prediction response
    """
    try:
        logger.info(f"Received prediction request: {request.features}")

        prediction, probability = await model.predict_async(request.features)

        response = PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=settings.MODEL_VERSION,
            timestamp=datetime.utcnow(),
        )

        # Log in background
        background_tasks.add_task(
            log_prediction_background, request, response
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model: MLModel = Depends(get_model),
) -> BatchPredictionResponse:
    """
    Make batch predictions.

    Args:
        request: Batch prediction request
        model: ML model instance

    Returns:
        Batch prediction response
    """
    try:
        logger.info(f"Received batch request: {len(request.instances)} instances")

        predictions = []
        for instance in request.instances:
            prediction, probability = await model.predict_async(instance.features)
            predictions.append(
                PredictionResponse(
                    prediction=prediction,
                    probability=probability,
                    model_version=settings.MODEL_VERSION,
                    timestamp=datetime.utcnow(),
                )
            )

        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            model_version=settings.MODEL_VERSION,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")
