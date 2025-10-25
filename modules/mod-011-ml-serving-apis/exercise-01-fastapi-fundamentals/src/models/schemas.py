"""Pydantic models for request/response validation."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Single prediction request."""

    features: List[float] = Field(
        ...,
        min_items=5,
        max_items=5,
        description="Input features for prediction"
    )

    @validator('features')
    def validate_features(cls, v):
        """Validate features are non-negative."""
        if any(x < 0 for x in v):
            raise ValueError('All features must be non-negative')
        return v

    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0, 5.0]
            }
        }


class PredictionResponse(BaseModel):
    """Single prediction response."""

    prediction: int = Field(..., description="Predicted class")
    probability: List[float] = Field(..., description="Class probabilities")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": [0.23, 0.77],
                "model_version": "v1.0.0",
                "timestamp": "2025-10-24T12:00:00Z"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    instances: List[PredictionRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of prediction instances"
    )

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {"features": [1.0, 2.0, 3.0, 4.0, 5.0]},
                    {"features": [2.0, 3.0, 4.0, 5.0, 6.0]}
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResponse]
    batch_size: int
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Model load status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class APIInfo(BaseModel):
    """API information response."""

    name: str
    version: str
    model_type: str
    model_version: str
    environment: str
