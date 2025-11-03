"""Pydantic models for request/response validation.

This module defines all data models used by the ML serving API.
Models provide:
- Automatic validation
- Type enforcement at runtime
- Schema generation for OpenAPI
- Clear documentation of expected formats
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response model.

    Returns current API status and timestamp.
    """
    status: HealthStatus = Field(
        ...,
        description="Current health status of the API"
    )
    timestamp: datetime = Field(
        ...,
        description="Time when health check was performed"
    )
    uptime_seconds: Optional[float] = Field(
        None,
        description="Number of seconds the service has been running"
    )
    model_loaded: bool = Field(
        True,
        description="Whether the ML model is loaded and ready"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00.000Z",
                "uptime_seconds": 3600.5,
                "model_loaded": True
            }
        }


class PredictionRequest(BaseModel):
    """Single prediction request model.

    Accepts a single feature vector and returns a prediction.
    """
    features: List[float] = Field(
        ...,
        min_items=10,
        max_items=10,
        description="List of exactly 10 numerical features in range [-1000, 1000]"
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional unique identifier for request tracking"
    )

    @validator('features')
    def validate_features(cls, v):
        """Ensure all features are valid numbers within acceptable range."""
        if not v:
            raise ValueError('Features list cannot be empty')

        for i, feature in enumerate(v):
            if not isinstance(feature, (int, float)):
                raise ValueError(f'Feature {i} must be a number, got {type(feature).__name__}')

            # Check for NaN or infinity
            if not (-1000 <= feature <= 1000):
                raise ValueError(
                    f'Feature {i} value {feature} out of valid range [-1000, 1000]'
                )

        return v

    class Config:
        schema_extra = {
            "example": {
                "features": [1.2, 3.4, 5.6, 7.8, 9.0, 2.3, 4.5, 6.7, 8.9, 0.1],
                "request_id": "req-12345"
            }
        }


class PredictionResponse(BaseModel):
    """Single prediction response model.

    Returns the model's prediction and metadata.
    """
    prediction: float = Field(
        ...,
        description="Model prediction value"
    )
    cached: bool = Field(
        False,
        description="Whether this result was served from cache"
    )
    model_version: str = Field(
        ...,
        description="Version of the model that generated this prediction"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID if provided in the request"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model confidence score (if available)"
    )

    class Config:
        schema_extra = {
            "example": {
                "prediction": 42.5,
                "cached": False,
                "model_version": "1.0",
                "request_id": "req-12345",
                "confidence": 0.95
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model.

    Accepts multiple feature vectors for batch processing.
    Maximum 100 samples to prevent resource exhaustion.
    """
    samples: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of feature vectors (each with 10 features, max 100 samples)"
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional unique identifier for batch request tracking"
    )

    @validator('samples')
    def validate_samples(cls, v):
        """Validate each sample has correct number of features and valid values."""
        if not v:
            raise ValueError('Samples list cannot be empty')

        for i, sample in enumerate(v):
            # Check it's a list
            if not isinstance(sample, list):
                raise ValueError(f'Sample {i} must be a list, got {type(sample).__name__}')

            # Check correct length
            if len(sample) != 10:
                raise ValueError(
                    f'Sample {i} must have exactly 10 features, got {len(sample)}'
                )

            # Validate each feature
            for j, feature in enumerate(sample):
                if not isinstance(feature, (int, float)):
                    raise ValueError(
                        f'Sample {i}, feature {j} must be a number, got {type(feature).__name__}'
                    )

                if not (-1000 <= feature <= 1000):
                    raise ValueError(
                        f'Sample {i}, feature {j} value {feature} out of valid range [-1000, 1000]'
                    )

        return v

    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    [1.2, 3.4, 5.6, 7.8, 9.0, 2.3, 4.5, 6.7, 8.9, 0.1],
                    [2.1, 4.3, 6.5, 8.7, 0.9, 3.2, 5.4, 7.6, 9.8, 1.0],
                    [3.0, 5.2, 7.4, 9.6, 1.8, 4.1, 6.3, 8.5, 0.7, 2.9]
                ],
                "request_id": "batch-12345"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model.

    Returns predictions for all samples in the batch.
    """
    predictions: List[float] = Field(
        ...,
        description="List of predictions, one per sample"
    )
    count: int = Field(
        ...,
        ge=1,
        description="Number of predictions returned"
    )
    model_version: str = Field(
        ...,
        description="Version of the model that generated these predictions"
    )
    request_id: Optional[str] = Field(
        None,
        description="Batch request ID if provided"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to process all predictions (milliseconds)"
    )

    @root_validator
    def validate_count_matches_predictions(cls, values):
        """Ensure count matches actual number of predictions."""
        predictions = values.get('predictions', [])
        count = values.get('count', 0)

        if len(predictions) != count:
            raise ValueError(
                f'Count mismatch: count={count} but got {len(predictions)} predictions'
            )

        return values

    class Config:
        schema_extra = {
            "example": {
                "predictions": [42.5, 38.2, 45.7],
                "count": 3,
                "model_version": "1.0",
                "request_id": "batch-12345",
                "processing_time_ms": 125.3
            }
        }


class ModelInfo(BaseModel):
    """Model information response.

    Provides metadata about the deployed model.
    """
    version: str = Field(
        ...,
        description="Model version identifier"
    )
    input_features: int = Field(
        ...,
        ge=1,
        description="Number of input features expected"
    )
    model_type: str = Field(
        ...,
        description="Type of machine learning model (e.g., random_forest, neural_network)"
    )
    training_date: str = Field(
        ...,
        description="Date when model was trained (ISO format)"
    )
    accuracy: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model accuracy on test set (if available)"
    )
    framework: Optional[str] = Field(
        None,
        description="ML framework used (e.g., scikit-learn, tensorflow)"
    )

    class Config:
        schema_extra = {
            "example": {
                "version": "1.0",
                "input_features": 10,
                "model_type": "random_forest",
                "training_date": "2024-01-15",
                "accuracy": 0.94,
                "framework": "scikit-learn"
            }
        }


class LoginRequest(BaseModel):
    """Login credentials request.

    Used to authenticate and receive a JWT token.
    """
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Username (3-50 characters)"
    )
    password: str = Field(
        ...,
        min_length=8,
        description="Password (minimum 8 characters)"
    )

    class Config:
        schema_extra = {
            "example": {
                "username": "admin",
                "password": "password123"
            }
        }


class LoginResponse(BaseModel):
    """Login response with JWT token.

    Returns an authentication token valid for 24 hours.
    """
    token: str = Field(
        ...,
        description="JWT authentication token"
    )
    token_type: str = Field(
        "bearer",
        description="Token type (always 'bearer')"
    )
    expires_in: int = Field(
        86400,
        description="Token expiration time in seconds (default: 24 hours)"
    )

    class Config:
        schema_extra = {
            "example": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 86400
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model.

    Used for all error responses to provide consistent error format.
    """
    error: str = Field(
        ...,
        description="Short error message"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error description"
    )
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="When the error occurred"
    )

    class Config:
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Feature 5 out of valid range",
                "error_code": "VALIDATION_ERROR",
                "timestamp": "2024-01-15T10:30:00.000Z"
            }
        }


class CacheStats(BaseModel):
    """Cache statistics model.

    Returns information about the prediction cache.
    """
    size: int = Field(
        ...,
        ge=0,
        description="Number of entries in cache"
    )
    hit_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cache hit rate (0.0 to 1.0)"
    )
    total_requests: int = Field(
        ...,
        ge=0,
        description="Total number of prediction requests"
    )
    cache_hits: int = Field(
        ...,
        ge=0,
        description="Number of cache hits"
    )
    cache_misses: int = Field(
        ...,
        ge=0,
        description="Number of cache misses"
    )

    class Config:
        schema_extra = {
            "example": {
                "size": 150,
                "hit_rate": 0.65,
                "total_requests": 1000,
                "cache_hits": 650,
                "cache_misses": 350
            }
        }
