"""
ML Model Inference API

This FastAPI application provides REST endpoints for image classification
using a pre-trained neural network model.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import io
from PIL import Image

from models.classifier import ImageClassifier
from preprocessing.image import ImagePreprocessor
from utils.logging import get_logger


# Initialize logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ML Image Classification API",
    description="REST API for image classification using PyTorch models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
classifier: Optional[ImageClassifier] = None
preprocessor: ImagePreprocessor = ImagePreprocessor()


# Response models
class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    success: bool
    predictions: List[Dict[str, float]] = Field(
        ...,
        description="List of class predictions with confidence scores"
    )
    top_prediction: str = Field(..., description="Most likely class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    {"class": "golden_retriever", "confidence": 0.92},
                    {"class": "labrador", "confidence": 0.05},
                    {"class": "german_shepherd", "confidence": 0.02}
                ],
                "top_prediction": "golden_retriever",
                "confidence": 0.92
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    """Response model for errors"""
    success: bool = False
    error: str
    detail: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize model on application startup
    """
    global classifier
    try:
        logger.info("Loading image classification model...")
        classifier = ImageClassifier()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Don't fail startup, but model won't be available
        classifier = None


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on application shutdown
    """
    logger.info("Shutting down ML inference API")


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "ML Image Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns:
        Health status including model availability
    """
    return HealthResponse(
        status="healthy" if classifier is not None else "degraded",
        model_loaded=classifier is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file to classify")
):
    """
    Classify an uploaded image

    Args:
        file: Image file (JPEG, PNG)

    Returns:
        Classification results with top predictions

    Raises:
        HTTPException: If model not loaded or invalid image
    """
    # Check if model is loaded
    if classifier is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Must be an image."
        )

    try:
        # Read image
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess
        processed_image = preprocessor.preprocess(image)

        # Predict
        predictions = classifier.predict(processed_image)

        # Format response
        top_class = predictions[0]["class"]
        top_confidence = predictions[0]["confidence"]

        logger.info(
            f"Prediction complete: {top_class} ({top_confidence:.2%})"
        )

        return PredictionResponse(
            success=True,
            predictions=predictions,
            top_prediction=top_class,
            confidence=top_confidence
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple image files")
):
    """
    Classify multiple images in batch

    Args:
        files: List of image files

    Returns:
        List of classification results
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images per batch request"
        )

    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            processed_image = preprocessor.preprocess(image)
            predictions = classifier.predict(processed_image)

            results.append(PredictionResponse(
                success=True,
                predictions=predictions,
                top_prediction=predictions[0]["class"],
                confidence=predictions[0]["confidence"]
            ))
        except Exception as e:
            logger.error(f"Batch prediction error for {file.filename}: {e}")
            results.append(ErrorResponse(
                success=False,
                error=str(e),
                detail=f"Failed to process {file.filename}"
            ))

    return results


@app.get("/classes", response_model=List[str])
async def get_classes():
    """
    Get list of available classification classes

    Returns:
        List of class names the model can predict
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return classifier.get_classes()


@app.get("/model/info", response_model=Dict[str, any])
async def get_model_info():
    """
    Get information about the loaded model

    Returns:
        Model metadata including architecture and classes
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "architecture": classifier.model_name,
        "num_classes": len(classifier.get_classes()),
        "input_size": preprocessor.target_size,
        "classes": classifier.get_classes()[:10]  # First 10 classes
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ML Inference API server...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
