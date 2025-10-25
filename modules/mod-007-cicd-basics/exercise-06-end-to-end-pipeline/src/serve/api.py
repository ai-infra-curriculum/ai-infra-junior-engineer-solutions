"""FastAPI serving endpoint for ML model."""

import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[Dict[str, Any]]
    model_uri: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    model_uri: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool


def create_app(model_uri: str = None) -> FastAPI:
    """Create FastAPI application.

    Args:
        model_uri: MLflow model URI

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="ML Model API",
        description="Production ML model serving API",
        version="1.0.0"
    )

    # Global model storage
    model_cache = {"model": None, "uri": None}

    def load_model(uri: str):
        """Load model from MLflow."""
        if model_cache["uri"] != uri:
            logger.info(f"Loading model from {uri}")
            model_cache["model"] = mlflow.sklearn.load_model(uri)
            model_cache["uri"] = uri
            logger.info("Model loaded successfully")
        return model_cache["model"]

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=model_cache["model"] is not None
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Make predictions.

        Args:
            request: Prediction request with features

        Returns:
            Prediction response
        """
        try:
            # Determine model URI
            uri = request.model_uri or model_uri
            if not uri:
                raise HTTPException(
                    status_code=400,
                    detail="No model URI provided"
                )

            # Load model
            model = load_model(uri)

            # Convert features to DataFrame
            df = pd.DataFrame(request.features)

            # Make predictions
            predictions = model.predict(df)

            return PredictionResponse(
                predictions=predictions.tolist(),
                model_uri=uri
            )

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )

    @app.get("/model/info")
    async def model_info():
        """Get information about loaded model."""
        if model_cache["model"] is None:
            raise HTTPException(
                status_code=404,
                detail="No model loaded"
            )

        return {
            "model_uri": model_cache["uri"],
            "model_type": type(model_cache["model"]).__name__
        }

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True, help="MLflow model URI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    app = create_app(model_uri=args.model_uri)
    uvicorn.run(app, host=args.host, port=args.port)
