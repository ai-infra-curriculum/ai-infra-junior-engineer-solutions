"""Model Inference Module"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Make predictions."""
    if X.shape[0] == 0:
        raise ValueError("Input data cannot be empty")

    logger.info(f"Making predictions on {X.shape[0]} samples")
    predictions = model.predict(X)
    return predictions


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict probabilities."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support probability predictions")

    logger.info(f"Predicting probabilities for {X.shape[0]} samples")
    probabilities = model.predict_proba(X)
    return probabilities


def predict_single(model: Any, features: dict) -> Any:
    """Predict for single instance."""
    X = pd.DataFrame([features])
    prediction = predict(model, X)[0]
    logger.info(f"Prediction for single instance: {prediction}")
    return prediction


def batch_predict(model: Any, X: pd.DataFrame, batch_size: int = 1000) -> np.ndarray:
    """Predict in batches for large datasets."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    all_predictions = []

    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i + batch_size]
        batch_predictions = predict(model, batch)
        all_predictions.extend(batch_predictions)

    logger.info(f"Completed batch prediction for {len(X)} samples")
    return np.array(all_predictions)
