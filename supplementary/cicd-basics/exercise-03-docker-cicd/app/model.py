"""
ML Model Loading and Inference

Handles model loading and prediction logic.
"""

import logging
import os
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MLModel:
    """ML Model wrapper for inference."""

    def __init__(self, model_path: str = None):
        """
        Initialize model.

        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path
        self.model = None
        self._loaded = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning("Model path not found, using dummy model")
            self._init_dummy_model()

    def load_model(self, path: str) -> None:
        """Load model from file."""
        try:
            import joblib
            self.model = joblib.load(path)
            self._loaded = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._init_dummy_model()

    def _init_dummy_model(self) -> None:
        """Initialize a dummy model for testing."""
        from sklearn.ensemble import RandomForestClassifier

        # Create and train a simple dummy model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Dummy training data
        X_dummy = np.random.rand(100, 4)
        y_dummy = np.random.randint(0, 2, 100)

        self.model.fit(X_dummy, y_dummy)
        self._loaded = True
        logger.info("Dummy model initialized")

    def predict(self, features: List[float]) -> Tuple[int, np.ndarray]:
        """
        Make prediction for single sample.

        Args:
            features: Feature vector

        Returns:
            Tuple of (prediction, probabilities)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Convert to numpy array
        X = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        return prediction, probabilities

    def predict_batch(self, features_list: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make batch predictions.

        Args:
            features_list: List of feature vectors

        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Convert to numpy array
        X = np.array(features_list)

        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return predictions, probabilities

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_info(self) -> dict:
        """Get model information."""
        if not self._loaded:
            return {"loaded": False}

        info = {
            "loaded": True,
            "model_type": type(self.model).__name__,
            "model_path": self.model_path,
        }

        # Add model-specific info
        if hasattr(self.model, "n_features_in_"):
            info["n_features"] = int(self.model.n_features_in_)

        if hasattr(self.model, "n_classes_"):
            info["n_classes"] = int(self.model.n_classes_)

        if hasattr(self.model, "feature_importances_"):
            info["has_feature_importance"] = True

        return info
