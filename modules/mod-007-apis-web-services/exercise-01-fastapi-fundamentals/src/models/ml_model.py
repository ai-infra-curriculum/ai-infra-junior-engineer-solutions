"""ML Model wrapper for predictions."""

import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MLModel:
    """Wrapper for ML model inference."""

    def __init__(self, model_path: str = "models/model.pkl"):
        """
        Initialize model.

        Args:
            model_path: Path to pickled model file
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model from disk."""
        try:
            if Path(self.model_path).exists():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                # Create dummy model for demo
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(n_estimators=10, random_state=42)
                # Fit with dummy data
                X_dummy = np.random.rand(100, 5)
                y_dummy = np.random.randint(0, 2, 100)
                self.model.fit(X_dummy, y_dummy)
                logger.warning(f"Model file not found. Using dummy model.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, features: List[float]) -> Tuple[int, List[float]]:
        """
        Make prediction.

        Args:
            features: Input features

        Returns:
            Tuple of (prediction, probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        X = np.array(features).reshape(1, -1)
        prediction = int(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0].tolist()

        return prediction, probabilities

    async def predict_async(self, features: List[float]) -> Tuple[int, List[float]]:
        """Async prediction wrapper."""
        return self.predict(features)
