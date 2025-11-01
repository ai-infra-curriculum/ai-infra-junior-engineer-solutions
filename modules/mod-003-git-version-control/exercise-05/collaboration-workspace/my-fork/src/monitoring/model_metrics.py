"""
Model Performance Metrics

Track model-specific performance indicators.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time


@dataclass
class ModelMetrics:
    """Track model performance metrics."""

    def __init__(self):
        self.predictions: List[float] = []
        self.ground_truth: List[float] = []
        self.latencies: List[float] = []
        self.start_time = time.time()

    def add_prediction(
        self,
        prediction: float,
        truth: Optional[float] = None,
        latency: Optional[float] = None
    ):
        """
        Record a prediction.

        Args:
            prediction: Model prediction value
            truth: Ground truth value (if available)
            latency: Prediction latency in seconds
        """
        self.predictions.append(prediction)

        if truth is not None:
            self.ground_truth.append(truth)

        if latency is not None:
            self.latencies.append(latency)

    def calculate_accuracy(self, threshold: float = 0.5) -> float:
        """
        Calculate prediction accuracy.

        Args:
            threshold: Classification threshold

        Returns:
            Accuracy percentage (0-100)
        """
        if not self.predictions or not self.ground_truth:
            return 0.0

        if len(self.predictions) != len(self.ground_truth):
            raise ValueError(
                "Predictions and ground truth lengths don't match"
            )

        correct = sum(
            1 for pred, truth in zip(self.predictions, self.ground_truth)
            if (pred >= threshold) == (truth >= threshold)
        )

        return (correct / len(self.predictions)) * 100

    def get_average_latency(self) -> float:
        """Get average prediction latency."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        return {
            "total_predictions": len(self.predictions),
            "accuracy": (
                self.calculate_accuracy()
                if self.ground_truth else None
            ),
            "average_latency": self.get_average_latency(),
            "uptime_seconds": time.time() - self.start_time
        }


# Global metrics instance
model_metrics = ModelMetrics()
