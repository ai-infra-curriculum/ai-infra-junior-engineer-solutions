"""
ML Model Example
A simple machine learning model for demonstration purposes
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModel:
    """
    A simple ML model wrapper for classification tasks.

    This class demonstrates basic ML operations including
    training, prediction, and evaluation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize the ML model.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self._is_trained = False

        logger.info(
            f"Initialized MLModel with n_estimators={n_estimators}, "
            f"max_depth={max_depth}"
        )

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> "MLModel":
        """
        Train the model on the provided data.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            self: The trained model instance

        Raises:
            ValueError: If input arrays are empty or incompatible
        """
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train ({X_train.shape[0]}) and y_train "
                f"({y_train.shape[0]}) must have the same number of samples"
            )

        logger.info(f"Training model on {X_train.shape[0]} samples...")

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        self.model.fit(X_train, y_train)
        self._is_trained = True

        logger.info("Training complete!")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features to predict

        Returns:
            np.ndarray: Predicted labels

        Raises:
            RuntimeError: If model is not trained
            ValueError: If input shape is incorrect
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError(
                "Model must be trained before making predictions. "
                "Call train() first."
            )

        if X.shape[0] == 0:
            raise ValueError("Input data cannot be empty")

        logger.info(f"Making predictions on {X.shape[0]} samples...")
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict

        Returns:
            np.ndarray: Class probabilities

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError(
                "Model must be trained before making predictions. "
                "Call train() first."
            )

        logger.info(
            f"Predicting probabilities for {X.shape[0]} samples..."
        )
        probabilities = self.model.predict_proba(X)

        return probabilities

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, str]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            tuple: (accuracy_score, classification_report)

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError(
                "Model must be trained before evaluation. Call train() first."
            )

        logger.info(f"Evaluating model on {X_test.shape[0]} samples...")

        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        logger.info(f"Accuracy: {accuracy:.4f}")

        return accuracy, report

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importances from the trained model.

        Returns:
            np.ndarray or None: Feature importances if model is trained

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError(
                "Model must be trained to get feature importances. "
                "Call train() first."
            )

        return self.model.feature_importances_

    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained


def load_sample_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load sample dataset for demonstration.

    Returns:
        tuple: (features, labels, feature_names)
    """
    logger.info("Loading sample dataset...")
    data = load_iris()
    return data.data, data.target, data.feature_names


def create_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Args:
        X: Features
        y: Labels
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}...")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def main():
    """Main function to demonstrate model usage."""
    logger.info("=== ML Model Demo ===")

    # Load data
    X, y, feature_names = load_sample_data()
    logger.info(
        f"Loaded dataset with {X.shape[0]} samples and "
        f"{X.shape[1]} features"
    )

    # Split data
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    logger.info(
        f"Training set: {X_train.shape[0]} samples, "
        f"Test set: {X_test.shape[0]} samples"
    )

    # Create and train model
    model = MLModel(n_estimators=100, max_depth=5)
    model.train(X_train, y_train)

    # Evaluate model
    accuracy, report = model.evaluate(X_test, y_test)
    logger.info(f"\nModel Performance:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    # Feature importance
    importances = model.get_feature_importance()
    logger.info("\nFeature Importances:")
    for name, importance in zip(feature_names, importances):
        logger.info(f"  {name}: {importance:.4f}")

    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
