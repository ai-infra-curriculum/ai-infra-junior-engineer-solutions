"""Model Training Module"""

import logging
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Any:
    """Train ML model."""
    if hyperparameters is None:
        hyperparameters = {}

    models = {
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = models[model_type]
    model = model_class(random_state=random_state, **hyperparameters)

    logger.info(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    logger.info("Training complete")

    return model


def save_model(model: Any, filepath: str) -> None:
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load trained model from disk."""
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Perform cross-validation.

    Returns:
        Dictionary with mean and std of scores
    """
    from sklearn.model_selection import cross_val_score

    model = train_model(
        X, y,
        model_type=model_type,
        random_state=random_state
    )

    scores = cross_val_score(model, X, y, cv=cv_folds)

    results = {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "scores": scores.tolist(),
    }

    logger.info(f"CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")

    return results
