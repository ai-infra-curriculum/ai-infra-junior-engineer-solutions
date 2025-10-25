"""Model training module with MLflow integration."""

import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train ML models with hyperparameter tuning and tracking."""

    def __init__(
        self,
        experiment_name: str = "ml-pipeline",
        model_type: str = "random_forest"
    ):
        """Initialize model trainer.

        Args:
            experiment_name: MLflow experiment name
            model_type: Type of model ("random_forest", "gradient_boosting", "logistic")
        """
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.model = None
        mlflow.set_experiment(experiment_name)
        logger.info(f"ModelTrainer initialized with {model_type}")

    def _get_model(self, **params) -> Any:
        """Get model instance based on model_type.

        Args:
            **params: Model hyperparameters

        Returns:
            Model instance
        """
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 10),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", 3),
                random_state=42
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_param_grid(self) -> Dict[str, list]:
        """Get parameter grid for hyperparameter tuning.

        Returns:
            Dictionary of parameters to search
        """
        if self.model_type == "random_forest":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10]
            }
        elif self.model_type == "gradient_boosting":
            return {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        elif self.model_type == "logistic":
            return {
                "C": [0.1, 1.0, 10.0],
                "max_iter": [500, 1000, 2000]
            }
        else:
            return {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        tune_hyperparameters: bool = False,
        cv_folds: int = 5
    ) -> Tuple[Any, Dict[str, float]]:
        """Train model with optional hyperparameter tuning.

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            tune_hyperparameters: Whether to tune hyperparameters
            cv_folds: Number of cross-validation folds

        Returns:
            Tuple of (trained model, metrics dict)
        """
        logger.info(f"Training {self.model_type} model")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        with mlflow.start_run(run_name=f"{self.model_type}-training"):
            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            # Train model
            if tune_hyperparameters:
                logger.info("Performing hyperparameter tuning")
                base_model = self._get_model()
                param_grid = self._get_param_grid()

                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_

                # Log best parameters
                for param, value in grid_search.best_params_.items():
                    mlflow.log_param(f"best_{param}", value)

                mlflow.log_metric("best_cv_score", grid_search.best_score_)
            else:
                self.model = self._get_model()
                self.model.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = self.model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"{metric_name}: {metric_value:.4f}")

            # Log feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                # Log top 10 features
                mlflow.log_dict(
                    feature_importance.head(10).to_dict(),
                    "top_features.json"
                )

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            logger.info(f"Model training complete. Accuracy: {metrics['accuracy']:.4f}")

        return self.model, metrics

    def save_model(self, path: str):
        """Save trained model to file.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model trained yet")

        mlflow.sklearn.save_model(self.model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from file.

        Args:
            path: Path to load model from
        """
        self.model = mlflow.sklearn.load_model(path)
        logger.info(f"Model loaded from {path}")
