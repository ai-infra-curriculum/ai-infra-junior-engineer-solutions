#!/usr/bin/env python3
"""
ML Pipeline Components

Complete ML pipeline with testable components for demonstration of
pytest testing best practices.

Components:
- DataLoader: Load and validate data
- Preprocessor: Clean and normalize data
- FeatureEngineer: Create features
- ModelTrainer: Train ML models
- ModelEvaluator: Evaluate model performance
- PredictionService: Serve predictions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================
# Custom Exceptions
# ===========================

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class DataLoadError(PipelineError):
    """Error loading data."""
    pass


class ValidationError(PipelineError):
    """Data validation error."""
    pass


class ModelError(PipelineError):
    """Model operation error."""
    pass


# ===========================
# Data Classes
# ===========================

@dataclass
class DataStats:
    """Statistics about a dataset."""
    n_samples: int
    n_features: int
    missing_values: int
    duplicate_rows: int


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float


# ===========================
# Data Loader
# ===========================

class DataLoader:
    """Load and validate data from various sources."""

    def __init__(self, validate: bool = True):
        """
        Initialize data loader.

        Args:
            validate: Whether to validate loaded data
        """
        self.validate = validate

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            Loaded DataFrame

        Raises:
            DataLoadError: If loading fails
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath}")

            if self.validate:
                self._validate_dataframe(df)

            return df

        except FileNotFoundError:
            raise DataLoadError(f"File not found: {filepath}")
        except pd.errors.EmptyDataError:
            raise DataLoadError(f"Empty file: {filepath}")
        except Exception as e:
            raise DataLoadError(f"Error loading CSV: {e}")

    def load_from_dict(self, data: Dict[str, List]) -> pd.DataFrame:
        """
        Load data from dictionary.

        Args:
            data: Dictionary with column names as keys

        Returns:
            DataFrame

        Raises:
            DataLoadError: If conversion fails
        """
        try:
            df = pd.DataFrame(data)
            logger.info(f"Created DataFrame with {len(df)} rows")

            if self.validate:
                self._validate_dataframe(df)

            return df

        except Exception as e:
            raise DataLoadError(f"Error creating DataFrame: {e}")

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame.

        Args:
            df: DataFrame to validate

        Raises:
            ValidationError: If validation fails
        """
        if df.empty:
            raise ValidationError("DataFrame is empty")

        if df.shape[1] == 0:
            raise ValidationError("DataFrame has no columns")

    def get_stats(self, df: pd.DataFrame) -> DataStats:
        """
        Get dataset statistics.

        Args:
            df: DataFrame

        Returns:
            DataStats object
        """
        return DataStats(
            n_samples=len(df),
            n_features=df.shape[1],
            missing_values=df.isnull().sum().sum(),
            duplicate_rows=df.duplicated().sum()
        )


# ===========================
# Preprocessor
# ===========================

class Preprocessor:
    """Preprocess data for ML models."""

    def __init__(self, handle_missing: str = 'drop'):
        """
        Initialize preprocessor.

        Args:
            handle_missing: How to handle missing values ('drop', 'mean', 'median')
        """
        self.handle_missing = handle_missing
        self.feature_means: Dict[str, float] = {}
        self.feature_medians: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> 'Preprocessor':
        """
        Fit preprocessor on data.

        Args:
            df: Training data

        Returns:
            Self for chaining
        """
        # Calculate statistics for imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            self.feature_means[col] = df[col].mean()
            self.feature_medians[col] = df[col].median()

        logger.info(f"Fitted preprocessor on {len(df)} samples")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.

        Args:
            df: Data to transform

        Returns:
            Transformed DataFrame
        """
        df = df.copy()

        # Handle missing values
        if self.handle_missing == 'drop':
            df = df.dropna()
        elif self.handle_missing == 'mean':
            for col, mean_val in self.feature_means.items():
                if col in df.columns:
                    df[col] = df[col].fillna(mean_val)
        elif self.handle_missing == 'median':
            for col, median_val in self.feature_medians.items():
                if col in df.columns:
                    df[col] = df[col].fillna(median_val)

        # Remove duplicates
        df = df.drop_duplicates()

        logger.info(f"Transformed data: {len(df)} rows")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: Data to fit and transform

        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)


# ===========================
# Feature Engineer
# ===========================

class FeatureEngineer:
    """Engineer features for ML models."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize feature engineer.

        Args:
            feature_names: Names of features to use
        """
        self.feature_names = feature_names
        self.feature_min: Dict[str, float] = {}
        self.feature_max: Dict[str, float] = {}

    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features.

        Args:
            df: Input DataFrame
            columns: Columns to create polynomials from
            degree: Polynomial degree

        Returns:
            DataFrame with additional polynomial features
        """
        df = df.copy()

        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f"{col}^{d}"] = df[col] ** d

        logger.info(f"Created polynomial features up to degree {degree}")
        return df

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str
    ) -> pd.DataFrame:
        """
        Create interaction features between two columns.

        Args:
            df: Input DataFrame
            col1: First column
            col2: Second column

        Returns:
            DataFrame with interaction feature
        """
        df = df.copy()

        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
            logger.info(f"Created interaction: {col1} * {col2}")

        return df

    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize features to [0, 1] range.

        Args:
            df: Input DataFrame
            columns: Columns to normalize (all numeric if None)

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df.columns:
                if col not in self.feature_min:
                    self.feature_min[col] = df[col].min()
                    self.feature_max[col] = df[col].max()

                min_val = self.feature_min[col]
                max_val = self.feature_max[col]

                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0.0

        logger.info(f"Normalized {len(columns)} features")
        return df


# ===========================
# Model Trainer
# ===========================

class ModelTrainer:
    """Train ML models."""

    def __init__(self, model_type: str = "simple"):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
        self.model = None
        self.is_fitted = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10
    ) -> None:
        """
        Train model.

        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs

        Raises:
            ModelError: If training fails
        """
        if len(X) == 0:
            raise ModelError("Cannot train on empty data")

        if len(X) != len(y):
            raise ModelError("X and y must have same length")

        try:
            # Simple mock training
            self.model = {
                'weights': np.random.randn(X.shape[1]),
                'bias': 0.0
            }
            self.is_fitted = True

            logger.info(f"Trained {self.model_type} model for {epochs} epochs")

        except Exception as e:
            raise ModelError(f"Training failed: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions

        Raises:
            ModelError: If model not trained
        """
        if not self.is_fitted:
            raise ModelError("Model must be trained before prediction")

        if len(X) == 0:
            return np.array([])

        # Simple mock prediction
        predictions = X @ self.model['weights'] + self.model['bias']
        predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid

        return (predictions > 0.5).astype(int)

    def save(self, filepath: str) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save model

        Raises:
            ModelError: If save fails
        """
        if not self.is_fitted:
            raise ModelError("Cannot save untrained model")

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved model to {filepath}")

        except Exception as e:
            raise ModelError(f"Failed to save model: {e}")

    def load(self, filepath: str) -> None:
        """
        Load model from file.

        Args:
            filepath: Path to model file

        Raises:
            ModelError: If load fails
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.is_fitted = True
            logger.info(f"Loaded model from {filepath}")

        except FileNotFoundError:
            raise ModelError(f"Model file not found: {filepath}")
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}")


# ===========================
# Model Evaluator
# ===========================

class ModelEvaluator:
    """Evaluate model performance."""

    @staticmethod
    def calculate_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate accuracy.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Accuracy score

        Raises:
            ValidationError: If inputs invalid
        """
        if len(y_true) != len(y_pred):
            raise ValidationError("y_true and y_pred must have same length")

        if len(y_true) == 0:
            return 0.0

        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

    @staticmethod
    def calculate_precision(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate precision.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Precision score
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)

        if predicted_positives == 0:
            return 0.0

        return true_positives / predicted_positives

    @staticmethod
    def calculate_recall(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate recall.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Recall score
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)

        if actual_positives == 0:
            return 0.0

        return true_positives / actual_positives

    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 score.

        Args:
            precision: Precision value
            recall: Recall value

        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> ModelMetrics:
        """
        Calculate all metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            ModelMetrics object
        """
        accuracy = self.calculate_accuracy(y_true, y_pred)
        precision = self.calculate_precision(y_true, y_pred)
        recall = self.calculate_recall(y_true, y_pred)
        f1 = self.calculate_f1_score(precision, recall)

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1
        )


# ===========================
# Prediction Service
# ===========================

class PredictionService:
    """Service for making predictions."""

    def __init__(self, model: ModelTrainer):
        """
        Initialize prediction service.

        Args:
            model: Trained model
        """
        self.model = model
        self.prediction_count = 0

    def predict_single(self, features: List[float]) -> int:
        """
        Predict single sample.

        Args:
            features: Feature values

        Returns:
            Prediction

        Raises:
            ModelError: If prediction fails
        """
        try:
            X = np.array([features])
            prediction = self.model.predict(X)[0]
            self.prediction_count += 1

            logger.info(f"Prediction {self.prediction_count}: {prediction}")
            return int(prediction)

        except Exception as e:
            raise ModelError(f"Prediction failed: {e}")

    def predict_batch(self, features: List[List[float]]) -> List[int]:
        """
        Predict batch of samples.

        Args:
            features: List of feature vectors

        Returns:
            List of predictions
        """
        X = np.array(features)
        predictions = self.model.predict(X)
        self.prediction_count += len(predictions)

        logger.info(f"Batch prediction: {len(predictions)} samples")
        return predictions.tolist()

    def get_prediction_count(self) -> int:
        """Get total number of predictions made."""
        return self.prediction_count


# ===========================
# Complete Pipeline
# ===========================

class MLPipeline:
    """Complete ML pipeline."""

    def __init__(self):
        """Initialize pipeline components."""
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.prediction_service = None

    def train_pipeline(
        self,
        train_data: pd.DataFrame,
        target_column: str
    ) -> ModelMetrics:
        """
        Train complete pipeline.

        Args:
            train_data: Training data
            target_column: Name of target column

        Returns:
            Training metrics
        """
        # Preprocess
        train_data = self.preprocessor.fit_transform(train_data)

        # Separate features and target
        X = train_data.drop(columns=[target_column]).values
        y = train_data[target_column].values

        # Train model
        self.model_trainer.train(X, y)

        # Evaluate
        predictions = self.model_trainer.predict(X)
        metrics = self.model_evaluator.evaluate(y, predictions)

        # Initialize prediction service
        self.prediction_service = PredictionService(self.model_trainer)

        logger.info(f"Pipeline trained - Accuracy: {metrics.accuracy:.4f}")
        return metrics

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            data: Data to predict on

        Returns:
            Predictions
        """
        if self.prediction_service is None:
            raise ModelError("Pipeline must be trained before prediction")

        # Preprocess
        data = self.preprocessor.transform(data)

        # Predict
        X = data.values
        return self.model_trainer.predict(X)
