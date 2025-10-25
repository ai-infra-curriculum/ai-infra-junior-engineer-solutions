#!/usr/bin/env python3
"""
Comprehensive test suite for ML Pipeline.

Demonstrates pytest best practices:
- Fixtures for test setup
- Parameterized tests
- Mocking
- Test organization with markers
- Integration tests
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add solutions directory to path
solutions_dir = Path(__file__).parent.parent / "solutions"
sys.path.insert(0, str(solutions_dir))

from ml_pipeline import (
    DataLoader,
    Preprocessor,
    FeatureEngineer,
    ModelTrainer,
    ModelEvaluator,
    PredictionService,
    MLPipeline,
    DataLoadError,
    ValidationError,
    ModelError,
    DataStats,
    ModelMetrics
)


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [2.0, 4.0, 6.0, 8.0, 10.0],
        'target': [0, 0, 1, 1, 1]
    })


@pytest.fixture
def sample_dataframe_with_missing():
    """Create DataFrame with missing values."""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature2': [2.0, np.nan, 6.0, 8.0, 10.0],
        'target': [0, 0, 1, 1, 1]
    })


@pytest.fixture
def sample_array():
    """Create sample numpy array."""
    return np.array([[1, 2], [3, 4], [5, 6]])


@pytest.fixture
def sample_labels():
    """Create sample labels."""
    return np.array([0, 1, 1])


@pytest.fixture
def data_loader():
    """Create DataLoader instance."""
    return DataLoader()


@pytest.fixture
def preprocessor():
    """Create Preprocessor instance."""
    return Preprocessor()


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer()


@pytest.fixture
def model_trainer():
    """Create ModelTrainer instance."""
    return ModelTrainer()


@pytest.fixture
def trained_model(model_trainer, sample_array, sample_labels):
    """Create trained model."""
    model_trainer.train(sample_array, sample_labels)
    return model_trainer


@pytest.fixture
def model_evaluator():
    """Create ModelEvaluator instance."""
    return ModelEvaluator()


@pytest.fixture
def prediction_service(trained_model):
    """Create PredictionService with trained model."""
    return PredictionService(trained_model)


# ===========================
# DataLoader Tests
# ===========================

class TestDataLoader:
    """Test DataLoader class."""

    def test_initialization(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader.validate is True

    def test_load_from_dict(self, data_loader):
        """Test loading data from dictionary."""
        data = {
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        }

        df = data_loader.load_from_dict(data)

        assert len(df) == 3
        assert list(df.columns) == ['col1', 'col2']

    def test_load_from_empty_dict(self, data_loader):
        """Test loading from empty dictionary raises error."""
        with pytest.raises(ValidationError, match="empty"):
            data_loader.load_from_dict({})

    def test_load_csv_file_not_found(self, data_loader):
        """Test loading nonexistent CSV raises error."""
        with pytest.raises(DataLoadError, match="not found"):
            data_loader.load_csv("nonexistent.csv")

    def test_get_stats(self, data_loader, sample_dataframe):
        """Test getting dataset statistics."""
        stats = data_loader.get_stats(sample_dataframe)

        assert isinstance(stats, DataStats)
        assert stats.n_samples == 5
        assert stats.n_features == 3
        assert stats.missing_values == 0

    def test_get_stats_with_missing(self, data_loader, sample_dataframe_with_missing):
        """Test statistics with missing values."""
        stats = data_loader.get_stats(sample_dataframe_with_missing)

        assert stats.missing_values == 2

    @pytest.mark.parametrize("validate", [True, False])
    def test_validation_flag(self, validate):
        """Test validation flag behavior."""
        loader = DataLoader(validate=validate)
        assert loader.validate == validate


# ===========================
# Preprocessor Tests
# ===========================

class TestPreprocessor:
    """Test Preprocessor class."""

    def test_fit(self, preprocessor, sample_dataframe):
        """Test fitting preprocessor."""
        result = preprocessor.fit(sample_dataframe)

        assert result is preprocessor  # Check chaining
        assert 'feature1' in preprocessor.feature_means
        assert 'feature2' in preprocessor.feature_means

    def test_transform_drop_missing(self, sample_dataframe_with_missing):
        """Test transform with drop strategy."""
        preprocessor = Preprocessor(handle_missing='drop')
        preprocessor.fit(sample_dataframe_with_missing)

        transformed = preprocessor.transform(sample_dataframe_with_missing)

        assert len(transformed) == 3  # 2 rows with missing dropped

    def test_transform_mean_imputation(self, sample_dataframe_with_missing):
        """Test transform with mean imputation."""
        preprocessor = Preprocessor(handle_missing='mean')
        preprocessor.fit(sample_dataframe_with_missing)

        transformed = preprocessor.transform(sample_dataframe_with_missing)

        assert len(transformed) == 5  # No rows dropped
        assert not transformed.isnull().any().any()  # No missing values

    def test_transform_median_imputation(self, sample_dataframe_with_missing):
        """Test transform with median imputation."""
        preprocessor = Preprocessor(handle_missing='median')
        preprocessor.fit(sample_dataframe_with_missing)

        transformed = preprocessor.transform(sample_dataframe_with_missing)

        assert not transformed.isnull().any().any()

    def test_fit_transform(self, preprocessor, sample_dataframe):
        """Test fit_transform combines fit and transform."""
        transformed = preprocessor.fit_transform(sample_dataframe)

        assert isinstance(transformed, pd.DataFrame)
        assert 'feature1' in preprocessor.feature_means

    def test_remove_duplicates(self, preprocessor):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': [4, 4, 5, 6]
        })

        preprocessor.fit(df)
        transformed = preprocessor.transform(df)

        assert len(transformed) == 3  # One duplicate removed


# ===========================
# FeatureEngineer Tests
# ===========================

class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    def test_create_polynomial_features(self, feature_engineer, sample_dataframe):
        """Test creating polynomial features."""
        result = feature_engineer.create_polynomial_features(
            sample_dataframe,
            columns=['feature1'],
            degree=2
        )

        assert 'feature1^2' in result.columns
        assert result['feature1^2'].iloc[0] == 1.0  # 1^2 = 1
        assert result['feature1^2'].iloc[1] == 4.0  # 2^2 = 4

    def test_create_polynomial_degree_3(self, feature_engineer, sample_dataframe):
        """Test polynomial features with degree 3."""
        result = feature_engineer.create_polynomial_features(
            sample_dataframe,
            columns=['feature1'],
            degree=3
        )

        assert 'feature1^2' in result.columns
        assert 'feature1^3' in result.columns
        assert result['feature1^3'].iloc[1] == 8.0  # 2^3 = 8

    def test_create_interaction_features(self, feature_engineer, sample_dataframe):
        """Test creating interaction features."""
        result = feature_engineer.create_interaction_features(
            sample_dataframe,
            'feature1',
            'feature2'
        )

        assert 'feature1_feature2_interaction' in result.columns
        assert result['feature1_feature2_interaction'].iloc[0] == 2.0  # 1 * 2 = 2
        assert result['feature1_feature2_interaction'].iloc[1] == 8.0  # 2 * 4 = 8

    def test_normalize_features(self, feature_engineer, sample_dataframe):
        """Test feature normalization."""
        result = feature_engineer.normalize_features(
            sample_dataframe,
            columns=['feature1']
        )

        # Check normalized to [0, 1]
        assert result['feature1'].min() == 0.0
        assert result['feature1'].max() == 1.0

    def test_normalize_all_numeric(self, feature_engineer, sample_dataframe):
        """Test normalizing all numeric columns."""
        result = feature_engineer.normalize_features(sample_dataframe)

        # All numeric columns should be normalized
        assert result['feature1'].min() == 0.0
        assert result['feature2'].min() == 0.0


# ===========================
# ModelTrainer Tests
# ===========================

class TestModelTrainer:
    """Test ModelTrainer class."""

    def test_initialization(self, model_trainer):
        """Test ModelTrainer initialization."""
        assert model_trainer.model_type == "simple"
        assert model_trainer.model is None
        assert not model_trainer.is_fitted

    def test_train(self, model_trainer, sample_array, sample_labels):
        """Test model training."""
        model_trainer.train(sample_array, sample_labels)

        assert model_trainer.is_fitted
        assert model_trainer.model is not None
        assert 'weights' in model_trainer.model

    def test_train_empty_data(self, model_trainer):
        """Test training with empty data raises error."""
        with pytest.raises(ModelError, match="empty"):
            model_trainer.train(np.array([]), np.array([]))

    def test_train_mismatched_lengths(self, model_trainer, sample_array):
        """Test training with mismatched X and y lengths."""
        with pytest.raises(ModelError, match="same length"):
            model_trainer.train(sample_array, np.array([0, 1]))  # Wrong length

    def test_predict(self, trained_model, sample_array):
        """Test making predictions."""
        predictions = trained_model.predict(sample_array)

        assert len(predictions) == len(sample_array)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_before_training(self, model_trainer, sample_array):
        """Test prediction before training raises error."""
        with pytest.raises(ModelError, match="must be trained"):
            model_trainer.predict(sample_array)

    def test_predict_empty_array(self, trained_model):
        """Test prediction with empty array."""
        predictions = trained_model.predict(np.array([]).reshape(0, 2))
        assert len(predictions) == 0

    def test_save_model(self, trained_model, tmp_path):
        """Test saving model."""
        model_path = tmp_path / "model.pkl"
        trained_model.save(str(model_path))

        assert model_path.exists()

    def test_save_untrained_model(self, model_trainer, tmp_path):
        """Test saving untrained model raises error."""
        with pytest.raises(ModelError, match="untrained"):
            model_trainer.save(str(tmp_path / "model.pkl"))

    def test_load_model(self, trained_model, tmp_path):
        """Test loading model."""
        # Save model
        model_path = tmp_path / "model.pkl"
        trained_model.save(str(model_path))

        # Load into new trainer
        new_trainer = ModelTrainer()
        new_trainer.load(str(model_path))

        assert new_trainer.is_fitted
        assert new_trainer.model is not None

    def test_load_nonexistent_model(self, model_trainer):
        """Test loading nonexistent model raises error."""
        with pytest.raises(ModelError, match="not found"):
            model_trainer.load("nonexistent.pkl")


# ===========================
# ModelEvaluator Tests
# ===========================

class TestModelEvaluator:
    """Test ModelEvaluator class."""

    @pytest.mark.parametrize("y_true,y_pred,expected", [
        ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),  # Perfect
        ([1, 0, 1, 0], [0, 1, 0, 1], 0.0),  # All wrong
        ([1, 1, 0, 0], [1, 1, 0, 0], 1.0),  # Perfect
        ([1, 1, 1, 0], [1, 1, 0, 0], 0.75),  # 3/4 correct
    ])
    def test_calculate_accuracy(self, model_evaluator, y_true, y_pred, expected):
        """Test accuracy calculation with various inputs."""
        accuracy = model_evaluator.calculate_accuracy(
            np.array(y_true),
            np.array(y_pred)
        )
        assert accuracy == pytest.approx(expected)

    def test_calculate_accuracy_empty(self, model_evaluator):
        """Test accuracy with empty arrays."""
        accuracy = model_evaluator.calculate_accuracy(
            np.array([]),
            np.array([])
        )
        assert accuracy == 0.0

    def test_calculate_accuracy_mismatched(self, model_evaluator):
        """Test accuracy with mismatched lengths."""
        with pytest.raises(ValidationError, match="same length"):
            model_evaluator.calculate_accuracy(
                np.array([1, 0]),
                np.array([1])
            )

    def test_calculate_precision(self, model_evaluator):
        """Test precision calculation."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])

        precision = model_evaluator.calculate_precision(y_true, y_pred)

        assert precision == 1.0  # 1 TP, 0 FP

    def test_calculate_precision_no_predictions(self, model_evaluator):
        """Test precision when no positive predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        precision = model_evaluator.calculate_precision(y_true, y_pred)

        assert precision == 0.0

    def test_calculate_recall(self, model_evaluator):
        """Test recall calculation."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])

        recall = model_evaluator.calculate_recall(y_true, y_pred)

        assert recall == 0.5  # 1 TP out of 2 positives

    def test_calculate_recall_no_positives(self, model_evaluator):
        """Test recall when no positive labels."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 0, 0, 0])

        recall = model_evaluator.calculate_recall(y_true, y_pred)

        assert recall == 0.0

    @pytest.mark.parametrize("precision,recall,expected", [
        (1.0, 1.0, 1.0),
        (0.8, 0.9, 0.847),
        (0.5, 0.5, 0.5),
        (0.0, 0.0, 0.0),
    ])
    def test_calculate_f1_score(self, model_evaluator, precision, recall, expected):
        """Test F1 score calculation."""
        f1 = model_evaluator.calculate_f1_score(precision, recall)
        assert f1 == pytest.approx(expected, rel=0.01)

    def test_evaluate(self, model_evaluator):
        """Test complete evaluation."""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 0])

        metrics = model_evaluator.evaluate(y_true, y_pred)

        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1


# ===========================
# PredictionService Tests
# ===========================

class TestPredictionService:
    """Test PredictionService class."""

    def test_initialization(self, prediction_service):
        """Test PredictionService initialization."""
        assert prediction_service.model is not None
        assert prediction_service.prediction_count == 0

    def test_predict_single(self, prediction_service):
        """Test single prediction."""
        features = [1.0, 2.0]

        prediction = prediction_service.predict_single(features)

        assert prediction in [0, 1]
        assert prediction_service.prediction_count == 1

    def test_predict_batch(self, prediction_service):
        """Test batch prediction."""
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        predictions = prediction_service.predict_batch(features)

        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)
        assert prediction_service.prediction_count == 3

    def test_get_prediction_count(self, prediction_service):
        """Test getting prediction count."""
        prediction_service.predict_single([1.0, 2.0])
        prediction_service.predict_single([3.0, 4.0])

        assert prediction_service.get_prediction_count() == 2


# ===========================
# Integration Tests
# ===========================

@pytest.mark.integration
class TestMLPipeline:
    """Integration tests for complete pipeline."""

    def test_complete_pipeline(self, sample_dataframe):
        """Test training and using complete pipeline."""
        pipeline = MLPipeline()

        # Train pipeline
        metrics = pipeline.train_pipeline(sample_dataframe, 'target')

        # Check metrics
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1

        # Make predictions
        test_data = sample_dataframe.drop(columns=['target'])
        predictions = pipeline.predict(test_data)

        assert len(predictions) == len(test_data)

    def test_pipeline_predict_before_train(self, sample_dataframe):
        """Test prediction before training raises error."""
        pipeline = MLPipeline()

        test_data = sample_dataframe.drop(columns=['target'])

        with pytest.raises(ModelError, match="must be trained"):
            pipeline.predict(test_data)


# ===========================
# Mocking Tests
# ===========================

class TestMocking:
    """Demonstrate mocking techniques."""

    def test_mock_model_prediction(self, mocker):
        """Test with mocked model."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, 1])

        service = PredictionService(mock_model)
        predictions = service.predict_batch([[1, 2], [3, 4], [5, 6]])

        assert len(predictions) == 3
        mock_model.predict.assert_called_once()

    @patch('ml_pipeline.pickle.dump')
    def test_mock_save(self, mock_dump, trained_model, tmp_path):
        """Test save with mocked pickle."""
        model_path = tmp_path / "model.pkl"

        trained_model.save(str(model_path))

        mock_dump.assert_called_once()

    def test_mock_logger(self, mocker, model_trainer, sample_array, sample_labels):
        """Test with mocked logger."""
        mock_logger = mocker.patch('ml_pipeline.logger')

        model_trainer.train(sample_array, sample_labels)

        mock_logger.info.assert_called()


# ===========================
# Parametrized Test Examples
# ===========================

@pytest.mark.parametrize("handle_missing", ["drop", "mean", "median"])
def test_preprocessing_strategies(handle_missing, sample_dataframe_with_missing):
    """Test all preprocessing strategies."""
    preprocessor = Preprocessor(handle_missing=handle_missing)
    preprocessor.fit(sample_dataframe_with_missing)
    result = preprocessor.transform(sample_dataframe_with_missing)

    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_polynomial_degrees(degree, feature_engineer, sample_dataframe):
    """Test polynomial features with different degrees."""
    result = feature_engineer.create_polynomial_features(
        sample_dataframe,
        columns=['feature1'],
        degree=degree
    )

    for d in range(2, degree + 1):
        assert f'feature1^{d}' in result.columns


# ===========================
# Performance Tests
# ===========================

@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing large dataset (marked as slow)."""
    # Create large dataset
    large_df = pd.DataFrame({
        'feature1': np.random.randn(10000),
        'feature2': np.random.randn(10000),
        'target': np.random.randint(0, 2, 10000)
    })

    preprocessor = Preprocessor()
    result = preprocessor.fit_transform(large_df)

    assert len(result) <= len(large_df)


# ===========================
# Run Tests
# ===========================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=ml_pipeline", "--cov-report=term-missing"])
