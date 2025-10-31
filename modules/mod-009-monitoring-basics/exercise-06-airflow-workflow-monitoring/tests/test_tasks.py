"""
Task Tests

Tests for individual task functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import download_data, validate_data, preprocess_data
from src.model_training import feature_engineering, train_model, evaluate_model
from src.model_deployment import deploy_model


class MockTaskInstance:
    """Mock Airflow TaskInstance for testing."""

    def __init__(self):
        self.xcom_data = {}

    def xcom_pull(self, task_ids=None, key='return_value'):
        """Mock XCom pull."""
        if task_ids:
            return self.xcom_data.get(task_ids)
        return None

    def xcom_push(self, key, value):
        """Mock XCom push."""
        self.xcom_data[key] = value


class TestDataProcessingTasks:
    """Test data processing task functions."""

    def test_download_data(self):
        """Test download_data task."""
        # Create mock context
        context = {
            'ds': '2024-01-15',
            'ti': MockTaskInstance()
        }

        # Run task
        result = download_data(**context)

        # Validate result
        assert result is not None
        assert 'dataset_size_gb' in result
        assert 'num_samples' in result
        assert 'download_time_seconds' in result

        # Validate data types
        assert isinstance(result['dataset_size_gb'], (int, float))
        assert isinstance(result['num_samples'], int)
        assert result['num_samples'] > 0

    def test_validate_data_success(self):
        """Test validate_data task with good data."""
        # Create mock TI with download data
        ti = MockTaskInstance()
        ti.xcom_data['download_data'] = {
            'dataset_size_gb': 1.5,
            'num_samples': 10000
        }

        context = {
            'ds': '2024-01-15',
            'ti': ti
        }

        # Run task (may pass or fail randomly, so we catch potential errors)
        try:
            result = validate_data(**context)

            # If it passes, validate result
            assert result is not None
            assert 'validation_passed' in result
            assert 'null_percentage' in result
            assert 'duplicate_percentage' in result
        except ValueError as e:
            # Task can fail due to simulated data quality issues
            assert 'null' in str(e).lower() or 'duplicate' in str(e).lower()

    def test_validate_data_missing_input(self):
        """Test validate_data fails without input data."""
        context = {
            'ds': '2024-01-15',
            'ti': MockTaskInstance()
        }

        # Should raise error due to missing input
        with pytest.raises(ValueError, match="No dataset metadata"):
            validate_data(**context)

    def test_preprocess_data(self):
        """Test preprocess_data task."""
        # Create mock TI with validation data
        ti = MockTaskInstance()
        ti.xcom_data['validate_data'] = {
            'valid_samples': 9500,
            'validation_passed': True
        }

        context = {
            'ds': '2024-01-15',
            'ti': ti
        }

        # Run task
        result = preprocess_data(**context)

        # Validate result
        assert result is not None
        assert 'train_samples' in result
        assert 'validation_samples' in result
        assert 'num_features' in result

        # Validate split
        total = result['train_samples'] + result['validation_samples']
        assert total == result['total_samples']


class TestModelTrainingTasks:
    """Test model training task functions."""

    def test_feature_engineering(self):
        """Test feature_engineering task."""
        # Create mock TI
        ti = MockTaskInstance()
        ti.xcom_data['preprocess_data'] = {
            'num_features': 50,
            'train_samples': 8000
        }

        context = {
            'ds': '2024-01-15',
            'ti': ti
        }

        # Run task
        result = feature_engineering(**context)

        # Validate result
        assert result is not None
        assert 'total_features' in result
        assert 'base_features' in result

        # Total features should be more than base
        assert result['total_features'] > result['base_features']

    def test_train_model(self):
        """Test train_model task."""
        # Create mock TI
        ti = MockTaskInstance()
        ti.xcom_data['feature_engineering'] = {
            'total_features': 120
        }
        ti.xcom_data['preprocess_data'] = {
            'train_samples': 8000,
            'processed_samples': 10000
        }

        context = {
            'ds': '2024-01-15',
            'ti': ti
        }

        # Run task
        result = train_model(**context)

        # Validate result
        assert result is not None
        assert 'train_accuracy' in result
        assert 'validation_accuracy' in result
        assert 'model_size_mb' in result
        assert 'model_path' in result

        # Validate accuracy range
        assert 0 <= result['train_accuracy'] <= 1
        assert 0 <= result['validation_accuracy'] <= 1

    def test_evaluate_model_success(self):
        """Test evaluate_model task with good model."""
        # Create mock TI
        ti = MockTaskInstance()
        ti.xcom_data['train_model'] = {
            'validation_accuracy': 0.92,
            'model_size_mb': 450
        }
        ti.xcom_data['preprocess_data'] = {
            'validation_samples': 2000
        }

        context = {
            'ds': '2024-01-15',
            'ti': ti
        }

        # Run task (may pass or fail due to randomness)
        try:
            result = evaluate_model(**context)

            # If passes, validate result
            assert result is not None
            assert 'accuracy' in result
            assert 'precision' in result
            assert 'recall' in result
            assert 'f1_score' in result
            assert 'quality_gates_passed' in result

            # All metrics should be between 0 and 1
            assert 0 <= result['accuracy'] <= 1
            assert 0 <= result['precision'] <= 1
            assert 0 <= result['recall'] <= 1
            assert 0 <= result['f1_score'] <= 1

        except ValueError as e:
            # Can fail if random accuracy is too low
            assert 'accuracy' in str(e).lower() or 'f1' in str(e).lower()


class TestDeploymentTasks:
    """Test deployment task functions."""

    def test_deploy_model(self):
        """Test deploy_model task."""
        # Create mock TI
        ti = MockTaskInstance()
        ti.xcom_data['train_model'] = {
            'model_path': '/models/model_2024-01-15.pkl',
            'model_size_mb': 450
        }
        ti.xcom_data['evaluate_model'] = {
            'accuracy': 0.92,
            'f1_score': 0.89,
            'quality_gates_passed': True
        }

        context = {
            'ds': '2024-01-15',
            'ti': ti
        }

        # Run task (may fail due to random canary metrics)
        try:
            result = deploy_model(**context)

            # Validate result
            assert result is not None
            assert 'model_version' in result
            assert 'deployment_url' in result
            assert 'deployment_status' in result
            assert 'health_checks_passed' in result

            # Check deployment succeeded
            assert result['deployment_status'] == 'SUCCESS'
            assert result['health_checks_passed'] is True

        except ValueError as e:
            # Can fail if canary error rate is too high
            assert 'error rate' in str(e).lower()


class TestTaskDataFlow:
    """Test data flow between tasks."""

    def test_full_pipeline_data_flow(self):
        """Test data flows correctly through entire pipeline."""
        # Create mock TI
        ti = MockTaskInstance()

        # 1. Download
        context1 = {'ds': '2024-01-15', 'ti': ti}
        download_result = download_data(**context1)
        ti.xcom_data['download_data'] = download_result

        # 2. Validate (may fail, so wrap in try)
        try:
            validate_result = validate_data(**context1)
            ti.xcom_data['validate_data'] = validate_result

            # 3. Preprocess
            preprocess_result = preprocess_data(**context1)
            ti.xcom_data['preprocess_data'] = preprocess_result

            # 4. Feature engineering
            feature_result = feature_engineering(**context1)
            ti.xcom_data['feature_engineering'] = feature_result

            # Validate data propagation
            assert download_result['num_samples'] > 0

            if validate_result:
                assert validate_result['total_samples'] == download_result['num_samples']

            assert preprocess_result['total_samples'] > 0
            assert feature_result['base_features'] > 0

        except ValueError:
            # Data quality check can fail randomly
            pass


class TestErrorHandling:
    """Test error handling in tasks."""

    def test_missing_xcom_data(self):
        """Test tasks fail gracefully with missing XCom data."""
        context = {
            'ds': '2024-01-15',
            'ti': MockTaskInstance()
        }

        # These should raise errors due to missing input
        with pytest.raises(ValueError):
            validate_data(**context)

        with pytest.raises(ValueError):
            preprocess_data(**context)

        with pytest.raises(ValueError):
            feature_engineering(**context)

    def test_task_with_bad_data(self):
        """Test tasks handle bad data appropriately."""
        # Test with extremely low sample count
        ti = MockTaskInstance()
        ti.xcom_data['download_data'] = {
            'dataset_size_gb': 0.1,
            'num_samples': 100  # Very low
        }

        context = {
            'ds': '2024-01-15',
            'ti': ti
        }

        # Validation might fail or pass depending on random checks
        # Just ensure it doesn't crash
        try:
            result = validate_data(**context)
            assert result is not None
        except ValueError as e:
            # Expected for bad data
            assert len(str(e)) > 0


class TestTaskOutput:
    """Test task output formats."""

    def test_download_output_format(self):
        """Test download_data returns correct format."""
        context = {'ds': '2024-01-15', 'ti': MockTaskInstance()}
        result = download_data(**context)

        # Check all required keys
        required_keys = [
            'dataset_size_gb',
            'num_samples',
            'download_time_seconds',
            'source',
            'timestamp'
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_training_output_format(self):
        """Test train_model returns correct format."""
        ti = MockTaskInstance()
        ti.xcom_data['feature_engineering'] = {'total_features': 120}
        ti.xcom_data['preprocess_data'] = {
            'train_samples': 8000,
            'processed_samples': 10000
        }

        context = {'ds': '2024-01-15', 'ti': ti}
        result = train_model(**context)

        # Check all required keys
        required_keys = [
            'model_type',
            'train_accuracy',
            'validation_accuracy',
            'training_time_minutes',
            'model_size_mb',
            'model_path'
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
