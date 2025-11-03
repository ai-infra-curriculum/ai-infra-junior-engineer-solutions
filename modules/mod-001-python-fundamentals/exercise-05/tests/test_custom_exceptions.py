"""
Tests for Custom Exceptions

Demonstrates proper testing of custom exception classes.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.custom_exceptions import (
    MLException,
    ModelNotFoundError,
    InvalidDataError,
    GPUOutOfMemoryError,
    TrainingFailedError,
    ConfigurationError,
    DataValidationError,
    CheckpointError,
    validate_data,
    validate_config,
    check_gpu_memory,
    validate_training_data
)


class TestMLException:
    """Test base MLException class"""

    def test_is_exception(self):
        """Base class should inherit from Exception"""
        assert issubclass(MLException, Exception)

    def test_can_be_raised(self):
        """Base exception should be raisable"""
        with pytest.raises(MLException):
            raise MLException("Test error")


class TestModelNotFoundError:
    """Test ModelNotFoundError"""

    def test_stores_model_path(self):
        """Exception should store model path"""
        path = "/path/to/model.h5"
        error = ModelNotFoundError(path)

        assert error.model_path == path
        assert path in str(error)

    def test_inherits_from_ml_exception(self):
        """Should inherit from MLException"""
        assert issubclass(ModelNotFoundError, MLException)

    def test_can_be_caught_as_ml_exception(self):
        """Should be catchable as MLException"""
        with pytest.raises(MLException):
            raise ModelNotFoundError("/test/path")


class TestInvalidDataError:
    """Test InvalidDataError"""

    def test_stores_message_and_data_info(self):
        """Exception should store message and data info"""
        message = "Invalid data format"
        data_info = {"size": 0, "type": "list"}

        error = InvalidDataError(message, data_info)

        assert str(error) == message or message in str(error)
        assert error.data_info == data_info

    def test_works_without_data_info(self):
        """Exception should work without data_info"""
        error = InvalidDataError("Error message")
        assert error.data_info == {}

    def test_validate_data_empty_list(self):
        """validate_data should raise for empty list"""
        with pytest.raises(InvalidDataError) as exc_info:
            validate_data([])

        assert "empty" in str(exc_info.value).lower()

    def test_validate_data_too_small(self):
        """validate_data should raise for insufficient samples"""
        with pytest.raises(InvalidDataError) as exc_info:
            validate_data([1, 2, 3], min_size=10)

        assert exc_info.value.data_info["size"] == 3
        assert exc_info.value.data_info["minimum"] == 10

    def test_validate_data_success(self):
        """validate_data should pass for valid data"""
        data = list(range(20))
        # Should not raise
        validate_data(data, min_size=10)


class TestGPUOutOfMemoryError:
    """Test GPUOutOfMemoryError"""

    def test_stores_batch_and_model_size(self):
        """Exception should store batch_size and model_size"""
        error = GPUOutOfMemoryError(
            batch_size=128,
            model_size=2000,
            available_memory=1000
        )

        assert error.batch_size == 128
        assert error.model_size == 2000
        assert error.available_memory == 1000
        assert "128" in str(error)
        assert "2000" in str(error)

    def test_check_gpu_memory_insufficient(self):
        """check_gpu_memory should raise for insufficient memory"""
        with pytest.raises(GPUOutOfMemoryError) as exc_info:
            check_gpu_memory(
                batch_size=128,
                model_size=2000,
                available_memory=1000
            )

        assert exc_info.value.batch_size == 128


class TestTrainingFailedError:
    """Test TrainingFailedError"""

    def test_stores_epoch_and_reason(self):
        """Exception should store epoch and reason"""
        error = TrainingFailedError(
            epoch=42,
            reason="Loss diverged",
            metrics={"loss": float('nan')}
        )

        assert error.epoch == 42
        assert error.reason == "Loss diverged"
        assert error.metrics["loss"] != error.metrics["loss"]  # NaN check


class TestConfigurationError:
    """Test ConfigurationError"""

    def test_stores_param_value_expected(self):
        """Exception should store param, value, and expected"""
        error = ConfigurationError(
            param="learning_rate",
            value=1.5,
            expected="0 < lr < 1"
        )

        assert error.param == "learning_rate"
        assert error.value == 1.5
        assert error.expected == "0 < lr < 1"

    def test_validate_config_missing_param(self):
        """validate_config should raise for missing parameter"""
        config = {"learning_rate": 0.001, "batch_size": 32}  # Missing epochs

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        assert exc_info.value.param == "epochs"

    def test_validate_config_invalid_learning_rate(self):
        """validate_config should raise for invalid learning_rate"""
        config = {
            "learning_rate": 1.5,  # Out of range
            "batch_size": 32,
            "epochs": 100
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        assert exc_info.value.param == "learning_rate"
        assert exc_info.value.value == 1.5

    def test_validate_config_invalid_batch_size(self):
        """validate_config should raise for invalid batch_size"""
        config = {
            "learning_rate": 0.001,
            "batch_size": -10,  # Negative
            "epochs": 100
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        assert exc_info.value.param == "batch_size"

    def test_validate_config_success(self):
        """validate_config should pass for valid config"""
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        }

        # Should not raise
        validate_config(config)


class TestDataValidationError:
    """Test DataValidationError"""

    def test_stores_validation_type_and_details(self):
        """Exception should store validation_type and details"""
        error = DataValidationError(
            validation_type="shape_mismatch",
            details="Features (100) != Labels (80)"
        )

        assert error.validation_type == "shape_mismatch"
        assert error.details == "Features (100) != Labels (80)"

    def test_validate_training_data_shape_mismatch(self):
        """validate_training_data should raise for shape mismatch"""
        features = [1, 2, 3, 4, 5]
        labels = [0, 1, 0]

        with pytest.raises(DataValidationError) as exc_info:
            validate_training_data(features, labels)

        assert exc_info.value.validation_type == "shape_mismatch"

    def test_validate_training_data_empty(self):
        """validate_training_data should raise for empty data"""
        with pytest.raises(DataValidationError) as exc_info:
            validate_training_data([], [])

        assert exc_info.value.validation_type == "empty_data"

    def test_validate_training_data_success(self):
        """validate_training_data should pass for valid data"""
        features = [1, 2, 3, 4, 5]
        labels = [0, 1, 0, 1, 0]

        # Should not raise
        validate_training_data(features, labels)


class TestCheckpointError:
    """Test CheckpointError"""

    def test_stores_path_and_reason(self):
        """Exception should store checkpoint_path and reason"""
        error = CheckpointError(
            checkpoint_path="/models/checkpoint.h5",
            reason="Disk full"
        )

        assert error.checkpoint_path == "/models/checkpoint.h5"
        assert error.reason == "Disk full"
        assert "checkpoint.h5" in str(error)
        assert "Disk full" in str(error)


class TestExceptionHierarchy:
    """Test exception hierarchy and catching"""

    def test_all_inherit_from_ml_exception(self):
        """All custom exceptions should inherit from MLException"""
        exceptions = [
            ModelNotFoundError,
            InvalidDataError,
            GPUOutOfMemoryError,
            TrainingFailedError,
            ConfigurationError,
            DataValidationError,
            CheckpointError
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, MLException)

    def test_can_catch_with_base_class(self):
        """Should be able to catch all ML exceptions with base class"""
        for exc_class in [ModelNotFoundError, InvalidDataError]:
            with pytest.raises(MLException):
                if exc_class == ModelNotFoundError:
                    raise exc_class("/test/path")
                else:
                    raise exc_class("Test error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
