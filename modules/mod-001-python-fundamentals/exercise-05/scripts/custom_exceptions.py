#!/usr/bin/env python3
"""
Custom Exceptions for ML Workflows

Demonstrates creating domain-specific exception classes for ML applications.
"""

import os
from typing import Dict, Any, Optional


# Base exception for ML operations
class MLException(Exception):
    """Base exception for all ML-related errors"""
    pass


# Domain-specific exceptions
class ModelNotFoundError(MLException):
    """Raised when model file is not found"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.message = f"Model not found: {model_path}"
        super().__init__(self.message)


class InvalidDataError(MLException):
    """Raised when data validation fails"""

    def __init__(self, message: str, data_info: Optional[Dict] = None):
        self.data_info = data_info or {}
        self.message = message
        super().__init__(message)

    def __str__(self):
        if self.data_info:
            return f"{self.message} | Data info: {self.data_info}"
        return self.message


class GPUOutOfMemoryError(MLException):
    """Raised when GPU runs out of memory"""

    def __init__(self, batch_size: int, model_size: int, available_memory: int = 0):
        self.batch_size = batch_size
        self.model_size = model_size
        self.available_memory = available_memory
        self.message = (
            f"GPU Out of Memory: batch_size={batch_size}, "
            f"model_size={model_size}MB"
        )
        if available_memory:
            self.message += f", available={available_memory}MB"
        super().__init__(self.message)


class TrainingFailedError(MLException):
    """Raised when training fails"""

    def __init__(self, epoch: int, reason: str, metrics: Optional[Dict] = None):
        self.epoch = epoch
        self.reason = reason
        self.metrics = metrics or {}
        self.message = f"Training failed at epoch {epoch}: {reason}"
        super().__init__(self.message)


class ConfigurationError(MLException):
    """Raised when configuration is invalid"""

    def __init__(self, param: str, value: Any, expected: str):
        self.param = param
        self.value = value
        self.expected = expected
        self.message = f"Invalid config: {param}={value}, expected {expected}"
        super().__init__(self.message)


class DataValidationError(MLException):
    """Raised when data validation fails"""

    def __init__(self, validation_type: str, details: str):
        self.validation_type = validation_type
        self.details = details
        self.message = f"Data validation failed [{validation_type}]: {details}"
        super().__init__(self.message)


class CheckpointError(MLException):
    """Raised when model checkpointing fails"""

    def __init__(self, checkpoint_path: str, reason: str):
        self.checkpoint_path = checkpoint_path
        self.reason = reason
        self.message = f"Checkpoint failed at {checkpoint_path}: {reason}"
        super().__init__(self.message)


# Functions that use custom exceptions
def load_model(model_path: str) -> Dict[str, Any]:
    """
    Load model or raise custom exception.

    Args:
        model_path: Path to model file

    Returns:
        Model dictionary

    Raises:
        ModelNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise ModelNotFoundError(model_path)

    # Simulate loading
    return {"path": model_path, "loaded": True}


def validate_data(data: list, min_size: int = 10) -> None:
    """
    Validate data or raise exception.

    Args:
        data: Dataset to validate
        min_size: Minimum required size

    Raises:
        InvalidDataError: If data is invalid
    """
    if not data:
        raise InvalidDataError(
            "Dataset is empty",
            data_info={"size": 0, "type": type(data).__name__}
        )

    if len(data) < min_size:
        raise InvalidDataError(
            f"Dataset too small (minimum {min_size} samples required)",
            data_info={"size": len(data), "minimum": min_size}
        )

    # Check for None values
    none_count = sum(1 for item in data if item is None)
    if none_count > 0:
        raise InvalidDataError(
            f"Dataset contains {none_count} None values",
            data_info={"total": len(data), "none_count": none_count}
        )


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration or raise exceptions.

    Args:
        config: Configuration dictionary

    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_params = ["learning_rate", "batch_size", "epochs"]

    for param in required_params:
        if param not in config:
            raise ConfigurationError(param, None, "required parameter")

    # Validate learning rate
    if not 0 < config["learning_rate"] < 1:
        raise ConfigurationError(
            "learning_rate",
            config["learning_rate"],
            "0 < lr < 1"
        )

    # Validate batch size
    if config["batch_size"] <= 0:
        raise ConfigurationError(
            "batch_size",
            config["batch_size"],
            "positive integer"
        )

    # Validate epochs
    if config["epochs"] <= 0:
        raise ConfigurationError(
            "epochs",
            config["epochs"],
            "positive integer"
        )


def check_gpu_memory(batch_size: int, model_size: int,
                    available_memory: int) -> None:
    """
    Check if GPU has sufficient memory.

    Args:
        batch_size: Batch size
        model_size: Model size in MB
        available_memory: Available GPU memory in MB

    Raises:
        GPUOutOfMemoryError: If insufficient memory
    """
    estimated_memory = batch_size * 0.5 + model_size  # Simplified estimate

    if estimated_memory > available_memory:
        raise GPUOutOfMemoryError(
            batch_size=batch_size,
            model_size=model_size,
            available_memory=available_memory
        )


def validate_training_data(features: list, labels: list) -> None:
    """
    Validate training data format and consistency.

    Args:
        features: Feature data
        labels: Label data

    Raises:
        DataValidationError: If data is invalid
    """
    if len(features) != len(labels):
        raise DataValidationError(
            "shape_mismatch",
            f"Features ({len(features)}) and labels ({len(labels)}) "
            f"have different lengths"
        )

    if not features or not labels:
        raise DataValidationError(
            "empty_data",
            "Features or labels are empty"
        )


def main():
    """Demonstrate custom exceptions"""
    print("=" * 70)
    print("Custom Exceptions for ML Workflows")
    print("=" * 70)
    print()

    # Test 1: ModelNotFoundError
    print("Test 1: ModelNotFoundError")
    print("-" * 70)
    try:
        model = load_model("/nonexistent/model.h5")
    except ModelNotFoundError as e:
        print(f"✓ Caught: {e}")
        print(f"  Model path: {e.model_path}")
    print()

    # Test 2: InvalidDataError - Empty dataset
    print("Test 2: InvalidDataError - Empty Dataset")
    print("-" * 70)
    try:
        validate_data([])
    except InvalidDataError as e:
        print(f"✓ Caught: {e}")
        print(f"  Data info: {e.data_info}")
    print()

    # Test 3: InvalidDataError - Too small
    print("Test 3: InvalidDataError - Dataset Too Small")
    print("-" * 70)
    try:
        validate_data([1, 2, 3], min_size=10)
    except InvalidDataError as e:
        print(f"✓ Caught: {e}")
        print(f"  Data info: {e.data_info}")
    print()

    # Test 4: ConfigurationError - Missing parameter
    print("Test 4: ConfigurationError - Missing Parameter")
    print("-" * 70)
    try:
        config = {"learning_rate": 0.001, "batch_size": 32}
        validate_config(config)
    except ConfigurationError as e:
        print(f"✓ Caught: {e}")
        print(f"  Parameter: {e.param}, Value: {e.value}")
    print()

    # Test 5: ConfigurationError - Invalid value
    print("Test 5: ConfigurationError - Invalid Value")
    print("-" * 70)
    try:
        config = {"learning_rate": 1.5, "batch_size": 32, "epochs": 100}
        validate_config(config)
    except ConfigurationError as e:
        print(f"✓ Caught: {e}")
        print(f"  Parameter: {e.param}, Value: {e.value}, Expected: {e.expected}")
    print()

    # Test 6: GPUOutOfMemoryError
    print("Test 6: GPUOutOfMemoryError")
    print("-" * 70)
    try:
        check_gpu_memory(batch_size=128, model_size=2000, available_memory=1000)
    except GPUOutOfMemoryError as e:
        print(f"✓ Caught: {e}")
        print(f"  Batch size: {e.batch_size}, Model size: {e.model_size}MB")
        print(f"  Available: {e.available_memory}MB")
    print()

    # Test 7: DataValidationError
    print("Test 7: DataValidationError - Shape Mismatch")
    print("-" * 70)
    try:
        features = [1, 2, 3, 4, 5]
        labels = [0, 1, 0]
        validate_training_data(features, labels)
    except DataValidationError as e:
        print(f"✓ Caught: {e}")
        print(f"  Validation type: {e.validation_type}")
        print(f"  Details: {e.details}")
    print()

    # Test 8: TrainingFailedError
    print("Test 8: TrainingFailedError")
    print("-" * 70)
    try:
        raise TrainingFailedError(
            epoch=42,
            reason="Loss diverged to NaN",
            metrics={"loss": float('nan'), "accuracy": 0.23}
        )
    except TrainingFailedError as e:
        print(f"✓ Caught: {e}")
        print(f"  Epoch: {e.epoch}, Reason: {e.reason}")
        print(f"  Metrics: {e.metrics}")
    print()

    # Test 9: CheckpointError
    print("Test 9: CheckpointError")
    print("-" * 70)
    try:
        raise CheckpointError(
            checkpoint_path="/models/checkpoint_epoch_50.h5",
            reason="Disk full"
        )
    except CheckpointError as e:
        print(f"✓ Caught: {e}")
        print(f"  Path: {e.checkpoint_path}, Reason: {e.reason}")
    print()

    # Test 10: Exception Hierarchy
    print("Test 10: Catching Base MLException")
    print("-" * 70)
    try:
        raise ModelNotFoundError("/some/model.h5")
    except MLException as e:
        print(f"✓ Caught as MLException: {e}")
        print(f"  Specific type: {type(e).__name__}")
    print()

    print("=" * 70)
    print("✓ Custom exceptions demonstration complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
