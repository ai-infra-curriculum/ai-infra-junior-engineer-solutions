#!/usr/bin/env python3
"""
Exception Handling Patterns for ML Workflows

Demonstrates try-except-else-finally patterns with practical ML examples.
"""

from typing import List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_divide(a: float, b: float) -> Optional[float]:
    """
    Safely divide two numbers with comprehensive error handling.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Division result or None if operation fails
    """
    try:
        result = a / b
    except ZeroDivisionError:
        logger.error(f"Cannot divide {a} by zero")
        return None
    except TypeError as e:
        logger.error(f"Invalid types for division: {e}")
        return None
    else:
        # Runs only if no exception occurred
        logger.info(f"Division successful: {a} / {b} = {result}")
        return result
    finally:
        # Always runs, regardless of exception
        logger.debug("Division operation completed")


def load_model_with_fallback(primary_path: str,
                             backup_path: str) -> Optional[Dict[str, Any]]:
    """
    Load model with fallback to backup path.

    Args:
        primary_path: Primary model file path
        backup_path: Backup model file path

    Returns:
        Loaded model dictionary or None if both fail
    """
    try:
        # Try primary path
        logger.info(f"Loading model from {primary_path}")
        with open(primary_path, 'r') as f:
            model = {"path": primary_path, "data": f.read()}
            logger.info("✓ Successfully loaded from primary path")
            return model
    except FileNotFoundError:
        logger.warning(f"Primary model not found, trying backup")

        try:
            # Try backup path
            with open(backup_path, 'r') as f:
                model = {"path": backup_path, "data": f.read()}
                logger.info("✓ Successfully loaded from backup path")
                return model
        except FileNotFoundError:
            logger.error("Both primary and backup models not found")
            return None
    finally:
        logger.info("Model loading attempt completed")


def process_batch_safe(batch: List[Any]) -> Dict[str, Any]:
    """
    Process batch with comprehensive error handling.

    Args:
        batch: List of items to process

    Returns:
        Dictionary with processed results, errors, and statistics
    """
    results = {
        "processed": [],
        "errors": [],
        "stats": {}
    }

    try:
        if not batch:
            raise ValueError("Empty batch provided")

        logger.info(f"Processing batch of {len(batch)} items")

        for i, value in enumerate(batch):
            try:
                # Simulate processing that might fail
                if isinstance(value, str):
                    raise TypeError(f"Cannot process string value: {value}")

                processed = value * 2
                results["processed"].append(processed)
            except TypeError as e:
                error_msg = f"Index {i}: {str(e)}"
                results["errors"].append(error_msg)
                results["processed"].append(None)
                logger.warning(error_msg)

        # Calculate statistics
        valid_values = [v for v in results["processed"] if v is not None]
        if valid_values:
            results["stats"] = {
                "total": len(batch),
                "successful": len(valid_values),
                "failed": len(results["errors"]),
                "mean": sum(valid_values) / len(valid_values)
            }
        else:
            results["stats"] = {
                "total": len(batch),
                "successful": 0,
                "failed": len(results["errors"]),
                "mean": 0
            }

    except ValueError as e:
        logger.error(f"Batch processing failed: {e}")
        results["errors"].append(str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        results["errors"].append(f"Unexpected: {str(e)}")
    finally:
        logger.info(f"Processed {len(results['processed'])} items, "
                   f"{len(results['errors'])} errors")

    return results


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration with multiple checks.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["learning_rate", "batch_size", "epochs"]

    try:
        # Check required keys
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required key: {key}")

        # Validate learning rate
        lr = config["learning_rate"]
        if not isinstance(lr, (int, float)):
            raise TypeError(f"learning_rate must be numeric, got {type(lr)}")
        if not 0 < lr < 1:
            raise ValueError(f"learning_rate must be in (0, 1), got {lr}")

        # Validate batch size
        batch_size = config["batch_size"]
        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be int, got {type(batch_size)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Validate epochs
        epochs = config["epochs"]
        if not isinstance(epochs, int):
            raise TypeError(f"epochs must be int, got {type(epochs)}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        logger.info("✓ Configuration validation passed")
        return True

    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def train_with_error_handling(data: List[float],
                              config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simulate training with comprehensive error handling.

    Args:
        data: Training data
        config: Training configuration

    Returns:
        Training results or None if failed
    """
    try:
        # Validate inputs
        if not data:
            raise ValueError("Empty dataset provided")

        if not validate_config(config):
            raise ValueError("Invalid configuration")

        logger.info("Starting training...")

        # Simulate training process
        results = {
            "status": "success",
            "epochs_completed": config["epochs"],
            "final_loss": 0.15,
            "final_accuracy": 0.92
        }

        logger.info(f"✓ Training completed successfully")
        return results

    except ValueError as e:
        logger.error(f"Training failed due to validation error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected training error: {e}")
        return None
    finally:
        logger.info("Training session ended")


def main():
    """Demonstrate exception handling patterns"""
    print("=" * 70)
    print("Exception Handling Patterns")
    print("=" * 70)
    print()

    # Example 1: Safe Division
    print("Example 1: Safe Division with Try-Except-Else-Finally")
    print("-" * 70)
    print(f"10 / 2 = {safe_divide(10, 2)}")
    print(f"10 / 0 = {safe_divide(10, 0)}")
    print(f"'10' / 2 = {safe_divide('10', 2)}")
    print()

    # Example 2: Model Loading with Fallback
    print("Example 2: Model Loading with Fallback")
    print("-" * 70)
    model = load_model_with_fallback(
        "/tmp/primary_model.h5",
        "/tmp/backup_model.h5"
    )
    if model:
        print(f"✓ Model loaded from: {model['path']}")
    else:
        print("✗ Failed to load model from any source")
    print()

    # Example 3: Batch Processing
    print("Example 3: Batch Processing with Error Collection")
    print("-" * 70)
    batch = [1.0, 2.0, "invalid", 4.0, 5.0]
    results = process_batch_safe(batch)
    print(f"Processed: {results['processed']}")
    print(f"Errors: {results['errors']}")
    print(f"Stats: {results['stats']}")
    print()

    # Example 4: Configuration Validation
    print("Example 4: Configuration Validation")
    print("-" * 70)
    valid_config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
    invalid_config = {
        "learning_rate": 1.5,  # Out of range
        "batch_size": 32,
        "epochs": 100
    }
    print(f"Valid config: {validate_config(valid_config)}")
    print(f"Invalid config: {validate_config(invalid_config)}")
    print()

    # Example 5: Training with Error Handling
    print("Example 5: Training with Comprehensive Error Handling")
    print("-" * 70)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
    results = train_with_error_handling(data, config)
    if results:
        print(f"✓ Training results: {results}")
    else:
        print("✗ Training failed")
    print()

    print("=" * 70)
    print("✓ Exception handling patterns demonstration complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
