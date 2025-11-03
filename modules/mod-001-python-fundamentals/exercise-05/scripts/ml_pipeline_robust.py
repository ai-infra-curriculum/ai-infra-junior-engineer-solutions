#!/usr/bin/env python3
"""
Robust ML Pipeline with Comprehensive Error Handling

Demonstrates building a production-ready ML pipeline with comprehensive error handling,
graceful degradation, and structured error reporting.
"""

from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass
from enum import Enum
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status"""
    SUCCESS = "success"
    PARTIAL_FAILURE = "partial_failure"
    COMPLETE_FAILURE = "failure"
    RETRYING = "retrying"


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    status: PipelineStatus
    data: Optional[Any]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

    def __str__(self):
        return (
            f"PipelineResult(\n"
            f"  status={self.status.value},\n"
            f"  errors={len(self.errors)},\n"
            f"  warnings={len(self.warnings)},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )


class RobustMLPipeline:
    """ML pipeline with comprehensive error handling"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metadata: Dict[str, Any] = {
            "steps_completed": 0,
            "start_time": None,
            "end_time": None
        }

    def run(self) -> PipelineResult:
        """
        Execute pipeline with comprehensive error handling.

        Returns:
            PipelineResult with status, data, errors, and metadata
        """
        logger.info("=" * 70)
        logger.info("Starting ML Pipeline")
        logger.info("=" * 70)

        self.metadata["start_time"] = time.time()
        data = None

        try:
            # Step 1: Load data
            logger.info("Step 1/5: Loading data...")
            data = self._load_data_safe()
            if data is None:
                return self._create_failure_result("Data loading failed")
            self.metadata["steps_completed"] = 1

            # Step 2: Validate data
            logger.info("Step 2/5: Validating data...")
            if not self._validate_data_safe(data):
                return self._create_failure_result("Data validation failed")
            self.metadata["steps_completed"] = 2

            # Step 3: Preprocess
            logger.info("Step 3/5: Preprocessing data...")
            data = self._preprocess_safe(data)
            if data is None:
                return self._create_failure_result("Preprocessing failed")
            self.metadata["steps_completed"] = 3

            # Step 4: Train model
            logger.info("Step 4/5: Training model...")
            model = self._train_safe(data)
            if model is None:
                return self._create_failure_result("Training failed")
            self.metadata["steps_completed"] = 4

            # Step 5: Evaluate
            logger.info("Step 5/5: Evaluating model...")
            metrics = self._evaluate_safe(model, data)
            self.metadata["steps_completed"] = 5

            # Success
            self.metadata["end_time"] = time.time()
            self.metadata["duration_seconds"] = (
                self.metadata["end_time"] - self.metadata["start_time"]
            )

            logger.info("=" * 70)
            logger.info("✓ Pipeline completed successfully")
            logger.info("=" * 70)

            return PipelineResult(
                status=PipelineStatus.SUCCESS,
                data={"model": model, "metrics": metrics},
                errors=self.errors,
                warnings=self.warnings,
                metadata=self.metadata
            )

        except Exception as e:
            logger.error(f"Unexpected pipeline error: {e}")
            return self._create_failure_result(f"Unexpected error: {str(e)}")

    def _load_data_safe(self) -> Optional[Dict[str, Any]]:
        """
        Load data with error handling.

        Returns:
            Data dictionary or None if loading fails
        """
        try:
            # Simulate data loading
            logger.info("  → Loading dataset from storage...")
            time.sleep(0.1)

            # Simulated data
            data = {
                "samples": 10000,
                "features": 50,
                "labels": list(range(10000)),
                "feature_names": [f"feature_{i}" for i in range(50)]
            }

            logger.info(f"  ✓ Loaded {data['samples']} samples with {data['features']} features")
            return data

        except FileNotFoundError as e:
            self.errors.append(f"Data file not found: {e}")
            logger.error(f"  ✗ Data loading failed: {e}")
            return None
        except MemoryError as e:
            self.errors.append("Insufficient memory to load data")
            logger.error(f"  ✗ Data loading failed: Out of memory")
            return None
        except Exception as e:
            self.errors.append(f"Data loading error: {e}")
            logger.error(f"  ✗ Unexpected data loading error: {e}")
            return None

    def _validate_data_safe(self, data: Dict[str, Any]) -> bool:
        """
        Validate data with error handling.

        Args:
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check minimum samples
            if data.get("samples", 0) < 10:
                self.errors.append(f"Insufficient samples: {data.get('samples', 0)} < 10")
                logger.error("  ✗ Insufficient samples")
                return False

            # Check features
            if data.get("features", 0) < 1:
                self.errors.append("No features found")
                logger.error("  ✗ No features found")
                return False

            # Check labels
            if not data.get("labels"):
                self.errors.append("No labels found")
                logger.error("  ✗ No labels found")
                return False

            # Check consistency
            if len(data["labels"]) != data["samples"]:
                self.errors.append(
                    f"Sample/label mismatch: {data['samples']} != {len(data['labels'])}"
                )
                logger.error("  ✗ Sample/label count mismatch")
                return False

            logger.info("  ✓ Data validation passed")
            return True

        except KeyError as e:
            self.errors.append(f"Missing required field: {e}")
            logger.error(f"  ✗ Validation failed: Missing field {e}")
            return False
        except Exception as e:
            self.errors.append(f"Validation error: {e}")
            logger.error(f"  ✗ Validation failed: {e}")
            return False

    def _preprocess_safe(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Preprocess data with error handling.

        Args:
            data: Raw data

        Returns:
            Preprocessed data or None if preprocessing fails
        """
        try:
            logger.info("  → Normalizing features...")
            time.sleep(0.1)

            # Simulate preprocessing
            processed = {
                **data,
                "preprocessed": True,
                "normalized": True,
                "missing_values_handled": True
            }

            # Add warning if any issues found
            if data["samples"] < 1000:
                self.warnings.append("Small dataset - results may not generalize well")
                logger.warning("  ⚠ Small dataset warning")

            logger.info("  ✓ Preprocessing complete")
            return processed

        except MemoryError:
            self.errors.append("Out of memory during preprocessing")
            logger.error("  ✗ Preprocessing failed: Out of memory")
            return None
        except Exception as e:
            self.errors.append(f"Preprocessing error: {e}")
            logger.error(f"  ✗ Preprocessing failed: {e}")
            return None

    def _train_safe(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Train model with error handling.

        Args:
            data: Training data

        Returns:
            Trained model or None if training fails
        """
        try:
            logger.info(f"  → Training model for {self.config.get('epochs', 10)} epochs...")
            time.sleep(0.15)

            # Simulate training
            model = {
                "type": self.config.get("model_type", "default"),
                "trained": True,
                "epochs": self.config.get("epochs", 10),
                "parameters": 1000000,
                "accuracy": 0.92,
                "loss": 0.15
            }

            logger.info(f"  ✓ Training complete - Accuracy: {model['accuracy']:.3f}")
            return model

        except MemoryError:
            self.errors.append("GPU/CPU out of memory during training")
            logger.error("  ✗ Training failed: Out of memory")

            # Suggest solution
            self.warnings.append(
                "Try reducing batch_size or model complexity"
            )
            return None
        except RuntimeError as e:
            self.errors.append(f"Training runtime error: {e}")
            logger.error(f"  ✗ Training failed: {e}")
            return None
        except Exception as e:
            self.errors.append(f"Training error: {e}")
            logger.error(f"  ✗ Training failed: {e}")
            return None

    def _evaluate_safe(self, model: Dict[str, Any],
                      data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model with error handling.

        Args:
            model: Trained model
            data: Test data

        Returns:
            Evaluation metrics (empty dict if evaluation fails)
        """
        try:
            logger.info("  → Evaluating model performance...")
            time.sleep(0.1)

            metrics = {
                "accuracy": 0.92,
                "precision": 0.90,
                "recall": 0.89,
                "f1_score": 0.895,
                "loss": 0.15
            }

            logger.info(f"  ✓ Evaluation complete - Accuracy: {metrics['accuracy']:.3f}")
            return metrics

        except Exception as e:
            self.warnings.append(f"Evaluation error: {e}")
            logger.warning(f"  ⚠ Evaluation failed: {e}")
            # Return empty metrics instead of failing entire pipeline
            return {}

    def _create_failure_result(self, error_msg: str) -> PipelineResult:
        """
        Create failure result.

        Args:
            error_msg: Error message

        Returns:
            PipelineResult with failure status
        """
        self.errors.append(error_msg)
        self.metadata["end_time"] = time.time()
        if self.metadata["start_time"]:
            self.metadata["duration_seconds"] = (
                self.metadata["end_time"] - self.metadata["start_time"]
            )

        logger.error("=" * 70)
        logger.error(f"✗ Pipeline failed: {error_msg}")
        logger.error("=" * 70)

        return PipelineResult(
            status=PipelineStatus.COMPLETE_FAILURE,
            data=None,
            errors=self.errors,
            warnings=self.warnings,
            metadata=self.metadata
        )


def run_successful_pipeline():
    """Run pipeline with valid configuration"""
    print("\n" + "=" * 70)
    print("Example 1: Successful Pipeline Execution")
    print("=" * 70)

    config = {
        "data_path": "/data/train.csv",
        "model_type": "resnet",
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.001
    }

    pipeline = RobustMLPipeline(config)
    result = pipeline.run()

    print(f"\n{result}")
    print(f"\nErrors: {result.errors}")
    print(f"Warnings: {result.warnings}")

    if result.status == PipelineStatus.SUCCESS:
        print(f"\n✓ Model trained successfully!")
        if result.data and "metrics" in result.data:
            print(f"Metrics: {result.data['metrics']}")


def run_pipeline_with_warnings():
    """Run pipeline that generates warnings"""
    print("\n" + "=" * 70)
    print("Example 2: Pipeline with Warnings")
    print("=" * 70)

    config = {
        "data_path": "/data/small_dataset.csv",
        "model_type": "simple",
        "epochs": 5
    }

    pipeline = RobustMLPipeline(config)
    result = pipeline.run()

    print(f"\n{result}")
    print(f"\nWarnings: {result.warnings}")


def main():
    """Run all pipeline examples"""
    print("=" * 70)
    print("Robust ML Pipeline Demonstration")
    print("=" * 70)

    run_successful_pipeline()
    run_pipeline_with_warnings()

    print("\n" + "=" * 70)
    print("✓ Robust ML pipeline demonstration complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
