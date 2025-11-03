#!/usr/bin/env python3
"""
Context Managers for ML Resource Management

Demonstrates using context managers for safe resource management in ML workflows.
"""

import time
import logging
from typing import Optional, Any
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPUContext:
    """Context manager for GPU operations"""

    def __init__(self, device_id: int = 0):
        """
        Initialize GPU context.

        Args:
            device_id: GPU device ID to use
        """
        self.device_id = device_id
        self.previous_device = None
        self.allocated = False

    def __enter__(self):
        """Allocate GPU resources"""
        logger.info(f"→ Allocating GPU {self.device_id}")
        # Simulate GPU allocation
        self.previous_device = 0
        self.allocated = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release GPU resources"""
        if self.allocated:
            logger.info(f"← Releasing GPU {self.device_id}")

            if exc_type is not None:
                logger.error(f"GPU operation failed: {exc_type.__name__}: {exc_val}")
            else:
                logger.info("GPU operation completed successfully")

        # Return False to propagate exceptions
        return False


class ModelCheckpoint:
    """Context manager for safe model checkpointing"""

    def __init__(self, checkpoint_path: str):
        """
        Initialize checkpoint context.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.temp_path = Path(f"{checkpoint_path}.tmp")
        self.checkpoint_file = None

    def __enter__(self):
        """Start checkpoint operation"""
        logger.info(f"→ Starting checkpoint to {self.checkpoint_path}")
        # Create temp file (in real implementation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize or cleanup checkpoint"""
        if exc_type is None:
            # Success: rename temp to final (atomic operation)
            logger.info(f"✓ Checkpoint successful: {self.checkpoint_path}")
            # In real implementation:
            # self.temp_path.rename(self.checkpoint_path)
        else:
            # Failure: remove temp file
            logger.error(f"✗ Checkpoint failed: {exc_val}")
            # In real implementation:
            # if self.temp_path.exists():
            #     self.temp_path.unlink()

        return False


class TimerContext:
    """Context manager for timing code blocks"""

    def __init__(self, name: str = "Operation"):
        """
        Initialize timer context.

        Args:
            name: Name of operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timer"""
        self.start_time = time.time()
        logger.info(f"⏱  Starting: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log elapsed time"""
        self.elapsed = time.time() - self.start_time

        if exc_type is None:
            logger.info(f"✓ {self.name} completed in {self.elapsed:.4f} seconds")
        else:
            logger.warning(f"✗ {self.name} failed after {self.elapsed:.4f} seconds")

        return False


class MemoryTracker:
    """Context manager for tracking memory usage"""

    def __init__(self, name: str = "Operation"):
        """
        Initialize memory tracker.

        Args:
            name: Name of operation to track
        """
        self.name = name
        self.start_memory = None
        self.peak_memory = None

    def __enter__(self):
        """Start tracking memory"""
        logger.info(f"📊 Tracking memory for: {self.name}")
        # In real implementation, would use psutil or similar
        self.start_memory = 100  # Simulated MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Report memory usage"""
        # Simulated memory tracking
        end_memory = 250  # Simulated MB
        peak_memory = 300  # Simulated MB
        memory_increase = end_memory - self.start_memory

        logger.info(
            f"📊 Memory report for {self.name}:\n"
            f"     Start: {self.start_memory}MB\n"
            f"     End: {end_memory}MB\n"
            f"     Peak: {peak_memory}MB\n"
            f"     Increase: +{memory_increase}MB"
        )

        return False


class ResourcePool:
    """Context manager for managing a pool of resources"""

    def __init__(self, resource_type: str, count: int):
        """
        Initialize resource pool.

        Args:
            resource_type: Type of resource
            count: Number of resources in pool
        """
        self.resource_type = resource_type
        self.count = count
        self.resources = []

    def __enter__(self):
        """Acquire resources"""
        logger.info(f"→ Acquiring {self.count} {self.resource_type} resources")

        for i in range(self.count):
            resource = {"id": i, "type": self.resource_type, "in_use": True}
            self.resources.append(resource)

        logger.info(f"✓ Acquired {len(self.resources)} resources")
        return self.resources

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release all resources"""
        logger.info(f"← Releasing {len(self.resources)} {self.resource_type} resources")

        for resource in self.resources:
            resource["in_use"] = False

        self.resources.clear()
        logger.info("✓ All resources released")

        return False


class ExceptionSuppressor:
    """Context manager that optionally suppresses exceptions"""

    def __init__(self, suppress: bool = True, log_exceptions: bool = True):
        """
        Initialize exception suppressor.

        Args:
            suppress: Whether to suppress exceptions
            log_exceptions: Whether to log suppressed exceptions
        """
        self.suppress = suppress
        self.log_exceptions = log_exceptions

    def __enter__(self):
        """Enter context"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle exceptions"""
        if exc_type is not None:
            if self.log_exceptions:
                logger.warning(
                    f"Exception {'suppressed' if self.suppress else 'propagated'}: "
                    f"{exc_type.__name__}: {exc_val}"
                )

            # Return True to suppress exception, False to propagate
            return self.suppress

        return False


def train_with_context():
    """Demonstrate training with context managers"""
    print("=" * 70)
    print("Training with Context Managers")
    print("=" * 70)
    print()

    with GPUContext(device_id=0):
        logger.info("Training on GPU...")

        with TimerContext("Training epoch"):
            # Simulate training
            time.sleep(0.3)

        with ModelCheckpoint("model_checkpoint_epoch_10.h5"):
            logger.info("Saving model checkpoint...")
            # Simulate saving
            time.sleep(0.1)


def data_processing_with_context():
    """Demonstrate data processing with multiple context managers"""
    print("=" * 70)
    print("Data Processing with Multiple Contexts")
    print("=" * 70)
    print()

    with MemoryTracker("Data Loading"):
        with TimerContext("Load Dataset"):
            # Simulate data loading
            time.sleep(0.2)
            logger.info("Dataset loaded")


def resource_pool_example():
    """Demonstrate resource pool management"""
    print("=" * 70)
    print("Resource Pool Management")
    print("=" * 70)
    print()

    with ResourcePool("GPU", count=4) as gpus:
        logger.info(f"Using {len(gpus)} GPUs for distributed training")
        # Simulate using resources
        time.sleep(0.2)


def exception_handling_in_context():
    """Demonstrate exception handling in context managers"""
    print("=" * 70)
    print("Exception Handling in Context Managers")
    print("=" * 70)
    print()

    # Example 1: Context manager with exception
    print("Example 1: GPU Context with Exception")
    print("-" * 70)
    try:
        with GPUContext(device_id=0):
            logger.info("Starting GPU operation...")
            raise RuntimeError("Simulated GPU error")
    except RuntimeError as e:
        logger.info(f"Exception caught outside context: {e}")
    print()

    # Example 2: Checkpoint with exception
    print("Example 2: Checkpoint with Exception")
    print("-" * 70)
    try:
        with ModelCheckpoint("model_failed.h5"):
            logger.info("Starting checkpoint...")
            raise IOError("Simulated disk error")
    except IOError as e:
        logger.info(f"Exception caught outside context: {e}")
    print()

    # Example 3: Suppressed exceptions
    print("Example 3: Suppressed Exceptions")
    print("-" * 70)
    with ExceptionSuppressor(suppress=True):
        logger.info("This exception will be suppressed")
        raise ValueError("This error is suppressed")
    logger.info("Execution continues after suppressed exception")
    print()


def nested_context_managers():
    """Demonstrate nested context managers"""
    print("=" * 70)
    print("Nested Context Managers")
    print("=" * 70)
    print()

    with TimerContext("Complete Training Pipeline"):
        with MemoryTracker("Pipeline"):
            with GPUContext(device_id=0):
                with TimerContext("Data Loading"):
                    time.sleep(0.1)
                    logger.info("Data loaded")

                with TimerContext("Training"):
                    time.sleep(0.2)
                    logger.info("Model trained")

                with ModelCheckpoint("final_model.h5"):
                    time.sleep(0.05)
                    logger.info("Model saved")


def main():
    """Run all context manager demonstrations"""
    train_with_context()
    print()

    data_processing_with_context()
    print()

    resource_pool_example()
    print()

    exception_handling_in_context()
    print()

    nested_context_managers()
    print()

    print("=" * 70)
    print("✓ Context manager demonstrations complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
