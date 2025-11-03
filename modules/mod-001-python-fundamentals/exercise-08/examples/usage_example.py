"""
Example usage of ml-infra-utils package.

This demonstrates how to use the various utilities provided by the package.
"""

import time
from ml_infra_utils import (
    normalize,
    standardize,
    accuracy,
    precision,
    recall,
    f1_score,
    timer,
    retry,
    StructuredLogger,
)


def preprocessing_example():
    """Demonstrate preprocessing utilities."""
    print("\n=== Preprocessing Example ===")

    # Normalize data
    raw_data = [10, 20, 30, 40, 50]
    normalized = normalize(raw_data)
    print(f"Original data: {raw_data}")
    print(f"Normalized: {normalized}")

    # Standardize data
    standardized = standardize(raw_data)
    print(f"Standardized: {standardized}")


def metrics_example():
    """Demonstrate classification metrics."""
    print("\n=== Classification Metrics Example ===")

    # Example predictions and true labels
    predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    labels = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]

    print(f"Predictions: {predictions}")
    print(f"True labels: {labels}")

    # Calculate metrics
    acc = accuracy(predictions, labels)
    prec = precision(predictions, labels)
    rec = recall(predictions, labels)
    f1 = f1_score(predictions, labels)

    print(f"\nAccuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"F1 Score:  {f1:.2%}")


def decorator_examples():
    """Demonstrate decorator utilities."""
    print("\n=== Decorator Examples ===")

    # Timer decorator
    @timer
    def simulate_training():
        """Simulate ML model training."""
        time.sleep(0.5)
        return "Model trained"

    print("\n1. Timer decorator:")
    result = simulate_training()
    print(f"Result: {result}")

    # Retry decorator
    attempt_count = [0]

    @retry(max_attempts=3, delay=0.1, backoff=2.0)
    def unreliable_api_call():
        """Simulate unreliable API call that succeeds on third try."""
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ConnectionError(f"Connection failed (attempt {attempt_count[0]})")
        return "Success!"

    print("\n2. Retry decorator:")
    result = unreliable_api_call()
    print(f"Final result: {result}")


def logging_example():
    """Demonstrate structured logging."""
    print("\n=== Structured Logging Example ===\n")

    # Create logger
    logger = StructuredLogger("ml_pipeline")

    # Log various events
    logger.info("Pipeline started", pipeline_id="pipe_123", version="1.0")

    logger.info(
        "Data loaded",
        context={"dataset": "train_v1", "num_samples": 10000, "num_features": 50},
    )

    logger.info(
        "Model training completed",
        context={
            "model_name": "random_forest",
            "metrics": {"accuracy": 0.95, "f1": 0.93},
            "training_time_sec": 120.5,
        },
    )

    logger.warning("High memory usage detected", memory_mb=8192, threshold_mb=8000)

    logger.info("Pipeline completed successfully")


def complete_ml_workflow_example():
    """Demonstrate a complete ML workflow using all utilities."""
    print("\n=== Complete ML Workflow Example ===\n")

    logger = StructuredLogger("ml_workflow")

    @timer
    def load_and_preprocess_data():
        """Load and preprocess data."""
        logger.info("Loading data")
        # Simulate loading data
        raw_data = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

        logger.info("Preprocessing data")
        # Normalize features
        processed_data = normalize(raw_data)
        return processed_data

    @timer
    def train_model(data):
        """Train ML model."""
        logger.info("Training model")
        time.sleep(0.2)  # Simulate training
        return "trained_model"

    @timer
    def evaluate_model():
        """Evaluate model performance."""
        logger.info("Evaluating model")
        # Simulate predictions
        predictions = [1, 0, 1, 1, 0]
        labels = [1, 0, 1, 0, 0]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy(predictions, labels),
            "precision": precision(predictions, labels),
            "recall": recall(predictions, labels),
            "f1_score": f1_score(predictions, labels),
        }

        logger.info("Model evaluation complete", context={"metrics": metrics})
        return metrics

    # Run workflow
    logger.info("Starting ML workflow")

    data = load_and_preprocess_data()
    model = train_model(data)
    metrics = evaluate_model()

    logger.info("ML workflow completed", context={"final_metrics": metrics})

    print("\n=== Workflow Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1_score']:.2%}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("ML Infrastructure Utilities - Usage Examples")
    print("=" * 60)

    preprocessing_example()
    metrics_example()
    decorator_examples()
    logging_example()
    complete_ml_workflow_example()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
