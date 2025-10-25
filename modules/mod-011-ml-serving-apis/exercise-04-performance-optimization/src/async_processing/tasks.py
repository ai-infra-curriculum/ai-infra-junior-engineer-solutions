"""Celery tasks for async ML inference.

This module defines async tasks for:
- Single predictions
- Batch predictions
- Model retraining
- Data preprocessing
"""

from celery import group, chain, chord
from .celery_app import app, MLTask
import time
import random
import numpy as np
from typing import List, Dict


@app.task(base=MLTask, bind=True, max_retries=3)
def predict_single(self, features: List[float]) -> Dict:
    """
    Async single prediction task.

    Args:
        features: Input features

    Returns:
        Prediction result
    """
    try:
        # Simulate model loading (cached in worker)
        model = self.model

        # Simulate preprocessing
        time.sleep(0.01)

        # Simulate inference
        time.sleep(0.05)

        # Generate prediction (mock)
        prediction = random.choice([0, 1, 2])
        probability = random.random()

        return {
            "prediction": prediction,
            "probability": float(probability),
            "features": features,
            "model_version": "v1.0",
            "task_id": self.request.id,
        }

    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


@app.task(base=MLTask, bind=True, max_retries=3)
def predict_batch(self, instances: List[List[float]]) -> Dict:
    """
    Async batch prediction task.

    More efficient than individual predictions due to:
    - Vectorized operations
    - Reduced overhead
    - Better GPU utilization

    Args:
        instances: List of feature arrays

    Returns:
        Batch prediction results
    """
    try:
        model = self.model

        # Simulate batch preprocessing
        time.sleep(0.02)

        # Simulate batch inference (faster per-item than single)
        batch_size = len(instances)
        time.sleep(0.02 * batch_size)  # Sub-linear scaling

        # Generate predictions
        predictions = []
        for features in instances:
            pred = random.choice([0, 1, 2])
            prob = random.random()
            predictions.append({
                "prediction": pred,
                "probability": float(prob),
                "features": features,
            })

        return {
            "predictions": predictions,
            "batch_size": batch_size,
            "model_version": "v1.0",
            "task_id": self.request.id,
        }

    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


@app.task(base=MLTask)
def preprocess_features(features: List[float]) -> List[float]:
    """
    Preprocess features before prediction.

    Can be chained with prediction task.

    Args:
        features: Raw features

    Returns:
        Preprocessed features
    """
    # Simulate preprocessing
    time.sleep(0.01)

    # Normalize (mock)
    features_array = np.array(features)
    normalized = ((features_array - features_array.mean()) / features_array.std()).tolist()

    return normalized


@app.task(bind=True, max_retries=3)
def train_model(self, training_data: Dict) -> Dict:
    """
    Long-running model training task.

    Args:
        training_data: Training dataset and configuration

    Returns:
        Training results
    """
    try:
        # Simulate long training
        epochs = training_data.get("epochs", 10)

        for epoch in range(epochs):
            # Simulate epoch training
            time.sleep(1)

            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': epoch + 1,
                    'total': epochs,
                    'status': f'Training epoch {epoch + 1}/{epochs}'
                }
            )

        # Return results
        return {
            "status": "completed",
            "epochs": epochs,
            "final_loss": 0.123,
            "final_accuracy": 0.95,
            "model_path": "/models/trained_model.pt",
        }

    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)  # Retry after 1 minute


@app.task
def aggregate_results(results: List[Dict]) -> Dict:
    """
    Aggregate results from multiple tasks.

    Used with chord to combine results.

    Args:
        results: List of task results

    Returns:
        Aggregated statistics
    """
    predictions = [r["prediction"] for r in results]
    probabilities = [r["probability"] for r in results]

    return {
        "total_predictions": len(results),
        "prediction_distribution": {
            str(i): predictions.count(i) for i in set(predictions)
        },
        "mean_probability": float(np.mean(probabilities)),
        "std_probability": float(np.std(probabilities)),
    }


# Task compositions (chains and chords)

def create_preprocessing_pipeline(features_list: List[List[float]]):
    """
    Create a pipeline: preprocess -> predict -> aggregate.

    Args:
        features_list: List of feature arrays

    Returns:
        Celery workflow
    """
    # Chain: preprocess then predict for each instance
    workflow = chord(
        (chain(preprocess_features.s(features), predict_single.s())
         for features in features_list),
        aggregate_results.s()
    )

    return workflow


def create_batch_prediction_workflow(instances: List[List[float]], batch_size: int = 10):
    """
    Create a batch prediction workflow.

    Splits large dataset into batches and processes in parallel.

    Args:
        instances: All instances to predict
        batch_size: Batch size for each task

    Returns:
        Celery group of batch tasks
    """
    # Split into batches
    batches = [
        instances[i:i + batch_size]
        for i in range(0, len(instances), batch_size)
    ]

    # Create group of batch prediction tasks
    workflow = group(predict_batch.s(batch) for batch in batches)

    return workflow


# Scheduled periodic tasks (requires celery beat)

@app.task
def cleanup_old_results():
    """Periodic task to clean up old prediction results."""
    # Clean up results older than 24 hours
    print("Cleaning up old results...")
    # Implementation here
    return {"cleaned": 0}


@app.task
def update_model():
    """Periodic task to check for new model versions."""
    print("Checking for model updates...")
    # Implementation here
    return {"updated": False}


@app.task
def health_check():
    """Periodic health check task."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
    }


# Configure periodic tasks
app.conf.beat_schedule = {
    'cleanup-every-hour': {
        'task': 'tasks.cleanup_old_results',
        'schedule': 3600.0,  # Every hour
    },
    'update-model-daily': {
        'task': 'tasks.update_model',
        'schedule': 86400.0,  # Every day
    },
    'health-check-every-minute': {
        'task': 'tasks.health_check',
        'schedule': 60.0,  # Every minute
    },
}


if __name__ == "__main__":
    # Example usage
    print("Example Celery Task Usage")
    print("=" * 60)

    # 1. Single prediction
    print("\n1. Single async prediction:")
    result = predict_single.delay([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Task ID: {result.id}")
    print(f"Status: {result.status}")

    # 2. Batch prediction
    print("\n2. Batch prediction:")
    instances = [[1.0, 2.0, 3.0, 4.0, 5.0] for _ in range(100)]
    result = predict_batch.delay(instances)
    print(f"Task ID: {result.id}")

    # 3. Chained tasks (preprocess -> predict)
    print("\n3. Chained tasks:")
    result = chain(
        preprocess_features.s([1.0, 2.0, 3.0, 4.0, 5.0]),
        predict_single.s()
    ).apply_async()
    print(f"Chain ID: {result.id}")

    # 4. Parallel tasks with chord
    print("\n4. Parallel predictions with aggregation:")
    features_list = [[random.random() for _ in range(5)] for _ in range(10)]
    workflow = create_preprocessing_pipeline(features_list)
    result = workflow.apply_async()
    print(f"Chord ID: {result.id}")

    # 5. Batch workflow
    print("\n5. Large-scale batch workflow:")
    large_dataset = [[random.random() for _ in range(5)] for _ in range(1000)]
    workflow = create_batch_prediction_workflow(large_dataset, batch_size=50)
    result = workflow.apply_async()
    print(f"Group ID: {result.id}")

    print("\n" + "=" * 60)
    print("Tasks submitted! Check Celery worker logs for execution.")
