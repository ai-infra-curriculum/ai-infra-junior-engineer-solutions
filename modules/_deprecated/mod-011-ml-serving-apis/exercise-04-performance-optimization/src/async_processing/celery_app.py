"""Celery application configuration for async ML inference.

This module configures Celery for handling asynchronous ML predictions,
batch processing, and long-running tasks.
"""

from celery import Celery
from celery.signals import worker_ready, worker_shutdown
import os


# Celery configuration
BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Create Celery app
app = Celery(
    "ml_tasks",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

# Celery configuration
app.conf.update(
    # Task settings
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,  # Reject task if worker dies
    task_time_limit=300,  # 5 minutes max
    task_soft_time_limit=240,  # Soft limit warning at 4 minutes

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },

    # Worker settings
    worker_prefetch_multiplier=1,  # Only fetch 1 task at a time (for ML workloads)
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (prevent memory leaks)

    # Monitoring
    task_send_sent_event=True,
    worker_send_task_events=True,

    # Rate limiting
    task_default_rate_limit="100/m",  # 100 tasks per minute default

    # Task routes (can route different tasks to different queues)
    task_routes={
        "ml_tasks.tasks.predict_single": {"queue": "predictions"},
        "ml_tasks.tasks.predict_batch": {"queue": "batch"},
        "ml_tasks.tasks.train_model": {"queue": "training"},
    },

    # Priority queues
    task_queue_max_priority=10,
    task_default_priority=5,
)


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """
    Called when worker is ready.
    Load ML models into memory here.
    """
    print("=" * 60)
    print("CELERY WORKER READY")
    print("=" * 60)
    print(f"Worker: {sender}")
    print(f"Broker: {BROKER_URL}")
    print(f"Backend: {RESULT_BACKEND}")
    print("=" * 60)


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    """
    Called when worker shuts down.
    Clean up resources here.
    """
    print("=" * 60)
    print("CELERY WORKER SHUTTING DOWN")
    print("=" * 60)
    print("Cleaning up resources...")
    print("=" * 60)


# Task base class with custom behavior
class MLTask(app.Task):
    """Base task class for ML tasks with model caching."""

    _model = None

    @property
    def model(self):
        """Lazy load model (cached in worker process)."""
        if self._model is None:
            print("Loading ML model into memory...")
            # In production, load actual model here
            # self._model = torch.jit.load("model.pt")
            self._model = "MockModel"  # Placeholder
            print("✓ Model loaded")
        return self._model

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        print(f"Task {task_id} failed: {exc}")
        # Log to monitoring system, send alert, etc.

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        print(f"Task {task_id} completed successfully")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        print(f"Task {task_id} retrying after exception: {exc}")


if __name__ == "__main__":
    # Start worker: celery -A celery_app worker --loglevel=info
    app.start()
