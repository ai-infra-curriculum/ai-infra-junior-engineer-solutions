"""Celery application configuration for async ML inference.

Sets up:
- Broker / result backend wiring from environment.
- Conservative defaults appropriate for GPU-backed ML workers
  (``prefetch_multiplier=1``, ``acks_late``, child recycling).
- A reusable ``MLTask`` base class with lazy model loading so each
  worker process loads the model exactly once.
- A simple structured logger so worker output is grep-friendly in
  production.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from celery import Celery
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    worker_ready,
    worker_shutdown,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
MODEL_PATH = os.getenv("ML_MODEL_PATH", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ml-celery")


# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------

app = Celery(
    "ml_tasks",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    result_expires=3600,
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    task_send_sent_event=True,
    worker_send_task_events=True,
    task_default_rate_limit="100/m",
    task_routes={
        "ml_tasks.tasks.predict_single": {"queue": "predictions"},
        "ml_tasks.tasks.predict_batch": {"queue": "batch"},
        "ml_tasks.tasks.train_model": {"queue": "training"},
    },
    task_queue_max_priority=10,
    task_default_priority=5,
)


# ---------------------------------------------------------------------------
# Worker lifecycle hooks
# ---------------------------------------------------------------------------


@worker_ready.connect
def on_worker_ready(sender: Any = None, **_kwargs: Any) -> None:
    logger.info(
        "celery_worker_ready worker=%s broker=%s backend=%s",
        sender,
        BROKER_URL,
        RESULT_BACKEND,
    )


@worker_shutdown.connect
def on_worker_shutdown(sender: Any = None, **_kwargs: Any) -> None:
    logger.info("celery_worker_shutdown worker=%s", sender)


@task_prerun.connect
def _task_prerun(task_id: str = "", task: Any = None, **_kwargs: Any) -> None:
    logger.info("task_started id=%s name=%s", task_id, getattr(task, "name", "?"))


@task_postrun.connect
def _task_postrun(
    task_id: str = "",
    task: Any = None,
    state: Optional[str] = None,
    **_kwargs: Any,
) -> None:
    logger.info(
        "task_finished id=%s name=%s state=%s",
        task_id,
        getattr(task, "name", "?"),
        state,
    )


@task_failure.connect
def _task_failure(
    task_id: str = "",
    exception: Optional[BaseException] = None,
    **_kwargs: Any,
) -> None:
    logger.error("task_failed id=%s exception=%s", task_id, exception)


# ---------------------------------------------------------------------------
# Reusable base class with cached model loading
# ---------------------------------------------------------------------------


class _StubModel:
    """Deterministic stub model used when no real artifact is configured."""

    def __init__(self) -> None:
        self.loaded_at = time.time()

    def predict(self, instances: Any) -> Any:
        # Echo a single deterministic prediction per input.
        if isinstance(instances, list):
            return [{"class_index": idx % 4, "confidence": 0.9} for idx in range(len(instances))]
        return {"class_index": 0, "confidence": 0.9}


class MLTask(app.Task):
    """Base task class that caches a model instance per worker process."""

    abstract = True
    _model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            logger.info("Loading ML model into worker process...")
            self._model = self._load_model()
            logger.info("Model loaded (type=%s)", type(self._model).__name__)
        return self._model

    @staticmethod
    def _load_model() -> Any:
        """Override in subclasses to load the real artifact.

        Default behaviour: if ``ML_MODEL_PATH`` is set, attempt to load
        a torch JIT model; otherwise fall back to the deterministic
        stub. Either way the task code path is identical.
        """
        if MODEL_PATH:
            try:
                import torch  # noqa: F401 (imported only when needed)

                return torch.jit.load(MODEL_PATH)
            except Exception:
                logger.exception(
                    "Failed to load torch model from %s; using stub.", MODEL_PATH
                )
        return _StubModel()

    def on_failure(
        self,
        exc: BaseException,
        task_id: str,
        args: Any,
        kwargs: Dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.error("Task failed id=%s exc=%s", task_id, exc)

    def on_success(self, retval: Any, task_id: str, args: Any, kwargs: Dict[str, Any]) -> None:
        logger.info("Task succeeded id=%s", task_id)

    def on_retry(
        self,
        exc: BaseException,
        task_id: str,
        args: Any,
        kwargs: Dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.warning("Task retrying id=%s exc=%s", task_id, exc)


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    # Run with: celery -A celery_app worker --loglevel=info
    app.start()
