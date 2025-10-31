"""FastAPI application with async task processing.

This API provides endpoints for:
- Submitting async prediction tasks
- Checking task status
- Retrieving task results
- Batch processing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from celery.result import AsyncResult
import uvicorn

from .celery_app import app as celery_app
from .tasks import (
    predict_single,
    predict_batch,
    train_model,
    create_preprocessing_pipeline,
    create_batch_prediction_workflow,
)


# FastAPI app
app = FastAPI(
    title="Async ML Prediction API",
    description="ML API with async task processing using Celery",
    version="1.0.0",
)


# Request/Response models

class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: List[float] = Field(..., min_items=5, max_items=5)


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    instances: List[List[float]] = Field(..., min_items=1)
    batch_size: Optional[int] = Field(10, gt=0, le=100)


class TrainingRequest(BaseModel):
    """Model training request."""
    epochs: int = Field(10, gt=0, le=100)
    learning_rate: float = Field(0.001, gt=0)
    batch_size: int = Field(32, gt=0)


class TaskResponse(BaseModel):
    """Task submission response."""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Task status response."""
    task_id: str
    status: str
    result: Optional[Dict] = None
    progress: Optional[Dict] = None
    error: Optional[str] = None


# Endpoints

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Async ML Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "async_predict": "/api/v1/predict/async",
            "batch_predict": "/api/v1/predict/batch",
            "task_status": "/api/v1/tasks/{task_id}",
            "task_result": "/api/v1/tasks/{task_id}/result",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check Celery connection
    try:
        celery_status = celery_app.control.inspect().stats()
        celery_healthy = celery_status is not None
    except:
        celery_healthy = False

    return {
        "status": "healthy" if celery_healthy else "degraded",
        "celery": "connected" if celery_healthy else "disconnected",
    }


@app.post("/api/v1/predict/async", response_model=TaskResponse)
async def async_predict(request: PredictionRequest):
    """
    Submit async prediction task.

    Returns task ID immediately. Client can poll for result.

    Args:
        request: Prediction request with features

    Returns:
        Task ID and status
    """
    # Submit task to Celery
    task = predict_single.delay(request.features)

    return TaskResponse(
        task_id=task.id,
        status="pending",
        message="Prediction task submitted. Use task_id to check status."
    )


@app.post("/api/v1/predict/async/pipeline", response_model=TaskResponse)
async def async_predict_pipeline(request: PredictionRequest):
    """
    Submit async prediction with preprocessing pipeline.

    Chains preprocessing -> prediction tasks.

    Args:
        request: Prediction request

    Returns:
        Task ID
    """
    # Create preprocessing pipeline
    workflow = create_preprocessing_pipeline([request.features])

    # Submit workflow
    task = workflow.apply_async()

    return TaskResponse(
        task_id=task.id,
        status="pending",
        message="Preprocessing pipeline submitted."
    )


@app.post("/api/v1/predict/batch", response_model=TaskResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Submit batch prediction task.

    Processes multiple instances efficiently.

    Args:
        request: Batch prediction request

    Returns:
        Task ID
    """
    # Submit batch task
    task = predict_batch.delay(request.instances)

    return TaskResponse(
        task_id=task.id,
        status="pending",
        message=f"Batch prediction task submitted for {len(request.instances)} instances."
    )


@app.post("/api/v1/predict/batch/distributed", response_model=TaskResponse)
async def distributed_batch_predict(request: BatchPredictionRequest):
    """
    Submit distributed batch prediction.

    Splits large batch into smaller chunks processed in parallel.

    Args:
        request: Batch prediction request

    Returns:
        Task ID
    """
    # Create distributed workflow
    workflow = create_batch_prediction_workflow(
        request.instances,
        batch_size=request.batch_size
    )

    # Submit group
    task = workflow.apply_async()

    num_batches = len(request.instances) // request.batch_size + 1

    return TaskResponse(
        task_id=task.id,
        status="pending",
        message=f"Distributed batch prediction submitted ({num_batches} parallel tasks)."
    )


@app.post("/api/v1/train", response_model=TaskResponse)
async def train_async(request: TrainingRequest):
    """
    Submit async model training task.

    Long-running task with progress updates.

    Args:
        request: Training configuration

    Returns:
        Task ID
    """
    training_data = {
        "epochs": request.epochs,
        "learning_rate": request.learning_rate,
        "batch_size": request.batch_size,
    }

    task = train_model.delay(training_data)

    return TaskResponse(
        task_id=task.id,
        status="pending",
        message="Training task submitted. This may take several minutes."
    )


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get task status.

    Args:
        task_id: Celery task ID

    Returns:
        Task status and metadata
    """
    task_result = AsyncResult(task_id, app=celery_app)

    response = TaskStatusResponse(
        task_id=task_id,
        status=task_result.status,
    )

    if task_result.status == "PENDING":
        response.result = None
        response.progress = None

    elif task_result.status == "PROGRESS":
        response.result = None
        response.progress = task_result.info

    elif task_result.status == "SUCCESS":
        response.result = task_result.result
        response.progress = None

    elif task_result.status == "FAILURE":
        response.error = str(task_result.info)
        response.result = None

    return response


@app.get("/api/v1/tasks/{task_id}/result")
async def get_task_result(task_id: str, wait: bool = False):
    """
    Get task result.

    Args:
        task_id: Celery task ID
        wait: If True, wait for task completion (with timeout)

    Returns:
        Task result

    Raises:
        HTTPException: If task not found or failed
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if wait:
        # Wait up to 30 seconds
        try:
            result = task_result.get(timeout=30)
            return {"task_id": task_id, "status": "success", "result": result}
        except TimeoutError:
            raise HTTPException(
                status_code=408,
                detail="Task execution timeout"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Task failed: {str(e)}"
            )

    # Non-blocking check
    if task_result.status == "SUCCESS":
        return {"task_id": task_id, "status": "success", "result": task_result.result}
    elif task_result.status == "FAILURE":
        raise HTTPException(
            status_code=500,
            detail=f"Task failed: {str(task_result.info)}"
        )
    elif task_result.status == "PENDING":
        raise HTTPException(
            status_code=202,
            detail="Task is still pending"
        )
    else:
        raise HTTPException(
            status_code=202,
            detail=f"Task status: {task_result.status}"
        )


@app.delete("/api/v1/tasks/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a pending task.

    Args:
        task_id: Celery task ID

    Returns:
        Cancellation status
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.status in ["PENDING", "PROGRESS"]:
        celery_app.control.revoke(task_id, terminate=True)
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancellation requested"
        }
    else:
        return {
            "task_id": task_id,
            "status": task_result.status,
            "message": f"Cannot cancel task in {task_result.status} state"
        }


@app.get("/api/v1/workers")
async def get_workers():
    """
    Get active Celery workers.

    Returns:
        List of active workers
    """
    inspect = celery_app.control.inspect()

    active_workers = inspect.active()
    registered_tasks = inspect.registered()
    stats = inspect.stats()

    return {
        "active_workers": list(active_workers.keys()) if active_workers else [],
        "registered_tasks": registered_tasks,
        "stats": stats,
    }


@app.get("/api/v1/queue/stats")
async def get_queue_stats():
    """
    Get queue statistics.

    Returns:
        Queue depth and worker stats
    """
    inspect = celery_app.control.inspect()

    active = inspect.active()
    scheduled = inspect.scheduled()
    reserved = inspect.reserved()

    total_active = sum(len(tasks) for tasks in (active or {}).values())
    total_scheduled = sum(len(tasks) for tasks in (scheduled or {}).values())
    total_reserved = sum(len(tasks) for tasks in (reserved or {}).values())

    return {
        "active_tasks": total_active,
        "scheduled_tasks": total_scheduled,
        "reserved_tasks": total_reserved,
        "total_pending": total_active + total_scheduled + total_reserved,
    }


if __name__ == "__main__":
    # Run with: uvicorn api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
