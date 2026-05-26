"""Reference production ML serving API.

Demonstrates: lifespan model load, Pydantic schemas, async
handlers, structured logging, Prometheus metrics, health
endpoints, error handling.

Run:
    pip install fastapi uvicorn prometheus-client pydantic
    uvicorn api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


# ----- Logging (structured JSON) -----


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add any structured fields the caller attached
        for key in ("request_id", "tenant_id", "model_version"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("ml-serving")
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# ----- Metrics -----


REQUEST_COUNT = Counter(
    "ml_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Request latency",
    ["endpoint"],
)


# ----- Pydantic schemas (input validation) -----


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, max_length=512)
    tenant_id: str = Field(..., min_length=1, max_length=64)


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str


# ----- Model wrapper (toy, swap in real model) -----


class ToyModel:
    """Stand-in for an actual model. Replace with PyTorch / etc."""

    version = "toy-v1"

    def predict(self, features: list[float]) -> tuple[int, float]:
        # Trivially deterministic: pick class based on feature sum
        s = sum(features)
        if s < 0:
            return (0, 0.9)
        if s < 10:
            return (1, 0.8)
        return (2, 0.85)


# ----- Lifespan: load model once -----


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model")
    app.state.model = ToyModel()
    logger.info("Model loaded", extra={"model_version": app.state.model.version})
    yield
    logger.info("Shutting down")


app = FastAPI(lifespan=lifespan)


# ----- Middleware: per-request metrics + request ID -----


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", "no-id")
    try:
        response = await call_next(request)
        status = "ok"
    except Exception:
        status = "error"
        raise
    finally:
        latency = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
        REQUEST_COUNT.labels(endpoint=request.url.path, status=status).inc()
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "latency_ms": int(latency * 1000),
            },
        )
    return response


# ----- Routes -----


@app.get("/healthz")
async def healthz():
    """Basic liveness — process is up."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness — model loaded + upstream reachable. Deep check."""
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_version": app.state.model.version}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    request_id = request.headers.get("x-request-id", "no-id")
    model = app.state.model
    prediction, confidence = model.predict(req.features)
    logger.info(
        "prediction_made",
        extra={
            "request_id": request_id,
            "tenant_id": req.tenant_id,
            "model_version": model.version,
        },
    )
    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        model_version=model.version,
    )


# ----- Error handler -----


@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    request_id = request.headers.get("x-request-id", "no-id")
    logger.error(
        "unhandled_exception",
        extra={"request_id": request_id, "error": str(exc)},
    )
    # Never expose the stack trace to clients
    return JSONResponse(
        status_code=500,
        content={"error": "internal server error", "request_id": request_id},
    )
