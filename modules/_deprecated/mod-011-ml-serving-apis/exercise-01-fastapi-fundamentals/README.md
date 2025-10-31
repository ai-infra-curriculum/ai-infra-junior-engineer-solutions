# Exercise 01: FastAPI Fundamentals

## Overview

Build a production-ready ML prediction API using FastAPI. This exercise covers FastAPI basics, async programming, request validation, error handling, and testing.

## Learning Objectives

- Understand FastAPI framework architecture
- Implement async API endpoints
- Use Pydantic for data validation
- Handle errors gracefully
- Write comprehensive API tests
- Generate interactive API documentation
- Containerize FastAPI applications

## Prerequisites

- Python 3.8+
- Basic understanding of HTTP and REST APIs
- Familiarity with async programming concepts
- Docker installed

## Project Structure

```
exercise-01-fastapi-fundamentals/
├── src/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py           # Pydantic models
│   │   └── ml_model.py          # ML model wrapper
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py            # Health check endpoints
│   │   └── predictions.py       # Prediction endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py # Business logic
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_health.py
│   └── test_predictions.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Locally

```bash
# Start the API
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Access interactive docs
open http://localhost:8000/docs
```

### 3. Run with Docker

```bash
# Build and run
docker-compose up --build

# Test the API
curl http://localhost:8000/health
```

### 4. Run Tests

```bash
pytest tests/ -v --cov=src
```

## API Endpoints

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Get API Info

```http
GET /api/v1/info
```

Response:
```json
{
  "name": "ML Prediction API",
  "version": "1.0.0",
  "model_type": "random_forest",
  "model_version": "v1.0.0"
}
```

### Single Prediction

```http
POST /api/v1/predict
Content-Type: application/json

{
  "features": [1.0, 2.0, 3.0, 4.0, 5.0]
}
```

Response:
```json
{
  "prediction": 1,
  "probability": [0.23, 0.77],
  "model_version": "v1.0.0",
  "timestamp": "2025-10-24T12:00:00Z"
}
```

### Batch Prediction

```http
POST /api/v1/predict/batch
Content-Type: application/json

{
  "instances": [
    {"features": [1.0, 2.0, 3.0, 4.0, 5.0]},
    {"features": [2.0, 3.0, 4.0, 5.0, 6.0]}
  ]
}
```

Response:
```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability": [0.23, 0.77]
    },
    {
      "prediction": 0,
      "probability": [0.85, 0.15]
    }
  ],
  "batch_size": 2,
  "model_version": "v1.0.0",
  "timestamp": "2025-10-24T12:00:00Z"
}
```

## FastAPI Features Demonstrated

### 1. Async Endpoints

```python
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Async prediction endpoint."""
    result = await prediction_service.predict_async(request.features)
    return result
```

### 2. Pydantic Validation

```python
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=5,
        max_items=5,
        description="Input features for prediction"
    )

    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0, 5.0]
            }
        }
```

### 3. Dependency Injection

```python
def get_model() -> MLModel:
    """Dependency to get ML model."""
    return model_instance

@router.post("/predict")
async def predict(
    request: PredictionRequest,
    model: MLModel = Depends(get_model)
):
    return model.predict(request.features)
```

### 4. Error Handling

```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )
```

### 5. Background Tasks

```python
@router.post("/predict")
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    result = await predict_async(request)

    # Log prediction in background
    background_tasks.add_task(log_prediction, request, result)

    return result
```

### 6. Middleware

```python
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## Testing

### Unit Tests

```python
def test_predict_endpoint(client):
    """Test single prediction endpoint."""
    response = client.post(
        "/api/v1/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_batch_prediction(client):
    """Test batch prediction endpoint."""
    response = client.post(
        "/api/v1/predict/batch",
        json={
            "instances": [
                {"features": [1.0, 2.0, 3.0, 4.0, 5.0]},
                {"features": [2.0, 3.0, 4.0, 5.0, 6.0]}
            ]
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
```

### Load Testing

```python
# Using locust
from locust import HttpUser, task, between

class MLAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        self.client.post(
            "/api/v1/predict",
            json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/model.pkl
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
```

## Performance Tips

### 1. Use Async Everywhere

```python
# Good - Async
@router.get("/data")
async def get_data():
    data = await fetch_from_database()
    return data

# Bad - Blocking
@router.get("/data")
def get_data():
    data = fetch_from_database()  # Blocks event loop
    return data
```

### 2. Enable Response Caching

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

@router.get("/cached")
@cache(expire=60)
async def cached_endpoint():
    return {"data": "This is cached for 60 seconds"}
```

### 3. Optimize JSON Serialization

```python
# Use orjson for faster JSON
from fastapi.responses import ORJSONResponse

app = FastAPI(default_response_class=ORJSONResponse)
```

### 4. Connection Pooling

```python
# Use connection pools for databases
from databases import Database

database = Database("postgresql://user:pass@localhost/db")

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
```

## Best Practices

### 1. API Versioning

```python
# Version in URL
app.include_router(
    predictions_router,
    prefix="/api/v1",
    tags=["predictions"]
)
```

### 2. Request Validation

```python
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_items=1, max_items=100)

    @validator('features')
    def validate_features(cls, v):
        if any(x < 0 for x in v):
            raise ValueError('All features must be non-negative')
        return v
```

### 3. Error Responses

```python
class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            timestamp=datetime.utcnow()
        ).dict()
    )
```

### 4. Logging

```python
import logging

logger = logging.getLogger(__name__)

@router.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(f"Received prediction request: {request.features}")
    try:
        result = await predict_async(request)
        logger.info(f"Prediction successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
```

## Common Pitfalls

### ❌ Blocking I/O Operations

```python
# Don't do this
@router.get("/slow")
async def slow_endpoint():
    time.sleep(5)  # Blocks entire event loop!
    return {"status": "done"}
```

### ✅ Use Async Sleep

```python
@router.get("/slow")
async def slow_endpoint():
    await asyncio.sleep(5)  # Doesn't block
    return {"status": "done"}
```

### ❌ No Error Handling

```python
# Don't do this
@router.post("/predict")
async def predict(request: PredictionRequest):
    return model.predict(request.features)  # What if model.predict fails?
```

### ✅ Proper Error Handling

```python
@router.post("/predict")
async def predict(request: PredictionRequest):
    try:
        result = model.predict(request.features)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Async Programming in Python](https://docs.python.org/3/library/asyncio.html)

## Next Steps

After completing this exercise:

1. ✅ Understand FastAPI framework
2. ✅ Build async API endpoints
3. ✅ Validate requests with Pydantic
4. ✅ Handle errors gracefully
5. ✅ Test FastAPI applications
6. ✅ Deploy with Docker

**Move on to**: Exercise 02 - ML Model Serving Frameworks
