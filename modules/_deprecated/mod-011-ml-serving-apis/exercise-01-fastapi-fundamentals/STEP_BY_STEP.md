# Step-by-Step Guide: FastAPI Fundamentals

## Overview
Master FastAPI basics including routing, request validation, async operations, and automatic API documentation for building production ML APIs.

## Phase 1: Basic FastAPI Setup (10 minutes)

### Install FastAPI and Dependencies
```bash
# Create project directory
mkdir -p fastapi-fundamentals
cd fastapi-fundamentals

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install FastAPI and server
pip install "fastapi[all]==0.104.1" uvicorn[standard]==0.24.0
pip freeze > requirements.txt
```

### Create First API
Create `main.py`:
```python
from fastapi import FastAPI

app = FastAPI(
    title="ML API Fundamentals",
    description="Learning FastAPI for ML serving",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run and Test
```bash
# Start server
uvicorn main:app --reload

# In another terminal, test endpoints
curl http://localhost:8000/
curl http://localhost:8000/health

# View automatic docs
# Open browser: http://localhost:8000/docs (Swagger UI)
# Open browser: http://localhost:8000/redoc (ReDoc)
```

**Validation**: Verify API responds and docs are accessible.

## Phase 2: Request Validation with Pydantic (15 minutes)

### Create Data Models
Create `models.py`:
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[float] = Field(..., min_items=1, max_items=100)
    model_version: Optional[str] = Field(default="latest", max_length=20)

    @validator('features')
    def validate_features(cls, v):
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError('All features must be numeric')
        return v

    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.5, 3.7, 4.2],
                "model_version": "v1.0"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)
```

### Update API with Validation
Update `main.py`:
```python
from fastapi import FastAPI, HTTPException, status
from models import PredictionRequest, PredictionResponse, ErrorResponse
import random

app = FastAPI(title="ML API Fundamentals", version="1.0.0")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction based on input features"""
    try:
        # Simulate prediction
        prediction = sum(request.features) / len(request.features)
        confidence = random.uniform(0.7, 0.99)

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=request.model_version
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/batch-predict", response_model=List[PredictionResponse])
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    results = []
    for req in requests:
        prediction = sum(req.features) / len(req.features)
        results.append(PredictionResponse(
            prediction=prediction,
            confidence=random.uniform(0.7, 0.99),
            model_version=req.model_version
        ))
    return results
```

**Validation**: Test with `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [1.0, 2.0, 3.0]}'`

## Phase 3: Path and Query Parameters (15 minutes)

### Add Path Parameters
```python
from fastapi import Path, Query

@app.get("/models/{model_id}")
async def get_model_info(
    model_id: str = Path(..., description="Model identifier", min_length=1)
):
    """Get information about a specific model"""
    return {
        "model_id": model_id,
        "status": "active",
        "version": "1.0.0",
        "accuracy": 0.95
    }

@app.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: int = Path(..., description="Prediction ID", ge=1)
):
    """Retrieve a specific prediction by ID"""
    return {
        "prediction_id": prediction_id,
        "result": random.uniform(0, 1),
        "timestamp": "2024-01-01T00:00:00"
    }
```

### Add Query Parameters
```python
from typing import Optional

@app.get("/search")
async def search_predictions(
    model_version: Optional[str] = Query(None, max_length=20),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    max_results: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    """Search predictions with filtering"""
    return {
        "filters": {
            "model_version": model_version,
            "min_confidence": min_confidence
        },
        "pagination": {
            "skip": skip,
            "limit": max_results
        },
        "results": [
            {"id": i, "confidence": random.uniform(min_confidence, 1.0)}
            for i in range(max_results)
        ]
    }
```

**Validation**: Test different parameter combinations via Swagger UI.

## Phase 4: Async Operations (15 minutes)

### Create Async Database Simulation
Create `database.py`:
```python
import asyncio
from typing import Dict, List

class AsyncDatabase:
    def __init__(self):
        self.predictions = {}

    async def save_prediction(self, prediction_id: int, data: Dict):
        """Simulate async database write"""
        await asyncio.sleep(0.1)  # Simulate I/O
        self.predictions[prediction_id] = data
        return prediction_id

    async def get_prediction(self, prediction_id: int) -> Dict:
        """Simulate async database read"""
        await asyncio.sleep(0.05)  # Simulate I/O
        return self.predictions.get(prediction_id)

    async def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """Get recent predictions"""
        await asyncio.sleep(0.1)
        return list(self.predictions.values())[-limit:]

# Global database instance
db = AsyncDatabase()
```

### Update API with Async Operations
```python
from database import db
import asyncio

@app.post("/predict-async")
async def predict_async(request: PredictionRequest):
    """Async prediction with database storage"""
    # Simulate async prediction
    await asyncio.sleep(0.1)
    prediction = sum(request.features) / len(request.features)

    # Save to database
    prediction_id = len(db.predictions) + 1
    await db.save_prediction(prediction_id, {
        "prediction": prediction,
        "features": request.features,
        "model_version": request.model_version
    })

    return {
        "prediction_id": prediction_id,
        "prediction": prediction
    }

@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = Query(10, ge=1, le=100)):
    """Get recent predictions asynchronously"""
    predictions = await db.get_recent_predictions(limit)
    return {"count": len(predictions), "predictions": predictions}
```

**Validation**: Compare response times between sync and async endpoints.

## Phase 5: Dependency Injection (15 minutes)

### Create Dependencies
Create `dependencies.py`:
```python
from fastapi import Header, HTTPException, status
from typing import Optional

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from header"""
    if x_api_key != "secret-key-123":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key

async def get_current_user(x_user_id: Optional[str] = Header(None)):
    """Get current user from header"""
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID required"
        )
    return {"user_id": x_user_id, "role": "user"}

def get_db():
    """Database dependency"""
    from database import db
    try:
        yield db
    finally:
        pass  # Cleanup if needed
```

### Use Dependencies in Routes
```python
from fastapi import Depends
from dependencies import verify_api_key, get_current_user, get_db

@app.post("/secure-predict", dependencies=[Depends(verify_api_key)])
async def secure_predict(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user),
    database = Depends(get_db)
):
    """Secured prediction endpoint"""
    prediction = sum(request.features) / len(request.features)

    prediction_id = len(database.predictions) + 1
    await database.save_prediction(prediction_id, {
        "prediction": prediction,
        "user_id": current_user["user_id"]
    })

    return {
        "prediction": prediction,
        "user": current_user["user_id"]
    }
```

**Validation**: Test with and without required headers.

## Phase 6: Error Handling and Middleware (10 minutes)

### Create Custom Exception Handlers
```python
from fastapi import Request
from fastapi.responses import JSONResponse
import time

class PredictionError(Exception):
    def __init__(self, message: str, code: str):
        self.message = message
        self.code = code

@app.exception_handler(PredictionError)
async def prediction_exception_handler(request: Request, exc: PredictionError):
    return JSONResponse(
        status_code=422,
        content={
            "error": exc.code,
            "message": exc.message,
            "path": request.url.path
        }
    )

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    print(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response
```

### Add Startup and Shutdown Events
```python
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print("Starting up...")
    # Load models, connect to databases, etc.

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down...")
    # Close connections, save state, etc.
```

**Validation**: Check response headers for `X-Process-Time`.

## Summary

You've mastered FastAPI fundamentals including:
- **Basic routing** with automatic OpenAPI documentation generation
- **Pydantic validation** for request/response models with type safety
- **Path and query parameters** with validation constraints
- **Async operations** for improved I/O performance
- **Dependency injection** for authentication, database access, and code reuse
- **Error handling** with custom exceptions and middleware
- **Lifecycle events** for startup/shutdown resource management

These skills form the foundation for building production-grade ML serving APIs with automatic validation, documentation, and high performance.
