# Step-by-Step Implementation Guide: FastAPI Fundamentals

## Overview

Build production-ready APIs with FastAPI! Learn request handling, validation, documentation, middleware, dependency injection, and async patterns.

**Time**: 2-3 hours | **Difficulty**: Beginner to Intermediate

---

## Learning Objectives

✅ Create FastAPI applications
✅ Define route handlers
✅ Implement request validation
✅ Use Pydantic models
✅ Add middleware and dependencies
✅ Generate API documentation
✅ Handle errors gracefully
✅ Test API endpoints

---

## Phase 1: Basic FastAPI Application

### Installation

```bash
# Create project
mkdir fastapi-ml-api
cd fastapi-ml-api
python3 -m venv venv
source venv/bin/activate

# Install FastAPI and Uvicorn
pip install fastapi uvicorn[standard] pydantic python-multipart
```

### Hello World API

```python
# main.py
from fastapi import FastAPI

app = FastAPI(
    title="ML Inference API",
    description="API for machine learning model inference",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to ML Inference API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run with: uvicorn main:app --reload
```

```bash
# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/
curl http://localhost:8000/health

# View auto-generated docs
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

---

## Phase 2: Request and Response Models

### Pydantic Models

```python
# models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    """Input data for model prediction"""
    features: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Input features for iris classification"
    )
    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use"
    )

    @validator('features')
    def validate_features(cls, v):
        if any(x < 0 for x in v):
            raise ValueError('Features must be non-negative')
        return v

    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2],
                "model_version": "v1.0"
            }
        }

class PredictionResponse(BaseModel):
    """Prediction result"""
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "prediction": "setosa",
                "confidence": 0.95,
                "model_version": "v1.0",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### Using Models in Routes

```python
# main.py
from fastapi import FastAPI, HTTPException
from models import PredictionRequest, PredictionResponse, ErrorResponse
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Train simple model (for demo)
iris = load_iris()
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(iris.data, iris.target)
class_names = iris.target_names

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def predict(request: PredictionRequest):
    """Make prediction using ML model"""
    try:
        # Prepare input
        features = np.array(request.features).reshape(1, -1)

        # Predict
        prediction_idx = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction_idx])

        return PredictionResponse(
            prediction=class_names[prediction_idx],
            confidence=confidence,
            model_version=request.model_version or "v1.0"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
```

---

## Phase 3: Path and Query Parameters

### Path Parameters

```python
@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model information by ID"""
    models = {
        "iris-v1": {"name": "Iris Classifier", "version": "1.0"},
        "iris-v2": {"name": "Iris Classifier", "version": "2.0"}
    }

    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return models[model_id]

@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: int):
    """Get prediction by ID"""
    # In real app, query from database
    return {
        "id": prediction_id,
        "result": "setosa",
        "timestamp": "2024-01-01T12:00:00Z"
    }
```

### Query Parameters

```python
from typing import Optional

@app.get("/predictions")
async def list_predictions(
    skip: int = 0,
    limit: int = 10,
    model_version: Optional[str] = None,
    min_confidence: float = 0.0
):
    """List predictions with filtering"""
    # In real app, query from database
    predictions = [
        {"id": i, "result": "setosa", "confidence": 0.95}
        for i in range(skip, skip + limit)
    ]

    # Filter by confidence
    predictions = [p for p in predictions if p["confidence"] >= min_confidence]

    return {
        "total": len(predictions),
        "skip": skip,
        "limit": limit,
        "predictions": predictions
    }
```

---

## Phase 4: Dependency Injection

### Simple Dependencies

```python
# dependencies.py
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key"""
    valid_keys = {"secret-key-1", "secret-key-2"}
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

async def get_current_user(x_user_id: str = Header(...)):
    """Get current user from header"""
    # In real app, validate user ID and fetch user data
    return {"user_id": x_user_id, "username": f"user_{x_user_id}"}
```

### Using Dependencies

```python
from fastapi import Depends
from dependencies import verify_api_key, get_current_user

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict_with_auth(request: PredictionRequest):
    """Protected prediction endpoint"""
    # ... prediction logic ...
    pass

@app.get("/user/predictions")
async def get_user_predictions(user: dict = Depends(get_current_user)):
    """Get predictions for current user"""
    return {
        "user_id": user["user_id"],
        "predictions": []  # Query user's predictions
    }
```

### Database Dependency

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

SQLALCHEMY_DATABASE_URL = "sqlite:///./ml_api.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Usage
from fastapi import Depends
from sqlalchemy.orm import Session

@app.get("/predictions/db")
async def get_predictions_from_db(db: Session = Depends(get_db)):
    """Query predictions from database"""
    # predictions = db.query(Prediction).all()
    return {"predictions": []}
```

---

## Phase 5: Middleware

### CORS Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://app.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Custom Middleware

```python
import time
from fastapi import Request
import structlog

logger = structlog.get_logger()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()

    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        client=request.client.host
    )

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration
    )

    response.headers["X-Process-Time"] = str(duration)
    return response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response
```

---

## Phase 6: Error Handling

### Custom Exception Handlers

```python
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

class ModelNotFoundError(Exception):
    """Model not found exception"""
    def __init__(self, model_id: str):
        self.model_id = model_id

@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Model not found",
            "detail": f"Model {exc.model_id} does not exist",
            "model_id": exc.model_id
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "errors": exc.errors()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error("internal_error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )
```

---

## Phase 7: Background Tasks

```python
from fastapi import BackgroundTasks
import asyncio

async def log_prediction(prediction_id: int, result: str):
    """Log prediction to database (background task)"""
    await asyncio.sleep(1)  # Simulate database write
    logger.info("prediction_logged", prediction_id=prediction_id, result=result)

@app.post("/predict/async")
async def predict_with_logging(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Prediction with background logging"""
    # Make prediction
    features = np.array(request.features).reshape(1, -1)
    prediction_idx = model.predict(features)[0]
    result = class_names[prediction_idx]

    # Schedule background task
    prediction_id = 123  # In real app, generate ID
    background_tasks.add_task(log_prediction, prediction_id, result)

    return {"prediction": result, "id": prediction_id}
```

---

## Phase 8: File Uploads

```python
from fastapi import File, UploadFile
import pandas as pd
from io import StringIO

@app.post("/predict/batch")
async def batch_predict(file: UploadFile = File(...)):
    """Batch prediction from CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Read CSV
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Validate columns
    required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_cols}")

    # Predict
    features = df[required_cols].values
    predictions = model.predict(features)
    results = [class_names[pred] for pred in predictions]

    return {
        "total": len(results),
        "predictions": results
    }
```

---

## Phase 9: WebSockets

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/predictions")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time prediction updates via WebSocket"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            features = data['features']

            # Make prediction
            prediction = model.predict([features])[0]
            result = class_names[prediction]

            # Send result
            await websocket.send_json({
                "prediction": result,
                "timestamp": datetime.utcnow().isoformat()
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

## Phase 10: Testing

### Test Setup

```python
# test_main.py
from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid():
    response = client.post(
        "/predict",
        json={
            "features": [5.1, 3.5, 1.4, 0.2],
            "model_version": "v1.0"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1

def test_predict_invalid_features():
    response = client.post(
        "/predict",
        json={
            "features": [5.1, 3.5],  # Too few features
            "model_version": "v1.0"
        }
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_batch_predict():
    csv_content = """sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2"""

    response = client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_content, "text/csv")}
    )
    assert response.status_code == 200
    assert response.json()["total"] == 2
```

---

## Complete Example Application

```python
# app/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.models import PredictionRequest, PredictionResponse
from app.ml_service import MLService
from app.dependencies import verify_api_key
import structlog

logger = structlog.get_logger()

app = FastAPI(
    title="ML Inference API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML service
ml_service = MLService()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    await ml_service.load_models()
    logger.info("api_started", models_loaded=ml_service.model_count)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": ml_service.model_count}

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Make prediction"""
    try:
        result = await ml_service.predict(request.features, request.model_version)
        background_tasks.add_task(ml_service.log_prediction, result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Best Practices

✅ Use Pydantic models for validation
✅ Implement proper error handling
✅ Add API documentation with examples
✅ Use dependency injection
✅ Implement authentication/authorization
✅ Add request/response logging
✅ Use async/await for I/O operations
✅ Add health check endpoints
✅ Version your API (e.g., /api/v1/)
✅ Write comprehensive tests

---

**FastAPI Fundamentals mastered!** 🚀

**Next Exercise**: Model Serving
