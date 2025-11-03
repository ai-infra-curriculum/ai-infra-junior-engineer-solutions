# Flask to FastAPI Migration - Implementation Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Setup](#phase-1-setup)
4. [Phase 2: Create Pydantic Models](#phase-2-create-pydantic-models)
5. [Phase 3: Migrate Core Endpoints](#phase-3-migrate-core-endpoints)
6. [Phase 4: Add Advanced Features](#phase-4-add-advanced-features)
7. [Phase 5: Testing & Benchmarking](#phase-5-testing--benchmarking)
8. [Phase 6: Deployment](#phase-6-deployment)
9. [Migration Checklist](#migration-checklist)

---

## Overview

This guide provides step-by-step instructions for migrating a Flask ML serving API to FastAPI, demonstrating improvements in performance, type safety, and developer experience.

**Time Required**: 3-4 hours
**Difficulty**: Intermediate
**Outcome**: Fully functional FastAPI implementation with better performance and automatic documentation

---

## Prerequisites

### Knowledge Requirements

- Python 3.11+ basics
- Flask fundamentals
- HTTP/REST API concepts
- JWT authentication basics
- Docker basics (optional)

### Tools Required

```bash
# Check Python version (3.11+ required)
python --version

# Install tools
pip install pip-tools  # For dependency management
pip install black isort mypy  # For code quality
```

---

## Phase 1: Setup

### Step 1.1: Create Project Structure

```bash
mkdir exercise-07
cd exercise-07

# Create directories
mkdir -p src tests configs examples docs

# Create empty files
touch src/{models.py,flask_app.py,fastapi_app.py,create_model.py}
touch tests/{test_comparison.py,locustfile.py}
touch configs/{requirements-flask.txt,requirements-fastapi.txt,requirements-test.txt}
touch examples/{Dockerfile.flask,Dockerfile.fastapi,docker-compose.yml}
touch README.md docs/IMPLEMENTATION_GUIDE.md
```

### Step 1.2: Setup Dependencies

Create `configs/requirements-flask.txt`:
```
Flask==3.0.0
gunicorn==21.2.0
PyJWT==2.8.0
numpy==1.26.2
scikit-learn==1.3.2
python-dotenv==1.0.0
```

Create `configs/requirements-fastapi.txt`:
```
fastapi==0.108.0
pydantic==2.5.3
uvicorn[standard]==0.25.0
PyJWT==2.8.0
python-multipart==0.0.6
numpy==1.26.2
scikit-learn==1.3.2
python-dotenv==1.0.0
```

Install both:
```bash
pip install -r configs/requirements-flask.txt
pip install -r configs/requirements-fastapi.txt
```

### Step 1.3: Create ML Model

Create `src/create_model.py` to generate a dummy model:

```python
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

def create_and_save_model(output_path='model.pkl'):
    # Generate synthetic dataset (10 features)
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=10,
        random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model saved to {output_path}")

if __name__ == '__main__':
    create_and_save_model()
```

Run it:
```bash
cd src/
python create_model.py
# ✅ Model saved to model.pkl
```

---

## Phase 2: Create Pydantic Models

### Step 2.1: Define Base Models

Create `src/models.py`:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    uptime_seconds: Optional[float] = None
    model_loaded: bool = True
```

**Why**: Defines structure for health check endpoint with automatic validation.

### Step 2.2: Add Prediction Models

```python
class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: List[float] = Field(
        ...,
        min_items=10,
        max_items=10,
        description="List of 10 numerical features"
    )
    request_id: Optional[str] = None

    @validator('features')
    def validate_features(cls, v):
        """Ensure features are in valid range."""
        for i, feature in enumerate(v):
            if not isinstance(feature, (int, float)):
                raise ValueError(f'Feature {i} must be a number')
            if not (-1000 <= feature <= 1000):
                raise ValueError(f'Feature {i} out of range [-1000, 1000]')
        return v

    class Config:
        schema_extra = {
            "example": {
                "features": [1.2, 3.4, 5.6, 7.8, 9.0, 2.3, 4.5, 6.7, 8.9, 0.1]
            }
        }

class PredictionResponse(BaseModel):
    """Single prediction response."""
    prediction: float
    cached: bool = False
    model_version: str
    request_id: Optional[str] = None
```

**Why**: Automatic validation eliminates manual if/else checks. Example in schema generates documentation.

### Step 2.3: Add Batch Prediction Models

```python
class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    samples: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of feature vectors (max 100 samples)"
    )
    request_id: Optional[str] = None

    @validator('samples')
    def validate_samples(cls, v):
        """Validate each sample."""
        for i, sample in enumerate(v):
            if not isinstance(sample, list):
                raise ValueError(f'Sample {i} must be a list')
            if len(sample) != 10:
                raise ValueError(f'Sample {i} must have 10 features')
            for j, feature in enumerate(sample):
                if not (-1000 <= feature <= 1000):
                    raise ValueError(f'Sample {i}, feature {j} out of range')
        return v

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[float]
    count: int
    model_version: str
    request_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
```

### Step 2.4: Add Authentication Models

```python
class LoginRequest(BaseModel):
    """Login credentials."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

class LoginResponse(BaseModel):
    """Login response with token."""
    token: str
    token_type: str = "bearer"
    expires_in: int = 86400

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
```

**Complete models.py**: See `src/models.py` in solution for full implementation with all models.

---

## Phase 3: Migrate Core Endpoints

### Step 3.1: Create FastAPI Application

Create `src/fastapi_app.py`:

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pickle
import numpy as np
import jwt
from datetime import datetime, timedelta
import time
import os

# Import models
from models import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    LoginRequest,
    LoginResponse
)

# Create app
app = FastAPI(
    title="ML Model Serving API",
    description="FastAPI-based ML model serving",
    version="2.0.0"
)

# Configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
ALGORITHM = "HS256"

# Security
security = HTTPBearer()

# Global variables
model = None
prediction_cache = {}
start_time = time.time()
```

**Why**: FastAPI app with configuration, similar to Flask but with more metadata.

### Step 3.2: Add Model Loading

```python
# Load model at startup
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

**Better approach with lifecycle**:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded")
    yield
    # Shutdown
    print("✅ Cleanup complete")

app = FastAPI(..., lifespan=lifespan)
```

### Step 3.3: Implement Authentication

```python
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Verify JWT token (dependency)."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
```

**Why**: Dependency injection is cleaner than decorators. Reusable across endpoints.

### Step 3.4: Migrate Health Check

Flask version:
```python
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })
```

FastAPI version:
```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - start_time
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime,
        model_loaded=model is not None
    )
```

**Key differences**:
- `@app.get` instead of `@app.route`
- `response_model=HealthResponse` for automatic validation
- `async def` for async support
- Return Pydantic model directly (no jsonify needed)

### Step 3.5: Migrate Prediction Endpoint

Flask version (manual validation):
```python
@app.route('/predict', methods=['POST'])
@token_required
def predict():
    data = request.get_json()

    # Manual validation
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    if 'features' not in data:
        return jsonify({'error': 'Missing features field'}), 400
    if not isinstance(data['features'], list):
        return jsonify({'error': 'Features must be a list'}), 400
    if len(data['features']) != 10:
        return jsonify({'error': 'Expected 10 features'}), 400

    # ... prediction logic ...
```

FastAPI version (automatic validation):
```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    user: dict = Depends(verify_token)
):
    """Make prediction on input features."""
    # No validation needed! Pydantic handles it.

    # Check cache
    feature_key = str(request.features)
    if feature_key in prediction_cache:
        return PredictionResponse(
            prediction=prediction_cache[feature_key],
            cached=True,
            model_version="1.0",
            request_id=request.request_id
        )

    # Make prediction
    features_array = np.array([request.features])
    prediction = float(model.predict(features_array)[0])

    # Cache result
    prediction_cache[feature_key] = prediction

    return PredictionResponse(
        prediction=prediction,
        cached=False,
        model_version="1.0",
        request_id=request.request_id
    )
```

**Key improvements**:
- Zero validation code (handled by Pydantic)
- Dependency injection for auth
- Type-safe (request.features guaranteed to be valid)
- Return Pydantic model directly

### Step 3.6: Migrate Login Endpoint

FastAPI version:
```python
@app.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest):
    """Authenticate and receive JWT token."""
    # No manual validation needed!

    if credentials.username == "admin" and credentials.password == "password":
        token = jwt.encode(
            {
                "user": credentials.username,
                "exp": datetime.utcnow() + timedelta(hours=24)
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        return LoginResponse(
            token=token,
            token_type="bearer",
            expires_in=86400
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )
```

---

## Phase 4: Add Advanced Features

### Step 4.1: Add Background Tasks

```python
from fastapi import BackgroundTasks

def log_prediction(features: list, prediction: float, user: str):
    """Log prediction to file (runs in background)."""
    with open('predictions.log', 'a') as f:
        timestamp = datetime.utcnow().isoformat()
        f.write(f"{timestamp},{user},{features},{prediction}\n")

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,  # Add this parameter
    user: dict = Depends(verify_token)
):
    # ... prediction logic ...

    # Add background task (doesn't block response)
    background_tasks.add_task(
        log_prediction,
        request.features,
        prediction,
        user['user']
    )

    return PredictionResponse(...)
```

**Why**: Logging happens asynchronously, reducing response time.

### Step 4.2: Add Middleware

```python
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response
```

**Why**: Automatically tracks request duration for all endpoints.

### Step 4.3: Add CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Step 4.4: Enhance Documentation

```python
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],  # Group in docs
    summary="Make single prediction",
    description="Make a prediction on a single feature vector",
    responses={
        200: {"description": "Successful prediction"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"}
    }
)
async def predict(...):
    """
    Make a prediction on input features.

    **Authentication required** - Include JWT token in Authorization header.

    Args:
        request: Prediction request with features
        user: Authenticated user (from token)

    Returns:
        Prediction response with value and metadata
    """
```

**Complete fastapi_app.py**: See `src/fastapi_app.py` in solution for full implementation.

---

## Phase 5: Testing & Benchmarking

### Step 5.1: Manual Testing

```bash
# Start FastAPI
python src/fastapi_app.py

# In another terminal:

# Get token
TOKEN=$(curl -s -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}' \
  | jq -r '.token')

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features":[1,2,3,4,5,6,7,8,9,10]}'
```

### Step 5.2: Test Validation

```bash
# Should fail: missing features
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'

# Should fail: wrong number of features
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features":[1,2,3]}'

# Should fail: feature out of range
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features":[1,2,3,4,5,6,7,8,9,2000]}'
```

### Step 5.3: Performance Benchmark

See `tests/test_comparison.py` for complete implementation.

```bash
# Start both servers
python src/flask_app.py &
python src/fastapi_app.py &

# Run benchmark
python tests/test_comparison.py
```

Expected results:
- FastAPI: 20-40% better throughput under load
- FastAPI: 15-30% lower P95 latency
- FastAPI: Better consistency (lower max latency)

---

## Phase 6: Deployment

### Step 6.1: Dockerize FastAPI

Create `examples/Dockerfile.fastapi`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY configs/requirements-fastapi.txt .
RUN pip install --no-cache-dir -r requirements-fastapi.txt

COPY src/fastapi_app.py src/models.py src/model.pkl ./

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:
```bash
docker build -f examples/Dockerfile.fastapi -t fastapi-ml .
docker run -p 8000:8000 fastapi-ml
```

### Step 6.2: Docker Compose

See `examples/docker-compose.yml` for complete multi-service deployment.

```bash
docker-compose -f examples/docker-compose.yml up -d
```

---

## Migration Checklist

### Pre-Migration

- [ ] Document all Flask endpoints
- [ ] Create test suite for existing API
- [ ] Benchmark current performance
- [ ] Review authentication mechanism
- [ ] Plan backward compatibility strategy

### Implementation

- [ ] Create Pydantic models for all request/response types
- [ ] Implement custom validators for complex validation
- [ ] Migrate endpoints one-by-one
- [ ] Add dependency injection for authentication
- [ ] Implement async endpoints where beneficial
- [ ] Add background tasks for non-blocking operations
- [ ] Configure middleware (CORS, timing, request ID)
- [ ] Set up lifecycle management (startup/shutdown)

### Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Validation tests comprehensive
- [ ] Performance benchmarks meet targets
- [ ] Backward compatibility verified
- [ ] Load testing completed
- [ ] Error handling tested

### Documentation

- [ ] API documentation generated (Swagger/ReDoc)
- [ ] README updated
- [ ] Migration guide written
- [ ] Example requests documented
- [ ] Troubleshooting guide created

### Deployment

- [ ] Dockerfile created and tested
- [ ] Docker Compose configuration ready
- [ ] Environment variables configured
- [ ] Health checks working
- [ ] Logging configured
- [ ] Monitoring ready (optional)

### Production Rollout

- [ ] Deploy to staging environment
- [ ] Run smoke tests
- [ ] Performance test in staging
- [ ] Monitor metrics (latency, error rate)
- [ ] Gradual rollout (canary/blue-green)
- [ ] Monitor production metrics
- [ ] Document any issues

---

## Common Migration Patterns

### Pattern 1: Convert Flask Decorators to FastAPI Dependencies

Flask:
```python
@token_required
def endpoint():
    pass
```

FastAPI:
```python
async def endpoint(user: dict = Depends(verify_token)):
    pass
```

### Pattern 2: Convert Manual Validation to Pydantic

Flask:
```python
if not data:
    return jsonify({'error': 'No data'}), 400
if 'field' not in data:
    return jsonify({'error': 'Missing field'}), 400
```

FastAPI:
```python
class Request(BaseModel):
    field: str  # Automatically validated
```

### Pattern 3: Convert Synchronous to Async

Flask:
```python
def endpoint():
    result = expensive_operation()
    return jsonify(result)
```

FastAPI:
```python
async def endpoint():
    result = await expensive_operation_async()
    return result
```

---

## Performance Optimization Tips

1. **Use Async for I/O**: Database queries, external APIs, file operations
2. **Keep CPU Work Sync**: Model inference, data processing
3. **Enable Caching**: For repeated identical requests
4. **Use Background Tasks**: For logging, notifications, non-critical operations
5. **Configure Workers**: Match CPU cores for optimal throughput
6. **Profile Bottlenecks**: Use `cProfile` or `py-spy`

---

## Next Steps After Migration

1. **Add Unit Tests**: Test individual functions
2. **Add Integration Tests**: Test full request/response cycle
3. **Set Up CI/CD**: Automate testing and deployment
4. **Add Monitoring**: Prometheus metrics, application logs
5. **Implement Rate Limiting**: Protect against abuse
6. **Add Database**: Replace in-memory cache with Redis
7. **API Versioning**: Plan for v2, v3, etc.
8. **Generate Client SDKs**: Use OpenAPI spec

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [JWT.io](https://jwt.io/) - JWT debugger
- [Swagger UI](https://swagger.io/tools/swagger-ui/)

---

**Guide Version**: 1.0.0
**Last Updated**: 2025-10-30
**Estimated Time**: 3-4 hours
**Difficulty**: Intermediate
