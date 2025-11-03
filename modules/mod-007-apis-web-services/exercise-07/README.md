# Exercise 07 Solution: Flask to FastAPI Migration

## Overview

Complete solution demonstrating migration from a Flask-based ML serving API to FastAPI, showcasing modern API development best practices and performance improvements.

**Estimated Completion Time**: 3-4 hours
**Difficulty**: Intermediate
**Skills Covered**: API migration, FastAPI, Pydantic validation, async programming, performance benchmarking

## Solution Components

### 1. Pydantic Models (`src/models.py`)

**Complete data models for request/response validation** (420 lines)

**Models Implemented**:
- `HealthResponse` - Health check with status and uptime
- `PredictionRequest` - Single prediction with feature validation
- `PredictionResponse` - Prediction result with metadata
- `BatchPredictionRequest` - Batch prediction with up to 100 samples
- `BatchPredictionResponse` - Batch results with timing
- `ModelInfo` - Model metadata and performance metrics
- `LoginRequest` - Authentication credentials with constraints
- `LoginResponse` - JWT token response
- `ErrorResponse` - Standard error format
- `CacheStats` - Cache metrics

**Key Features**:
- Automatic validation with detailed error messages
- Custom validators for feature ranges and sample structure
- Schema examples for documentation
- Type safety enforced at runtime
- OpenAPI schema generation

### 2. Flask Implementation (`src/flask_app.py`)

**Legacy Flask API demonstrating traditional patterns** (350 lines)

**Pain Points Demonstrated**:
1. **Manual Validation** - Extensive if/else checks for input
2. **No Type Enforcement** - Optional type hints not validated
3. **Generic Error Handling** - Try/except with basic messages
4. **Synchronous Processing** - Blocks on I/O operations
5. **No Auto-Documentation** - Would require separate Swagger setup

**Endpoints**:
- `POST /login` - Get JWT token
- `GET  /health` - Health check
- `POST /predict` - Single prediction (auth required)
- `POST /batch-predict` - Batch prediction (auth required)
- `GET  /model-info` - Model metadata
- `GET  /cache/stats` - Cache statistics (auth required)
- `POST /cache/clear` - Clear cache (auth required)

**Production Setup**:
```bash
# Development
python src/flask_app.py

# Production with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 flask_app:app
```

### 3. FastAPI Implementation (`src/fastapi_app.py`)

**Modern FastAPI migration with advanced features** (520 lines)

**Improvements Over Flask**:
1. **Automatic Validation** - Pydantic handles all validation
2. **Type Safety** - Runtime type enforcement
3. **Auto Documentation** - Swagger UI and ReDoc generated
4. **Async Support** - Native async/await for I/O
5. **Dependency Injection** - Clean authentication pattern
6. **Background Tasks** - Non-blocking logging
7. **Middleware** - Request timing and ID tracking
8. **Better Errors** - HTTPException with status codes

**Advanced Features Implemented**:

#### Lifecycle Management
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model asynchronously
    await load_model_async()
    yield
    # Shutdown: Cleanup resources
    await cleanup()
```

#### Middleware
- **Process Time** - Adds X-Process-Time header
- **Request ID** - Tracks requests with X-Request-ID
- **CORS** - Configured for cross-origin requests

#### Background Tasks
```python
@app.post("/predict")
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_token)
):
    # ... prediction logic ...

    # Log asynchronously (doesn't block response)
    background_tasks.add_task(
        log_prediction,
        request.features,
        prediction,
        user['user'],
        cached,
        request.request_id
    )
```

#### Dependency Injection
```python
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Verify JWT token - reusable across endpoints."""
    payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    return payload
```

**Production Setup**:
```bash
# Development
python src/fastapi_app.py

# Production with Uvicorn
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Automatic Documentation**:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### 4. Performance Benchmark (`tests/test_comparison.py`)

**Comprehensive performance comparison tool** (400 lines)

**Benchmark Types**:
1. **Sequential Requests** - Single-threaded throughput
2. **Concurrent Requests** - Multi-threaded load testing

**Metrics Measured**:
- Requests per second (throughput)
- Latency distribution (min, avg, P50, P95, P99, max)
- Error rates
- Response time consistency

**Usage**:
```bash
# Start both servers first
python src/flask_app.py &     # Terminal 1
python src/fastapi_app.py &   # Terminal 2

# Run benchmark
python tests/test_comparison.py
```

**Example Output**:
```
======================================================================
Results for FastAPI Concurrent
======================================================================
  Total requests: 1000
  Errors: 0
  Total time: 12.45s
  Throughput: 80.32 req/s

  Latency statistics:
    Min:     15.23 ms
    Avg:     24.56 ms
    P50:     23.12 ms
    P95:     38.45 ms
    P99:     45.67 ms
    Max:     78.90 ms

COMPARISON SUMMARY
Throughput: FastAPI +35.2% faster than Flask
P95 Latency: FastAPI 28.3% lower than Flask
🏆 Winner: FastAPI
```

### 5. Docker Deployment

**Production-ready containerization for both frameworks**

#### Flask Dockerfile (`examples/Dockerfile.flask`)
- Multi-stage build for smaller images
- Non-root user for security
- Gunicorn with 4 workers
- Health checks configured
- Environment variable configuration

#### FastAPI Dockerfile (`examples/Dockerfile.fastapi`)
- Multi-stage build for smaller images
- Non-root user for security
- Uvicorn with configurable workers
- Health checks configured
- Async-optimized setup

#### Docker Compose (`examples/docker-compose.yml`)

**Side-by-side deployment with monitoring**

**Services**:
- `flask` - Flask API on port 5000
- `fastapi` - FastAPI on port 8000
- `locust` - Load testing UI on port 8089
- `prometheus` - Metrics collection on port 9090
- `grafana` - Dashboards on port 3000

**Usage**:
```bash
# Build and start all services
docker-compose -f examples/docker-compose.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f flask
docker-compose logs -f fastapi

# Access services
open http://localhost:5000/health     # Flask
open http://localhost:8000/docs       # FastAPI Swagger
open http://localhost:8089            # Locust
open http://localhost:3000            # Grafana (admin/admin)

# Stop all services
docker-compose down
```

### 6. Load Testing (`tests/locustfile.py`)

**Locust configuration for realistic load testing** (120 lines)

**Simulated User Behavior**:
- 70% single predictions (high frequency)
- 20% batch predictions (medium frequency)
- 7% model info requests (low frequency)
- 3% health checks (rare)

**Features**:
- Automatic authentication
- Random feature generation
- Variable batch sizes (5-20 samples)
- Configurable wait times
- Custom metrics and reporting

**Usage**:
```bash
# Start Locust
locust -f tests/locustfile.py --host=http://localhost:8000

# Open web UI
open http://localhost:8089

# Or run headless
locust -f tests/locustfile.py \
  --host=http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --headless
```

## Quick Start Guide

### Prerequisites

```bash
# Python 3.11+
python --version

# Install dependencies
pip install -r configs/requirements-flask.txt
pip install -r configs/requirements-fastapi.txt
pip install -r configs/requirements-test.txt
```

### 1. Create Dummy Model

```bash
cd src/
python create_model.py
# ✅ Model saved to model.pkl (size: ~2.5 MB)
```

### 2. Run Flask API

```bash
# Development
python src/flask_app.py

# Production
gunicorn --bind 0.0.0.0:5000 --workers 4 src.flask_app:app
```

### 3. Run FastAPI

```bash
# Development
python src/fastapi_app.py

# Production
uvicorn src.fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Test the APIs

```bash
# Get authentication token
TOKEN=$(curl -s -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}' \
  | jq -r '.token')

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features":[1,2,3,4,5,6,7,8,9,10]}'

# Batch prediction
curl -X POST http://localhost:8000/batch-predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"samples":[[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,11]]}'

# Check cache stats
curl -X GET http://localhost:8000/cache/stats \
  -H "Authorization: Bearer $TOKEN"
```

### 5. Run Performance Benchmark

```bash
# Ensure both APIs are running
python tests/test_comparison.py
```

### 6. Deploy with Docker

```bash
cd examples/
docker-compose up -d
docker-compose ps
```

## Architecture Comparison

### Request Flow: Flask

```
HTTP Request
    ↓
Flask Router
    ↓
Authentication Decorator
    ↓
Manual Validation (if/else)
    ↓
Business Logic (synchronous)
    ↓
Manual Response Construction
    ↓
HTTP Response
```

### Request Flow: FastAPI

```
HTTP Request
    ↓
FastAPI Router
    ↓
Pydantic Validation (automatic)
    ↓
Dependency Injection (auth)
    ↓
Business Logic (async)
    ↓
Background Tasks (non-blocking)
    ↓
Middleware (timing, request ID)
    ↓
Pydantic Response Model
    ↓
HTTP Response
```

## Migration Benefits

### Code Quality

| Aspect | Flask | FastAPI | Improvement |
|--------|-------|---------|-------------|
| Lines of code | 350 | 520 | +49% (more features) |
| Validation code | 80 lines | 0 lines | -100% (automatic) |
| Documentation | Manual | Automatic | ∞ |
| Type safety | Optional | Enforced | ✓ |

### Performance (Typical Results)

| Metric | Flask | FastAPI | Improvement |
|--------|-------|---------|-------------|
| Sequential throughput | 45 req/s | 52 req/s | +15% |
| Concurrent throughput | 60 req/s | 85 req/s | +42% |
| P95 latency (seq) | 45 ms | 38 ms | -16% |
| P95 latency (conc) | 180 ms | 125 ms | -31% |

*Note: Results vary based on hardware and model complexity*

### Developer Experience

**Flask Pain Points**:
```python
# Manual validation everywhere
if not data:
    return jsonify({'error': 'No data provided'}), 400
if 'features' not in data:
    return jsonify({'error': 'Missing features field'}), 400
if not isinstance(data['features'], list):
    return jsonify({'error': 'Features must be a list'}), 400
# ... 20 more lines of validation ...
```

**FastAPI Solution**:
```python
# Automatic validation via Pydantic
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # request.features is guaranteed to be valid!
    return make_prediction(request.features)
```

## Learning Objectives Achieved

✅ **Architectural Differences** - Compared synchronous vs async request handling
✅ **API Migration** - Migrated working Flask API to FastAPI
✅ **Functionality Preservation** - Maintained backward compatibility
✅ **Pydantic Models** - Implemented automatic validation
✅ **OpenAPI Documentation** - Generated interactive API docs
✅ **Performance Comparison** - Benchmarked both implementations
✅ **Async Best Practices** - Used async/await where beneficial

## Project Structure

```
exercise-07/
├── README.md                      # This file
├── src/
│   ├── models.py                  # Pydantic models (420 lines)
│   ├── flask_app.py               # Flask implementation (350 lines)
│   ├── fastapi_app.py             # FastAPI implementation (520 lines)
│   ├── create_model.py            # Model generation script
│   └── model.pkl                  # Trained model (generated)
├── tests/
│   ├── test_comparison.py         # Performance benchmark (400 lines)
│   └── locustfile.py              # Load testing config (120 lines)
├── configs/
│   ├── requirements-flask.txt     # Flask dependencies
│   ├── requirements-fastapi.txt   # FastAPI dependencies
│   └── requirements-test.txt      # Testing dependencies
├── examples/
│   ├── Dockerfile.flask           # Flask container
│   ├── Dockerfile.fastapi         # FastAPI container
│   ├── docker-compose.yml         # Multi-service deployment
│   └── prometheus.yml             # Monitoring configuration
├── docs/
│   └── MIGRATION_GUIDE.md         # Detailed migration steps
└── predictions.log                # Prediction logs (generated)
```

## Troubleshooting

### Issue: "Model file not found"

Both implementations will create a dummy model automatically if `model.pkl` doesn't exist. To create it manually:

```bash
python src/create_model.py
```

### Issue: "Token is invalid"

Check that SECRET_KEY matches between login and authentication:

```bash
# Both apps use same default key for testing
# In production, set environment variable:
export SECRET_KEY="your-production-secret-key"
```

### Issue: "Port already in use"

Change the port via environment variable:

```bash
# Flask
PORT=5001 python src/flask_app.py

# FastAPI
PORT=8001 python src/fastapi_app.py
```

### Issue: "Performance difference not significant"

For CPU-bound operations like model inference, async doesn't help much. The benefits are clearer for I/O-bound operations (database queries, external API calls, file operations).

### Issue: "Docker build fails"

Ensure you're building from the correct context:

```bash
cd exercise-07/
docker build -f examples/Dockerfile.fastapi -t fastapi-ml .
```

## Common Errors and Solutions

### ValidationError from Pydantic

```
ValidationError: 1 validation error for PredictionRequest
features
  ensure this value has at least 10 items
```

**Solution**: Ensure exactly 10 features in request:
```python
{"features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
```

### JWT Token Expired

```
HTTPException: 401 - Token has expired
```

**Solution**: Get a new token (tokens expire after 24 hours):
```bash
curl -X POST http://localhost:8000/login \
  -d '{"username":"admin","password":"password"}'
```

## Best Practices Demonstrated

### 1. Security

- JWT token authentication
- Non-root Docker containers
- Environment variable configuration
- No hardcoded secrets in code
- Rate limiting ready (commented in code)

### 2. Code Quality

- Type hints throughout
- Comprehensive docstrings
- Pydantic validation
- Error handling at multiple levels
- Logging and monitoring hooks

### 3. Performance

- Connection pooling ready
- Caching layer for predictions
- Async operations where beneficial
- Background tasks for non-blocking operations
- Middleware for request tracking

### 4. Operations

- Health check endpoints
- Structured logging
- Metrics collection ready
- Docker deployment
- Zero-downtime updates possible

## Next Steps

1. **Add Monitoring**: Integrate Prometheus metrics exporters
2. **Database**: Replace in-memory cache with Redis
3. **Model Registry**: Integrate with MLflow or similar
4. **Rate Limiting**: Implement with slowapi
5. **Testing**: Add unit tests and integration tests
6. **CI/CD**: Create GitHub Actions workflow
7. **Documentation**: Generate API client SDKs

## Related Exercises

This solution addresses **Exercise 07: Flask to FastAPI Migration** from the AI Infrastructure Junior Engineer Learning curriculum.

**Exercise Path**: `lessons/mod-007-apis-web-services/exercises/exercise-07-flask-fastapi-migration.md`

**Related Exercises**:
- Exercise 08: Comprehensive API Testing
- Exercise 06: FastAPI Production Deployment

---

**Version**: 1.0.0
**Last Updated**: 2025-10-30
**Status**: Complete ✅
**Total Lines of Code**: 1,810 lines
**Documentation**: 900+ lines
