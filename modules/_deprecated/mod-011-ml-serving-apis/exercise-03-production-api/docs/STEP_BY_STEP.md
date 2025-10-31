# Step-by-Step Implementation Guide: Production API

## Overview

Deploy production-ready ML APIs! Learn authentication, rate limiting, observability, error handling, Docker deployment, and Kubernetes orchestration.

**Time**: 3-4 hours | **Difficulty**: Intermediate to Advanced

---

## Learning Objectives

✅ Implement authentication and authorization
✅ Add rate limiting
✅ Configure logging and tracing
✅ Handle errors robustly
✅ Deploy with Docker
✅ Orchestrate with Kubernetes
✅ Implement health checks
✅ Set up monitoring and alerts

---

## Phase 1: Authentication

### API Key Authentication

```python
# auth.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Optional
import secrets
import hashlib

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

class APIKeyAuth:
    """API key authentication"""

    def __init__(self):
        # In production, store in database
        self.valid_keys = {
            hashlib.sha256(b"secret-key-1").hexdigest(): {"user": "user1", "tier": "premium"},
            hashlib.sha256(b"secret-key-2").hexdigest(): {"user": "user2", "tier": "free"},
        }

    def verify_key(self, api_key: str) -> dict:
        """Verify API key and return user info"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash not in self.valid_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing API key"
            )

        return self.valid_keys[key_hash]

auth_service = APIKeyAuth()

async def get_current_user(api_key: str = Security(API_KEY_HEADER)):
    """Dependency to get current user"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key required"
        )

    return auth_service.verify_key(api_key)

# Usage
from fastapi import Depends

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    user: dict = Depends(get_current_user)
):
    """Protected prediction endpoint"""
    logger.info("prediction_request", user=user["user"])
    # ... prediction logic ...
```

### JWT Authentication

```python
# jwt_auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel

SECRET_KEY = "your-secret-key"  # Use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

class TokenData(BaseModel):
    user_id: str
    tier: str

def create_access_token(data: dict):
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        tier: str = payload.get("tier", "free")

        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        return TokenData(user_id=user_id, tier=tier)

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/token")
async def login(username: str, password: str):
    """Login and get JWT token"""
    # Verify credentials (implement your logic)
    if username != "testuser" or password != "testpass":
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": username, "tier": "premium"})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/predict/jwt")
async def predict_jwt(
    request: PredictionRequest,
    token_data: TokenData = Depends(verify_token)
):
    """JWT-protected prediction"""
    logger.info("jwt_prediction", user=token_data.user_id, tier=token_data.tier)
    # ... prediction logic ...
```

---

## Phase 2: Rate Limiting

### Simple Rate Limiter

```python
# rate_limiter.py
from fastapi import HTTPException, Request
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.buckets = defaultdict(list)

    async def check_rate_limit(self, key: str):
        """Check if request is within rate limit"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        # Clean old requests
        self.buckets[key] = [
            req_time for req_time in self.buckets[key]
            if req_time > minute_ago
        ]

        # Check limit
        if len(self.buckets[key]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later."
            )

        # Add current request
        self.buckets[key].append(now)

rate_limiter = RateLimiter(requests_per_minute=100)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Use API key or IP as identifier
    api_key = request.headers.get("X-API-Key", request.client.host)

    await rate_limiter.check_rate_limit(api_key)

    response = await call_next(request)
    return response
```

### Tier-based Rate Limiting

```python
# tiered_rate_limiter.py
from enum import Enum

class Tier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class TieredRateLimiter:
    """Rate limiter with different tiers"""

    TIER_LIMITS = {
        Tier.FREE: 10,
        Tier.PREMIUM: 100,
        Tier.ENTERPRISE: 1000,
    }

    def __init__(self):
        self.buckets = defaultdict(list)

    async def check_rate_limit(self, user_id: str, tier: Tier):
        """Check tier-based rate limit"""
        limit = self.TIER_LIMITS[tier]
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        # Clean old requests
        self.buckets[user_id] = [
            req_time for req_time in self.buckets[user_id]
            if req_time > minute_ago
        ]

        if len(self.buckets[user_id]) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for {tier} tier ({limit}/min)"
            )

        self.buckets[user_id].append(now)

tiered_limiter = TieredRateLimiter()

@app.post("/predict/tiered")
async def predict_tiered(
    request: PredictionRequest,
    user: dict = Depends(get_current_user)
):
    """Prediction with tier-based rate limiting"""
    await tiered_limiter.check_rate_limit(user["user"], Tier(user["tier"]))
    # ... prediction logic ...
```

---

## Phase 3: Structured Logging

### Configure Logging

```python
# logging_config.py
import structlog
import logging
from datetime import datetime

def configure_logging():
    """Configure structured logging"""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

configure_logging()
logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.utcnow()

    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        client=request.client.host
    )

    response = await call_next(request)

    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_seconds=duration
    )

    return response

@app.post("/predict/logged")
async def predict_with_logging(request: PredictionRequest):
    """Prediction with detailed logging"""
    logger.info(
        "prediction_started",
        model_version=request.model_version,
        feature_count=len(request.features)
    )

    try:
        result = await predict(request)

        logger.info(
            "prediction_completed",
            prediction=result.prediction,
            confidence=result.confidence
        )

        return result

    except Exception as e:
        logger.error(
            "prediction_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise
```

---

## Phase 4: Distributed Tracing

### OpenTelemetry Integration

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Configure tracer
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

tracer = trace.get_tracer(__name__)

@app.post("/predict/traced")
async def predict_with_tracing(request: PredictionRequest):
    """Prediction with distributed tracing"""
    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("model.version", request.model_version)
        span.set_attribute("features.count", len(request.features))

        # Load model (traced)
        with tracer.start_as_current_span("load_model"):
            model = model_loader.load_sklearn_model("iris", request.model_version)

        # Inference (traced)
        with tracer.start_as_current_span("inference"):
            features = np.array(request.features).reshape(1, -1)
            prediction = model.predict(features)[0]

        span.set_attribute("prediction.value", float(prediction))

        return {"prediction": float(prediction)}
```

---

## Phase 5: Error Handling

### Comprehensive Error Handling

```python
# error_handlers.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import traceback

class ModelError(Exception):
    """Base exception for model errors"""
    pass

class ModelNotFoundError(ModelError):
    """Model not found"""
    pass

class InferenceError(ModelError):
    """Inference failed"""
    pass

@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    logger.error("model_not_found", error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "ModelNotFound",
            "message": str(exc),
            "path": request.url.path
        }
    )

@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    logger.error("inference_error", error=str(exc), traceback=traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InferenceFailed",
            "message": "An error occurred during inference",
            "path": request.url.path
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Invalid request parameters",
            "details": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), traceback=traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred"
        }
    )
```

---

## Phase 6: Health Checks

### Comprehensive Health Checks

```python
# health.py
from fastapi import status
from pydantic import BaseModel
from typing import Dict
import asyncio

class HealthStatus(BaseModel):
    status: str
    version: str
    checks: Dict[str, dict]

class HealthChecker:
    """Health check service"""

    async def check_database(self) -> dict:
        """Check database connection"""
        try:
            # Check DB connection
            # db.execute("SELECT 1")
            return {"status": "healthy", "latency_ms": 5}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_model_loaded(self) -> dict:
        """Check if models are loaded"""
        try:
            if len(model_loader.models) > 0:
                return {
                    "status": "healthy",
                    "models_loaded": len(model_loader.models)
                }
            return {"status": "unhealthy", "models_loaded": 0}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_redis(self) -> dict:
        """Check Redis connection"""
        try:
            # cache.redis_client.ping()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

health_checker = HealthChecker()

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Basic health check"""
    return HealthStatus(
        status="healthy",
        version="1.0.0",
        checks={}
    )

@app.get("/health/detailed", response_model=HealthStatus)
async def detailed_health_check():
    """Detailed health check with dependencies"""
    checks = await asyncio.gather(
        health_checker.check_database(),
        health_checker.check_model_loaded(),
        health_checker.check_redis(),
        return_exceptions=True
    )

    all_healthy = all(
        isinstance(check, dict) and check.get("status") == "healthy"
        for check in checks
    )

    return HealthStatus(
        status="healthy" if all_healthy else "degraded",
        version="1.0.0",
        checks={
            "database": checks[0],
            "models": checks[1],
            "cache": checks[2]
        }
    )

@app.get("/readiness")
async def readiness_check():
    """Kubernetes readiness probe"""
    # Check if app is ready to serve traffic
    if len(model_loader.models) == 0:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "reason": "Models not loaded"}
        )
    return {"ready": True}

@app.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe"""
    # Check if app is alive
    return {"alive": True}
```

---

## Phase 7: Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Multi-stage Build

```dockerfile
# Dockerfile.multistage
# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/mlapi
    depends_on:
      - redis
      - db
    volumes:
      - ./models:/app/models
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mlapi
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Phase 8: Kubernetes Deployment

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  labels:
    app: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: api
        image: ml-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ml-api-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /liveness
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readiness
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: ml-models-pvc
```

### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### HorizontalPodAutoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Best Practices

✅ Implement authentication and authorization
✅ Add rate limiting per tier
✅ Use structured logging
✅ Implement distributed tracing
✅ Handle all errors gracefully
✅ Add comprehensive health checks
✅ Use multi-stage Docker builds
✅ Configure resource limits
✅ Implement autoscaling
✅ Monitor all metrics

---

**Production API mastered!** 🚀

**Next Exercise**: Performance Optimization
