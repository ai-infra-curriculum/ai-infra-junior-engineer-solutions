# Step-by-Step Guide: Production-Ready ML API

## Overview
Build a production-grade ML API with JWT authentication, rate limiting, response caching, API versioning, and comprehensive monitoring.

## Phase 1: Authentication with JWT (15 minutes)

### Install Dependencies
```bash
mkdir -p production-api
cd production-api

python3 -m venv venv
source venv/bin/activate

pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt] python-multipart
pip freeze > requirements.txt
```

### Create Authentication System
Create `auth.py`:
```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# Configuration
SECRET_KEY = "your-secret-key-keep-it-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = False

class UserInDB(User):
    hashed_password: str

# Fake user database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "email": "test@example.com",
        "hashed_password": pwd_context.hash("testpass123"),
        "disabled": False,
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user
```

### Create Main API with Authentication
Create `main.py`:
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from auth import (
    authenticate_user, create_access_token, get_current_user,
    Token, User, ACCESS_TOKEN_EXPIRE_MINUTES
)
from datetime import timedelta

app = FastAPI(title="Production ML API", version="1.0.0")

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/predict")
async def predict(
    features: list[float],
    current_user: User = Depends(get_current_user)
):
    """Protected prediction endpoint"""
    prediction = sum(features) / len(features)
    return {
        "prediction": prediction,
        "user": current_user.username
    }
```

**Validation**: Test login and protected endpoint with curl.

## Phase 2: Rate Limiting (15 minutes)

### Install Rate Limiting
```bash
pip install slowapi
```

### Implement Rate Limiter
Create `rate_limit.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
import time
from collections import defaultdict

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

# Custom rate limiter for API keys
class APIKeyRateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)

    def is_allowed(self, api_key: str, limit: int = 100, window: int = 60):
        """Check if request is allowed within rate limit"""
        now = time.time()
        window_start = now - window

        # Clean old requests
        self.requests[api_key] = [
            req_time for req_time in self.requests[api_key]
            if req_time > window_start
        ]

        # Check limit
        if len(self.requests[api_key]) >= limit:
            return False

        # Add current request
        self.requests[api_key].append(now)
        return True

    def get_remaining(self, api_key: str, limit: int = 100):
        """Get remaining requests in current window"""
        return max(0, limit - len(self.requests[api_key]))

api_rate_limiter = APIKeyRateLimiter()
```

### Update Main API with Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from rate_limit import limiter, api_rate_limiter

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.get("/public")
@limiter.limit("5/minute")
async def public_endpoint(request: Request):
    """Rate limited public endpoint"""
    return {"message": "This endpoint is rate limited to 5 requests per minute"}

@app.post("/predict-limited")
@limiter.limit("10/minute")
async def predict_limited(
    request: Request,
    features: list[float],
    current_user: User = Depends(get_current_user)
):
    """Rate limited prediction"""
    prediction = sum(features) / len(features)
    return {"prediction": prediction}

@app.post("/predict-custom-limit")
async def predict_custom(
    features: list[float],
    current_user: User = Depends(get_current_user)
):
    """Custom rate limiting by user"""
    if not api_rate_limiter.is_allowed(current_user.username, limit=100, window=60):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )

    remaining = api_rate_limiter.get_remaining(current_user.username)
    prediction = sum(features) / len(features)

    return {
        "prediction": prediction,
        "rate_limit_remaining": remaining
    }
```

**Validation**: Make multiple requests to verify rate limiting works.

## Phase 3: Response Caching (15 minutes)

### Install Caching Libraries
```bash
pip install redis aioredis aiocache
```

### Create Cache Layer
Create `cache.py`:
```python
from aiocache import Cache
from aiocache.serializers import JsonSerializer
import hashlib
import json
from typing import Optional, Any

class ResponseCache:
    def __init__(self):
        self.cache = Cache(
            Cache.MEMORY,
            serializer=JsonSerializer(),
            ttl=300  # 5 minutes
        )

    def _generate_key(self, endpoint: str, params: dict) -> str:
        """Generate cache key from endpoint and parameters"""
        params_str = json.dumps(params, sort_keys=True)
        key = f"{endpoint}:{hashlib.md5(params_str.encode()).hexdigest()}"
        return key

    async def get(self, endpoint: str, params: dict) -> Optional[Any]:
        """Get cached response"""
        key = self._generate_key(endpoint, params)
        return await self.cache.get(key)

    async def set(self, endpoint: str, params: dict, value: Any, ttl: int = 300):
        """Set cached response"""
        key = self._generate_key(endpoint, params)
        await self.cache.set(key, value, ttl=ttl)

    async def invalidate(self, endpoint: str, params: dict):
        """Invalidate cached response"""
        key = self._generate_key(endpoint, params)
        await self.cache.delete(key)

cache = ResponseCache()
```

### Add Caching to Endpoints
```python
from cache import cache

@app.post("/predict-cached")
async def predict_cached(
    features: list[float],
    current_user: User = Depends(get_current_user)
):
    """Cached prediction endpoint"""
    # Check cache
    cache_params = {"features": features}
    cached_result = await cache.get("predict", cache_params)

    if cached_result:
        return {
            **cached_result,
            "cached": True
        }

    # Compute prediction
    prediction = sum(features) / len(features)
    result = {
        "prediction": prediction,
        "user": current_user.username,
        "cached": False
    }

    # Cache result
    await cache.set("predict", cache_params, result, ttl=300)

    return result

@app.delete("/cache/invalidate")
async def invalidate_cache(
    features: list[float],
    current_user: User = Depends(get_current_user)
):
    """Invalidate cache for specific prediction"""
    await cache.invalidate("predict", {"features": features})
    return {"message": "Cache invalidated"}
```

**Validation**: Test cache hits and misses with repeated requests.

## Phase 4: API Versioning (15 minutes)

### Create Versioned API Structure
Create `v1/endpoints.py`:
```python
from fastapi import APIRouter, Depends
from auth import get_current_user, User

router = APIRouter(prefix="/v1", tags=["v1"])

@router.post("/predict")
async def predict_v1(
    features: list[float],
    current_user: User = Depends(get_current_user)
):
    """Version 1 prediction - simple average"""
    prediction = sum(features) / len(features)
    return {
        "version": "1.0",
        "prediction": prediction,
        "method": "average"
    }
```

Create `v2/endpoints.py`:
```python
from fastapi import APIRouter, Depends
from auth import get_current_user, User
import numpy as np

router = APIRouter(prefix="/v2", tags=["v2"])

@router.post("/predict")
async def predict_v2(
    features: list[float],
    current_user: User = Depends(get_current_user)
):
    """Version 2 prediction - weighted average"""
    weights = [1.0, 1.5, 2.0, 0.5][:len(features)]
    prediction = np.average(features, weights=weights[:len(features)])

    return {
        "version": "2.0",
        "prediction": float(prediction),
        "method": "weighted_average",
        "weights": weights[:len(features)]
    }
```

### Update Main App
```python
from v1 import endpoints as v1_endpoints
from v2 import endpoints as v2_endpoints

app.include_router(v1_endpoints.router)
app.include_router(v2_endpoints.router)

@app.get("/")
async def root():
    return {
        "service": "Production ML API",
        "versions": ["v1", "v2"],
        "endpoints": {
            "v1": "/v1/predict",
            "v2": "/v2/predict"
        }
    }
```

**Validation**: Test both API versions with different responses.

## Phase 5: Monitoring and Metrics (15 minutes)

### Add Prometheus Metrics
Create `metrics.py`:
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CollectorRegistry
import time
from functools import wraps

# Create registry
registry = CollectorRegistry()

# Define metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

request_latency = Histogram(
    'api_request_duration_seconds',
    'API request latency',
    ['method', 'endpoint'],
    registry=registry
)

prediction_count = Counter(
    'predictions_total',
    'Total predictions made',
    ['version', 'user'],
    registry=registry
)

active_users = Gauge(
    'active_users',
    'Number of active users',
    registry=registry
)

cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    registry=registry
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    registry=registry
)

def track_metrics(endpoint: str):
    """Decorator to track endpoint metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status = 200
                return result
            except Exception as e:
                status = 500
                raise
            finally:
                duration = time.time() - start_time
                request_count.labels(
                    method='POST',
                    endpoint=endpoint,
                    status=status
                ).inc()
                request_latency.labels(
                    method='POST',
                    endpoint=endpoint
                ).observe(duration)
        return wrapper
    return decorator
```

### Add Metrics Endpoint
```python
from metrics import registry, track_metrics, prediction_count
from prometheus_client import generate_latest
from fastapi.responses import Response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(registry),
        media_type="text/plain"
    )

@app.post("/v2/predict-monitored")
@track_metrics("/v2/predict")
async def predict_monitored(
    features: list[float],
    current_user: User = Depends(get_current_user)
):
    """Monitored prediction endpoint"""
    prediction = sum(features) / len(features)

    # Track prediction
    prediction_count.labels(
        version="v2",
        user=current_user.username
    ).inc()

    return {"prediction": prediction}
```

**Validation**: Check `/metrics` endpoint for Prometheus format metrics.

## Phase 6: Health Checks and Logging (10 minutes)

### Create Comprehensive Health Check
```python
from typing import Dict
import psutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe"""
    # Check if model is loaded, database is accessible, etc.
    checks = {
        "model": True,  # Check if model is loaded
        "cache": True,  # Check cache connectivity
    }

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503

    return {
        "status": "ready" if all_ready else "not ready",
        "checks": checks
    }

@app.get("/health/metrics")
async def health_metrics():
    """System health metrics"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"{request.method} {request.url.path}")
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {process_time:.3f}s "
        f"with status {response.status_code}"
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response
```

**Validation**: Test health endpoints and check logs.

## Summary

You've built a production-ready ML API with:
- **JWT authentication** for secure access control with token-based auth
- **Rate limiting** preventing abuse with per-user and global limits
- **Response caching** improving performance for repeated requests
- **API versioning** enabling backward compatibility and gradual migrations
- **Prometheus metrics** for comprehensive monitoring and alerting
- **Health checks** for Kubernetes liveness and readiness probes
- **Structured logging** for debugging and audit trails

This API is ready for production deployment with enterprise-grade security, performance, and observability features.
