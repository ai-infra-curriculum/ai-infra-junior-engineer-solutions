# Exercise 03: Production API Design

## Overview

Build a production-grade ML API with enterprise features: authentication, authorization, rate limiting, caching, API versioning, and security best practices. This exercise transforms a basic API into a production-ready service.

## Learning Objectives

- Implement JWT authentication
- Add OAuth2 authorization
- Configure rate limiting and throttling
- Implement caching strategies with Redis
- Design API versioning schemes
- Build API gateway patterns
- Handle security best practices
- Implement comprehensive error handling

## Prerequisites

- Completed Exercise 01 (FastAPI Fundamentals)
- Completed Exercise 02 (Model Serving Frameworks)
- Understanding of authentication concepts
- Redis installed (for caching)
- PostgreSQL (for user management)

## Project Structure

```
exercise-03-production-api/
├── src/
│   ├── main.py                      # Main application
│   ├── config.py                    # Configuration
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── jwt_handler.py          # JWT token management
│   │   ├── oauth2.py               # OAuth2 implementation
│   │   ├── api_key.py              # API key authentication
│   │   └── models.py               # User models
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py         # Rate limiting
│   │   ├── auth_middleware.py      # Auth middleware
│   │   ├── cors_middleware.py      # CORS handling
│   │   └── logging_middleware.py   # Request logging
│   ├── versioning/
│   │   ├── __init__.py
│   │   ├── v1/                     # API v1
│   │   │   └── routes.py
│   │   └── v2/                     # API v2
│   │       └── routes.py
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── load_balancer.py        # Load balancing
│   │   └── circuit_breaker.py      # Circuit breaker pattern
│   └── cache/
│       ├── __init__.py
│       └── redis_cache.py          # Redis caching
├── tests/
│   ├── test_auth.py
│   ├── test_rate_limiting.py
│   ├── test_caching.py
│   └── test_versioning.py
├── docker-compose.yml               # Full stack
├── requirements.txt
└── README.md
```

## Features

### 1. Authentication & Authorization

#### JWT Authentication
- Token-based authentication
- Refresh token mechanism
- Token expiration handling
- User registration and login

#### OAuth2
- OAuth2 Password flow
- Scopes and permissions
- Role-based access control (RBAC)

#### API Keys
- API key generation
- Key rotation
- Usage tracking

### 2. Rate Limiting

- Per-user rate limits
- Per-endpoint limits
- Token bucket algorithm
- Redis-backed rate limiting
- Custom rate limit headers

### 3. Caching

- Response caching
- Model prediction caching
- Cache invalidation strategies
- Redis integration
- TTL configuration

### 4. API Versioning

- URL versioning (`/api/v1/`, `/api/v2/`)
- Header versioning
- Query parameter versioning
- Deprecation strategies

### 5. Security

- Input validation
- SQL injection prevention
- XSS protection
- CORS configuration
- Security headers
- Request sanitization

## Quick Start

### 1. Start Services

```bash
# Start Redis, PostgreSQL, and API
docker-compose up -d

# Check services
docker-compose ps
```

### 2. Create User

```bash
# Register new user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "SecurePass123!"
  }'
```

### 3. Login and Get Token

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "SecurePass123!"
  }'

# Response:
{
  "access_token": "eyJhbGc...",
  "refresh_token": "eyJhbGc...",
  "token_type": "bearer"
}
```

### 4. Make Authenticated Request

```bash
# Predict with JWT token
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer eyJhbGc..." \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.0, 2.0, 3.0, 4.0, 5.0]
  }'
```

### 5. Use API Key

```bash
# Generate API key
curl -X POST http://localhost:8000/api/v1/auth/api-keys \
  -H "Authorization: Bearer eyJhbGc..." \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-api-key",
    "scopes": ["predict:read"]
  }'

# Use API key
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: sk_live_xxx..." \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.0, 2.0, 3.0, 4.0, 5.0]
  }'
```

## Authentication Implementation

### JWT Handler

```python
# src/auth/jwt_handler.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### OAuth2 Implementation

```python
# src/auth/oauth2.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    payload = verify_token(token)
    user_id = payload.get("sub")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
```

## Rate Limiting Implementation

### Redis-Based Rate Limiter

```python
# src/middleware/rate_limiter.py
import time
from fastapi import Request, HTTPException
import redis
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_client: redis.Redis):
        super().__init__(app)
        self.redis = redis_client
        self.default_limit = 100  # requests per minute
        self.window = 60  # seconds

    async def dispatch(self, request: Request, call_next):
        # Get user identifier (IP or user_id)
        user_id = self.get_user_identifier(request)

        # Check rate limit
        key = f"rate_limit:{user_id}"
        current = self.redis.get(key)

        if current is None:
            # First request in window
            self.redis.setex(key, self.window, 1)
            remaining = self.default_limit - 1
        else:
            current = int(current)
            if current >= self.default_limit:
                # Rate limit exceeded
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(self.default_limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + self.window)
                    }
                )

            # Increment counter
            self.redis.incr(key)
            remaining = self.default_limit - current - 1

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.default_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def get_user_identifier(self, request: Request) -> str:
        """Get user identifier from request."""
        # Try to get from JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header:
            # Extract user from token
            # ...
            pass

        # Fall back to IP address
        return request.client.host
```

### Token Bucket Algorithm

```python
# src/middleware/token_bucket.py
import time
from typing import Dict

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Returns:
            True if tokens consumed, False if insufficient tokens
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill

        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

# Usage
buckets: Dict[str, TokenBucket] = {}

def get_bucket(user_id: str) -> TokenBucket:
    """Get or create token bucket for user."""
    if user_id not in buckets:
        buckets[user_id] = TokenBucket(capacity=100, refill_rate=10.0)
    return buckets[user_id]

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    user_id = get_user_id(request)
    bucket = get_bucket(user_id)

    if not bucket.consume():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return await call_next(request)
```

## Caching Implementation

### Redis Caching

```python
# src/cache/redis_cache.py
import json
import hashlib
from typing import Optional, Any
import redis
from functools import wraps

class RedisCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL."""
        self.redis.setex(key, ttl, json.dumps(value))

    def delete(self, key: str):
        """Delete key from cache."""
        self.redis.delete(key)

    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern."""
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)

# Global cache instance
cache = RedisCache()

def cached(ttl: int = 300, key_prefix: str = ""):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))

            cache_key = hashlib.md5(
                ":".join(key_parts).encode()
            ).hexdigest()

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl=ttl)

            return result
        return wrapper
    return decorator

# Usage
@router.post("/predict")
@cached(ttl=60, key_prefix="predict")
async def predict(request: PredictionRequest):
    """Cached prediction endpoint."""
    result = model.predict(request.features)
    return result
```

### Cache Invalidation

```python
# src/cache/invalidation.py
from enum import Enum

class CacheStrategy(Enum):
    TTL = "ttl"                    # Time-to-live
    LRU = "lru"                    # Least recently used
    EVENT_DRIVEN = "event_driven"  # Invalidate on events

class CacheInvalidator:
    def __init__(self, cache: RedisCache):
        self.cache = cache

    def invalidate_user(self, user_id: str):
        """Invalidate all cache entries for user."""
        self.cache.clear_pattern(f"user:{user_id}:*")

    def invalidate_model(self, model_version: str):
        """Invalidate cache when model updates."""
        self.cache.clear_pattern(f"predict:*:model:{model_version}")

    def invalidate_all(self):
        """Clear all cache (use with caution)."""
        self.cache.redis.flushdb()
```

## API Versioning

### URL Versioning

```python
# src/versioning/v1/routes.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["v1"])

@router.post("/predict")
async def predict_v1(request: PredictionRequest):
    """Version 1 prediction endpoint."""
    # V1 implementation
    return {"version": "1.0", "prediction": result}

# src/versioning/v2/routes.py
router = APIRouter(prefix="/api/v2", tags=["v2"])

@router.post("/predict")
async def predict_v2(request: PredictionRequestV2):
    """Version 2 prediction endpoint with enhanced features."""
    # V2 implementation with new features
    return {
        "version": "2.0",
        "prediction": result,
        "confidence": confidence,
        "explanation": explanation  # New in v2
    }
```

### Header Versioning

```python
# src/versioning/header_version.py
from fastapi import Header, HTTPException

async def get_api_version(
    accept_version: str = Header(default="1.0", alias="Accept-Version")
) -> str:
    """Get API version from header."""
    if accept_version not in ["1.0", "2.0"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported API version: {accept_version}"
        )
    return accept_version

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    version: str = Depends(get_api_version)
):
    """Version-aware prediction endpoint."""
    if version == "1.0":
        return predict_v1(request)
    elif version == "2.0":
        return predict_v2(request)
```

### Deprecation Strategy

```python
# src/versioning/deprecation.py
from datetime import datetime
from fastapi import Header

DEPRECATED_VERSIONS = {
    "1.0": datetime(2025, 12, 31),  # Deprecated on
    "1.5": datetime(2026, 6, 30)
}

def check_deprecation(version: str) -> dict:
    """Check if version is deprecated."""
    if version in DEPRECATED_VERSIONS:
        deprecation_date = DEPRECATED_VERSIONS[version]
        return {
            "X-API-Deprecation": "true",
            "X-API-Sunset": deprecation_date.isoformat(),
            "X-API-Replacement": "2.0"
        }
    return {}

@app.middleware("http")
async def deprecation_middleware(request: Request, call_next):
    """Add deprecation headers."""
    response = await call_next(request)

    version = request.headers.get("Accept-Version", "1.0")
    headers = check_deprecation(version)

    for key, value in headers.items():
        response.headers[key] = value

    return response
```

## API Gateway Pattern

### Load Balancer

```python
# src/gateway/load_balancer.py
import random
from typing import List
from enum import Enum

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"

class LoadBalancer:
    def __init__(self, backends: List[str], strategy: LoadBalancingStrategy):
        self.backends = backends
        self.strategy = strategy
        self.current_index = 0
        self.connections = {backend: 0 for backend in backends}

    def get_backend(self) -> str:
        """Get next backend based on strategy."""
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            backend = self.backends[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.backends)
            return backend

        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(self.backends)

        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(self.connections, key=self.connections.get)

    def increment_connections(self, backend: str):
        """Track active connections."""
        self.connections[backend] += 1

    def decrement_connections(self, backend: str):
        """Decrement active connections."""
        self.connections[backend] -= 1
```

### Circuit Breaker

```python
# src/gateway/circuit_breaker.py
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures exceeded threshold
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold

        self.failures = 0
        self.successes = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failures = 0

        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            if self.successes >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.successes = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.timeout
        )
```

## Security Best Practices

### Input Validation

```python
# src/middleware/validation.py
from fastapi import Request, HTTPException
import re

class InputValidator:
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize user input."""
        # Remove potential SQL injection
        value = re.sub(r"[;']", "", value)

        # Remove XSS attempts
        value = re.sub(r"<script.*?>.*?</script>", "", value, flags=re.IGNORECASE)

        return value.strip()

    @staticmethod
    def validate_features(features: List[float]):
        """Validate prediction features."""
        if not features:
            raise ValueError("Features cannot be empty")

        if len(features) > 1000:
            raise ValueError("Too many features")

        if any(abs(f) > 1e6 for f in features):
            raise ValueError("Feature values too large")
```

### Security Headers

```python
# src/middleware/security_headers.py
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response
```

## Testing

### Authentication Tests

```python
# tests/test_auth.py
def test_register_user(client):
    response = client.post("/api/v1/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "SecurePass123!"
    })
    assert response.status_code == 201

def test_login(client):
    response = client.post("/api/v1/auth/login", json={
        "username": "testuser",
        "password": "SecurePass123!"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_protected_endpoint_without_token(client):
    response = client.post("/api/v1/predict", json={
        "features": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    assert response.status_code == 401
```

### Rate Limiting Tests

```python
# tests/test_rate_limiting.py
def test_rate_limit_exceeded(client, auth_token):
    # Make requests until rate limit hit
    for i in range(101):
        response = client.post(
            "/api/v1/predict",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

    assert response.status_code == 429
    assert "X-RateLimit-Remaining" in response.headers
```

## Best Practices

### 1. Authentication

✅ Use strong password hashing (bcrypt)
✅ Implement token refresh mechanism
✅ Set appropriate token expiration
✅ Use HTTPS in production
✅ Implement account lockout after failed attempts

### 2. Rate Limiting

✅ Set reasonable default limits
✅ Allow per-user customization
✅ Provide clear error messages
✅ Include rate limit headers
✅ Use distributed rate limiting (Redis)

### 3. Caching

✅ Cache expensive operations
✅ Set appropriate TTLs
✅ Implement cache invalidation
✅ Monitor cache hit rates
✅ Use cache warming strategies

### 4. Security

✅ Validate all inputs
✅ Use parameterized queries
✅ Implement CORS properly
✅ Add security headers
✅ Keep dependencies updated
✅ Conduct security audits

## Resources

- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [OAuth2 Specification](https://oauth.net/2/)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)
- [API Security Best Practices](https://owasp.org/www-project-api-security/)

## Next Steps

After completing this exercise:

1. ✅ Implement JWT authentication
2. ✅ Add rate limiting
3. ✅ Configure caching
4. ✅ Design API versioning
5. ✅ Implement security best practices
6. ✅ Build API gateway patterns

**Move on to**: Exercise 04 - Performance & Optimization
