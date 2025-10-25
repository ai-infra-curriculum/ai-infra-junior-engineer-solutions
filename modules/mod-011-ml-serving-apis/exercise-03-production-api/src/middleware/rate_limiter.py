"""Rate limiting middleware using Redis."""

import time
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import redis
import hashlib


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm with Redis.

    Supports:
    - Per-user rate limits
    - Per-endpoint rate limits
    - Custom rate limit headers
    - Redis-backed storage
    """

    def __init__(
        self,
        app,
        redis_client: redis.Redis,
        default_limit: int = 100,
        window_seconds: int = 60
    ):
        """
        Initialize rate limiter.

        Args:
            app: FastAPI application
            redis_client: Redis client instance
            default_limit: Default requests per window
            window_seconds: Time window in seconds
        """
        super().__init__(app)
        self.redis = redis_client
        self.default_limit = default_limit
        self.window = window_seconds

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/live", "/ready"]:
            return await call_next(request)

        # Get user identifier
        user_id = self.get_user_identifier(request)

        # Get endpoint-specific limit
        limit = self.get_endpoint_limit(request.url.path)

        # Check rate limit
        allowed, remaining, reset_time = self.check_rate_limit(
            user_id,
            request.url.path,
            limit
        )

        if not allowed:
            # Rate limit exceeded
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time - int(time.time()))
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response

    def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        limit: int
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.

        Args:
            user_id: User identifier
            endpoint: API endpoint
            limit: Rate limit

        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        # Create Redis key
        key = self.create_rate_limit_key(user_id, endpoint)

        # Get current count
        current = self.redis.get(key)

        if current is None:
            # First request in window
            pipe = self.redis.pipeline()
            pipe.set(key, 1)
            pipe.expire(key, self.window)
            pipe.execute()

            reset_time = int(time.time()) + self.window
            return True, limit - 1, reset_time

        current = int(current)

        if current >= limit:
            # Rate limit exceeded
            ttl = self.redis.ttl(key)
            reset_time = int(time.time()) + ttl
            return False, 0, reset_time

        # Increment counter
        self.redis.incr(key)
        ttl = self.redis.ttl(key)
        reset_time = int(time.time()) + ttl
        remaining = limit - current - 1

        return True, remaining, reset_time

    def get_user_identifier(self, request: Request) -> str:
        """
        Get user identifier from request.

        Tries in order:
        1. User ID from JWT token
        2. API key
        3. Client IP address

        Args:
            request: HTTP request

        Returns:
            User identifier string
        """
        # Try to get from JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            # In production, decode token to get user_id
            # For now, hash the token
            return hashlib.md5(token.encode()).hexdigest()[:16]

        # Try to get API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return hashlib.md5(api_key.encode()).hexdigest()[:16]

        # Fall back to IP address
        return request.client.host

    def get_endpoint_limit(self, path: str) -> int:
        """
        Get rate limit for specific endpoint.

        Args:
            path: Endpoint path

        Returns:
            Rate limit for endpoint
        """
        # Define custom limits for specific endpoints
        endpoint_limits = {
            "/api/v1/predict": 50,          # Lower limit for expensive operations
            "/api/v1/predict/batch": 10,    # Even lower for batch
            "/api/v1/auth/login": 5,        # Prevent brute force
            "/api/v1/auth/register": 3,     # Prevent spam
        }

        # Check for exact match
        for endpoint, limit in endpoint_limits.items():
            if path.startswith(endpoint):
                return limit

        # Return default limit
        return self.default_limit

    def create_rate_limit_key(self, user_id: str, endpoint: str) -> str:
        """
        Create Redis key for rate limiting.

        Args:
            user_id: User identifier
            endpoint: API endpoint

        Returns:
            Redis key
        """
        # Hash endpoint to keep key short
        endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
        return f"rate_limit:{user_id}:{endpoint_hash}"


class TokenBucket:
    """
    Token bucket algorithm for rate limiting.

    More sophisticated than simple counter, allows for burst traffic.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        capacity: int,
        refill_rate: float
    ):
        """
        Initialize token bucket.

        Args:
            redis_client: Redis client
            capacity: Maximum tokens
            refill_rate: Tokens added per second
        """
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate

    def consume(self, user_id: str, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            user_id: User identifier
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False otherwise
        """
        key = f"token_bucket:{user_id}"
        key_time = f"{key}:time"

        # Get current tokens and last refill time
        pipe = self.redis.pipeline()
        pipe.get(key)
        pipe.get(key_time)
        current_tokens, last_refill = pipe.execute()

        now = time.time()

        if current_tokens is None:
            # Initialize bucket
            current_tokens = self.capacity
            last_refill = now
        else:
            current_tokens = float(current_tokens)
            last_refill = float(last_refill)

            # Refill tokens
            elapsed = now - last_refill
            tokens_to_add = elapsed * self.refill_rate
            current_tokens = min(self.capacity, current_tokens + tokens_to_add)

        # Try to consume
        if current_tokens >= tokens:
            current_tokens -= tokens

            # Update Redis
            pipe = self.redis.pipeline()
            pipe.set(key, current_tokens)
            pipe.set(key_time, now)
            pipe.expire(key, 3600)  # 1 hour expiry
            pipe.expire(key_time, 3600)
            pipe.execute()

            return True

        return False
