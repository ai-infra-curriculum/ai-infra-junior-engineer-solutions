"""Redis caching implementation."""

import json
import hashlib
import pickle
from typing import Optional, Any, Callable
from functools import wraps
import redis
from datetime import timedelta


class RedisCache:
    """Redis-based caching layer."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 300
    ):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
        """
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        value = self.redis.get(key)
        if value:
            try:
                return pickle.loads(value)
            except:
                # Fall back to JSON
                return json.loads(value.decode())
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        ttl = ttl or self.default_ttl

        try:
            # Try pickle first (supports more types)
            serialized = pickle.dumps(value)
        except:
            # Fall back to JSON
            serialized = json.dumps(value).encode()

        self.redis.setex(key, ttl, serialized)

    def delete(self, key: str):
        """
        Delete key from cache.

        Args:
            key: Cache key
        """
        self.redis.delete(key)

    def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        return self.redis.exists(key) > 0

    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Redis key pattern (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        keys = self.redis.keys(pattern)
        if keys:
            return self.redis.delete(*keys)
        return 0

    def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no expiry, -2 if doesn't exist
        """
        return self.redis.ttl(key)

    def extend_ttl(self, key: str, additional_seconds: int):
        """
        Extend TTL for existing key.

        Args:
            key: Cache key
            additional_seconds: Seconds to add to TTL
        """
        current_ttl = self.redis.ttl(key)
        if current_ttl > 0:
            self.redis.expire(key, current_ttl + additional_seconds)


# Global cache instance
cache = RedisCache()


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    key_builder: Optional[Callable] = None
):
    """
    Decorator for caching function results.

    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache key
        key_builder: Custom function to build cache key

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = build_cache_key(func, key_prefix, *args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl=ttl)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = build_cache_key(func, key_prefix, *args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl=ttl)

            return result

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def build_cache_key(func, prefix: str, *args, **kwargs) -> str:
    """
    Build cache key from function and arguments.

    Args:
        func: Function
        prefix: Key prefix
        args: Function arguments
        kwargs: Function keyword arguments

    Returns:
        Cache key
    """
    # Build key components
    key_parts = [prefix or "", func.__module__, func.__name__]

    # Add args
    for arg in args:
        if hasattr(arg, "__dict__"):
            # For objects, use string representation
            key_parts.append(str(arg))
        else:
            key_parts.append(str(arg))

    # Add kwargs
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    # Create hash
    key_string = ":".join(key_parts)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()

    return f"cache:{key_hash}"


class CacheInvalidator:
    """Handle cache invalidation strategies."""

    def __init__(self, cache: RedisCache):
        """
        Initialize invalidator.

        Args:
            cache: Redis cache instance
        """
        self.cache = cache

    def invalidate_user(self, user_id: str):
        """
        Invalidate all cache entries for user.

        Args:
            user_id: User identifier
        """
        pattern = f"cache:*user:{user_id}*"
        deleted = self.cache.clear_pattern(pattern)
        return deleted

    def invalidate_model(self, model_version: str):
        """
        Invalidate cache when model updates.

        Args:
            model_version: Model version
        """
        pattern = f"cache:*model:{model_version}*"
        deleted = self.cache.clear_pattern(pattern)
        return deleted

    def invalidate_predictions(self):
        """Invalidate all prediction caches."""
        pattern = "cache:*predict*"
        deleted = self.cache.clear_pattern(pattern)
        return deleted

    def invalidate_all(self):
        """Clear all cache (use with caution)."""
        self.cache.redis.flushdb()


# Example usage
if __name__ == "__main__":
    # Initialize cache
    cache = RedisCache()

    # Simple get/set
    cache.set("my_key", {"data": "value"}, ttl=60)
    result = cache.get("my_key")
    print(f"Cached result: {result}")

    # Using decorator
    @cached(ttl=120, key_prefix="expensive_op")
    def expensive_operation(x: int, y: int) -> int:
        """Simulate expensive operation."""
        import time
        time.sleep(2)
        return x + y

    # First call - slow
    result1 = expensive_operation(5, 3)
    print(f"Result 1: {result1}")

    # Second call - fast (from cache)
    result2 = expensive_operation(5, 3)
    print(f"Result 2 (cached): {result2}")
