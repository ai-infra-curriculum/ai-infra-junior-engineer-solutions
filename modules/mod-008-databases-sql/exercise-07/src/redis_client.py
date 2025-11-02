"""
Redis Client for ML Platform

Handles high-speed caching and feature serving with <10ms latency.
Use cases:
- Model prediction caching with TTL
- Real-time feature stores
- Session management
- Leaderboards and rankings
- Rate limiting
"""

import redis
import json
import time
import logging
from typing import Dict, List, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "mlpass123"

# Create Redis client
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,  # Automatically decode bytes to strings
    socket_connect_timeout=5,
    socket_keepalive=True,
    health_check_interval=30
)


def test_connection() -> bool:
    """Test Redis connection."""
    try:
        redis_client.ping()
        info = redis_client.info("server")
        logger.info(f"✓ Redis connected: v{info['redis_version']}")
        return True
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        return False


# ============================================================================
# PREDICTION CACHING
# ============================================================================

def cache_prediction(
    user_id: str,
    features: List[float],
    prediction: float,
    model_version: str = "v1",
    ttl: int = 300
):
    """
    Cache prediction results with TTL (time-to-live).
    Redis excels at high-speed reads/writes with automatic expiration.

    Args:
        user_id: User identifier
        features: Input features used for prediction
        prediction: Model prediction result
        model_version: Version of model used
        ttl: Time-to-live in seconds (default: 5 minutes)
    """
    cache_key = f"prediction:{model_version}:{user_id}"

    data = {
        "user_id": user_id,
        "features": features,
        "prediction": prediction,
        "model_version": model_version,
        "cached_at": time.time()
    }

    # Store as JSON with expiration
    redis_client.setex(
        cache_key,
        ttl,
        json.dumps(data)
    )

    logger.info(f"✓ Cached prediction for {user_id} (TTL: {ttl}s)")


def get_cached_prediction(
    user_id: str,
    model_version: str = "v1"
) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached prediction if available.

    Returns:
        Prediction data if cache hit, None if cache miss
    """
    cache_key = f"prediction:{model_version}:{user_id}"

    cached = redis_client.get(cache_key)

    if cached:
        data = json.loads(cached)
        age = time.time() - data['cached_at']
        logger.info(f"✓ Cache HIT for {user_id} (age: {age:.1f}s)")
        return data
    else:
        logger.info(f"✗ Cache MISS for {user_id}")
        return None


def invalidate_user_cache(user_id: str, model_version: str = "v1"):
    """Manually invalidate cached prediction for a user."""
    cache_key = f"prediction:{model_version}:{user_id}"
    deleted = redis_client.delete(cache_key)
    if deleted:
        logger.info(f"✓ Invalidated cache for {user_id}")
    return deleted


def get_cache_stats() -> Dict[str, Any]:
    """Get Redis cache statistics."""
    info = redis_client.info("stats")
    memory = redis_client.info("memory")

    return {
        "total_connections": info.get("total_connections_received", 0),
        "total_commands": info.get("total_commands_processed", 0),
        "keyspace_hits": info.get("keyspace_hits", 0),
        "keyspace_misses": info.get("keyspace_misses", 0),
        "hit_rate": (
            info.get("keyspace_hits", 0) /
            max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
        ) * 100,
        "used_memory_mb": memory.get("used_memory", 0) / 1048576,
        "used_memory_peak_mb": memory.get("used_memory_peak", 0) / 1048576
    }


# ============================================================================
# FEATURE STORE
# ============================================================================

def store_user_features(user_id: str, features: Dict[str, Any], ttl: int = 3600):
    """
    Store user features in Redis Hash.
    Hashes are perfect for storing related key-value pairs.

    Args:
        user_id: User identifier
        features: Dictionary of feature names and values
        ttl: Time-to-live in seconds (default: 1 hour)
    """
    feature_key = f"features:user:{user_id}"

    # Convert all values to strings for Redis
    str_features = {k: str(v) for k, v in features.items()}

    # Use Redis hash for structured features
    redis_client.hset(feature_key, mapping=str_features)
    redis_client.expire(feature_key, ttl)

    logger.info(f"✓ Stored {len(features)} features for {user_id} (TTL: {ttl}s)")


def get_user_features(user_id: str) -> Optional[Dict[str, float]]:
    """Retrieve all features for a user."""
    feature_key = f"features:user:{user_id}"

    features = redis_client.hgetall(feature_key)

    if features:
        # Convert string values back to floats
        try:
            typed_features = {k: float(v) for k, v in features.items()}
            logger.info(f"✓ Retrieved {len(typed_features)} features for {user_id}")
            return typed_features
        except ValueError:
            logger.warning(f"Failed to convert features for {user_id}")
            return None
    else:
        logger.info(f"✗ No features found for {user_id}")
        return None


def get_specific_feature(user_id: str, feature_name: str) -> Optional[float]:
    """Retrieve a specific feature for a user (faster than getting all)."""
    feature_key = f"features:user:{user_id}"
    value = redis_client.hget(feature_key, feature_name)

    if value:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def batch_get_features(user_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Efficiently retrieve features for multiple users using pipeline.
    Pipelines batch commands to reduce round-trip latency.
    """
    pipe = redis_client.pipeline()

    # Queue all reads in pipeline
    for user_id in user_ids:
        feature_key = f"features:user:{user_id}"
        pipe.hgetall(feature_key)

    # Execute all commands at once
    start = time.time()
    results = pipe.execute()
    duration = (time.time() - start) * 1000

    logger.info(f"✓ Retrieved features for {len(user_ids)} users in {duration:.2f}ms")

    # Convert results
    features_dict = {}
    for user_id, features in zip(user_ids, results):
        if features:
            features_dict[user_id] = {k: float(v) for k, v in features.items()}

    return features_dict


def update_feature(user_id: str, feature_name: str, value: float):
    """Update a single feature for a user."""
    feature_key = f"features:user:{user_id}"
    redis_client.hset(feature_key, feature_name, str(value))
    logger.info(f"✓ Updated {feature_name} for {user_id}: {value}")


# ============================================================================
# LEADERBOARDS (Sorted Sets)
# ============================================================================

def update_model_leaderboard(model_name: str, accuracy: float):
    """
    Redis Sorted Sets are perfect for leaderboards and rankings.
    Score = accuracy, member = model_name.
    """
    leaderboard_key = "leaderboard:model_accuracy"
    redis_client.zadd(leaderboard_key, {model_name: accuracy})
    logger.info(f"✓ Updated leaderboard: {model_name} = {accuracy:.4f}")


def get_top_models(top_n: int = 10) -> List[Dict[str, Any]]:
    """Get top N models by accuracy from leaderboard."""
    leaderboard_key = "leaderboard:model_accuracy"

    # Get top models (highest scores first)
    top_models = redis_client.zrevrange(leaderboard_key, 0, top_n - 1, withscores=True)

    results = []
    for rank, (model_name, accuracy) in enumerate(top_models, 1):
        results.append({
            "rank": rank,
            "model_name": model_name,
            "accuracy": accuracy
        })

    logger.info(f"✓ Retrieved top {len(results)} models from leaderboard")
    return results


def get_model_rank(model_name: str) -> Optional[int]:
    """Get the rank of a specific model in the leaderboard."""
    leaderboard_key = "leaderboard:model_accuracy"
    rank = redis_client.zrevrank(leaderboard_key, model_name)

    if rank is not None:
        return rank + 1  # Convert 0-indexed to 1-indexed
    return None


def get_models_in_range(min_accuracy: float, max_accuracy: float) -> List[Dict[str, Any]]:
    """Get all models with accuracy in a specific range."""
    leaderboard_key = "leaderboard:model_accuracy"

    models = redis_client.zrangebyscore(
        leaderboard_key,
        min_accuracy,
        max_accuracy,
        withscores=True
    )

    return [
        {"model_name": name, "accuracy": score}
        for name, score in models
    ]


# ============================================================================
# COUNTERS AND METRICS
# ============================================================================

def increment_prediction_counter(model_name: str, amount: int = 1) -> int:
    """
    Increment prediction counter for a model.
    Redis atomic operations are perfect for counters.
    """
    counter_key = f"counter:predictions:{model_name}"
    new_count = redis_client.incrby(counter_key, amount)
    return new_count


def get_prediction_count(model_name: str) -> int:
    """Get total predictions for a model."""
    counter_key = f"counter:predictions:{model_name}"
    count = redis_client.get(counter_key)
    return int(count) if count else 0


def reset_prediction_counter(model_name: str):
    """Reset prediction counter for a model."""
    counter_key = f"counter:predictions:{model_name}"
    redis_client.delete(counter_key)
    logger.info(f"✓ Reset prediction counter for {model_name}")


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def create_session(session_id: str, user_data: Dict[str, Any], ttl: int = 1800):
    """
    Create user session with automatic expiration.

    Args:
        session_id: Session identifier
        user_data: Session data
        ttl: Time-to-live in seconds (default: 30 minutes)
    """
    session_key = f"session:{session_id}"
    redis_client.setex(
        session_key,
        ttl,
        json.dumps(user_data)
    )
    logger.info(f"✓ Created session {session_id} (TTL: {ttl}s)")


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve session data."""
    session_key = f"session:{session_id}"
    data = redis_client.get(session_key)

    if data:
        return json.loads(data)
    return None


def extend_session(session_id: str, ttl: int = 1800):
    """Extend session TTL (e.g., on user activity)."""
    session_key = f"session:{session_id}"
    if redis_client.exists(session_key):
        redis_client.expire(session_key, ttl)
        logger.info(f"✓ Extended session {session_id} by {ttl}s")
        return True
    return False


def delete_session(session_id: str):
    """Delete session (e.g., on logout)."""
    session_key = f"session:{session_id}"
    deleted = redis_client.delete(session_key)
    if deleted:
        logger.info(f"✓ Deleted session {session_id}")
    return deleted


# ============================================================================
# RATE LIMITING
# ============================================================================

def check_rate_limit(
    user_id: str,
    max_requests: int = 100,
    window_seconds: int = 60
) -> Dict[str, Any]:
    """
    Implement sliding window rate limiting.

    Args:
        user_id: User identifier
        max_requests: Maximum requests allowed in window
        window_seconds: Time window in seconds

    Returns:
        Dictionary with allowed status and remaining requests
    """
    key = f"ratelimit:{user_id}"
    current_time = time.time()
    window_start = current_time - window_seconds

    pipe = redis_client.pipeline()

    # Remove old entries outside the window
    pipe.zremrangebyscore(key, 0, window_start)

    # Count requests in current window
    pipe.zcard(key)

    # Add current request
    pipe.zadd(key, {str(current_time): current_time})

    # Set expiration
    pipe.expire(key, window_seconds)

    results = pipe.execute()
    request_count = results[1]

    allowed = request_count < max_requests
    remaining = max(0, max_requests - request_count - 1)

    return {
        "allowed": allowed,
        "current_requests": request_count,
        "remaining": remaining,
        "reset_in": window_seconds
    }


# ============================================================================
# UTILITIES
# ============================================================================

def flush_all_data():
    """⚠️ DANGER: Delete all data in Redis. Use with caution!"""
    redis_client.flushall()
    logger.warning("⚠️ Flushed all Redis data")


def get_memory_info() -> Dict[str, Any]:
    """Get Redis memory usage information."""
    info = redis_client.info("memory")
    return {
        "used_memory_mb": info["used_memory"] / 1048576,
        "used_memory_human": info["used_memory_human"],
        "used_memory_peak_mb": info["used_memory_peak"] / 1048576,
        "used_memory_peak_human": info["used_memory_peak_human"],
        "memory_fragmentation_ratio": info.get("mem_fragmentation_ratio", 1.0),
        "maxmemory_mb": info.get("maxmemory", 0) / 1048576,
        "maxmemory_policy": info.get("maxmemory_policy", "noeviction")
    }


def demonstrate_ttl():
    """Demonstrate Redis TTL (time-to-live) feature."""
    print("\n=== Redis TTL Demonstration ===\n")

    test_key = "test:ttl:demo"
    redis_client.setex(test_key, 5, "This will expire in 5 seconds")
    print(f"✓ Set key '{test_key}' with TTL of 5 seconds")

    for i in range(6):
        ttl = redis_client.ttl(test_key)
        value = redis_client.get(test_key)
        print(f"  After {i}s: TTL = {ttl}s, Value = {value}")
        time.sleep(1)

    print("\n✓ Key automatically expired and was deleted by Redis")


if __name__ == "__main__":
    print("="*70)
    print("Redis Client - ML Platform")
    print("="*70)

    # Test connection
    test_connection()

    # Demo: Prediction caching
    print("\n=== Prediction Caching ===")
    cache_prediction("user_123", [0.5, 0.3, 0.8, 0.2], 0.92, ttl=10)
    pred = get_cached_prediction("user_123")
    print(f"  Cached prediction: {pred['prediction']}")

    # Demo: Feature store
    print("\n=== Feature Store ===")
    store_user_features("user_001", {
        "age": 35.0,
        "account_balance": 5000.50,
        "days_since_signup": 120.0,
        "num_transactions": 45.0,
        "avg_transaction_amount": 234.56
    }, ttl=3600)

    features = get_user_features("user_001")
    print(f"  Features: {list(features.keys())}")

    # Demo: Batch retrieval
    print("\n=== Batch Feature Retrieval ===")
    store_user_features("user_002", {"age": 28.0, "account_balance": 12000.0}, ttl=3600)
    store_user_features("user_003", {"age": 42.0, "account_balance": 8500.0}, ttl=3600)

    batch_features = batch_get_features(["user_001", "user_002", "user_003"])
    print(f"  Retrieved features for {len(batch_features)} users")

    # Demo: Leaderboard
    print("\n=== Model Accuracy Leaderboard ===")
    models = [
        ("fraud-detector-v1", 0.9845),
        ("fraud-detector-v2", 0.9821),
        ("sentiment-analyzer-v2", 0.9234),
        ("churn-predictor-v3", 0.8934),
        ("recommender-v1", 0.8567)
    ]

    for model, acc in models:
        update_model_leaderboard(model, acc)

    top_3 = get_top_models(3)
    for model in top_3:
        print(f"  {model['rank']}. {model['model_name']:<30} | Acc: {model['accuracy']:.4f}")

    # Demo: Counters
    print("\n=== Prediction Counters ===")
    for i in range(100):
        increment_prediction_counter("fraud-detector-v1")

    count = get_prediction_count("fraud-detector-v1")
    print(f"  Total predictions for fraud-detector-v1: {count}")

    # Demo: Rate limiting
    print("\n=== Rate Limiting ===")
    for i in range(5):
        result = check_rate_limit("user_999", max_requests=3, window_seconds=60)
        print(f"  Request {i+1}: Allowed = {result['allowed']}, Remaining = {result['remaining']}")

    # Display cache stats
    print("\n=== Cache Statistics ===")
    stats = get_cache_stats()
    print(f"  Hit Rate: {stats['hit_rate']:.2f}%")
    print(f"  Total Commands: {stats['total_commands']}")
    print(f"  Memory Used: {stats['used_memory_mb']:.2f} MB")

    # Display memory info
    print("\n=== Memory Information ===")
    memory = get_memory_info()
    print(f"  Used: {memory['used_memory_human']}")
    print(f"  Peak: {memory['used_memory_peak_human']}")
    print(f"  Fragmentation Ratio: {memory['memory_fragmentation_ratio']:.2f}")
    print(f"  Max Memory: {memory['maxmemory_mb']:.0f} MB")
    print(f"  Eviction Policy: {memory['maxmemory_policy']}")

    print("\n" + "="*70)
    print("✓ Redis demonstration complete")
    print("="*70)
