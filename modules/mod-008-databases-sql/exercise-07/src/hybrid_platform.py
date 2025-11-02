"""
Hybrid ML Platform - Combining PostgreSQL, MongoDB, and Redis

Demonstrates how to leverage the strengths of each database type:
- PostgreSQL: Structured training run data requiring JOINs and ACID
- MongoDB: Flexible model configurations with varying schemas
- Redis: High-speed feature caching and prediction serving
"""

from postgres_client import get_session as pg_session, engine
from mongodb_client import db, insert_model_config, get_model_config
from redis_client import (
    redis_client,
    cache_prediction,
    get_cached_prediction,
    store_user_features,
    get_user_features,
    increment_prediction_counter
)
from sqlalchemy import text
import json
import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MLPlatformHybrid:
    """
    Hybrid ML Platform that intelligently routes data to the right database.

    Architecture:
    - PostgreSQL: Training runs, datasets (structured + relational)
    - MongoDB: Model configs, experiments (flexible + nested)
    - Redis: Predictions, features (fast + ephemeral)
    """

    def __init__(self):
        self.pg_engine = engine
        self.mongo_db = db
        self.redis_client = redis_client

    def register_training_run(
        self,
        dataset_name: str,
        model_config: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Register a new training run across multiple databases.

        1. PostgreSQL: Structured training metadata (for analytics)
        2. MongoDB: Full model configuration (for flexibility)
        3. Link them with a common model_id

        Returns:
            Dictionary with IDs from each database
        """
        model_id = f"{model_config['model_name']}-{int(time.time())}"

        # Step 1: Insert into PostgreSQL
        with pg_session() as session:
            result = session.execute(
                text("SELECT id FROM datasets WHERE name = :name"),
                {"name": dataset_name}
            )
            dataset_row = result.fetchone()
            if not dataset_row:
                raise ValueError(f"Dataset '{dataset_name}' not found")

            dataset_id = dataset_row[0]

            result = session.execute(
                text("""
                    INSERT INTO training_runs
                    (dataset_id, model_name, framework, status)
                    VALUES (:did, :model_id, :fw, 'running')
                    RETURNING id
                """),
                {
                    "did": dataset_id,
                    "model_id": model_id,
                    "fw": model_config['framework']
                }
            )
            pg_id = result.fetchone()[0]

        logger.info(f"✓ PostgreSQL: Created training_run (id: {pg_id})")

        # Step 2: Insert flexible config into MongoDB
        mongo_doc = {
            "model_id": model_id,
            "model_name": model_config['model_name'],
            "framework": model_config['framework'],
            "config": model_config.get('hyperparameters', {}),
            "postgres_training_run_id": pg_id
        }
        mongo_id = insert_model_config(mongo_doc)
        logger.info(f"✓ MongoDB: Stored model config ({model_id})")

        return {
            "model_id": model_id,
            "postgres_id": pg_id,
            "mongodb_id": mongo_id
        }

    def get_complete_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Fetch comprehensive model info from multiple databases.
        Shows how to join data across different database systems.
        """
        info = {"model_id": model_id}

        # Get structured data from PostgreSQL
        with pg_session() as session:
            result = session.execute(
                text("""
                    SELECT
                        tr.status, tr.accuracy, tr.loss,
                        tr.training_time_seconds, d.name, tr.started_at
                    FROM training_runs tr
                    JOIN datasets d ON tr.dataset_id = d.id
                    WHERE tr.model_name = :model_id
                """),
                {"model_id": model_id}
            )
            row = result.fetchone()

            if row:
                info['postgres'] = {
                    "status": row[0],
                    "accuracy": row[1],
                    "loss": row[2],
                    "training_time_seconds": row[3],
                    "dataset": row[4],
                    "started_at": str(row[5])
                }

        # Get flexible config from MongoDB
        mongo_doc = get_model_config(model_id)
        if mongo_doc:
            info['mongodb'] = {
                "framework": mongo_doc['framework'],
                "config": mongo_doc.get('config', {}),
                "metrics": mongo_doc.get('metrics', {})
            }

        # Get prediction stats from Redis
        pred_count = self.redis_client.get(f"counter:predictions:{model_id}")
        info['redis'] = {
            "total_predictions": int(pred_count) if pred_count else 0
        }

        return info

    def serve_prediction_with_cache(
        self,
        user_id: str,
        features: List[float],
        model_version: str = "v1"
    ) -> Dict[str, Any]:
        """
        Prediction serving with multi-layer architecture:
        1. Check Redis cache (fast path)
        2. If miss, load features from Redis and compute
        3. Cache result for future requests
        """
        start_time = time.time()

        # Layer 1: Check cache
        cached = get_cached_prediction(user_id, model_version)
        if cached:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "user_id": user_id,
                "prediction": cached['prediction'],
                "source": "cache",
                "latency_ms": latency_ms
            }

        # Layer 2: Cache miss - compute prediction
        logger.info(f"Cache miss - computing prediction for {user_id}")

        # Simulate model inference
        time.sleep(0.05)  # 50ms inference time
        prediction = sum(features) / len(features)  # Mock prediction

        # Layer 3: Cache the result
        cache_prediction(user_id, features, prediction, model_version, ttl=300)

        # Layer 4: Update prediction counter
        increment_prediction_counter(model_version)

        latency_ms = (time.time() - start_time) * 1000
        return {
            "user_id": user_id,
            "prediction": prediction,
            "source": "computed",
            "latency_ms": latency_ms
        }

    def batch_serve_with_feature_store(
        self,
        user_ids: List[str],
        model_version: str = "v1"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch prediction serving using Redis feature store.
        Demonstrates efficient batch operations with pipelines.
        """
        results = {}

        # Get features from Redis for all users (single pipeline)
        from redis_client import batch_get_features
        user_features = batch_get_features(user_ids)

        for user_id in user_ids:
            if user_id in user_features:
                features_dict = user_features[user_id]
                feature_vector = list(features_dict.values())

                # Serve prediction (will use cache if available)
                result = self.serve_prediction_with_cache(
                    user_id,
                    feature_vector,
                    model_version
                )
                results[user_id] = result
            else:
                results[user_id] = {"error": "Features not found"}

        return results

    def update_training_progress(
        self,
        model_id: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Update training progress in real-time:
        - MongoDB: Store full metrics history (flexible)
        - Redis: Cache latest metrics (fast access)
        """
        # MongoDB: Append to metrics history
        self.mongo_db.model_configs.update_one(
            {"model_id": model_id},
            {
                "$push": {
                    "training_history": {
                        "epoch": epoch,
                        "metrics": metrics,
                        "timestamp": time.time()
                    }
                }
            }
        )

        # Redis: Cache latest metrics
        cache_key = f"training:latest:{model_id}"
        self.redis_client.setex(
            cache_key,
            3600,
            json.dumps({"epoch": epoch, "metrics": metrics})
        )

        logger.info(f"✓ Updated training progress: {model_id} epoch {epoch}")


def demonstrate_hybrid_operations():
    """Demonstrate the power of the hybrid architecture."""
    print("\n" + "="*70)
    print("HYBRID ML PLATFORM DEMONSTRATION")
    print("="*70)

    platform = MLPlatformHybrid()

    # Demo 1: Register training run
    print("\n=== 1. Register Training Run (PostgreSQL + MongoDB) ===")
    model_config = {
        "model_name": "demo-hybrid-model",
        "framework": "pytorch",
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 20
        }
    }

    ids = platform.register_training_run("fraud-train-v1", model_config)
    print(f"  PostgreSQL ID: {ids['postgres_id']}")
    print(f"  MongoDB ID: {ids['mongodb_id']}")
    print(f"  Model ID: {ids['model_id']}")

    # Demo 2: Feature storage and prediction serving
    print("\n=== 2. Feature Storage & Prediction Serving (Redis) ===")

    # Store features for multiple users
    for i in range(1, 4):
        user_id = f"user_{i:03d}"
        store_user_features(user_id, {
            f"feature_{j}": float(i * j) for j in range(5)
        })

    # Batch prediction (first call - cache miss)
    print("\n  First batch (cache miss):")
    user_ids = ["user_001", "user_002", "user_003"]
    start = time.time()
    results1 = platform.batch_serve_with_feature_store(user_ids)
    time1 = (time.time() - start) * 1000
    print(f"  Time: {time1:.2f}ms")
    for uid, result in results1.items():
        print(f"    {uid}: {result.get('source', 'N/A')} - {result.get('latency_ms', 0):.2f}ms")

    # Second call (cache hit)
    print("\n  Second batch (cache hit):")
    start = time.time()
    results2 = platform.batch_serve_with_feature_store(user_ids)
    time2 = (time.time() - start) * 1000
    print(f"  Time: {time2:.2f}ms (speedup: {time1/time2:.1f}x)")
    for uid, result in results2.items():
        print(f"    {uid}: {result.get('source', 'N/A')} - {result.get('latency_ms', 0):.2f}ms")

    # Demo 3: Complete model info
    print("\n=== 3. Get Complete Model Info (All 3 Databases) ===")
    info = platform.get_complete_model_info(ids['model_id'])
    print(f"  PostgreSQL data: {list(info.get('postgres', {}).keys())}")
    print(f"  MongoDB data: {list(info.get('mongodb', {}).keys())}")
    print(f"  Redis data: {list(info.get('redis', {}).keys())}")

    print("\n" + "="*70)
    print("✓ Hybrid platform demonstration complete")
    print("="*70)


if __name__ == "__main__":
    demonstrate_hybrid_operations()
