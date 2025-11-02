# IMPLEMENTATION GUIDE: ML System Design Basics

## Exercise Overview

**Module**: MOD-004 Machine Learning Basics
**Exercise**: 06 - ML System Design and Workflow Planning
**Difficulty**: Intermediate to Advanced
**Estimated Time**: 3-4 hours
**Focus**: Production ML infrastructure, system design, and architectural patterns

This guide provides comprehensive implementation strategies for designing production-grade ML systems with emphasis on infrastructure engineering, scalability, reliability, and operational excellence.

---

## Table of Contents

1. [ML System Architecture Patterns](#1-ml-system-architecture-patterns)
2. [Offline vs Online Inference](#2-offline-vs-online-inference)
3. [Complete ML Inference Service Design](#3-complete-ml-inference-service-design)
4. [Request Handling and Batching](#4-request-handling-and-batching)
5. [Caching Strategies](#5-caching-strategies)
6. [Monitoring and Logging](#6-monitoring-and-logging)
7. [Error Handling and Failover](#7-error-handling-and-failover)
8. [Production Implementation](#8-production-implementation)
9. [Project-Specific Architectures](#9-project-specific-architectures)
10. [Cost Optimization](#10-cost-optimization)

---

## 1. ML System Architecture Patterns

### 1.1 Core Architecture Components

Every production ML system consists of several key components:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML SYSTEM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ Data Layer   │ ───> │Training Layer│ ───> │Serving Layer │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         ▼                      ▼                      ▼          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │Data Ingestion│      │Model Registry│      │  API Gateway │  │
│  │Feature Store │      │  Experiment  │      │Load Balancer │  │
│  │Preprocessing │      │   Tracking   │      │   Caching    │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Monitoring & Observability Platform                 │ │
│  │  - Metrics (Prometheus)  - Logs (ELK)  - Traces (Jaeger)  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Pattern 1: Lambda Architecture for ML

**Use Case**: Systems requiring both real-time predictions and batch retraining

```yaml
Architecture:
  Batch Layer:
    Purpose: Comprehensive model retraining
    Frequency: Daily/Weekly
    Data Source: Complete historical dataset
    Output: New model version
    Tools: Apache Spark, Airflow, Kubeflow

  Speed Layer:
    Purpose: Real-time predictions
    Latency: <100ms
    Data Source: Live requests
    Output: Predictions
    Tools: FastAPI, TensorFlow Serving, Redis

  Serving Layer:
    Purpose: Combine batch and real-time
    Strategy: Pre-computed recommendations + real-time signals
    Output: Final predictions
```

**Implementation Pattern**:

```python
class LambdaMLArchitecture:
    """
    Lambda architecture for ML systems combining batch and real-time processing.
    """
    def __init__(self):
        self.batch_predictions = BatchPredictionStore()  # Pre-computed
        self.realtime_model = RealtimeModel()  # Online inference
        self.feature_store = FeatureStore()

    async def predict(self, user_id: str, context: dict):
        """
        Hybrid prediction combining batch and real-time.
        """
        # Fast path: Get pre-computed recommendations (batch layer)
        batch_results = await self.batch_predictions.get(user_id)

        # Enrich with real-time signals (speed layer)
        realtime_features = await self.feature_store.get_realtime(user_id, context)

        # Re-rank or adjust using real-time model
        final_predictions = self.realtime_model.adjust(
            batch_results,
            realtime_features
        )

        return final_predictions
```

### 1.3 Pattern 2: Microservices for ML

**Use Case**: Multiple models serving different purposes, need independent scaling

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  API Gateway    │ (Rate limiting, Auth, Routing)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌─────────┐
│Feature  │ │Model    │
│Service  │ │Service  │
│(Redis)  │ │(FastAPI)│
└─────────┘ └────┬────┘
                 │
         ┌───────┼───────┐
         ▼       ▼       ▼
    ┌────────┐ ┌───┐ ┌────────┐
    │Model A │ │B  │ │Model C │
    │(Fraud) │ │...│ │(Recom) │
    └────────┘ └───┘ └────────┘
```

### 1.4 Pattern 3: Edge Inference

**Use Case**: Low latency requirements, offline capability, privacy concerns

```yaml
Architecture:
  Device Layer:
    - Lightweight model (TFLite, ONNX)
    - On-device inference
    - Periodic model updates

  Edge Layer:
    - Model compression
    - Quantization (INT8)
    - Model size: <50MB

  Cloud Layer:
    - Model training
    - Version management
    - Fallback for complex cases
```

---

## 2. Offline vs Online Inference

### 2.1 Decision Framework

Use this decision matrix to choose inference strategy:

| Criteria | Offline (Batch) | Online (Real-time) | Hybrid |
|----------|----------------|-------------------|--------|
| **Latency SLA** | Hours to days | <100ms | <100ms |
| **Scale** | Millions of predictions | 1000s req/sec | 1000s req/sec |
| **Cost** | Low (batch compute) | High (always-on) | Medium |
| **Freshness** | Stale (hours old) | Real-time | Near real-time |
| **Use Cases** | Recommendations, Email targeting | Fraud, Real-time bidding | Personalization |
| **Complexity** | Low | Medium | High |

### 2.2 Offline Inference Implementation

**Best for**: Product recommendations, email campaigns, content ranking

```python
"""
Offline batch inference pipeline for pre-computing recommendations.
Run daily via Airflow/Kubeflow.
"""

import pandas as pd
from typing import List, Dict
import logging
from datetime import datetime

class BatchInferencePipeline:
    """
    Batch inference pipeline for generating predictions offline.
    """

    def __init__(self, model_path: str, output_store: str):
        self.model = self.load_model(model_path)
        self.output_store = output_store
        self.logger = logging.getLogger(__name__)

    def run(self, date: str) -> Dict[str, any]:
        """
        Execute full batch inference pipeline.

        Args:
            date: Date for which to generate predictions (YYYY-MM-DD)

        Returns:
            Metrics and statistics about the run
        """
        start_time = datetime.now()
        stats = {
            'date': date,
            'start_time': start_time,
            'users_processed': 0,
            'predictions_generated': 0,
            'errors': 0
        }

        try:
            # Step 1: Load all users to process
            self.logger.info(f"Loading users for {date}")
            users = self.load_active_users(date)
            stats['users_processed'] = len(users)

            # Step 2: Batch feature extraction
            self.logger.info("Extracting features")
            features = self.extract_features_batch(users)

            # Step 3: Batch prediction (utilize GPU efficiently)
            self.logger.info("Generating predictions")
            predictions = self.predict_batch(features, batch_size=1024)
            stats['predictions_generated'] = len(predictions)

            # Step 4: Post-process and rank
            self.logger.info("Post-processing results")
            ranked_results = self.post_process(predictions)

            # Step 5: Write to cache/database
            self.logger.info("Writing results to store")
            self.write_results(ranked_results)

            stats['duration'] = (datetime.now() - start_time).total_seconds()
            stats['status'] = 'success'

        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}")
            stats['status'] = 'failed'
            stats['error'] = str(e)
            raise

        return stats

    def predict_batch(self, features: pd.DataFrame, batch_size: int = 1024):
        """
        Process predictions in batches for efficiency.
        """
        predictions = []

        for i in range(0, len(features), batch_size):
            batch = features.iloc[i:i+batch_size]
            batch_preds = self.model.predict(batch)
            predictions.extend(batch_preds)

            if i % 10000 == 0:
                self.logger.info(f"Processed {i}/{len(features)} users")

        return predictions

    def write_results(self, results: List[Dict]):
        """
        Write results to cache (Redis) for fast serving.
        """
        # Write to Redis with TTL
        # Structure: user_id -> top 100 recommendations
        import redis
        r = redis.Redis(host=self.output_store)

        pipe = r.pipeline()
        for user_id, recs in results.items():
            key = f"recs:user:{user_id}"
            # Store as sorted set with scores
            for rank, (item_id, score) in enumerate(recs[:100]):
                pipe.zadd(key, {item_id: score})
            pipe.expire(key, 86400)  # 24 hour TTL

        pipe.execute()
        self.logger.info(f"Wrote {len(results)} user recommendations to Redis")

# Airflow DAG example
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def run_batch_inference(**kwargs):
    pipeline = BatchInferencePipeline(
        model_path="/models/recommendation_v1.pkl",
        output_store="redis://cache:6379"
    )
    stats = pipeline.run(kwargs['ds'])  # Airflow execution date
    return stats

with DAG(
    'batch_recommendations',
    default_args={
        'owner': 'ml-platform',
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
    },
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False
) as dag:

    inference_task = PythonOperator(
        task_id='generate_recommendations',
        python_callable=run_batch_inference,
        provide_context=True
    )
```

### 2.3 Online Inference Implementation

**Best for**: Fraud detection, real-time bidding, instant personalization

```python
"""
Online inference service for real-time predictions.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import asyncio
from typing import List, Optional
import logging

class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    user_id: str
    features: Optional[dict] = None
    context: Optional[dict] = None

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    user_id: str
    predictions: List[dict]
    latency_ms: float
    model_version: str

class OnlineInferenceService:
    """
    Real-time inference service with optimizations.
    """

    def __init__(self):
        self.model = self.load_optimized_model()
        self.feature_store = FeatureStoreClient()
        self.cache = CacheClient()
        self.logger = logging.getLogger(__name__)
        self.model_version = "v1.2.3"

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate predictions with strict latency requirements.
        """
        import time
        start_time = time.time()

        try:
            # Parallel feature fetching
            user_features, item_features = await asyncio.gather(
                self.feature_store.get_user_features(request.user_id),
                self.feature_store.get_item_features()
            )

            # Combine with request features
            all_features = self.combine_features(
                user_features,
                item_features,
                request.features
            )

            # Model inference
            predictions = self.model.predict(all_features)

            # Post-process
            results = self.format_predictions(predictions)

            latency_ms = (time.time() - start_time) * 1000

            return PredictionResponse(
                user_id=request.user_id,
                predictions=results,
                latency_ms=latency_ms,
                model_version=self.model_version
            )

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return fallback predictions
            return self.get_fallback_predictions(request.user_id)

    def load_optimized_model(self):
        """
        Load model with optimizations for inference.
        """
        # Use ONNX for faster inference
        import onnxruntime as ort

        session = ort.InferenceSession(
            "/models/model_optimized.onnx",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        return session
```

### 2.4 Hybrid Inference (Best of Both)

```python
"""
Hybrid inference combining batch pre-computation with real-time adjustments.
"""

class HybridInferenceService:
    """
    Hybrid approach: Fast batch results + real-time signals.
    """

    def __init__(self):
        self.batch_cache = RedisClient()  # Pre-computed results
        self.realtime_model = LightweightModel()  # Fast re-ranker

    async def predict(self, user_id: str, context: dict):
        """
        Hybrid prediction flow.
        """
        # Step 1: Get pre-computed candidates (fast, from batch job)
        cached_candidates = await self.batch_cache.get(f"recs:{user_id}")

        if not cached_candidates:
            # Fallback to real-time computation (slower)
            return await self.realtime_full_prediction(user_id)

        # Step 2: Get real-time context features
        realtime_features = await self.get_realtime_features(user_id, context)

        # Step 3: Re-rank using lightweight real-time model
        # This is fast because we only re-rank ~100 items vs scoring all items
        reranked = self.realtime_model.rerank(
            candidates=cached_candidates,
            features=realtime_features
        )

        return reranked[:10]  # Top 10 results
```

---

## 3. Complete ML Inference Service Design

### 3.1 Production-Grade FastAPI Service

```python
"""
production_ml_service.py

Complete production ML inference service with all best practices.
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from pydantic import BaseModel, validator
import uvicorn
import logging
import time
import asyncio
from typing import List, Optional, Dict
import redis.asyncio as redis
import numpy as np
from contextlib import asynccontextmanager

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Service configuration."""
    MODEL_PATH = "/models/production/model.onnx"
    REDIS_URL = "redis://localhost:6379"
    MAX_BATCH_SIZE = 32
    MAX_BATCH_WAIT_MS = 10
    FEATURE_TIMEOUT_MS = 50
    PREDICTION_TIMEOUT_MS = 100
    CACHE_TTL_SECONDS = 3600
    MODEL_VERSION = "1.2.3"

# ============================================================================
# METRICS
# ============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    'ml_service_requests_total',
    'Total prediction requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'ml_service_request_duration_seconds',
    'Request latency',
    ['endpoint'],
    buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5]
)

# Model metrics
MODEL_PREDICTION_LATENCY = Histogram(
    'ml_service_model_inference_seconds',
    'Model inference latency',
    buckets=[.001, .0025, .005, .0075, .01, .025, .05, .075, .1]
)

FEATURE_FETCH_LATENCY = Histogram(
    'ml_service_feature_fetch_seconds',
    'Feature fetching latency',
    buckets=[.001, .0025, .005, .0075, .01, .025, .05]
)

CACHE_HIT_RATE = Counter(
    'ml_service_cache_hits_total',
    'Cache hits',
    ['cache_type']
)

CACHE_MISS_RATE = Counter(
    'ml_service_cache_misses_total',
    'Cache misses',
    ['cache_type']
)

ACTIVE_REQUESTS = Gauge(
    'ml_service_active_requests',
    'Number of requests being processed'
)

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    """Schema for prediction requests."""
    request_id: Optional[str] = None
    user_id: str
    item_ids: Optional[List[str]] = None
    context: Optional[Dict] = {}

    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v) == 0:
            raise ValueError('user_id cannot be empty')
        return v

class PredictionResponse(BaseModel):
    """Schema for prediction responses."""
    request_id: Optional[str]
    user_id: str
    predictions: List[Dict]
    model_version: str
    latency_ms: float
    cached: bool = False

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    cache_connected: bool
    uptime_seconds: float

# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    Manages model loading, inference, and versioning.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.model_version = Config.MODEL_VERSION
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load optimized model for inference."""
        try:
            import onnxruntime as ort

            # Configure ONNX Runtime for optimal performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            self.model = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

            self.logger.info(f"Model loaded successfully: {self.model_version}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run model inference with latency tracking.
        """
        with MODEL_PREDICTION_LATENCY.time():
            try:
                input_name = self.model.get_inputs()[0].name
                outputs = self.model.run(None, {input_name: features})
                return outputs[0]

            except Exception as e:
                self.logger.error(f"Model inference failed: {e}")
                raise

    def is_healthy(self) -> bool:
        """Check if model is loaded and healthy."""
        return self.model is not None

# ============================================================================
# FEATURE STORE CLIENT
# ============================================================================

class FeatureStoreClient:
    """
    Client for fetching features from feature store (Redis).
    """

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Connect to Redis."""
        try:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()
            self.logger.info("Connected to feature store")

        except Exception as e:
            self.logger.error(f"Failed to connect to feature store: {e}")
            raise

    async def close(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()

    async def get_features(self, user_id: str) -> Optional[Dict]:
        """
        Fetch user features with timeout.
        """
        with FEATURE_FETCH_LATENCY.time():
            try:
                key = f"features:user:{user_id}"

                # Try cache first
                cached = await self.client.get(key)

                if cached:
                    CACHE_HIT_RATE.labels(cache_type='features').inc()
                    return self._deserialize_features(cached)
                else:
                    CACHE_MISS_RATE.labels(cache_type='features').inc()
                    return None

            except asyncio.TimeoutError:
                self.logger.warning(f"Feature fetch timeout for user {user_id}")
                return None
            except Exception as e:
                self.logger.error(f"Feature fetch error: {e}")
                return None

    async def cache_prediction(self, user_id: str, prediction: Dict, ttl: int = 3600):
        """
        Cache prediction result.
        """
        try:
            key = f"prediction:user:{user_id}"
            await self.client.setex(
                key,
                ttl,
                self._serialize_prediction(prediction)
            )
        except Exception as e:
            self.logger.error(f"Failed to cache prediction: {e}")

    def _deserialize_features(self, data: str) -> Dict:
        """Deserialize features from cache."""
        import json
        return json.loads(data)

    def _serialize_prediction(self, data: Dict) -> str:
        """Serialize prediction for caching."""
        import json
        return json.dumps(data)

    def is_healthy(self) -> bool:
        """Check if connected to feature store."""
        return self.client is not None

# ============================================================================
# REQUEST BATCHER
# ============================================================================

class RequestBatcher:
    """
    Batches multiple requests for efficient model inference.
    Implements dynamic batching with timeout.
    """

    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 10):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = []
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)

    async def add_request(self, request: PredictionRequest, features: np.ndarray):
        """
        Add request to batch and wait for processing.
        """
        future = asyncio.Future()

        async with self.lock:
            self.pending_requests.append((request, features, future))

            # Trigger batch if full
            if len(self.pending_requests) >= self.max_batch_size:
                await self._process_batch()

        # Wait for result (with timeout)
        try:
            result = await asyncio.wait_for(
                future,
                timeout=Config.PREDICTION_TIMEOUT_MS / 1000
            )
            return result
        except asyncio.TimeoutError:
            self.logger.warning("Request batch processing timeout")
            raise HTTPException(status_code=504, detail="Prediction timeout")

    async def _process_batch(self):
        """
        Process accumulated batch.
        """
        if not self.pending_requests:
            return

        batch = self.pending_requests
        self.pending_requests = []

        # Extract features
        features_batch = np.array([f for _, f, _ in batch])

        try:
            # Run batch inference
            predictions = model_manager.predict(features_batch)

            # Resolve futures
            for i, (req, _, future) in enumerate(batch):
                if not future.done():
                    future.set_result(predictions[i])

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(e)

# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

# Global instances
model_manager = ModelManager(Config.MODEL_PATH)
feature_store = FeatureStoreClient(Config.REDIS_URL)
batcher = RequestBatcher(Config.MAX_BATCH_SIZE, Config.MAX_BATCH_WAIT_MS)
start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    """
    # Startup
    logging.info("Starting ML service...")
    model_manager.load_model()
    await feature_store.connect()
    logging.info("ML service started successfully")

    yield

    # Shutdown
    logging.info("Shutting down ML service...")
    await feature_store.close()
    logging.info("ML service shut down successfully")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Production ML Inference Service",
    description="High-performance ML inference service with batching, caching, and monitoring",
    version=Config.MODEL_VERSION,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_healthy() and feature_store.is_healthy() else "unhealthy",
        model_loaded=model_manager.is_healthy(),
        model_version=Config.MODEL_VERSION,
        cache_connected=feature_store.is_healthy(),
        uptime_seconds=time.time() - start_time
    )

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Main prediction endpoint with full optimizations.
    """
    start_time_req = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        # Check cache first
        cached_result = await feature_store.client.get(f"prediction:user:{request.user_id}")
        if cached_result:
            CACHE_HIT_RATE.labels(cache_type='predictions').inc()
            REQUEST_COUNT.labels(endpoint='predict', status='cache_hit').inc()

            return PredictionResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                predictions=feature_store._deserialize_features(cached_result),
                model_version=Config.MODEL_VERSION,
                latency_ms=(time.time() - start_time_req) * 1000,
                cached=True
            )

        CACHE_MISS_RATE.labels(cache_type='predictions').inc()

        # Fetch features
        features = await feature_store.get_features(request.user_id)

        if not features:
            # Use default features or return error
            raise HTTPException(status_code=404, detail="User features not found")

        # Convert to model input format
        feature_array = np.array(list(features.values())).reshape(1, -1).astype(np.float32)

        # Process through batcher
        predictions = await batcher.add_request(request, feature_array)

        # Format response
        results = [
            {"item_id": f"item_{i}", "score": float(score)}
            for i, score in enumerate(predictions[:10])
        ]

        response = PredictionResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            predictions=results,
            model_version=Config.MODEL_VERSION,
            latency_ms=(time.time() - start_time_req) * 1000,
            cached=False
        )

        # Cache result asynchronously
        asyncio.create_task(
            feature_store.cache_prediction(request.user_id, results, Config.CACHE_TTL_SECONDS)
        )

        REQUEST_COUNT.labels(endpoint='predict', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='predict').observe(time.time() - start_time_req)

        return response

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.dec()

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Batch prediction endpoint for multiple requests.
    """
    results = []

    # Process in parallel with concurrency limit
    sem = asyncio.Semaphore(10)

    async def predict_with_semaphore(req):
        async with sem:
            return await predict(req)

    tasks = [predict_with_semaphore(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {"predictions": results}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True
    )
```

---

## 4. Request Handling and Batching

### 4.1 Dynamic Batching Strategy

Dynamic batching improves throughput by processing multiple requests together:

```python
"""
Advanced batching with adaptive sizing.
"""

import asyncio
import time
from collections import deque
from typing import List, Tuple, Any

class AdaptiveBatcher:
    """
    Adaptive batching that adjusts batch size based on load.
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        max_wait_ms: int = 10,
        target_latency_ms: int = 50
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.target_latency_ms = target_latency_ms

        self.current_batch_size = min_batch_size
        self.pending: deque = deque()
        self.lock = asyncio.Lock()

        # Performance tracking
        self.latency_history = deque(maxlen=100)

    async def add_request(self, request: Any) -> Any:
        """
        Add request to batch queue.
        """
        future = asyncio.Future()

        async with self.lock:
            self.pending.append((request, future, time.time()))

            # Trigger batch if conditions met
            if len(self.pending) >= self.current_batch_size:
                asyncio.create_task(self._process_batch())
            elif len(self.pending) == 1:
                # Start timer for first request in batch
                asyncio.create_task(self._wait_and_process())

        return await future

    async def _wait_and_process(self):
        """
        Wait for timeout then process partial batch.
        """
        await asyncio.sleep(self.max_wait_ms / 1000)

        async with self.lock:
            if self.pending:
                await self._process_batch()

    async def _process_batch(self):
        """
        Process current batch and adapt batch size.
        """
        if not self.pending:
            return

        # Extract batch
        batch_size = min(len(self.pending), self.current_batch_size)
        batch = [self.pending.popleft() for _ in range(batch_size)]

        start_time = time.time()

        try:
            # Process batch
            results = await self._execute_batch([req for req, _, _ in batch])

            # Resolve futures
            for i, (_, future, _) in enumerate(batch):
                if not future.done():
                    future.set_result(results[i])

            # Track performance
            latency_ms = (time.time() - start_time) * 1000
            self.latency_history.append(latency_ms)

            # Adapt batch size
            self._adapt_batch_size(latency_ms)

        except Exception as e:
            for _, future, _ in batch:
                if not future.done():
                    future.set_exception(e)

    def _adapt_batch_size(self, latency_ms: float):
        """
        Dynamically adjust batch size based on latency.
        """
        if len(self.latency_history) < 10:
            return

        avg_latency = sum(self.latency_history) / len(self.latency_history)

        if avg_latency < self.target_latency_ms * 0.8:
            # Under target, can increase batch size
            self.current_batch_size = min(
                self.current_batch_size + 2,
                self.max_batch_size
            )
        elif avg_latency > self.target_latency_ms:
            # Over target, decrease batch size
            self.current_batch_size = max(
                self.current_batch_size - 2,
                self.min_batch_size
            )

    async def _execute_batch(self, requests: List[Any]) -> List[Any]:
        """
        Override this method with actual batch processing logic.
        """
        # Placeholder - implement actual model inference
        await asyncio.sleep(0.01)  # Simulate processing
        return [f"result_{i}" for i in range(len(requests))]
```

---

## 5. Caching Strategies

### 5.1 Multi-Level Caching

```python
"""
Multi-level caching: L1 (in-memory) + L2 (Redis) + L3 (Database).
"""

import asyncio
from typing import Optional, Any
from functools import lru_cache
import hashlib
import json

class MultiLevelCache:
    """
    Three-tier caching strategy for ML features and predictions.
    """

    def __init__(self, redis_client, db_client):
        self.redis = redis_client
        self.db = db_client
        self.l1_cache = {}  # In-memory cache
        self.l1_max_size = 10000

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (check all levels).
        """
        # L1: In-memory (fastest, ~0.001ms)
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2: Redis (fast, ~1-5ms)
        redis_value = await self.redis.get(key)
        if redis_value:
            # Promote to L1
            self._set_l1(key, redis_value)
            return redis_value

        # L3: Database (slow, ~10-50ms)
        db_value = await self.db.get(key)
        if db_value:
            # Promote to L2 and L1
            await self.redis.setex(key, 3600, db_value)
            self._set_l1(key, db_value)
            return db_value

        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """
        Set value in all cache levels.
        """
        # Set in all levels
        self._set_l1(key, value)
        await self.redis.setex(key, ttl, value)
        await self.db.set(key, value)

    def _set_l1(self, key: str, value: Any):
        """
        Set value in L1 cache with LRU eviction.
        """
        if len(self.l1_cache) >= self.l1_max_size:
            # Evict oldest entry
            self.l1_cache.pop(next(iter(self.l1_cache)))
        self.l1_cache[key] = value
```

### 5.2 Intelligent Cache Warming

```python
"""
Proactive cache warming for better hit rates.
"""

class CacheWarmer:
    """
    Proactively warm cache for high-traffic users/items.
    """

    def __init__(self, cache, model, analytics):
        self.cache = cache
        self.model = model
        self.analytics = analytics

    async def warm_popular_items(self):
        """
        Pre-compute predictions for popular items.
        """
        # Get top 1000 active users from analytics
        top_users = await self.analytics.get_top_users(limit=1000)

        # Pre-compute predictions
        for user_id in top_users:
            features = await self.get_features(user_id)
            predictions = self.model.predict(features)

            # Cache for 1 hour
            await self.cache.set(
                f"predictions:{user_id}",
                predictions,
                ttl=3600
            )

    async def warm_new_items(self):
        """
        Pre-compute embeddings for new items.
        """
        new_items = await self.analytics.get_new_items(hours=24)

        for item in new_items:
            embedding = self.model.encode_item(item)
            await self.cache.set(
                f"embedding:item:{item.id}",
                embedding,
                ttl=86400
            )
```

---

## 6. Monitoring and Logging

### 6.1 Comprehensive Monitoring Stack

```python
"""
Complete monitoring setup for ML services.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary
import logging
import structlog
from typing import Dict

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

class MLServiceMetrics:
    """
    Centralized metrics for ML service.
    """

    # Request metrics
    requests_total = Counter(
        'ml_requests_total',
        'Total requests',
        ['method', 'endpoint', 'status']
    )

    request_duration = Histogram(
        'ml_request_duration_seconds',
        'Request duration',
        ['endpoint'],
        buckets=[.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10]
    )

    # Model metrics
    model_inference_duration = Histogram(
        'ml_model_inference_seconds',
        'Model inference time',
        ['model_name', 'model_version']
    )

    prediction_score_distribution = Histogram(
        'ml_prediction_score',
        'Distribution of prediction scores',
        buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    # Feature metrics
    feature_fetch_duration = Histogram(
        'ml_feature_fetch_seconds',
        'Feature fetching time',
        ['feature_store']
    )

    feature_staleness = Histogram(
        'ml_feature_age_seconds',
        'Age of features',
        buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600, 86400]
    )

    # Cache metrics
    cache_operations = Counter(
        'ml_cache_operations_total',
        'Cache operations',
        ['operation', 'result', 'cache_level']
    )

    # Error metrics
    errors_total = Counter(
        'ml_errors_total',
        'Total errors',
        ['error_type', 'component']
    )

    # System metrics
    active_requests = Gauge(
        'ml_active_requests',
        'Currently active requests'
    )

    batch_size = Histogram(
        'ml_batch_size',
        'Request batch sizes',
        buckets=[1, 2, 4, 8, 16, 32, 64, 128]
    )

# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

def setup_logging():
    """
    Configure structured logging.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

# Example usage
logger = structlog.get_logger()

def log_prediction(user_id: str, latency_ms: float, cached: bool):
    """
    Log prediction with structured data.
    """
    logger.info(
        "prediction_completed",
        user_id=user_id,
        latency_ms=latency_ms,
        cached=cached,
        model_version="1.2.3"
    )

# ============================================================================
# ALERTING RULES
# ============================================================================

# Prometheus alerting rules (prometheus_alerts.yml)
ALERTING_RULES = """
groups:
  - name: ml_service_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(ml_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.99, ml_request_duration_seconds) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High p99 latency"
          description: "P99 latency is {{ $value }}s"

      # Low cache hit rate
      - alert: LowCacheHitRate
        expr: |
          (rate(ml_cache_operations_total{result="hit"}[5m]) /
           rate(ml_cache_operations_total[5m])) < 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"

      # Model inference slow
      - alert: SlowModelInference
        expr: histogram_quantile(0.95, ml_model_inference_seconds) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Model inference is slow"
"""
```

---

## 7. Error Handling and Failover

### 7.1 Circuit Breaker Pattern

```python
"""
Circuit breaker for external dependencies.
"""

import time
from enum import Enum
from typing import Callable, Any
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

# Usage example
feature_store_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60
)

async def get_features_with_breaker(user_id: str):
    """
    Fetch features with circuit breaker.
    """
    try:
        return await feature_store_breaker.call(
            feature_store.get_features,
            user_id
        )
    except Exception:
        # Use fallback default features
        return get_default_features()
```

### 7.2 Graceful Degradation

```python
"""
Graceful degradation strategies for ML services.
"""

class FallbackStrategy:
    """
    Implements fallback strategies when primary system fails.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def get_prediction_with_fallback(self, user_id: str) -> Dict:
        """
        Try multiple strategies in order of preference.
        """
        strategies = [
            self._strategy_cached,
            self._strategy_simple_model,
            self._strategy_rule_based,
            self._strategy_popular_items
        ]

        for strategy in strategies:
            try:
                result = await strategy(user_id)
                if result:
                    self.logger.info(f"Used fallback: {strategy.__name__}")
                    return result
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.__name__} failed: {e}")
                continue

        # Last resort
        return self._strategy_default()

    async def _strategy_cached(self, user_id: str) -> Optional[Dict]:
        """Strategy 1: Return cached predictions."""
        return await cache.get(f"predictions:{user_id}")

    async def _strategy_simple_model(self, user_id: str) -> Optional[Dict]:
        """Strategy 2: Use lightweight backup model."""
        features = await self.get_minimal_features(user_id)
        return simple_model.predict(features)

    async def _strategy_rule_based(self, user_id: str) -> Optional[Dict]:
        """Strategy 3: Use rule-based recommendations."""
        user_history = await self.get_user_history(user_id)
        return self.generate_rule_based_recs(user_history)

    async def _strategy_popular_items(self, user_id: str) -> Dict:
        """Strategy 4: Return popular items."""
        return await self.get_trending_items()

    def _strategy_default(self) -> Dict:
        """Strategy 5: Default hardcoded response."""
        return {"items": ["default_1", "default_2", "default_3"]}
```

---

## 8. Production Implementation

### 8.1 Complete Deployment Configuration

```yaml
# kubernetes_deployment.yaml
# Production-grade Kubernetes deployment for ML service

apiVersion: v1
kind: Namespace
metadata:
  name: ml-production

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-service-config
  namespace: ml-production
data:
  MODEL_PATH: "/models/production/model.onnx"
  REDIS_URL: "redis://redis-service:6379"
  MAX_BATCH_SIZE: "32"
  MAX_BATCH_WAIT_MS: "10"
  LOG_LEVEL: "INFO"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: ml-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
        version: v1.2.3
    spec:
      containers:
      - name: ml-service
        image: ml-service:1.2.3
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        envFrom:
        - configMapRef:
            name: ml-service-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: ml-service
  namespace: ml-production
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
spec:
  selector:
    app: ml-service
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
  namespace: ml-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 3
  maxReplicas: 20
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
  - type: Pods
    pods:
      metric:
        name: ml_active_requests
      target:
        type: AverageValue
        averageValue: "100"

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-service-pdb
  namespace: ml-production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: ml-service
```

### 8.2 Docker Configuration

```dockerfile
# Dockerfile for ML service

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 9. Project-Specific Architectures

### 9.1 Project 1: Product Recommendations (Hybrid Architecture)

```yaml
Architecture:
  Type: Hybrid (Batch + Real-time)

  Batch Layer:
    Schedule: Daily at 2 AM
    Process:
      1. Load user-item interactions (past 90 days)
      2. Train collaborative filtering model
      3. Generate top 100 candidates per user
      4. Store in Redis with 24h TTL
    Compute:
      - 4x GPU instances (V100)
      - 5 hours training time
      - Cost: ~$40/day

  Real-time Layer:
    Purpose: Re-rank candidates with fresh signals
    Latency: <50ms
    Features:
      - Current session behavior
      - Time of day
      - Device type
      - Current cart contents
    Model: Lightweight gradient boosting

  Serving Infrastructure:
    Platform: Kubernetes
    Replicas: 10-50 (auto-scaling)
    Instance: 4 CPU, 8GB RAM
    Cache: Redis Cluster (30GB)
    API: FastAPI with batching

  Expected Performance:
    Latency p99: 80ms
    Throughput: 2000 req/sec
    Cache hit rate: 85%
    Monthly cost: $3,500
```

### 9.2 Project 2: Fraud Detection (Real-time Streaming)

```yaml
Architecture:
  Type: Real-time Streaming

  Data Flow:
    Transaction → Kafka → Feature Enrichment → Model → Decision

  Feature Engineering:
    Pre-computed (Redis):
      - User lifetime stats (account age, total spend)
      - Historical fraud patterns
      - Device fingerprints
    Real-time (computed on request):
      - Transaction velocity (txns in last 5 min)
      - Amount deviation from user average
      - Geographic anomalies
    Streaming (Flink/Spark):
      - Rolling windows (1min, 5min, 1hour)
      - Velocity features
      - Pattern detection

  Model Serving:
    Framework: TensorFlow Serving or ONNX Runtime
    Optimization: FP16 quantization
    Latency SLA: <100ms (p99)
    Fallback: Rule-based system

  Monitoring:
    Metrics:
      - Precision/Recall (real-time)
      - False positive rate
      - Average transaction value flagged
      - Model latency
    Alerts:
      - Precision drop >5%
      - Latency >150ms
      - Error rate >0.1%

  Deployment Strategy:
    1. Shadow mode (log decisions, don't block)
    2. Canary 5% with high-confidence threshold
    3. Gradual rollout with A/B testing
    4. Full rollout after 7 days validation
```

### 9.3 Project 3: Image Moderation (Async Queue)

```yaml
Architecture:
  Type: Asynchronous Queue Processing

  Upload Flow:
    1. User uploads image → S3
    2. Trigger SQS message
    3. Worker picks up message
    4. Download + preprocess image
    5. Run inference
    6. Store result in DynamoDB
    7. Notify seller via webhook

  Model Optimization:
    Base Model: EfficientNet-B0
    Optimizations:
      - ONNX conversion
      - INT8 quantization
      - Input size: 224x224
    Result:
      - Model size: 15MB (from 52MB)
      - Latency: 80ms on CPU
      - Accuracy drop: <1%

  Infrastructure:
    Workers: 10-50 (auto-scaling)
    Instance: 8 CPU, 16GB RAM (no GPU needed)
    Queue: SQS (FIFO)
    Storage: S3 + DynamoDB
    Processing time: 3 seconds end-to-end

  Human-in-the-Loop:
    Confidence Thresholds:
      - >0.95: Auto-approve
      - <0.3: Auto-reject
      - 0.3-0.95: Human review
    Review Queue:
      - Prioritize by upload time
      - SLA: 2 hours
      - Feedback loop for model retraining
```

---

## 10. Cost Optimization

### 10.1 Cost Analysis Framework

```python
"""
Calculate and optimize infrastructure costs.
"""

class CostCalculator:
    """
    Estimate monthly infrastructure costs.
    """

    # Pricing (approximate AWS costs)
    PRICING = {
        'compute': {
            'c5.xlarge': 0.17,  # per hour
            'c5.2xlarge': 0.34,
            'g4dn.xlarge': 0.526,  # GPU
        },
        'storage': {
            's3_standard': 0.023,  # per GB/month
            'ebs_gp3': 0.08,
        },
        'database': {
            'redis_cache': 0.068,  # per GB/hour
            'rds_postgres': 0.115,  # db.t3.medium
        },
        'data_transfer': 0.09,  # per GB
    }

    def calculate_serving_cost(
        self,
        requests_per_day: int,
        avg_latency_ms: float,
        instance_type: str = 'c5.xlarge',
        replicas: int = 10
    ) -> dict:
        """
        Calculate monthly serving infrastructure cost.
        """
        # Compute cost
        hours_per_month = 730
        compute_cost = (
            self.PRICING['compute'][instance_type] *
            hours_per_month *
            replicas
        )

        # Cache cost (estimate 30GB Redis)
        cache_gb = 30
        cache_cost = (
            self.PRICING['database']['redis_cache'] *
            cache_gb *
            hours_per_month
        )

        # Data transfer (estimate 10% of requests need external calls)
        avg_response_kb = 5
        data_transfer_gb = (
            requests_per_day * 30 *
            avg_response_kb / 1024 / 1024 *
            0.1  # 10% external
        )
        transfer_cost = data_transfer_gb * self.PRICING['data_transfer']

        total_cost = compute_cost + cache_cost + transfer_cost
        cost_per_1k_requests = total_cost / (requests_per_day * 30) * 1000

        return {
            'compute': compute_cost,
            'cache': cache_cost,
            'data_transfer': transfer_cost,
            'total_monthly': total_cost,
            'cost_per_1k_requests': cost_per_1k_requests
        }

    def calculate_training_cost(
        self,
        training_time_hours: float,
        instance_type: str = 'g4dn.xlarge',
        frequency_per_month: int = 30
    ) -> dict:
        """
        Calculate monthly training cost.
        """
        cost_per_run = (
            training_time_hours *
            self.PRICING['compute'][instance_type]
        )

        monthly_cost = cost_per_run * frequency_per_month

        return {
            'cost_per_run': cost_per_run,
            'runs_per_month': frequency_per_month,
            'total_monthly': monthly_cost
        }

# Example usage
calculator = CostCalculator()

# Project 1: Recommendations
reco_costs = calculator.calculate_serving_cost(
    requests_per_day=1_000_000,
    avg_latency_ms=80,
    instance_type='c5.xlarge',
    replicas=15
)

print(f"Recommendation Service Cost: ${reco_costs['total_monthly']:.2f}/month")
print(f"Cost per 1K requests: ${reco_costs['cost_per_1k_requests']:.4f}")
```

### 10.2 Optimization Strategies

```python
"""
Strategies to reduce infrastructure costs.
"""

OPTIMIZATION_STRATEGIES = {
    "1. Right-size instances": {
        "description": "Use smallest instance that meets SLA",
        "potential_savings": "30-50%",
        "implementation": [
            "Profile actual CPU/memory usage",
            "Test with smaller instances",
            "Use spot instances for batch jobs",
            "Implement auto-scaling"
        ]
    },

    "2. Aggressive caching": {
        "description": "Reduce compute by caching predictions",
        "potential_savings": "40-60%",
        "implementation": [
            "Cache at multiple levels",
            "Increase TTL for stable predictions",
            "Pre-compute common patterns",
            "Use CDN for static content"
        ]
    },

    "3. Model optimization": {
        "description": "Reduce inference cost",
        "potential_savings": "50-70%",
        "implementation": [
            "Quantization (FP32 → FP16 → INT8)",
            "Model pruning",
            "Knowledge distillation",
            "Use smaller model architecture"
        ]
    },

    "4. Batch processing": {
        "description": "Process multiple requests together",
        "potential_savings": "30-40%",
        "implementation": [
            "Dynamic batching",
            "Queue requests",
            "Optimize batch size",
            "Use GPU efficiently"
        ]
    },

    "5. Smart scheduling": {
        "description": "Run batch jobs during off-peak hours",
        "potential_savings": "20-30%",
        "implementation": [
            "Use spot instances",
            "Schedule training at night",
            "Reserved instances for base load",
            "Scale down during low traffic"
        ]
    }
}
```

---

## Summary and Best Practices

### Key Takeaways

1. **Architecture Patterns**
   - Choose offline/online/hybrid based on latency and freshness requirements
   - Lambda architecture for combining batch and real-time
   - Microservices for independent scaling

2. **Performance Optimization**
   - Dynamic batching for throughput
   - Multi-level caching for latency
   - Model optimization (quantization, ONNX)
   - Async processing where possible

3. **Reliability**
   - Circuit breakers for fault tolerance
   - Graceful degradation strategies
   - Health checks and monitoring
   - Pod disruption budgets

4. **Monitoring**
   - Request latency (p50, p95, p99)
   - Model performance metrics
   - Cache hit rates
   - Error rates and types

5. **Cost Management**
   - Right-size infrastructure
   - Aggressive caching
   - Spot instances for batch
   - Model optimization

### Next Steps

After completing this exercise, you should be able to:
- Design complete ML systems for production
- Make informed architectural decisions
- Estimate infrastructure costs
- Implement monitoring and alerting
- Handle failures gracefully

**Continue to**: Advanced MLOps, Model Monitoring, or A/B Testing exercises.

---

## References

- [Designing Machine Learning Systems - Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [Machine Learning Engineering - Andriy Burkov](http://www.mlebook.com/)
- [Google SRE Book - Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [AWS Well-Architected Framework - ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/)
- [Eugene Yan's Blog - Applied ML](https://eugeneyan.com/)
