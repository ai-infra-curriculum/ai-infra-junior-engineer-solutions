# Step-by-Step Implementation Guide: Performance Optimization

## Overview

Optimize ML API performance! Learn async processing, connection pooling, caching strategies, batch optimization, GPU utilization, and load testing.

**Time**: 3-4 hours | **Difficulty**: Advanced

---

## Learning Objectives

✅ Optimize async operations
✅ Implement connection pooling
✅ Use caching effectively
✅ Optimize batch processing
✅ Leverage GPU acceleration
✅ Conduct load testing
✅ Profile and benchmark
✅ Reduce latency

---

## Phase 1: Async Optimization

### Async Database Connections

```python
# async_db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/mlapi"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

@asynccontextmanager
async def get_async_db():
    """Async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Usage
@app.post("/predictions")
async def create_prediction(request: PredictionRequest):
    """Store prediction asynchronously"""
    async with get_async_db() as db:
        prediction = Prediction(
            features=request.features,
            result=await predict(request)
        )
        db.add(prediction)
        await db.flush()
        return {"id": prediction.id}
```

### Async Redis

```python
# async_cache.py
import aioredis
from typing import Optional
import json

class AsyncCache:
    """Async Redis cache"""

    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = None
        self.redis_url = redis_url

    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(self.redis_url)

    async def get(self, key: str) -> Optional[dict]:
        """Get cached value"""
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    async def set(self, key: str, value: dict, ttl: int = 3600):
        """Cache value"""
        await self.redis.setex(key, ttl, json.dumps(value))

    async def delete(self, key: str):
        """Delete cached value"""
        await self.redis.delete(key)

cache = AsyncCache()

@app.on_event("startup")
async def startup():
    await cache.connect()

@app.post("/predict/cached")
async def predict_with_cache(request: PredictionRequest):
    """Prediction with async cache"""
    cache_key = f"pred:{hash(tuple(request.features))}"

    # Check cache
    cached = await cache.get(cache_key)
    if cached:
        return cached

    # Predict
    result = await predict(request)

    # Cache result
    await cache.set(cache_key, result.dict())

    return result
```

### Concurrent Processing

```python
# concurrent.py
import asyncio
from typing import List

async def process_single_prediction(features: List[float]):
    """Process single prediction"""
    # Simulate I/O-bound operation
    await asyncio.sleep(0.1)
    return model.predict([features])[0]

async def process_predictions_concurrent(features_list: List[List[float]]):
    """Process multiple predictions concurrently"""
    tasks = [
        process_single_prediction(features)
        for features in features_list
    ]
    results = await asyncio.gather(*tasks)
    return results

@app.post("/predict/concurrent")
async def predict_concurrent(requests: List[PredictionRequest]):
    """Concurrent prediction processing"""
    features_list = [req.features for req in requests]
    results = await process_predictions_concurrent(features_list)
    return {"predictions": results}
```

---

## Phase 2: Connection Pooling

### HTTP Client Pool

```python
# http_client.py
import httpx
from typing import Optional

class HTTPClientPool:
    """Pooled HTTP client"""

    def __init__(self, max_connections: int = 100, max_keepalive: int = 20):
        self.client: Optional[httpx.AsyncClient] = None
        self.limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive
        )

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            limits=self.limits,
            timeout=httpx.Timeout(30.0)
        )
        return self.client

    async def __aexit__(self, *args):
        await self.client.aclose()

http_pool = HTTPClientPool()

async def fetch_external_data(url: str):
    """Fetch data with connection pooling"""
    async with http_pool as client:
        response = await client.get(url)
        return response.json()
```

### Database Connection Pool

```python
# db_pool.py
from sqlalchemy.pool import QueuePool
from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # Connections to keep open
    max_overflow=10,       # Additional connections allowed
    pool_timeout=30,       # Timeout for getting connection
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True,    # Verify connections before use
)

# Monitor pool
@app.get("/debug/pool")
async def get_pool_status():
    """Get connection pool status"""
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "timeout": pool._timeout,
    }
```

---

## Phase 3: Advanced Caching

### Multi-Level Cache

```python
# multi_level_cache.py
from functools import lru_cache
from typing import Optional, Tuple

class MultiLevelCache:
    """In-memory + Redis cache"""

    def __init__(self, redis_cache: AsyncCache, memory_size: int = 1000):
        self.redis = redis_cache
        self.memory_size = memory_size

    @lru_cache(maxsize=1000)
    def _memory_cache_get(self, key: str) -> Optional[str]:
        """In-memory cache (LRU)"""
        return None

    async def get(self, key: str) -> Optional[dict]:
        """Get from cache (memory first, then Redis)"""
        # Try memory cache
        cached = self._memory_cache_get(key)
        if cached:
            return json.loads(cached)

        # Try Redis
        cached = await self.redis.get(key)
        if cached:
            # Populate memory cache
            self._memory_cache_get.__wrapped__(self, key)
            return cached

        return None

    async def set(self, key: str, value: dict, ttl: int = 3600):
        """Set in both caches"""
        # Set in Redis
        await self.redis.set(key, value, ttl)

        # Set in memory
        self._memory_cache_get.__wrapped__(self, key)
```

### Cache Warming

```python
# cache_warmer.py
import asyncio
from typing import List

class CacheWarmer:
    """Pre-populate cache on startup"""

    def __init__(self, cache: MultiLevelCache):
        self.cache = cache

    async def warm_popular_predictions(self, popular_inputs: List[List[float]]):
        """Warm cache with popular predictions"""
        tasks = []
        for features in popular_inputs:
            cache_key = f"pred:{hash(tuple(features))}"

            # Check if already cached
            if await self.cache.get(cache_key):
                continue

            # Generate prediction and cache
            async def cache_prediction(feats):
                result = model.predict([feats])[0]
                await self.cache.set(
                    f"pred:{hash(tuple(feats))}",
                    {"prediction": float(result)}
                )

            tasks.append(cache_prediction(features))

        await asyncio.gather(*tasks)
        logger.info("cache_warmed", predictions=len(tasks))

@app.on_event("startup")
async def warm_cache():
    """Warm cache on startup"""
    popular_inputs = [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3],
        # ... more popular inputs
    ]

    warmer = CacheWarmer(cache)
    await warmer.warm_popular_predictions(popular_inputs)
```

---

## Phase 4: Batch Optimization

### Dynamic Batching

```python
# dynamic_batching.py
import asyncio
from collections import deque
from typing import List
import numpy as np

class DynamicBatcher:
    """Dynamically batch requests for inference"""

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: int = 10
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000
        self.queue = deque()
        self.futures = {}

    async def predict(self, features: List[float]) -> float:
        """Add prediction request to batch"""
        future = asyncio.Future()
        request_id = id(future)

        self.queue.append((request_id, features))
        self.futures[request_id] = future

        # Trigger batch processing
        if len(self.queue) >= self.max_batch_size:
            asyncio.create_task(self._process_batch())

        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return result
        finally:
            self.futures.pop(request_id, None)

    async def _process_batch(self):
        """Process accumulated batch"""
        if not self.queue:
            return

        # Collect batch
        batch = []
        request_ids = []

        while self.queue and len(batch) < self.max_batch_size:
            request_id, features = self.queue.popleft()
            batch.append(features)
            request_ids.append(request_id)

        # Run batch inference
        features_array = np.array(batch)
        predictions = model.predict(features_array)

        # Resolve futures
        for request_id, prediction in zip(request_ids, predictions):
            future = self.futures.get(request_id)
            if future and not future.done():
                future.set_result(float(prediction))

batcher = DynamicBatcher()

@app.post("/predict/batched")
async def predict_batched(request: PredictionRequest):
    """Prediction with dynamic batching"""
    result = await batcher.predict(request.features)
    return {"prediction": result}
```

### Batch Queue Processing

```python
# batch_queue.py
from fastapi import BackgroundTasks
import uuid

class BatchQueue:
    """Queue for batch processing"""

    def __init__(self):
        self.pending_requests = {}
        self.processing = False

    def add_request(self, request_id: str, features: List[float]):
        """Add request to queue"""
        self.pending_requests[request_id] = {
            "features": features,
            "status": "pending",
            "result": None
        }

    async def process_queue(self):
        """Process all queued requests"""
        if self.processing or not self.pending_requests:
            return

        self.processing = True

        try:
            # Collect all pending
            request_ids = list(self.pending_requests.keys())
            features_list = [
                self.pending_requests[rid]["features"]
                for rid in request_ids
            ]

            # Batch predict
            predictions = model.predict(np.array(features_list))

            # Update results
            for rid, pred in zip(request_ids, predictions):
                self.pending_requests[rid]["status"] = "completed"
                self.pending_requests[rid]["result"] = float(pred)

        finally:
            self.processing = False

batch_queue = BatchQueue()

@app.post("/predict/queue")
async def predict_queue(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Queue prediction for batch processing"""
    request_id = str(uuid.uuid4())

    batch_queue.add_request(request_id, request.features)
    background_tasks.add_task(batch_queue.process_queue)

    return {"request_id": request_id, "status": "queued"}

@app.get("/predict/queue/{request_id}")
async def get_queued_result(request_id: str):
    """Get result of queued prediction"""
    if request_id not in batch_queue.pending_requests:
        raise HTTPException(status_code=404, detail="Request not found")

    return batch_queue.pending_requests[request_id]
```

---

## Phase 5: GPU Optimization

### GPU Batch Processing

```python
# gpu_inference.py
import torch
from torch.utils.data import DataLoader, TensorDataset

class GPUInferenceEngine:
    """Optimized GPU inference"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, features: np.ndarray, batch_size: int = 64):
        """Batch prediction on GPU"""
        # Convert to tensor
        tensor_data = torch.FloatTensor(features)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        predictions = []

        for batch in dataloader:
            batch = batch[0].to(self.device)
            output = self.model(batch)
            predictions.append(output.cpu().numpy())

        return np.concatenate(predictions)

    @torch.no_grad()
    def predict_single(self, features: List[float]):
        """Single prediction (batched internally)"""
        tensor = torch.FloatTensor([features]).to(self.device)
        output = self.model(tensor)
        return output.cpu().numpy()[0]

# Initialize
gpu_engine = GPUInferenceEngine("models/model.pth")

@app.post("/predict/gpu")
async def predict_gpu(request: PredictionRequest):
    """GPU-accelerated prediction"""
    result = gpu_engine.predict_single(request.features)
    return {"prediction": result.tolist()}
```

### TensorRT Optimization

```python
# tensorrt_inference.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTEngine:
    """TensorRT optimized inference"""

    def __init__(self, engine_path: str):
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, input_data):
        """Run inference"""
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])

        # Run inference
        self.context.execute_v2(bindings=self.bindings)

        # Copy output to host
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])

        return self.outputs[0]['host']
```

---

## Phase 6: Load Testing

### Locust Load Test

```python
# locustfile.py
from locust import HttpUser, task, between
import random

class MLAPIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Login and get API key"""
        response = self.client.post("/token", json={
            "username": "testuser",
            "password": "testpass"
        })
        self.token = response.json()["access_token"]

    @task(3)
    def predict(self):
        """Test prediction endpoint"""
        features = [random.uniform(0, 10) for _ in range(4)]

        self.client.post(
            "/predict",
            json={"features": features},
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(1)
    def batch_predict(self):
        """Test batch prediction"""
        samples = [[random.uniform(0, 10) for _ in range(4)] for _ in range(10)]

        self.client.post(
            "/predict/batch",
            json={"samples": samples},
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/health")

# Run: locust -f locustfile.py --host=http://localhost:8000
```

### Custom Benchmark

```python
# benchmark.py
import asyncio
import time
import aiohttp
from statistics import mean, median, stdev

async def benchmark_endpoint(url: str, num_requests: int = 1000):
    """Benchmark API endpoint"""
    latencies = []

    async with aiohttp.ClientSession() as session:
        async def make_request():
            start = time.time()
            async with session.post(url, json={"features": [5.1, 3.5, 1.4, 0.2]}) as resp:
                await resp.json()
            return time.time() - start

        # Run concurrent requests
        tasks = [make_request() for _ in range(num_requests)]
        latencies = await asyncio.gather(*tasks)

    return {
        "total_requests": num_requests,
        "mean_latency_ms": mean(latencies) * 1000,
        "median_latency_ms": median(latencies) * 1000,
        "p95_latency_ms": sorted(latencies)[int(num_requests * 0.95)] * 1000,
        "p99_latency_ms": sorted(latencies)[int(num_requests * 0.99)] * 1000,
        "stdev_ms": stdev(latencies) * 1000,
        "throughput_rps": num_requests / sum(latencies)
    }

# Usage
if __name__ == "__main__":
    result = asyncio.run(benchmark_endpoint("http://localhost:8000/predict", 1000))
    print(json.dumps(result, indent=2))
```

---

## Phase 7: Profiling

### Python Profiling

```python
# profile_app.py
import cProfile
import pstats
from pstats import SortKey

def profile_prediction():
    """Profile prediction function"""
    profiler = cProfile.Profile()
    profiler.enable()

    # Run predictions
    for _ in range(1000):
        predict(PredictionRequest(features=[5.1, 3.5, 1.4, 0.2]))

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    """Function to profile memory usage"""
    large_list = [i for i in range(1000000)]
    return sum(large_list)
```

### Line Profiler

```python
# Use line_profiler for line-by-line profiling
# Install: pip install line_profiler
# Usage: kernprof -l -v script.py

@profile
def predict_optimized(features: List[float]):
    """Optimized prediction with profiling"""
    # Load model
    model = model_loader.load_sklearn_model("iris", "v1")

    # Prepare input
    features_array = np.array(features).reshape(1, -1)

    # Predict
    prediction = model.predict(features_array)[0]

    return prediction
```

---

## Phase 8: Response Compression

### Gzip Compression

```python
# compression.py
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/large-response")
async def large_response():
    """Response that benefits from compression"""
    return {
        "predictions": [
            {"id": i, "result": random.random()}
            for i in range(10000)
        ]
    }
```

---

## Best Practices

✅ Use async/await for I/O operations
✅ Implement connection pooling
✅ Cache frequently accessed data
✅ Batch requests when possible
✅ Leverage GPU for large models
✅ Profile before optimizing
✅ Load test regularly
✅ Monitor latency percentiles
✅ Use compression for large responses
✅ Optimize database queries

---

## Performance Metrics

### Target Latencies

- **P50**: < 50ms
- **P95**: < 200ms
- **P99**: < 500ms
- **Throughput**: > 1000 RPS

### Optimization Checklist

```python
# optimization_checklist.py
CHECKLIST = {
    "async_operations": True,
    "connection_pooling": True,
    "caching": True,
    "batch_processing": True,
    "gpu_acceleration": False,  # If applicable
    "compression": True,
    "load_tested": True,
    "profiled": True,
    "monitored": True,
    "autoscaling": True
}
```

---

**Performance Optimization mastered!** ⚡

**Congratulations!** You've completed the entire ML Serving APIs module!

**Next Module**: Capstone Projects (mod-010)
